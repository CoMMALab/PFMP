"""
T-block pushing task using PFMP with Panda robot.

Refactored to use the new PFMP API with Jet2 controls and cached IK.

Real world workspace is 53' 1/8" x 23' 1/4"
"""

import argparse
import os
import time
import pickle
from io import StringIO
from typing import NamedTuple, Tuple
from xml.etree import ElementTree as ET

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import jaxlie
import jaxls
import mujoco
import numpy as np
import pyroki as pk
import yourdfpy
from mujoco import mjx

import pfmp
from pfmp.interpolate import Jet2

# ===============================
# Configuration
# ===============================

ROBOT_URDF_PATH = os.path.expanduser("~/src/PFMP/models/panda_spherized.urdf")
SCENE_XML_PATH = os.path.expanduser("~/src/PFMP/models/mjx_t_push.xml")

# End-effector configuration
EE_ID = 12
EE_HEIGHT = 0.045

# Physical platform: 53-1/8" wide x 23-1/4" deep, 1-1/2" clearance from base
# 53.125" = 1.349 m (Y), 23.25" = 0.591 m (X), 1.5" = 0.038 m clearance
WORKSPACE_XY_MIN = jnp.array([0.1, -0.675])
WORKSPACE_XY_MAX = jnp.array([0.69, 0.675])
IK_CACHE_RESOLUTION = 64

# Fixed end-effector orientation (pointing down)
EE_ROTATION = jnp.array([
    [1., 0., 0.],
    [0., -1., 0.],
    [0., 0., -1.],
])

# Home joint configuration (from keyframe) — used as IK rest pose to keep
# null-space resolution consistent across the workspace.
HOME_Q = jnp.array([0., -0.785398, 0., -2.356194, 0., 1.570796, 0.785398])
# Null-space weights: heavy on joints 5-7 to prevent wrist flip / elbow swing.
# These must dominate the rest_cost in the null space (pose_cost ≈ 0 there).
REST_WEIGHT = jnp.array([0.0, 0.1, 0.1, 0.1, 1.0, 1.0, 1.0])

# T-block qpos offset in the MuJoCo state
T_QPOS_OFFSET = 7


# ===============================
# Robot Setup (Pyroki IK)
# ===============================


def load_panda_robot():
    """Load and configure the Panda robot for IK."""
    urdf = yourdfpy.URDF.load(ROBOT_URDF_PATH)

    # Convert prismatic joints to fixed
    xml_tree = urdf.write_xml()
    for joint in xml_tree.findall('.//joint[@type="prismatic"]'):
        joint.set("type", "fixed")
        for tag in ("axis", "limit", "dynamics"):
            child = joint.find(tag)
            if child is not None:
                joint.remove(child)

    xml_str = ET.tostring(xml_tree.getroot(), encoding="unicode")
    urdf = yourdfpy.URDF.load(StringIO(xml_str))

    return pk.Robot.from_urdf(urdf)


PANDA = load_panda_robot()


# ===============================
# IK Cache
# ===============================

class IKCache(NamedTuple):
    """Precomputed IK solutions on a 2D grid."""
    xy_min: jax.Array      # (2,) lower bounds
    xy_max: jax.Array      # (2,) upper bounds
    q_grid: jax.Array      # (Nx, Ny, 7) joint configurations
    z: float               # fixed height

@jax.jit
def ik_from_cache(xy: jax.Array, theta, _mjx_data=None) -> jax.Array:
    """Look up IK solution from cache using bilinear interpolation."""
    # Normalize to [0, 1]
    uv = (xy - theta.ik_cache.xy_min) / (theta.ik_cache.xy_max - theta.ik_cache.xy_min)
    uv = jnp.clip(uv, 0.0, 1.0)
    
    # Grid dimensions
    Nx, Ny = theta.ik_cache.q_grid.shape[:2]
    
    # Convert to grid coordinates
    gx = uv[0] * (Nx - 1)
    gy = uv[1] * (Ny - 1)
    
    # Integer indices
    ix = jnp.floor(gx).astype(jnp.int32)
    iy = jnp.floor(gy).astype(jnp.int32)
    ix = jnp.clip(ix, 0, Nx - 2)
    iy = jnp.clip(iy, 0, Ny - 2)
    
    # Fractional parts
    fx = gx - ix
    fy = gy - iy
    
    # Bilinear interpolation
    q00 = theta.ik_cache.q_grid[ix,     iy    ]
    q01 = theta.ik_cache.q_grid[ix,     iy + 1]
    q10 = theta.ik_cache.q_grid[ix + 1, iy    ]
    q11 = theta.ik_cache.q_grid[ix + 1, iy + 1]

    q0 = q00 * (1 - fy) + q01 * fy
    q1 = q10 * (1 - fy) + q11 * fy
    q = q0 * (1 - fx) + q1 * fx
    
    return q



@jax.jit
def mat4_from_pose(t: jax.Array, R: jax.Array) -> jax.Array:
    """Create 4x4 transformation matrix from translation and rotation."""
    return jnp.array([
        [R[0, 0], R[0, 1], R[0, 2], t[0]],
        [R[1, 0], R[1, 1], R[1, 2], t[1]],
        [R[2, 0], R[2, 1], R[2, 2], t[2]],
        [0., 0., 0., 1.]
    ])


def pose_from_mat4(mat4: jax.Array) -> Tuple[jax.Array, jax.Array]:
    """Extract translation and quaternion from 4x4 matrix."""
    if mat4.shape[-1] == 16:
        m = mat4.reshape((*mat4.shape[:-1], 4, 4))
    else:
        m = mat4
    R = m[:3, :3]
    t = m[:3, 3]
    q = jaxlie.SO3.from_matrix(R).wxyz
    return t.reshape(3,), q.reshape(4,)


def _ee_z_axis_residual(
    vals: jaxls.VarValues,
    robot: pk.Robot,
    joint_var,
    target_link_index: jax.Array,
    weight: float,
) -> jax.Array:
    """Residual: EE z-axis must point down [-z world]. Spin about z is free."""
    Ts = robot.forward_kinematics(vals[joint_var])
    R = jaxlie.SE3(Ts[..., target_link_index, :]).rotation().as_matrix()
    ee_z = R @ jnp.array([0., 0., 1.])
    return ((ee_z - jnp.array([0., 0., -1.])) * weight).flatten()

_ee_z_cost = jaxls.Cost.factory(_ee_z_axis_residual)


@jdc.jit
def make_ik_problem(
    target_mat4: jax.Array,
    pos_weight: float = 10.0,
    ori_weight: float = 5.0,
    limit_weight: float = 100.0
):
    """Create IK optimization problem."""
    target_p, target_q = pose_from_mat4(target_mat4)
    joint_var = PANDA.joint_var_cls(0)
    T_target = jaxlie.SE3.from_rotation_and_translation(
        jaxlie.SO3(target_q), target_p
    )
    factors = [
        pk.costs.pose_cost_analytic_jac(
            PANDA,
            joint_var,
            T_target,
            target_link_index=jnp.array(EE_ID),
            pos_weight=pos_weight,
            ori_weight=0.0,        # full ori replaced by z-axis cost below
        ),
        _ee_z_cost(PANDA, joint_var, jnp.array(EE_ID), ori_weight),
        pk.costs.limit_cost(
            PANDA,
            joint_var,
            weight=limit_weight,
        ),
        pk.costs.rest_cost(
            joint_var,
            HOME_Q,
            REST_WEIGHT * 0.1,
        ),
    ]
    return jaxls.LeastSquaresProblem(factors, [joint_var]).analyze(), joint_var


@jax.jit
def ik_7d_from_mat4(target_mat4: jax.Array) -> jax.Array:
    """Solve IK for target pose (cold start)."""
    prob, joint_var = make_ik_problem(target_mat4)
    sol = prob.solve(
        verbose=False,
        linear_solver="dense_cholesky",
        trust_region=jaxls.TrustRegionConfig(lambda_initial=1.),
    )
    return sol[joint_var]


@jax.jit
def ik_step_from_mat4(prev_q: jax.Array, target_mat4: jax.Array) -> Tuple[jax.Array, jax.Array]:
    """Solve IK with warm start."""
    prob, joint_var = make_ik_problem(target_mat4)
    sol = prob.solve(
        initial_vals=jaxls.VarValues.make([joint_var.with_value(prev_q)]),
        verbose=False,
        linear_solver="dense_cholesky",
        trust_region=jaxls.TrustRegionConfig(lambda_initial=1.),
    )
    q = sol[joint_var]
    return q, q


def ik_from_xy(xy: jax.Array, z: float = EE_HEIGHT) -> jax.Array:
    """Solve IK for xy position with fixed z and orientation."""
    xyz = jnp.array([xy[0], xy[1], z])
    mat4 = mat4_from_pose(xyz, EE_ROTATION)
    return ik_7d_from_mat4(mat4)


def ik_step_from_xy(prev_q: jax.Array, xy: jax.Array, z: float = EE_HEIGHT) -> jax.Array:
    """Solve IK for xy with warm start."""
    xyz = jnp.array([xy[0], xy[1], z])
    mat4 = mat4_from_pose(xyz, EE_ROTATION)
    q, _ = ik_step_from_mat4(prev_q, mat4)
    return q

CACHE_PATH = "./examples/ik_cache.pkl"

_CACHE_MAX_JUMP = np.pi / 3  # ~60° — larger jumps indicate a wrist flip

def validate_cache(cache: IKCache) -> bool:
    """Return True if the cache has no large joint-angle discontinuities."""
    q = np.array(cache.q_grid)  # (Nx, Ny, 7)
    def _wrapped_max(diff):
        # Wrap to [-π, π] so joints near ±π don't look like ~360° jumps
        d = (diff + np.pi) % (2 * np.pi) - np.pi
        return float(np.max(np.abs(d)))
    worst = max(_wrapped_max(np.diff(q, axis=0)),
                _wrapped_max(np.diff(q, axis=1)))
    if worst >= _CACHE_MAX_JUMP:
        print(f"  IK cache invalid: max neighbour jump = {np.degrees(worst):.1f}° "
              f"(limit {np.degrees(_CACHE_MAX_JUMP):.0f}°)")
        return False
    print(f"  IK cache valid: max neighbour jump = {np.degrees(worst):.1f}°")
    return True

def get_or_build_cache(**kwargs):
    if os.path.exists(CACHE_PATH):
        print("Loading cached IK...")
        with open(CACHE_PATH, 'rb') as f:
            cache = pickle.load(f)
        if validate_cache(cache):
            return cache
        print("  Cache failed validation — rebuilding...")
        os.remove(CACHE_PATH)

    cache = build_ik_cache(**kwargs)
    if not validate_cache(cache):
        print("WARNING: rebuilt cache still has discontinuities — check rest_cost weights")
    with open(CACHE_PATH, 'wb') as f:
        pickle.dump(cache, f)
    return cache

def build_ik_cache(
    xy_min: jax.Array,
    xy_max: jax.Array,
    resolution: int = 50,
    z: float = EE_HEIGHT,
    verbose: bool = True
) -> IKCache:
    """Build IK cache by solving IK on a grid."""
    xy_min = jnp.asarray(xy_min)
    xy_max = jnp.asarray(xy_max)
    
    xs = jnp.linspace(xy_min[0], xy_max[0], resolution)
    ys = jnp.linspace(xy_min[1], xy_max[1], resolution)
    
    if verbose:
        print(f"Building IK cache: {resolution}x{resolution} = {resolution**2} points")
        print(f"  x: [{float(xy_min[0]):.3f}, {float(xy_max[0]):.3f}]")
        print(f"  y: [{float(xy_min[1]):.3f}, {float(xy_max[1]):.3f}]")
        print(f"  z: {z:.3f} (fixed)")
    
    t0 = time.perf_counter()
    
    q_grid = np.zeros((resolution, resolution, 7))
    
    # Always warm-start from HOME_Q so rest_cost resolves the null space
    # consistently.  Raster warm-start (from previous neighbour) would lock in
    # the wrong null-space branch and rest_cost can't pull it back.
    home_q_np = np.array(HOME_Q)

    for i, x in enumerate(xs):
        if verbose and i % 10 == 0:
            print(f"  Row {i}/{resolution}...")

        for j, y in enumerate(ys):
            xy = jnp.array([float(x), float(y)])
            q = ik_step_from_xy(jnp.array(home_q_np), xy, z)
            q_grid[i, j] = np.array(q)
    
    elapsed = time.perf_counter() - t0
    
    if verbose:
        print(f"  Cache built in {elapsed:.2f}s ({resolution**2 / elapsed:.1f} solves/sec)")
    
    return IKCache(
        xy_min=xy_min, 
        xy_max=xy_max, 
        q_grid=jnp.array(q_grid), 
        z=z
    )


# ===============================
# Task Parameters
# ===============================


class PushParams(NamedTuple):
    """Parameters for the T-block pushing task."""
    mean: jax.Array
    mjx_model: mjx.Model
    ik_cache: IKCache
    goal_xy: jax.Array
    goal_angle: jax.Array
    pos_weight: float
    angle_weight: float
    u_stdev: float
    du_stdev: float


# ===============================
# Helper Functions
# ===============================


@jax.jit
def quat_yaw_wxyz(q: jax.Array) -> jax.Array:
    """Extract yaw angle from quaternion (w, x, y, z)."""
    w, x, y, z = q
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return jnp.arctan2(siny_cosp, cosy_cosp)


@jax.jit
def angle_wrap(a: jax.Array) -> jax.Array:
    """Wrap angle to (-pi, pi]."""
    return (a + jnp.pi) % (2.0 * jnp.pi) - jnp.pi


@jax.jit
def compute_error(th: PushParams, t_qpos: jax.Array) -> jax.Array:
    """Compute position and angle error for T-block."""
    t_xy = t_qpos[:2]
    t_quat = t_qpos[3:7]
    t_yaw = quat_yaw_wxyz(t_quat)

    x_err = t_xy[0] - th.goal_xy[0]
    y_err = t_xy[1] - th.goal_xy[1]
    ang_err = angle_wrap(t_yaw - th.goal_angle)

    return jnp.array([x_err, y_err, ang_err])


# ===============================
# PFMP Task Functions
# ===============================


def dynamics(x: mjx.Data, u: Jet2, th: PushParams, t: float) -> Tuple[mjx.Data, jax.Array]:
    """
    Dynamics function for the pushing task.

    Uses cached IK for fast lookup.
    """
    # Extract target position from Jet2
    target_xy = u.u  # (2,) array

    # Fast IK lookup from cache
    target_qpos = ik_from_cache(target_xy, th)

    # Step simulation with computed joint targets
    x_next = mjx.step(th.mjx_model, x.replace(ctrl=target_qpos))

    return x_next, target_xy


def running_cost(x: mjx.Data, u: Jet2, th: PushParams, t: float) -> jax.Array:
    """Running cost penalizing distance from goal."""
    t_qpos = x.qpos[T_QPOS_OFFSET:T_QPOS_OFFSET + 7]
    error = compute_error(th, t_qpos)

    pos_cost = th.pos_weight * (error[0] ** 2 + error[1] ** 2)
    angle_cost = th.angle_weight * error[2] ** 2

    return pos_cost + angle_cost


def terminal_cost(x: mjx.Data, th: PushParams) -> jax.Array:
    """Terminal cost for final state."""
    t_qpos = x.qpos[T_QPOS_OFFSET:T_QPOS_OFFSET + 7]
    error = compute_error(th, t_qpos)

    pos_cost = 10.0 * th.pos_weight * (error[0] ** 2 + error[1] ** 2)
    angle_cost = 10.0 * th.angle_weight * error[2] ** 2

    return pos_cost + angle_cost


def init_control(key: jax.Array, x: mjx.Data, th: PushParams) -> Jet2:
    """Initialize control near current T-block position."""
    key1, key2 = jax.random.split(key)

    t_xy = x.qpos[T_QPOS_OFFSET:T_QPOS_OFFSET + 2]
    u = t_xy + jax.random.normal(key1, (2,)) * th.u_stdev
    du = jax.random.normal(key2, (2,)) * th.du_stdev

    return Jet2(u=u, du=du)


def proposal(key: jax.Array, x: mjx.Data, u: Jet2, th: PushParams, t: float) -> Jet2:
    """Proposal distribution for control sampling."""
    key1, key2 = jax.random.split(key)

    u_next = u.u + jax.random.normal(key1, (2,)) * th.u_stdev
    du_next = u.du + jax.random.normal(key2, (2,)) * th.du_stdev

    return Jet2(u=u_next, du=du_next)


# ===============================
# Visualization Helpers
# ===============================


def add_goal_T(
    user_scn,
    goal_xy: np.ndarray,
    yaw_rad: float,
    rgba=(0., 0., 0., 0.25),
    z_shadow=1e-3,
    thickness=5e-4
):
    """Draw a translucent T-shaped goal marker."""
    if user_scn.ngeom + 2 > user_scn.maxgeom:
        return

    stem_half = np.array([0.0225, 0.0675, thickness], dtype=float)
    bar_half = np.array([0.0675, 0.0225, thickness], dtype=float)
    bar_off_y = 0.0675

    x, y = np.asarray(goal_xy, dtype=float)
    pos_world = np.array([x, y, float(z_shadow)], dtype=float)

    c, s = float(np.cos(yaw_rad)), float(np.sin(yaw_rad))
    R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=float)

    geom = user_scn.geoms[user_scn.ngeom]
    mujoco.mjv_initGeom(
        geom, mujoco.mjtGeom.mjGEOM_BOX,
        size=stem_half, pos=pos_world, mat=R.flatten(), rgba=rgba
    )
    user_scn.ngeom += 1

    bar_center = pos_world + R @ np.array([0.0, bar_off_y, 0.0], dtype=float)
    geom = user_scn.geoms[user_scn.ngeom]
    mujoco.mjv_initGeom(
        geom, mujoco.mjtGeom.mjGEOM_BOX,
        size=bar_half, pos=bar_center, mat=R.flatten(), rgba=rgba
    )
    user_scn.ngeom += 1


# ===============================
# Main
# ===============================


def load_scene(xml_path: str):
    """Load MuJoCo model and data."""
    mj_model = mujoco.MjModel.from_xml_path(xml_path)
    mj_data = mujoco.MjData(mj_model)

    key_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_KEY, "home")
    if key_id >= 0:
        mujoco.mj_resetDataKeyframe(mj_model, mj_data, key_id)

    return mj_model, mj_data


def main():
    parser = argparse.ArgumentParser(description="T-block pushing with PFMP")
    parser.add_argument('--cache-resolution', type=int, default=IK_CACHE_RESOLUTION,
                        help='IK cache grid resolution')
    subparsers = parser.add_subparsers(dest="algorithm", required=True)
    subparsers.add_parser("ps", help="Predictive Sampling (Sequential Importance Sampling)")
    subparsers.add_parser("sir", help="Sequential Importance Resampling")
    args = parser.parse_args()

    # Load scene
    mj_model, mj_data = load_scene(SCENE_XML_PATH)

    # Build IK cache (one-time cost)
    print("Building IK cache...")
    ik_cache = get_or_build_cache(
        xy_min=WORKSPACE_XY_MIN,
        xy_max=WORKSPACE_XY_MAX,
        resolution=args.cache_resolution,
        z=EE_HEIGHT,
        verbose=True
    )

    # Task configuration
    horizon = 4.0
    num_knots = 8
    particles = 4096

    # Create parameters
    theta = PushParams(
        mean=jnp.zeros((num_knots, 2)),
        mjx_model=mjx.put_model(mj_model),
        ik_cache=ik_cache,
        goal_xy=jnp.array([0.35, 0.15]),
        goal_angle=np.pi/3,
        pos_weight=3.0,
        angle_weight=0.5,
        u_stdev=0.05,
        du_stdev=0.1,
    )

    # Initial end-effector position
    init_xy = jnp.array([0.5, -0.2])

    # Get initial joint config from cache
    q0 = ik_from_cache(init_xy, theta)
    mj_data.qpos[:7] = np.array(q0)

    # Create PFMP controller
    ctrl = pfmp.PFMP(
        dynamics,
        running_cost,
        terminal_cost,
        init_control,
        proposal,
        horizon=horizon,
        num_knots=num_knots,
        particles=particles,
        dt=mj_model.opt.timestep,
        ess_resample_thresh=0.5,
    )

    def goal_marker_callback(user_scn, theta):
        add_goal_T(user_scn, np.array(theta.goal_xy), float(theta.goal_angle),
                rgba=(0.2, 0.8, 0.2, 0.5))

    # Run interactive simulation
    pfmp.run_interactive(
        ctrl,
        theta,
        mj_model,
        mj_data,
        ik_interp=ik_from_cache,
        frequency=0.5,
        method=args.algorithm,
        fixed_camera_id=None,
        record_video=False,
        record_traj=False,
        max_step=8000,
        goal_marker_fn=goal_marker_callback
    )


if __name__ == "__main__":
    main()