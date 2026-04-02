"""
T-block pushing task using PFMP with Panda robot.

Refactored to use the new PFMP API with Jet2 controls and cached IK.

Real world workspace is 53' 1/8" x 23' 1/4"
"""

import argparse
import os
from typing import NamedTuple, Tuple

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from mujoco import mjx

import pfmp
from pfmp.interpolate import Jet2

from ik_cache import (
    IKCache, ik_from_cache, get_or_build_cache,
    WORKSPACE_XY_MIN, WORKSPACE_XY_MAX, IK_CACHE_RESOLUTION, EE_HEIGHT,
)

# ===============================
# Configuration
# ===============================

SCENE_XML_PATH = os.path.expanduser("~/src/PFMP/models/mjx_t_push.xml")

# T-block qpos offset in the MuJoCo state
T_QPOS_OFFSET = 7


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