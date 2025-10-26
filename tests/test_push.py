import os
import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import yourdfpy
import mujoco
import time
import os, time
import numpy as np
import jaxls
import jaxlie
import pyroki as pk


from xml.etree import ElementTree as ET
from io import StringIO
from genjax import gen, mv_normal
from jax.scipy.spatial.transform import Rotation
from mujoco import mjx
from typing import NamedTuple, Tuple
from mujoco.viewer import launch_passive
from pfmp import PFMP, make_transition_model


# ===============================
# Robot setup
# ===============================

ROBOT_URDF_PATH = "/home/austin/src/panda_scripts/assets/panda_spherized.urdf"

# Load URDF
urdf = yourdfpy.URDF.load(ROBOT_URDF_PATH)

# Convert prismatic joints to fixed (like in IK-Beam benchmark)
xml_tree = urdf.write_xml()
for joint in xml_tree.findall('.//joint[@type="prismatic"]'):
    joint.set("type", "fixed")
    for tag in ("axis", "limit", "dynamics"):
        child = joint.find(tag)
        if child is not None:
            joint.remove(child)

# Reload robot with modified XML
xml_str = ET.tostring(xml_tree.getroot(), encoding="unicode")
urdf = yourdfpy.URDF.load(StringIO(xml_str))

# Build Pyroki robot object
PANDA = pk.Robot.from_urdf(urdf)
EE_ID = 12
T_QPOS_OFFSET = 9

class PParams(NamedTuple):
    mjx_model: mjx.Model
    t_qpos_offset: int
    goal_xy: jax.Array
    goal_angle: jax.Array
    cost_factors: jax.Array
    u_stdev: float
    du_stdev: float
    ddu_stdev: float

class StateTracker(NamedTuple):
    qpos: jax.Array
    txyz: jax.Array

def load_mj(name: str):
    os.chdir("tests/mjx_mod/assets")
    with open(f"{name}.xml", 'r') as f:
        xml = f.read()

    mj_model = mujoco.MjModel.from_xml_string(xml)
    mj_data = mujoco.MjData(mj_model)

    key_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_KEY, "home")
    mujoco.mj_resetDataKeyframe(mj_model, mj_data, key_id)
    return (mj_model, mj_data)

# Debug draw: red line + green target
def add_line(user_scn, p0, p1, radius=0.004, color=(1,0,0,0.7)):
    p0 = np.asarray(p0); p1 = np.asarray(p1)
    mid = 0.5*(p0+p1)
    v = p1 - p0
    L = np.linalg.norm(v) + 1e-12
    z = v / L
    a = np.array([1.0,0.0,0.0]) if abs(z[0]) < 0.9 else np.array([0.0,1.0,0.0])
    x = a - np.dot(a, z)*z; x /= (np.linalg.norm(x)+1e-12)
    y = np.cross(z, x)
    R = np.stack([x,y,z], axis=1)
    geom = user_scn.geoms[user_scn.ngeom]
    mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_CYLINDER,
                        size=[radius, 0.5*L, 0.0], pos=mid, mat=R.flatten(), rgba=color)
    user_scn.ngeom += 1

def add_sphere(scene, pos, radius=0.0025, rgba=(0,1,0,0.9)):
    if scene.ngeom >= scene.maxgeom: return
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_SPHERE,
                        size=[radius,0,0], pos=pos, mat=np.eye(3).flatten(), rgba=rgba)
    scene.ngeom += 1

def add_goal_T(user_scn,
               goal_xy,
               yaw_rad,
               rgba=(0., 0., 0., 0.25),
               z_shadow=1e-3,
               thickness=5e-4):
    """
    Draw a translucent T-shaped 'shadow' using the user scene (passive viewer).
    Geometry matches the model's t_stem/t_bar XY half-sizes; Z is made very thin.

    Args:
        user_scn: viewer.user_scn
        goal_xy: (x, y) world position for the T body origin (matching t_block frame)
        yaw_rad: rotation about +Z (radians)
        rgba: color/alpha of the shadow
        z_shadow: world Z to place the shadow (slightly above plane to avoid z-fighting)
        thickness: half-thickness along Z for the two box geoms (very small)
    """
    # Guard: need 2 geoms available
    if user_scn.ngeom + 2 > user_scn.maxgeom:
        return

    # ---- Match the model's T dimensions (half-sizes) ----
    # stem: centered at body origin
    stem_half = np.array([0.0225, 0.0675, thickness], dtype=float)
    # bar: centered at +y by 0.0675 in the T local frame
    bar_half  = np.array([0.0675, 0.0225, thickness], dtype=float)
    bar_off_y = 0.0675

    # ---- Pose ----
    x, y = np.asarray(goal_xy, dtype=float)
    pos_world = np.array([x, y, float(z_shadow)], dtype=float)

    c, s = float(np.cos(yaw_rad)), float(np.sin(yaw_rad))
    R = np.array([[ c, -s, 0.0],
                  [ s,  c, 0.0],
                  [0.0, 0.0, 1.0]], dtype=float)

    # ---- Stem (centered at body origin) ----
    geom = user_scn.geoms[user_scn.ngeom]
    mujoco.mjv_initGeom(
        geom,
        mujoco.mjtGeom.mjGEOM_BOX,
        size=stem_half,
        pos=pos_world,
        mat=R.flatten(),
        rgba=rgba
    )
    user_scn.ngeom += 1

    # ---- Bar (offset +y in local frame, then rotated to world) ----
    bar_center_world = pos_world + R @ np.array([0.0, bar_off_y, 0.0], dtype=float)

    geom = user_scn.geoms[user_scn.ngeom]
    mujoco.mjv_initGeom(
        geom,
        mujoco.mjtGeom.mjGEOM_BOX,
        size=bar_half,
        pos=bar_center_world,
        mat=R.flatten(),
        rgba=rgba
    )
    user_scn.ngeom += 1

# ---------------------- IK (pose -> q*) ----------------------

@jax.jit
def mat4_from_pose(t: jax.Array, R: jax.Array) -> jax.Array:
    return jnp.array([
        [R[0,0], R[0,1], R[0,2], t[0]],
        [R[1,0], R[1,1], R[1,2], t[1]],
        [R[2,0], R[2,1], R[2,2], t[2]],
        [0.    , 0.    , 0.    , 1.  ]
    ])

@jax.jit
def quat_wxyz_from_R(R: jax.Array):
    return Rotation.from_matrix(R).as_quat(scalar_first=True)

@jax.jit
def pose_from_mat4(mat4: jax.Array) -> Tuple[jax.Array, jax.Array]:
    if mat4.shape[-1] == 16:
        m = jnp.reshape(mat4, mat4.shape[:-1] + (4, 4))
    else:
        m = mat4
    R = m[..., :3, :3]
    t = m[..., :3, 3]
    return t, quat_wxyz_from_R(R)

@jdc.jit
def make_ik_problem(
        target_mat4: jax.Array,
        pos_weight: float = 50.0,
        ori_weight: float = 10.0,
        weight: float = 100.0
    ):
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
            ori_weight=ori_weight
        ),
        pk.costs.limit_cost(
            PANDA,
            joint_var,
            weight=weight,
        ),
    ]
    return jaxls.LeastSquaresProblem(factors, [joint_var]).analyze(), joint_var

@jax.jit
def ik_7d_from_mat4(target_mat4: jax.Array) -> jax.Array:
    prob0, joint_var0 = make_ik_problem(target_mat4)
    sol0 = prob0.solve(
        verbose=False,
        linear_solver="dense_cholesky",
        trust_region=jaxls.TrustRegionConfig(lambda_initial=1.0),
    )
    return sol0[joint_var0]

@jax.jit
def step_ik_7d_from_mat4(prev_q: jax.Array, target_mat4: jax.Array) -> tuple[jax.Array,jax.Array]:
    prob, joint_var = make_ik_problem(target_mat4)
    sol = prob.solve(
        initial_vals=jaxls.VarValues.make([joint_var.with_value(prev_q)]),
        verbose=False,
        linear_solver="dense_cholesky",
        trust_region=jaxls.TrustRegionConfig(lambda_initial=1.0),
    )
    q = sol[joint_var]
    return q, q

@jax.jit
def ik_7d_from_mat4s(target_mat4s: jax.Array) -> jax.Array:
    q0 = ik_7d_from_mat4(target_mat4s[0])
    _, qs = jax.lax.scan(step_ik_7d_from_mat4, q0, target_mat4s[1:])
    return jnp.concatenate([q0[None, ...], qs])

class ParParams(NamedTuple):
    # goal: jax.Array
    # obstacles: jax.Array
    # collision_multiplier: float
    u_stdev: float
    du_stdev: float
    ddu_stdev: float

def init_mjx(mj_model, mj_data):
    mjx_model = mjx.put_model(mj_model)
    x0 = mjx.put_data(mj_model, mj_data)

    return x0, PParams(
        mjx_model,
        t_qpos_offset=9,
        goal_xy=jnp.array([0.4, 0.]),
        goal_angle=3*jnp.pi/4,
        cost_factors=jnp.array([3.0, 3.0, 0.5]),
        u_stdev=0.0125,
        du_stdev=0.01,
        ddu_stdev=0.1
    )


HEIGHT = 0.06

@gen
def proposal_fn(x: mjx.Data, u: jax.Array, th: ParParams, t: int) -> jax.Array:
    nu = u.shape[1]  # eg. Cartesian 2D/3D 
    pred_x = mv_normal(x.qpos[T_QPOS_OFFSET:T_QPOS_OFFSET+2], jnp.eye(nu) * theta.u_stdev) @ "u"
    pred_v = mv_normal(u[1], jnp.eye(nu) * th.du_stdev) @ "du"
    return jnp.array([pred_x, pred_v])  # cubic

@jax.jit
def dynamics_fn(x, u, th, t) -> Tuple[jax.Array,jax.Array]:
    target_xyz = jnp.array([u[0,0], u[0,1], HEIGHT])
    target_R = jnp.array([
        [1.,  0.,  0.],
        [0., -1.,  0.],
        [0.,  0., -1.],
    ])
    target_eefk = mat4_from_pose(target_xyz, target_R)
    target_qpos = step_ik_7d_from_mat4(x.qpos[:7], target_eefk)[0]
    x = x.replace(ctrl = target_qpos)
    x_next = mjx.step(th.mjx_model, x)
    return x_next, StateTracker(qpos=x.qpos, txyz=target_xyz)

@jax.jit
def _quat_yaw_wxyz(q):
    # q = (w, x, y, z) per MuJoCo
    w, x, y, z = q
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return jnp.arctan2(siny_cosp, cosy_cosp)

@jax.jit
def _angle_wrap(a):
    # maps to (-pi, pi]
    return (a + jnp.pi) % (2.0 * jnp.pi) - jnp.pi

def error_fn(th: PParams, t_qpos: jax.Array) -> jax.Array:
    t_xy    = t_qpos[:2]
    t_quat  = t_qpos[3:7]
    t_yaw   = _quat_yaw_wxyz(t_quat)
    x_err, y_err = t_xy - th.goal_xy
    ang_err = _angle_wrap(t_yaw - th.goal_angle)
    return jnp.array([x_err, y_err, ang_err])

@jax.jit
def cost_fn(x: mjx.Data, _u, th: PParams, _t) -> float:
    # Free joint layout: [px, py, pz, qw, qx, qy, qz]
    error = error_fn(th, x.qpos[T_QPOS_OFFSET:T_QPOS_OFFSET+7])
    return jnp.dot(th.cost_factors, error ** 2)


if __name__ == "__main__":
    K = 1000
    H = 20
    L = 80
    RHO = 0.8
    (mj_model, mj_data) = load_mj("mjx_t_push")


    EE_SITE = "gripper"
    ee_sid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SITE, EE_SITE)

    u0 = jnp.array([
        [0.4, -0.3],  # ( x0,  y0)
        [0.0,  0.2]   # (vx0, vy0)
    ])
    
    # p0 = mj_data.site_xpos[ee_sid].copy()
    # R0 = mj_data.site_xmat[ee_sid].reshape(3,3).copy()
    txyz = jnp.array([u0[0,0], u0[0,1], HEIGHT])
    tR = jnp.array([
        [1.,  0.,  0.],
        [0., -1.,  0.],
        [0.,  0., -1.],
    ])
    q0 = ik_7d_from_mat4(mat4_from_pose(txyz, tR))
    mj_data.qpos[:7] = q0
    (x0, theta) = init_mjx(mj_model, mj_data)
    theta = theta._replace(goal_xy = jnp.array([-0.4, 0.3]), goal_angle=-0.2)
    # theta = theta._replace(goal_xy = jnp.array([0.4, -0.3]), goal_angle=-0.2)
    qpos = jnp.concat((q0,mj_data.qpos[7:]))
    x0 = x0.replace(qpos=qpos)

    key = jax.random.key(42)
    pfmp = PFMP(make_transition_model(proposal_fn, cost_fn, dynamics_fn, l=L))
    # trajs, _ = pfmp.run_sis(key, x0, u0, theta, H, K)
    # trajs = trajs.reshape(K,-1,3)

    print("starting")
    t0 = time.time()
    states, ancestors = pfmp.run_sir(key, x0, u0, theta, H, K, RHO)
    trajs = pfmp.choose_paths(states, ancestors, jnp.arange(K))
    trajs = jax.tree.map(lambda x: x.reshape(K,-1,x.shape[-1]), trajs)
    besti = jnp.argmin(
        jnp.array([
            jnp.dot(
                theta.cost_factors,
                error_fn(theta, tau[T_QPOS_OFFSET:T_QPOS_OFFSET+7]) ** 2
            )
        for tau in trajs.qpos[-1]])
    )
    traj = jax.tree.map(lambda x: x[besti], trajs)
    # traj = jax.tree.map(lambda x: x.reshape(-1,x.shape[-1]), traj)
    print(f"JIT time = {time.time() - t0}")
    # t0 = time.time()

    # states, ancestors = jax.block_until_ready(pfmp.run_sir(key, x0, u0, theta, H, K, RHO))
    # traj = jax.block_until_ready(pfmp.choose_path(states, ancestors, 0))
    # print(f"Run time = {time.time() - t0}")
    # t0 = time.time()

    # print(traj.qpos)

    num_samples = 20

    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  # needed for 3D projection

    # trajs.shape = (20, 960, 3)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    for i in range(trajs.txyz.shape[0]):
        ax.plot(
            trajs.txyz[i, :, 0],  # x
            trajs.txyz[i, :, 1],  # y
            trajs.txyz[i, :, 2],  # z
            color="m",
            alpha=0.5,
        )

    ax.plot(
        traj.txyz[:, 0],  # x
        traj.txyz[:, 1],  # y
        traj.txyz[:, 2],  # z
        color="green",
        alpha=1.0
    )

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    plt.show()

    t = 0
    with launch_passive(
            mj_model,
            data=mj_data,
            show_left_ui=False,
            show_right_ui=False,
        ) as viewer:
        while True:
            # txyz = traj[t]
            # mj_data.ctrl[:7] = step_ik_7d_from_mat4(mj_data.qpos[:7], mat4_from_pose(txyz, R_des))[0]
            txyz = traj.txyz[t]
            mj_data.qpos = traj.qpos[t]
            mujoco.mj_step(mj_model, mj_data)

            print(f"errors = {error_fn(theta, mj_data.qpos[T_QPOS_OFFSET:T_QPOS_OFFSET+7])}")

            # Clear previous markers
            viewer.user_scn.ngeom = 0

            # Add goal marker
            add_goal_T(viewer.user_scn, theta.goal_xy, theta.goal_angle, rgba = (0, 0, 1, 0.9))

            # Add knot markers
            for n in range(0,H*L,L):
                add_sphere(viewer.user_scn, traj.txyz[n], radius=0.01)

            # Add interpolated markers
            # for n in range(0,H*L):
            #     add_sphere(viewer.user_scn, traj.txyz[n], radius=0.0025)
            
            # Add continuous PBC marker
            add_sphere(viewer.user_scn, txyz, radius=0.02)

            # Add continuous PBC error line
            xpos = mj_data.site_xpos[ee_sid].copy()
            add_line(
                viewer.user_scn,
                xpos,
                txyz,
                radius=0.005,
                color=[1, 0, 0, 0.5]
            )
            
            time.sleep(0.01)
            viewer.sync()
            t += 1