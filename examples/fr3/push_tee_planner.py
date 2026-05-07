"""
T-block pushing task using PFMP with Panda robot.

Real world workspace is 53' 1/8" x 23' 1/4"
"""

import argparse
import os
import time
from typing import NamedTuple, Optional, Tuple

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
import zmq
from mujoco import mjx

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from geom_tee import STEM_HALF, CAP_HALF, CAP_CTR

import pfmp
from pfmp.run_batch import Trajectory
from pub_sub.protocol import Command, Response, TeeBelief
from pub_sub.planner import PlannerClient

from ik_cache import (
    IKCache, ik_from_cache, get_or_build_cache,
    WORKSPACE_XY_MIN, WORKSPACE_XY_MAX, IK_CACHE_RESOLUTION, EE_HEIGHT,
)

# ===============================
# Configuration
# ===============================

SCENE_XML_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "models", "mjx_t_push.xml")

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
    gripper_site_id: int


# ===============================
# Helper Functions
# ===============================


@jax.jit
def angle_wrap(a: jax.Array) -> jax.Array:
    return (a + jnp.pi) % (2.0 * jnp.pi) - jnp.pi


@jax.jit
def compute_error(th: PushParams, t_qpos: jax.Array) -> jax.Array:
    """Compute [x_err, y_err, yaw_err] for T-block. t_qpos = [x, y, yaw]."""
    return jnp.array([
        t_qpos[0] - th.goal_xy[0],
        t_qpos[1] - th.goal_xy[1],
        angle_wrap(t_qpos[2] - th.goal_angle),
    ])


# ===============================
# PFMP Task Functions
# ===============================


PROPOSAL_MIX   = 0.5   # probability of directed (vs random local) sample
STEP_LEN_VEL   = 0.05  # step size toward T per knot (m)
PUSH_OVERSHOOT = 1.0   # multiplier on step toward T centroid


def dynamics(x: mjx.Data, u: jax.Array, th: PushParams, t: float) -> Tuple[mjx.Data, jax.Array]:
    """Step MuJoCo with IK-resolved joint targets for the given EE waypoint."""
    target_qpos = ik_from_cache(u, th.ik_cache)
    ctrl = x.ctrl.at[:7].set(target_qpos)
    x_next = mjx.step(th.mjx_model, x.replace(ctrl=ctrl))
    return x_next, u


def running_cost(x: mjx.Data, u: jax.Array, th: PushParams, t: float) -> jax.Array:
    t_qpos    = x.qpos[T_QPOS_OFFSET:T_QPOS_OFFSET + 3]
    error     = compute_error(th, t_qpos)
    err_pos   = jnp.linalg.norm(error[:2])
    err_angle = jnp.abs(error[2])
    aw = th.angle_weight * jnp.clip(1.0 - err_pos / 0.25, 0.0, 1.0)
    return th.pos_weight * err_pos ** 2 + aw * err_angle ** 2


def terminal_cost(x: mjx.Data, th: PushParams) -> jax.Array:
    t_qpos    = x.qpos[T_QPOS_OFFSET:T_QPOS_OFFSET + 3]
    error     = compute_error(th, t_qpos)
    err_pos   = jnp.linalg.norm(error[:2])
    err_angle = jnp.abs(error[2])
    aw = th.angle_weight * jnp.clip(1.0 - err_pos / 0.25, 0.0, 1.0)
    return 10.0 * (th.pos_weight * err_pos + aw * err_angle)


def init_control(key: jax.Array, _x: mjx.Data, th: PushParams) -> jax.Array:
    """Initialize at current EE position (from warm-started mean) with small noise."""
    r_xy = th.mean[0]
    return r_xy + jax.random.normal(key, (2,)) * th.u_stdev * 0.1


def proposal(key: jax.Array, x: mjx.Data, u: jax.Array, th: PushParams, t: float) -> jax.Array:
    """Mixture: directed toward T centroid (50%) or random walk from current u (50%).
    Near the goal, noise is reduced to let the plan stabilize.
    """
    key_mix, key_a, key_b = jax.random.split(key, 3)
    t_xy      = x.qpos[T_QPOS_OFFSET:T_QPOS_OFFSET + 2]

    # err_pos = jnp.linalg.norm(t_xy - th.goal_xy)
    # stdev = th.u_stdev * jnp.where(err_pos < 0.05, 0.2, 1.0)

    direction = t_xy - u
    dist      = jnp.linalg.norm(direction) + 1e-7
    directed  = u + (direction / dist) * (jnp.minimum(dist, STEP_LEN_VEL) * PUSH_OVERSHOOT)
    directed  = directed + jax.random.normal(key_a, (2,)) * th.u_stdev
    random_loc = u + jax.random.normal(key_b, (2,)) * th.u_stdev * 2.0
    return jnp.where(jax.random.uniform(key_mix) < PROPOSAL_MIX, directed, random_loc)


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

    stem_half = np.array([STEM_HALF[0], STEM_HALF[1], thickness], dtype=float)
    bar_half  = np.array([CAP_HALF[0],  CAP_HALF[1],  thickness], dtype=float)
    bar_off_y = float(CAP_CTR[1])

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

    print(f"Loading model from: {xml_path}")
    cyl_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, "push_cylinder_visual")
    print(f"push_cylinder_visual geom id: {cyl_id} (−1 = not found)")
    return mj_model, mj_data


def simulate(                                                                                                                                                           
    init_belief: TeeBelief,                                                                                                                                             
    commands: list[Command],                                                                                                                                            
    push_params: PushParams,                                                                                                                                            
    steps_per_knot: int,                                                                                                                                                
    interactive: bool = True                                                                                                                                            
) -> Trajectory:                                                                                                                                                        
    pass   



def _apply_response_to_mj(mj_model, mj_data, response: dict) -> Optional[np.ndarray]:
    """Mirror the controller's latest response into mj_data.

    Updates the t-block (t_x/t_y/t_yaw from the latest belief) and the robot
    joints (q[:7] from the latest joint_q row), then runs forward kinematics.
    Returns the [x, y, yaw] belief mean used (or None if no belief was set).
    """
    mean = None
    beliefs = response.get("beliefs") or []
    if beliefs:
        m = np.asarray(beliefs[-1]["mean"], dtype=float)
        mj_data.qpos[T_QPOS_OFFSET]     = m[0]
        mj_data.qpos[T_QPOS_OFFSET + 1] = m[1]
        mj_data.qpos[T_QPOS_OFFSET + 2] = m[2]
        mean = m

    joint_q = response.get("joint_q") or []
    if joint_q:
        mj_data.qpos[:7] = np.asarray(joint_q[-1], dtype=float)

    mujoco.mj_forward(mj_model, mj_data)
    return mean


def debug_realtime(
    args: argparse.Namespace,
    mj_model: mujoco.MjModel,
    mj_data: mujoco.MjData,
    theta: Optional[PushParams] = None,
    goal_marker_fn: Optional[callable] = None,
) -> None:
    """Connect to the control server and visualize the live belief in MuJoCo.

    Seeds MuJoCo with the latest EKF belief from the controller, then opens a
    passive viewer that mirrors subsequent responses (t-block pose + robot
    joints) live. No motion commands are sent to the simulator or the robot.
    """
    cmd_addr = f"tcp://{args.server_host}:{args.cmd_port}"
    resp_addr = f"tcp://{args.server_host}:{args.resp_port}"

    print(f"[debug-realtime] Connecting to control server")
    print(f"  command socket  REQ → {cmd_addr}")
    print(f"  response socket SUB ← {resp_addr}")
    # NOTE: conflate=False is required because the controller publishes
    # multipart frames ([topic][json][rgb][depth]) and ZMQ's CONFLATE option
    # doesn't support multipart — it asserts (`!_more` in fq.cpp). We get
    # "latest only" semantics from recv_response_latest(drain=True) instead.
    planner = PlannerClient(
        command_addr=cmd_addr,
        response_addr=resp_addr,
        conflate=False,
    )

    try:
        # Initial recv: SUB sockets miss messages published before subscription
        # is established (slow joiner), so retry a handful of times.
        response = None
        last_err: Optional[Exception] = None
        for attempt in range(1, 21):
            try:
                response, _rgb, _depth = planner.recv_response_latest(drain=True)
                break
            except zmq.Again as e:
                last_err = e
                print(f"  [attempt {attempt}/20] no response yet; retrying...")
                time.sleep(0.25)

        if response is None:
            raise RuntimeError(f"No response from control server: {last_err}")

        mean = _apply_response_to_mj(mj_model, mj_data, response)
        if mean is None:
            raise RuntimeError("Control server response had no beliefs")

        print(
            f"[debug-realtime] Seeded MuJoCo t_block: "
            f"x={mean[0]:.3f} m, y={mean[1]:.3f} m, yaw={np.degrees(mean[2]):.1f}°"
        )

        joint_q = response.get("joint_q") or []
        if joint_q:
            latest_q_str = ", ".join(f"{q:+.3f}" for q in joint_q[-1])
            print(f"[debug-realtime] Latest robot joint_q: [{latest_q_str}]")

        # Switch SUB to non-blocking so the viewer loop never stalls on recv.
        planner.sub_sock.setsockopt(zmq.RCVTIMEO, 0)

        print("[debug-realtime] Launching passive viewer (close window to exit).")
        with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
            while viewer.is_running():
                try:
                    response, _rgb, _depth = planner.recv_response_latest(drain=True)
                except zmq.Again:
                    response = None

                if response is not None:
                    _apply_response_to_mj(mj_model, mj_data, response)

                if goal_marker_fn is not None and theta is not None:
                    with viewer.lock():
                        viewer.user_scn.ngeom = 0
                        goal_marker_fn(viewer.user_scn, theta)

                viewer.sync()
                time.sleep(0.03)  # ~33 Hz

    finally:
        planner.close()
        print("[debug-realtime] Viewer closed.")

def main():
    parser = argparse.ArgumentParser(description="T-block pushing with PFMP")
    parser.add_argument('--cache-resolution', type=int, default=IK_CACHE_RESOLUTION,
                        help='IK cache grid resolution')
    parser.add_argument('--debug-realtime', action='store_true')
    parser.add_argument('--server-host', default='localhost',
                        help='Hostname/IP of the controller server (real robot machine)')
    parser.add_argument('--cmd-port', type=int, default=5555,
                        help='ZMQ REQ port for sending commands to the controller')
    parser.add_argument('--resp-port', type=int, default=5556,
                        help='ZMQ SUB port for receiving responses from the controller')
    subparsers = parser.add_subparsers(dest="algorithm", required=True)
    subparsers.add_parser("ps", help="Predictive Sampling (Sequential Importance Sampling)")
    subparsers.add_parser("sir", help="Sequential Importance Resampling")
    subparsers.add_parser("realtime", help="Server Run")
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
    )

    # Task configuration
    horizon = 6.0
    num_knots = 8
    particles = 4096

    # Initial end-effector position
    init_xy = jnp.array([0.5, -0.2])
    q0 = ik_from_cache(init_xy, ik_cache)
    mj_data.qpos[:7] = np.array(q0)

    # Initialize mean trajectory: linear interp from EE to T-block position.
    gripper_site_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SITE, "gripper")
    mujoco.mj_fwdPosition(mj_model, mj_data)
    r_xy = jnp.array(mj_data.site_xpos[gripper_site_id, :2])

    t_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "t_block")
    t_world_xy = mj_data.xpos[t_id, :2]
    t_qpos_xy  = mj_data.qpos[T_QPOS_OFFSET:T_QPOS_OFFSET + 2]
    print(f"T world xy (xpos):  {t_world_xy}")
    print(f"T qpos[7:9]:        {t_qpos_xy}")
    t_xy = jnp.array(mj_data.qpos[T_QPOS_OFFSET:T_QPOS_OFFSET + 2])
    alphas = jnp.linspace(0.0, 1.0, num_knots)
    init_mean = (1.0 - alphas[:, None]) * r_xy + alphas[:, None] * t_xy

    # Create parameters
    theta = PushParams(
        mean=init_mean,
        mjx_model=mjx.put_model(mj_model),
        ik_cache=ik_cache,
        goal_xy=jnp.array([0.45, 0.15]),
        goal_angle=4*np.pi/3,
        pos_weight=15.0,
        angle_weight=1.5,
        u_stdev=0.025,
        gripper_site_id=gripper_site_id,
    )

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

    z_traj = float(EE_HEIGHT)
    traj_rgba = np.array([1.0, 0.5, 0.0, 1.0], dtype=np.float32)

    def goal_marker_callback(user_scn, theta):
        add_goal_T(user_scn, np.array(theta.goal_xy), float(theta.goal_angle),
                rgba=(0.2, 0.8, 0.2, 0.5))
        # Draw planned EE trajectory as orange line segments between knots.
        knots = np.array(theta.mean)  # (num_knots, 2)
        for k in range(len(knots) - 1):
            if user_scn.ngeom >= user_scn.maxgeom:
                break
            p0 = np.array([knots[k, 0],   knots[k, 1],   z_traj], dtype=np.float64)
            p1 = np.array([knots[k+1, 0], knots[k+1, 1], z_traj], dtype=np.float64)
            geom = user_scn.geoms[user_scn.ngeom]
            mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_LINE, 2.0, p0, p1)
            geom.rgba[:] = traj_rgba
            user_scn.ngeom += 1

    if args.debug_realtime:
        debug_realtime(args, mj_model, mj_data, theta, goal_marker_callback)
        return
    else:
        # Run interactive simulation
        pfmp.run_interactive(
            ctrl,
            theta,
            mj_model,
            mj_data,
            ik_interp=lambda target, theta, mjx_data: ik_from_cache(target, ik_cache),
            frequency=0.25,
            method=args.algorithm,
            fixed_camera_id=None,
            record_video=False,
            record_traj=False,
            max_step=8000,
            goal_marker_fn=goal_marker_callback
        )


if __name__ == "__main__":
    main()