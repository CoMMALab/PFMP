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
STEP_LEN_VEL   = 0.02  # step size toward T per knot (m)
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


# Hardcoded debug-realtime motion defaults: y-sweep at x=SWEEP_X.
# Pick SWEEP_X so the sweep line passes through the tee that the EKF reports;
# otherwise the robot just moves through empty space and neither sim nor real
# will show any tee motion.
SWEEP_X = 0.5
SWEEP_Y_LO = -0.15
SWEEP_Y_HI = 0.15
SWEEP_N_WP = 8


def _make_sweep_waypoints(
    sweep_x: float = SWEEP_X,
    y_lo: float = SWEEP_Y_LO,
    y_hi: float = SWEEP_Y_HI,
    n_wp: int = SWEEP_N_WP,
) -> np.ndarray:
    """Hardcoded EE waypoints: linear y-sweep across the workspace at x=sweep_x."""
    ys = np.linspace(y_lo, y_hi, n_wp)
    return np.stack([np.full_like(ys, sweep_x), ys], axis=1)


def _exec_motion_sim(
    mj_model: mujoco.MjModel,
    mj_data: mujoco.MjData,
    ik_cache,
    waypoints: np.ndarray,
    steps_per_wp: int = 500,
    theta: Optional[PushParams] = None,
    goal_marker_fn: Optional[callable] = None,
) -> np.ndarray:
    """Drive MuJoCo through the waypoints and record t-block pose at each.

    Each waypoint sets the position-controller targets to ik_from_cache(xy) and
    steps physics for `steps_per_wp` ticks. Returns an (N, 3) array of
    [x, y, yaw] for the t-block at the end of each waypoint.
    """
    traj = []
    with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
        for i, wp in enumerate(waypoints):
            target_q = np.asarray(ik_from_cache(jnp.asarray(wp), ik_cache))
            mj_data.ctrl[:7] = target_q

            for step in range(steps_per_wp):
                mujoco.mj_step(mj_model, mj_data)
                if (step % 5) == 0:
                    if goal_marker_fn is not None and theta is not None:
                        with viewer.lock():
                            viewer.user_scn.ngeom = 0
                            goal_marker_fn(viewer.user_scn, theta)
                    viewer.sync()
                if not viewer.is_running():
                    break

            tee = mj_data.qpos[T_QPOS_OFFSET:T_QPOS_OFFSET + 3].copy()
            traj.append(tee)
            print(
                f"  [sim ] wp {i+1:2d}/{len(waypoints)}: "
                f"target_xy=({wp[0]:+.3f}, {wp[1]:+.3f})  "
                f"tee=({tee[0]:+.3f}, {tee[1]:+.3f}, {np.degrees(tee[2]):+6.1f}°)"
            )

            if not viewer.is_running():
                print("  [sim ] viewer closed; stopping early.")
                break

    return np.asarray(traj)


def _reconnect_req(planner: PlannerClient, command_timeout_ms: int = 5000) -> None:
    """Recreate the REQ socket on the planner.

    After a recv timeout (`zmq.Again`) the REQ state machine is stuck — it
    expects a recv before it can send again. The only clean recovery is to
    close and re-open the socket.
    """
    planner.cmd_sock.close(linger=0)
    s = planner.ctx.socket(zmq.REQ)
    s.setsockopt(zmq.RCVTIMEO, command_timeout_ms)
    s.setsockopt(zmq.SNDTIMEO, command_timeout_ms)
    s.connect(planner.command_addr)
    planner.cmd_sock = s


def _send_with_retry(
    planner: PlannerClient,
    cmd: Command,
    label: str,
    max_attempts: int = 30,
    retry_pause_s: float = 1.0,
    command_timeout_ms: int = 5000,
) -> dict:
    """send_command, but reconnect-and-retry on zmq.Again.

    Each attempt waits up to command_timeout_ms for the ack. If the controller
    is busy (long motion + perception + publish + recv), the ack might miss
    one window but land on the next. Worst case (controller crash + restart),
    reconnecting also lets a freshly-launched controller take over.
    """
    for attempt in range(1, max_attempts + 1):
        try:
            return planner.send_command(cmd)
        except zmq.Again:
            print(
                f"  [real] {label}: send_command timed out "
                f"(attempt {attempt}/{max_attempts}); reconnecting REQ and retrying…"
            )
            _reconnect_req(planner, command_timeout_ms=command_timeout_ms)
            time.sleep(retry_pause_s)
    raise RuntimeError(f"send_command failed after {max_attempts} attempts")


def _exec_motion_real(
    planner: PlannerClient,
    waypoints: np.ndarray,
    motion_timeout_s: float = 15.0,
    ack_settle_s: float = 0.1,
    send_max_attempts: int = 30,
    send_retry_pause_s: float = 1.0,
) -> np.ndarray:
    """Send each waypoint to the controller and record the post-motion belief.

    Sequencing inside the controller's main loop:
        … perception → publish(seq=N) → recv_command → reply_command → execute_xy …
    So when our send_command() returns with the ack, the publish at seq=N has
    already happened. While execute_xy is blocking, no further publishes occur.
    We wait briefly for that pre-motion publish to settle into our SUB queue,
    drain to capture seq=N, then wait for a response with seq > N — that's the
    next iteration's publish, which runs after execute_xy returns.

    If the ack doesn't arrive within the REQ timeout, we reconnect and retry
    (up to send_max_attempts times) before giving up on a waypoint.
    """
    traj = []
    for i, wp in enumerate(waypoints):
        cmd = Command(xy=wp.tolist())
        label = f"wp {i+1:2d}/{len(waypoints)}"
        try:
            ack = _send_with_retry(
                planner, cmd, label,
                max_attempts=send_max_attempts,
                retry_pause_s=send_retry_pause_s,
            )
        except RuntimeError as e:
            print(f"  [real] {label}: {e}; skipping remaining {len(waypoints) - i} waypoints.")
            for _ in range(i, len(waypoints)):
                traj.append(np.full(3, np.nan))
            break
        if not ack.get("accepted", False):
            print(f"  [real] wp {i+1}: command rejected: {ack}")
            traj.append(np.full(3, np.nan))
            continue
        print(f"  [real] wp {i+1:2d}/{len(waypoints)}: sent xy=({wp[0]:+.3f}, {wp[1]:+.3f})")

        # Capture the pre-motion seq AFTER the ack — the most recent publish at
        # this point is the one that ran in the controller's iteration just
        # before reply_command. Any seq strictly greater is post-motion.
        time.sleep(ack_settle_s)
        pre_motion_seq = -1
        try:
            pre, _rgb, _depth = planner.recv_response_latest(drain=True)
            pre_motion_seq = int(pre.get("seq", -1))
        except zmq.Again:
            pass

        belief = None
        t0 = time.time()
        while time.time() - t0 < motion_timeout_s:
            try:
                response, _rgb, _depth = planner.recv_response_latest(drain=True)
            except zmq.Again:
                time.sleep(0.05)
                continue
            seq = int(response.get("seq", -1))
            if seq > pre_motion_seq:
                beliefs = response.get("beliefs") or []
                if beliefs:
                    belief = np.asarray(beliefs[-1]["mean"], dtype=float)
                break
            time.sleep(0.05)

        if belief is None:
            print(f"  [real] wp {i+1}: no fresh belief within {motion_timeout_s}s")
            belief = np.full(3, np.nan)
        else:
            print(
                f"  [real] wp {i+1:2d}/{len(waypoints)}: "
                f"tee=({belief[0]:+.3f}, {belief[1]:+.3f}, {np.degrees(belief[2]):+6.1f}°)"
            )
        traj.append(belief)

    return np.asarray(traj)


def _print_trajectory(label: str, waypoints: np.ndarray, traj: np.ndarray) -> None:
    print(f"\n=== {label} t-block trajectory ===")
    print(f"{'wp':>3}  {'cmd_x':>8} {'cmd_y':>8}    {'tee_x':>8} {'tee_y':>8} {'tee_yaw':>9}")
    for i, (wp, p) in enumerate(zip(waypoints, traj)):
        yaw_str = "    nan  " if np.isnan(p[2]) else f"{np.degrees(p[2]):+8.1f}°"
        print(
            f"{i+1:>3}  {wp[0]:+8.3f} {wp[1]:+8.3f}    "
            f"{p[0]:+8.3f} {p[1]:+8.3f} {yaw_str}"
        )


def debug_realtime(
    args: argparse.Namespace,
    mj_model: mujoco.MjModel,
    mj_data: mujoco.MjData,
    theta: PushParams,
    goal_marker_fn: Optional[callable] = None,
) -> None:
    """Compare a hardcoded EE sweep in sim vs. on the real robot.

    Flow:
      1. Connect to the control server, fetch the latest EKF belief, and seed
         MuJoCo with it (t-block pose + robot joints).
      2. Run a hardcoded y-sweep in MuJoCo, recording the t-block pose at each
         waypoint.
      3. Run the same sweep on the real robot via ZMQ, recording the EKF belief
         after each motion completes.
      4. Print both trajectories side-by-side.
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
        # Default 1 s is tight: the controller acks before motion starts, but
        # under load the REQ round-trip can spike. Bump to 5 s.
        command_timeout_ms=5000,
    )

    try:
        # ── 1. Initial belief: seed MuJoCo ───────────────────────────────────
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

        waypoints = _make_sweep_waypoints(
            sweep_x=args.sweep_x,
            y_lo=args.sweep_y_lo,
            y_hi=args.sweep_y_hi,
            n_wp=args.sweep_n_wp,
        )
        print(
            f"[debug-realtime] Sweep: {len(waypoints)} waypoints, "
            f"x={args.sweep_x:.3f}, y={args.sweep_y_lo:+.3f}→{args.sweep_y_hi:+.3f}"
        )
        if abs(args.sweep_x - mean[0]) > 0.10:
            print(
                f"[debug-realtime] WARNING: sweep_x={args.sweep_x:.3f} is "
                f"{abs(args.sweep_x - mean[0])*100:.1f} cm from the seeded tee_x="
                f"{mean[0]:.3f}; the EE will pass through empty space and the "
                f"tee won't be pushed. Consider --sweep-x close to tee_x."
            )

        # ── 2. Sim execution ─────────────────────────────────────────────────
        print("\n=== Executing sweep in MuJoCo (sim) ===")
        sim_traj = _exec_motion_sim(
            mj_model, mj_data, theta.ik_cache, waypoints,
            theta=theta, goal_marker_fn=goal_marker_fn,
        )

        # ── 3. Real-robot execution ──────────────────────────────────────────
        print("\n=== Executing sweep on real robot ===")
        real_traj = _exec_motion_real(planner, waypoints)

        # ── 4. Print both trajectories ───────────────────────────────────────
        _print_trajectory("SIM ", waypoints, sim_traj)
        _print_trajectory("REAL", waypoints, real_traj)

    finally:
        planner.close()
        print("\n[debug-realtime] Done.")

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
    parser.add_argument('--sweep-x', type=float, default=SWEEP_X,
                        help='X position of the debug-realtime y-sweep (m)')
    parser.add_argument('--sweep-y-lo', type=float, default=SWEEP_Y_LO,
                        help='start y of the debug-realtime sweep (m)')
    parser.add_argument('--sweep-y-hi', type=float, default=SWEEP_Y_HI,
                        help='end y of the debug-realtime sweep (m)')
    parser.add_argument('--sweep-n-wp', type=int, default=SWEEP_N_WP,
                        help='number of waypoints in the debug-realtime sweep')
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