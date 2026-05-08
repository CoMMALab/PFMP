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
import matplotlib.pyplot as plt
import mujoco
import numpy as np
import zmq
from mujoco import mjx

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from geom_tee import STEM_HALF, CAP_HALF, CAP_CTR

# Real-world tee dimensions (8" × 8" T) for top-down outline plotting; see
# examples/fr3/perception/tee.py and viz.py for the full 3D shape definition.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "perception"))
from tee import (
    TEE_WIDTH_M, TEE_LENGTH_M, TEE_XY_THICKNESS_M,
    TEE_CENTER_OFFSET_Y_M,
)

import pfmp
from pfmp.run_batch import Trajectory
from pub_sub.protocol import Command, Reply, TeeBelief
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



def _apply_reply_to_mj(mj_model, mj_data, reply: Reply) -> np.ndarray:
    """Mirror the controller's reply into mj_data: t_block pose + robot joints.

    Returns the [x, y, yaw] belief mean that was applied.
    """
    m = np.asarray(reply.belief.mean, dtype=float)
    mj_data.qpos[T_QPOS_OFFSET]     = m[0]
    mj_data.qpos[T_QPOS_OFFSET + 1] = m[1]
    mj_data.qpos[T_QPOS_OFFSET + 2] = m[2]
    if reply.joint_q and len(reply.joint_q) >= 7:
        mj_data.qpos[:7] = np.asarray(reply.joint_q[:7], dtype=float)
    mujoco.mj_forward(mj_model, mj_data)
    return m


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


def _tee_outline_2d_local() -> Tuple[np.ndarray, np.ndarray]:
    """Top-down 2D outlines (cap, stem) of the tee in local body coords.

    Mirrors the xy footprint from `examples/fr3/perception/viz.py`'s
    `_tee_box_vertices_local`. The local origin is at the AprilTag (top
    surface center, with `TEE_CENTER_OFFSET_Y_M` y-offset). At yaw=0 the
    stem points in the −y direction.
    """
    half_w = TEE_WIDTH_M / 2.0
    stem_half_w = TEE_XY_THICKNESS_M / 2.0
    stem_len = TEE_LENGTH_M - TEE_XY_THICKNESS_M
    cy = TEE_CENTER_OFFSET_Y_M

    cap = np.array([
        [-half_w, 0.0       + cy],
        [ half_w, 0.0       + cy],
        [ half_w, TEE_XY_THICKNESS_M + cy],
        [-half_w, TEE_XY_THICKNESS_M + cy],
    ])
    stem = np.array([
        [-stem_half_w, -stem_len + cy],
        [ stem_half_w, -stem_len + cy],
        [ stem_half_w, 0.0       + cy],
        [-stem_half_w, 0.0       + cy],
    ])
    return cap, stem


def _draw_tee(ax, x: float, y: float, yaw: float, color="C1", alpha: float = 0.3) -> None:
    """Draw a top-down tee outline at (x, y) with yaw in radians."""
    cap_local, stem_local = _tee_outline_2d_local()
    R = np.array([
        [np.cos(yaw), -np.sin(yaw)],
        [np.sin(yaw),  np.cos(yaw)],
    ])
    cap_w  = (R @ cap_local.T ).T + np.array([x, y])
    stem_w = (R @ stem_local.T).T + np.array([x, y])
    for poly in (cap_w, stem_w):
        ax.fill(poly[:, 0], poly[:, 1], color=color, alpha=alpha,
                edgecolor=color, linewidth=0.8)


def _plot_trajectories(
    waypoints: np.ndarray,
    sim_tee: np.ndarray,
    sim_ee: np.ndarray,
    real_tee: np.ndarray,
    real_ee: np.ndarray,
    save_path: Optional[str] = None,
) -> None:
    """Side-by-side top-down plot of tee outlines + EE trail for sim and real."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 7), sharex=True, sharey=True)

    panels = [
        ("SIM",  sim_tee,  sim_ee,  axes[0]),
        ("REAL", real_tee, real_ee, axes[1]),
    ]

    for name, tee_traj, ee_traj, ax in panels:
        # Tee outlines: faded (early) → opaque (late) over the trajectory.
        n = len(tee_traj)
        for i, p in enumerate(tee_traj):
            if not np.all(np.isfinite(p)):
                continue
            alpha = 0.15 + 0.65 * (i / max(n - 1, 1))
            _draw_tee(ax, p[0], p[1], p[2], color="C1", alpha=alpha)

        # Tee centroid trail.
        valid_t = np.all(np.isfinite(tee_traj), axis=1)
        if np.any(valid_t):
            ax.plot(tee_traj[valid_t, 0], tee_traj[valid_t, 1], "-",
                    color="C1", alpha=0.6, linewidth=1.2, label="tee centroid")
            ax.scatter(tee_traj[valid_t, 0][0], tee_traj[valid_t, 1][0],
                       s=60, marker="s", facecolor="white", edgecolor="C1",
                       zorder=5, label="tee start")

        # Commanded waypoints (gray dotted reference).
        ax.plot(waypoints[:, 0], waypoints[:, 1], ":", color="gray",
                alpha=0.5, label="commanded xy")

        # EE trail (sim: actual gripper; real: commanded wp as proxy).
        ee = np.asarray(ee_traj)
        valid_e = np.all(np.isfinite(ee), axis=1)
        if np.any(valid_e):
            ax.plot(ee[valid_e, 0], ee[valid_e, 1], "-o",
                    color="C0", linewidth=1.5, markersize=5, label="EE")
            ax.scatter(ee[valid_e, 0][0], ee[valid_e, 1][0], s=80,
                       facecolor="white", edgecolor="C0",
                       zorder=5, label="EE start")

        ax.set_title(name)
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8)

    fig.suptitle("debug-realtime: top-down tee + EE trajectories")
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
        print(f"[debug-realtime] Saved trajectory plot to {save_path}")

    plt.show()


def _exec_motion_sim(
    mj_model: mujoco.MjModel,
    mj_data: mujoco.MjData,
    ik_cache,
    waypoints: np.ndarray,
    steps_per_wp: int = 500,
    theta: Optional[PushParams] = None,
    goal_marker_fn: Optional[callable] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Drive MuJoCo through the waypoints and record (tee, EE) at each.

    Each waypoint sets the position-controller targets to ik_from_cache(xy) and
    steps physics for `steps_per_wp` ticks. Returns
        tee_traj: (N, 3) array of [x, y, yaw] for the t-block,
        ee_traj:  (N, 2) array of [x, y] for the gripper site (or commanded
                  xy if theta is None / no gripper site is available).
    """
    tee_traj = []
    ee_traj = []
    gripper_site_id = theta.gripper_site_id if theta is not None else None

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
            tee_traj.append(tee)
            if gripper_site_id is not None:
                ee = np.array(mj_data.site_xpos[gripper_site_id, :2], dtype=float)
            else:
                ee = wp.copy()
            ee_traj.append(ee)
            print(
                f"  [sim ] wp {i+1:2d}/{len(waypoints)}: "
                f"target_xy=({wp[0]:+.3f}, {wp[1]:+.3f})  "
                f"ee=({ee[0]:+.3f}, {ee[1]:+.3f})  "
                f"tee=({tee[0]:+.3f}, {tee[1]:+.3f}, {np.degrees(tee[2]):+6.1f}°)"
            )

            if not viewer.is_running():
                print("  [sim ] viewer closed; stopping early.")
                break

    return np.asarray(tee_traj), np.asarray(ee_traj)


def _exec_motion_real(
    planner: PlannerClient,
    waypoints: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Send each waypoint to the controller and record (tee, EE) at each.

    Synchronous flow: planner.move(wp) blocks until the controller finishes
    execute_xy + perception, then returns the post-motion belief. No need to
    track sequence numbers or wait for fresh publishes.

    On a REQ timeout (controller hung or restarting) we reconnect and bail
    out, NaN-filling the remaining waypoints. The real EE position isn't
    reported, so we use the commanded xy as a proxy.
    """
    tee_traj = []
    ee_traj = []
    for i, wp in enumerate(waypoints):
        label = f"wp {i+1:2d}/{len(waypoints)}"
        print(f"  [real] {label}: sending xy=({wp[0]:+.3f}, {wp[1]:+.3f}) and waiting…")

        try:
            reply = planner.move(wp)
        except zmq.Again:
            print(f"  [real] {label}: timed out waiting for reply; reconnecting and bailing.")
            planner.reconnect()
            for _ in range(i, len(waypoints)):
                tee_traj.append(np.full(3, np.nan))
                ee_traj.append(np.full(2, np.nan))
            break

        if not reply.success:
            print(f"  [real] {label}: motion failed on the controller: {reply.error}")
            tee_traj.append(np.full(3, np.nan))
            ee_traj.append(np.full(2, np.nan))
            continue

        belief_mean = np.asarray(reply.belief.mean, dtype=float)
        print(
            f"  [real] {label}: tee=({belief_mean[0]:+.3f}, "
            f"{belief_mean[1]:+.3f}, {np.degrees(belief_mean[2]):+6.1f}°)"
        )
        tee_traj.append(belief_mean)
        ee_traj.append(wp.copy())

    return np.asarray(tee_traj), np.asarray(ee_traj)


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

    print(f"[debug-realtime] Connecting to control server  REQ ⇄ {cmd_addr}")
    planner = PlannerClient(
        command_addr=cmd_addr,
        # The reply to a "move" command only comes back after the motion
        # completes, so the timeout has to cover the longest expected motion.
        timeout_ms=args.req_timeout_ms,
    )

    try:
        # ── 1. Initial belief: query and seed MuJoCo ─────────────────────────
        print("[debug-realtime] Querying initial belief…")
        try:
            reply = planner.query()
        except zmq.Again as e:
            raise RuntimeError(f"No reply from control server: {e}")
        if not reply.success:
            raise RuntimeError(f"Query rejected by controller: {reply.error}")

        mean = _apply_reply_to_mj(mj_model, mj_data, reply)
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
        sim_tee, sim_ee = _exec_motion_sim(
            mj_model, mj_data, theta.ik_cache, waypoints,
            theta=theta, goal_marker_fn=goal_marker_fn,
        )

        # ── 3. Real-robot execution ──────────────────────────────────────────
        print("\n=== Executing sweep on real robot ===")
        real_tee, real_ee = _exec_motion_real(planner, waypoints)

        # ── 4. Print both trajectories ───────────────────────────────────────
        _print_trajectory("SIM ", waypoints, sim_tee)
        _print_trajectory("REAL", waypoints, real_tee)

        # ── 5. Plot ──────────────────────────────────────────────────────────
        plot_path = os.path.join(os.path.dirname(__file__), "debug_realtime_traj.png")
        _plot_trajectories(waypoints, sim_tee, sim_ee, real_tee, real_ee,
                           save_path=plot_path)

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
                        help='ZMQ REQ/REP port for command round-trips with the controller')
    parser.add_argument('--req-timeout-ms', type=int, default=60_000,
                        help='REQ recv/send timeout in ms; must exceed the longest motion duration')
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