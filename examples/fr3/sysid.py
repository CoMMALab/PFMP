"""
sysid.py — System identification for the push-T task.

Runs a scripted EE sweep on both the real robot and MuJoCo, records
the T-block response in both, and reviews them frame-by-frame.

No planner required. Sends commands to push_tee_controller.py and subscribes
to push_tee_perception.py for RGBD frames and EKF beliefs.

Usage:
    python examples/fr3/sysid.py
    python examples/fr3/sysid.py --sweep-x 0.45 --sweep-y-lo -0.2 --sweep-y-hi 0.2 --n 10
"""

import argparse
import os
import sys
import time
from typing import Optional, Tuple

import cv2
import jax.numpy as jnp
import mujoco
import mujoco.viewer
import numpy as np
import zmq

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ik_cache import ik_from_cache, get_or_build_cache

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "perception"))
from camera import WorldFrame, SERIAL_NUMS
from draw import CYAN, draw_pose_marker, project_world_xy
from geom_tee import THICKNESS

from pub_sub.protocol import Command
from pub_sub.planner import PlannerClient
from pub_sub.perception import PerceptionClient

# ── Config ────────────────────────────────────────────────────────────────────

SCENE_XML_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "models", "mjx_t_push.xml")
T_QPOS_OFFSET  = 7

SWEEP_X    = 0.5
SWEEP_Y_LO = -0.15
SWEEP_Y_HI =  0.15
SWEEP_N_WP =  8


# ── Sweep waypoints ───────────────────────────────────────────────────────────

def make_sweep_waypoints(sweep_x=SWEEP_X, y_lo=SWEEP_Y_LO,
                          y_hi=SWEEP_Y_HI, n=SWEEP_N_WP) -> np.ndarray:
    ys = np.linspace(y_lo, y_hi, n)
    return np.stack([np.full_like(ys, sweep_x), ys], axis=1)


# ── MuJoCo helpers ────────────────────────────────────────────────────────────

def apply_response_to_mj(mj_model, mj_data, response: dict) -> Optional[np.ndarray]:
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


def exec_sim(mj_model, mj_data, ik_cache, waypoints,
             steps_per_wp=500) -> Tuple[np.ndarray, np.ndarray]:
    tee_traj, ee_traj = [], []
    with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
        for i, wp in enumerate(waypoints):
            target_q = np.asarray(ik_from_cache(jnp.asarray(wp), ik_cache))
            mj_data.ctrl[:7] = target_q
            for step in range(steps_per_wp):
                mujoco.mj_step(mj_model, mj_data)
                if step % 5 == 0:
                    viewer.sync()
                if not viewer.is_running():
                    break
            tee = mj_data.qpos[T_QPOS_OFFSET:T_QPOS_OFFSET + 3].copy()
            tee_traj.append(tee)
            ee_traj.append(wp.copy())
            print(f"  [sim ] wp {i+1}/{len(waypoints)}: "
                  f"tee=({tee[0]:+.3f}, {tee[1]:+.3f}, {np.degrees(tee[2]):+.1f}°)")
            if not viewer.is_running():
                break
    return np.asarray(tee_traj), np.asarray(ee_traj)


# ── ZMQ helpers ───────────────────────────────────────────────────────────────

def reconnect_req(planner, timeout_ms=5000):
    planner.cmd_sock.close(linger=0)
    s = planner.ctx.socket(zmq.REQ)
    s.setsockopt(zmq.RCVTIMEO, timeout_ms)
    s.setsockopt(zmq.SNDTIMEO, timeout_ms)
    s.connect(planner.command_addr)
    planner.cmd_sock = s


def send_with_retry(planner, cmd, label, max_attempts=30, retry_pause=1.0):
    for attempt in range(1, max_attempts + 1):
        try:
            return planner.send_command(cmd)
        except zmq.Again:
            print(f"  [real] {label}: timeout (attempt {attempt}/{max_attempts}); reconnecting…")
            reconnect_req(planner)
            time.sleep(retry_pause)
    raise RuntimeError(f"send_command failed after {max_attempts} attempts")


def exec_real(
    planner,
    perception: PerceptionClient,
    waypoints,
    motion_timeout=15.0,
    settle_s=0.2,
) -> Tuple[np.ndarray, np.ndarray, list[Optional[np.ndarray]]]:
    tee_traj, ee_traj, frames = [], [], []
    for i, wp in enumerate(waypoints):
        label = f"wp {i+1}/{len(waypoints)}"
        time.sleep(1.0)
        try:
            ack = send_with_retry(planner, Command(xy=wp.tolist()), label)
        except RuntimeError as e:
            print(f"  [real] {label}: {e}; aborting sweep.")
            for _ in range(i, len(waypoints)):
                tee_traj.append(np.full(3, np.nan))
                ee_traj.append(np.full(2, np.nan))
                frames.append(None)
            break
        if not ack.get("accepted", False):
            tee_traj.append(np.full(3, np.nan))
            ee_traj.append(np.full(2, np.nan))
            frames.append(None)
            continue

        min_timestamp = time.time() + settle_s

        belief = None
        frame = None
        t0 = time.time()
        while time.time() - t0 < motion_timeout:
            try:
                resp, rgb, _ = perception.recv_latest(drain=True)
            except zmq.Again:
                time.sleep(0.05)
                continue
            if float(resp.get("timestamp", 0.0)) >= min_timestamp:
                frame = rgb.copy()
                beliefs = resp.get("beliefs") or []
                if beliefs:
                    belief = np.asarray(beliefs[-1]["mean"], dtype=float)
                break
            time.sleep(0.05)

        if belief is None:
            belief = np.full(3, np.nan)
        else:
            print(f"  [real] {label}: "
                  f"tee=({belief[0]:+.3f}, {belief[1]:+.3f}, {np.degrees(belief[2]):+.1f}°)")
        tee_traj.append(belief)
        ee_traj.append(wp.copy())
        frames.append(frame)

    return np.asarray(tee_traj), np.asarray(ee_traj), frames


# ── Comparison ────────────────────────────────────────────────────────────────

def print_comparison(waypoints, sim_tee, real_tee):
    print(f"\n{'wp':>3}  {'cmd_x':>8} {'cmd_y':>8}  "
          f"{'sim_x':>8} {'sim_y':>8} {'sim_yaw':>9}  "
          f"{'real_x':>8} {'real_y':>8} {'real_yaw':>9}")
    for i, (wp, s, r) in enumerate(zip(waypoints, sim_tee, real_tee)):
        def fmt(p):
            if np.any(np.isnan(p)):
                return f"{'nan':>8} {'nan':>8} {'nan':>9}"
            return f"{p[0]:+8.3f} {p[1]:+8.3f} {np.degrees(p[2]):+8.1f}°"
        print(f"{i+1:>3}  {wp[0]:+8.3f} {wp[1]:+8.3f}  {fmt(s)}  {fmt(r)}")


def _valid_pose(pose: np.ndarray) -> bool:
    return pose is not None and np.asarray(pose).shape == (3,) and np.all(np.isfinite(pose))


def make_review_frame(
    frame: Optional[np.ndarray],
    wf: WorldFrame,
    serial_top: str,
    K_top: np.ndarray,
    sim_pose: np.ndarray,
    real_pose: np.ndarray,
    waypoint: np.ndarray,
    idx: int,
    n: int,
) -> np.ndarray:
    if frame is None:
        canvas = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(canvas, "no RGB frame captured", (30, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (180, 180, 180), 2, cv2.LINE_AA)
    else:
        canvas = frame.copy()

    T_w2c = np.linalg.inv(wf.t_cams[serial_top])
    z_body = THICKNESS / 2
    purple = (180, 80, 220)

    if _valid_pose(sim_pose):
        draw_pose_marker(
            canvas,
            sim_pose[:2],
            float(sim_pose[2]),
            z_body,
            T_w2c,
            K_top,
            purple,
            outline_thickness=2,
            center_radius=5,
        )
        label_px = project_world_xy(sim_pose[:2], z_body, T_w2c, K_top)[0]
        cv2.putText(canvas, "SIM", (int(label_px[0]) + 8, int(label_px[1]) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, purple, 2, cv2.LINE_AA)

    if _valid_pose(real_pose):
        draw_pose_marker(
            canvas,
            real_pose[:2],
            float(real_pose[2]),
            z_body,
            T_w2c,
            K_top,
            CYAN,
            outline_thickness=3,
            center_radius=7,
        )
        label_px = project_world_xy(real_pose[:2], z_body, T_w2c, K_top)[0]
        cv2.putText(canvas, "EKF", (int(label_px[0]) + 8, int(label_px[1]) + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, CYAN, 2, cv2.LINE_AA)

    cv2.putText(canvas, f"{idx + 1}/{n}  cmd=({waypoint[0]:+.3f},{waypoint[1]:+.3f})",
                (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, "cyan=EKF   purple=SIM",
                (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, "left/right: frame   q/esc: quit",
                (20, canvas.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX,
                0.65, (255, 255, 255), 2, cv2.LINE_AA)
    return canvas


def review_frames(
    waypoints: np.ndarray,
    frames: list[Optional[np.ndarray]],
    sim_tee: np.ndarray,
    real_tee: np.ndarray,
    wf: WorldFrame,
) -> None:
    serial_top = SERIAL_NUMS["top"]
    K_top = wf.Ks[serial_top]
    n = min(len(waypoints), len(frames), len(sim_tee), len(real_tee))
    if n == 0:
        print("[sysid] no frames to review")
        return

    print("[sysid] review: left/right arrows cycle frames, q/esc exits")
    idx = 0
    while True:
        view = make_review_frame(
            frames[idx],
            wf,
            serial_top,
            K_top,
            sim_tee[idx],
            real_tee[idx],
            waypoints[idx],
            idx,
            n,
        )
        cv2.imshow("sysid review", view)
        key = cv2.waitKeyEx(0)
        if key in (ord("q"), 27):
            break
        if key in (81, 2424832, ord("a")):
            idx = (idx - 1) % n
        elif key in (83, 2555904, ord("d"), ord(" ")):
            idx = (idx + 1) % n
    cv2.destroyWindow("sysid review")


def save_review_gif(
    out_path: str,
    waypoints: np.ndarray,
    frames: list[Optional[np.ndarray]],
    sim_tee: np.ndarray,
    real_tee: np.ndarray,
    wf: WorldFrame,
    fps: float,
) -> None:
    from PIL import Image

    serial_top = SERIAL_NUMS["top"]
    K_top = wf.Ks[serial_top]
    n = min(len(waypoints), len(frames), len(sim_tee), len(real_tee))
    if n == 0:
        print("[sysid] no frames to save")
        return

    gif_frames = []
    for idx in range(n):
        view_bgr = make_review_frame(
            frames[idx],
            wf,
            serial_top,
            K_top,
            sim_tee[idx],
            real_tee[idx],
            waypoints[idx],
            idx,
            n,
        )
        view_rgb = cv2.cvtColor(view_bgr, cv2.COLOR_BGR2RGB)
        gif_frames.append(Image.fromarray(view_rgb))

    duration_ms = int(round(1000.0 / max(fps, 1e-6)))
    gif_frames[0].save(
        out_path,
        save_all=True,
        append_images=gif_frames[1:],
        duration=duration_ms,
        loop=0,
    )
    print(f"[sysid] saved review gif to {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Push-T system identification")
    parser.add_argument("--server-host", default="localhost")
    parser.add_argument("--cmd-port",   type=int, default=5555)
    parser.add_argument("--perception-port", type=int, default=5557)
    parser.add_argument("--sweep-x",    type=float, default=SWEEP_X)
    parser.add_argument("--sweep-y-lo", type=float, default=SWEEP_Y_LO)
    parser.add_argument("--sweep-y-hi", type=float, default=SWEEP_Y_HI)
    parser.add_argument("--n",          type=int,   default=SWEEP_N_WP)
    parser.add_argument("--steps",      type=int,   default=500)
    parser.add_argument("--settle",     type=float, default=0.2,
                        help="Seconds to wait after command ack before sampling perception.")
    parser.add_argument("--cache",      default=None)
    parser.add_argument("--wf",         default="wf.npz")
    parser.add_argument("--gif-out",    default=None,
                        help="Optional path to save the sysid review as a GIF.")
    parser.add_argument("--gif-fps",    type=float, default=2.0)
    args = parser.parse_args()

    wf = WorldFrame.load(args.wf)
    mj_model  = mujoco.MjModel.from_xml_path(SCENE_XML_PATH)
    mj_data   = mujoco.MjData(mj_model)
    cache_path = args.cache or os.path.join(os.path.dirname(__file__), "..", "ik_cache.pkl")
    ik_cache   = get_or_build_cache(path=cache_path)

    waypoints = make_sweep_waypoints(args.sweep_x, args.sweep_y_lo,
                                     args.sweep_y_hi, args.n)
    print(f"[sysid] {len(waypoints)} waypoints, x={args.sweep_x:.3f}, "
          f"y={args.sweep_y_lo:+.3f}→{args.sweep_y_hi:+.3f}")

    planner = PlannerClient(
        command_addr=f"tcp://{args.server_host}:{args.cmd_port}",
        response_addr="tcp://localhost:5556",
        conflate=False, command_timeout_ms=5000,
    )
    perception = PerceptionClient(
        addr=f"tcp://{args.server_host}:{args.perception_port}",
        conflate=False,
        timeout_ms=1000,
    )

    try:
        response = None
        for attempt in range(1, 21):
            try:
                response, _, _ = perception.recv_latest(drain=True)
                break
            except zmq.Again:
                print(f"  [sysid] waiting for perception (attempt {attempt}/20)…")
                time.sleep(0.25)
        if response is None:
            raise RuntimeError("No response from perception server")

        mean = apply_response_to_mj(mj_model, mj_data, response)
        if mean is None:
            raise RuntimeError("No T-block belief in perception response")
        print(f"[sysid] seeded MuJoCo: tee=({mean[0]:.3f}, {mean[1]:.3f}, "
              f"{np.degrees(mean[2]):.1f}°)")

        if abs(args.sweep_x - mean[0]) > 0.10:
            print(f"[sysid] WARNING: sweep_x is {abs(args.sweep_x-mean[0])*100:.1f} cm "
                  f"from tee_x — EE may miss the tee")

        print("\n=== Sim sweep ===")
        sim_tee, sim_ee = exec_sim(mj_model, mj_data, ik_cache, waypoints, args.steps)

        print("\n=== Real sweep ===")
        real_tee, real_ee, real_frames = exec_real(
            planner,
            perception,
            waypoints,
            settle_s=args.settle,
        )

        print_comparison(waypoints, sim_tee, real_tee)
        if args.gif_out is not None:
            save_review_gif(
                args.gif_out,
                waypoints,
                real_frames,
                sim_tee,
                real_tee,
                wf,
                args.gif_fps,
            )
        review_frames(waypoints, real_frames, sim_tee, real_tee, wf)

    finally:
        perception.close()
        planner.close()


if __name__ == "__main__":
    main()
