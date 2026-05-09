"""
sysid.py — System identification for the push-T task.

Runs a scripted EE sweep on both the real robot and MuJoCo, records
the T-block response in both, and plots them side-by-side.

No planner required. Communicates with push_tee_controller.py via ZMQ.

Usage:
    python examples/fr3/sysid.py
    python examples/fr3/sysid.py --sweep-x 0.45 --sweep-y-lo -0.2 --sweep-y-hi 0.2 --n 10
"""

import argparse
import os
import sys
import time
from typing import Optional, Tuple

import jax.numpy as jnp
import matplotlib.pyplot as plt
import mujoco
import numpy as np
import zmq

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ik_cache import ik_from_cache, get_or_build_cache

from pub_sub.protocol import Command
from pub_sub.planner import PlannerClient

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


def exec_real(planner, waypoints, motion_timeout=15.0) -> Tuple[np.ndarray, np.ndarray]:
    tee_traj, ee_traj = [], []
    for i, wp in enumerate(waypoints):
        label = f"wp {i+1}/{len(waypoints)}"
        try:
            ack = send_with_retry(planner, Command(xy=wp.tolist()), label)
        except RuntimeError as e:
            print(f"  [real] {label}: {e}; aborting sweep.")
            for _ in range(i, len(waypoints)):
                tee_traj.append(np.full(3, np.nan))
                ee_traj.append(np.full(2, np.nan))
            break
        if not ack.get("accepted", False):
            tee_traj.append(np.full(3, np.nan))
            ee_traj.append(np.full(2, np.nan))
            continue

        time.sleep(0.1)
        pre_seq = -1
        try:
            pre, _, _ = planner.recv_response_latest(drain=True)
            pre_seq = int(pre.get("seq", -1))
        except zmq.Again:
            pass

        belief = None
        t0 = time.time()
        while time.time() - t0 < motion_timeout:
            try:
                resp, _, _ = planner.recv_response_latest(drain=True)
            except zmq.Again:
                time.sleep(0.05)
                continue
            if int(resp.get("seq", -1)) > pre_seq:
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

    return np.asarray(tee_traj), np.asarray(ee_traj)


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


def plot_comparison(waypoints, sim_tee, sim_ee, real_tee, real_ee, save_path=None):
    fig, axes = plt.subplots(1, 2, figsize=(14, 7), sharex=True, sharey=True)
    for (name, tee, ee), ax in zip(
        [("SIM", sim_tee, sim_ee), ("REAL", real_tee, real_ee)], axes
    ):
        valid_t = np.all(np.isfinite(tee), axis=1)
        if np.any(valid_t):
            ax.plot(tee[valid_t, 0], tee[valid_t, 1], "-o", color="C1",
                    linewidth=1.5, markersize=5, label="tee centroid")
        ax.plot(waypoints[:, 0], waypoints[:, 1], ":", color="gray",
                alpha=0.5, label="commanded xy")
        valid_e = np.all(np.isfinite(ee), axis=1)
        if np.any(valid_e):
            ax.plot(ee[valid_e, 0], ee[valid_e, 1], "-", color="C0",
                    linewidth=1.2, alpha=0.7, label="EE")
        ax.set_title(name)
        ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
        ax.set_aspect("equal"); ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("sysid: sim vs real T-block response to EE sweep")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
        print(f"[sysid] saved → {save_path}")
    plt.show()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Push-T system identification")
    parser.add_argument("--server-host", default="localhost")
    parser.add_argument("--cmd-port",   type=int, default=5555)
    parser.add_argument("--resp-port",  type=int, default=5556)
    parser.add_argument("--sweep-x",    type=float, default=SWEEP_X)
    parser.add_argument("--sweep-y-lo", type=float, default=SWEEP_Y_LO)
    parser.add_argument("--sweep-y-hi", type=float, default=SWEEP_Y_HI)
    parser.add_argument("--n",          type=int,   default=SWEEP_N_WP)
    parser.add_argument("--steps",      type=int,   default=500)
    parser.add_argument("--cache",      default=None)
    parser.add_argument("--out",        default="sysid_traj.png")
    args = parser.parse_args()

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
        response_addr=f"tcp://{args.server_host}:{args.resp_port}",
        conflate=False, command_timeout_ms=5000,
    )

    try:
        response = None
        for attempt in range(1, 21):
            try:
                response, _, _ = planner.recv_response_latest(drain=True)
                break
            except zmq.Again:
                print(f"  [sysid] waiting for controller (attempt {attempt}/20)…")
                time.sleep(0.25)
        if response is None:
            raise RuntimeError("No response from control server")

        mean = apply_response_to_mj(mj_model, mj_data, response)
        if mean is None:
            raise RuntimeError("No T-block belief in response")
        print(f"[sysid] seeded MuJoCo: tee=({mean[0]:.3f}, {mean[1]:.3f}, "
              f"{np.degrees(mean[2]):.1f}°)")

        if abs(args.sweep_x - mean[0]) > 0.10:
            print(f"[sysid] WARNING: sweep_x is {abs(args.sweep_x-mean[0])*100:.1f} cm "
                  f"from tee_x — EE may miss the tee")

        print("\n=== Sim sweep ===")
        sim_tee, sim_ee = exec_sim(mj_model, mj_data, ik_cache, waypoints, args.steps)

        print("\n=== Real sweep ===")
        real_tee, real_ee = exec_real(planner, waypoints)

        print_comparison(waypoints, sim_tee, real_tee)
        plot_comparison(waypoints, sim_tee, sim_ee, real_tee, real_ee, save_path=args.out)

    finally:
        planner.cmd_sock.close(linger=0)
        planner.resp_sock.close(linger=0)
        planner.ctx.term()


if __name__ == "__main__":
    main()
