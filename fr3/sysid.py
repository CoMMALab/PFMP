"""
sysid_opti.py — System identification using Optitrack for tee pose.

Runs a scripted EE sweep on both the real robot and MuJoCo, records
the T-block response in both, and prints a comparison table.

Usage:
    python examples/fr3/sysid_opti.py --server 192.168.50.181 --local <ip>
"""

import argparse
import os
import sys
import time
from typing import Optional, Tuple

import jax.numpy as jnp
import mujoco
import mujoco.viewer
import numpy as np
import zmq

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ik_cache import ik_from_cache, get_or_build_cache

sys.path.insert(0, os.path.dirname(__file__))
from push_tee_optitrack import tee_pose, T_OPTI_TO_WORLD, opti_to_mjx
from new_natnet_client.Client import NatNetClient, NatNetParams

from pub_sub.protocol import Command
from pub_sub.planner import PlannerClient

SCENE_XML_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "mjx_t_push.xml")
T_QPOS_OFFSET  = 7

SWEEP_X    = 0.5
SWEEP_Y_LO = 0.0
SWEEP_Y_HI = 0.45
SWEEP_N_WP = 12


def make_sweep_waypoints(sweep_x=SWEEP_X, y_lo=SWEEP_Y_LO,
                          y_hi=SWEEP_Y_HI, n=SWEEP_N_WP) -> np.ndarray:
    ys = np.linspace(y_lo, y_hi, n)
    return np.stack([np.full_like(ys, sweep_x), ys], axis=1)


def exec_sim(mj_model, mj_data, ik_cache, waypoints, steps_per_wp=500):
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
            ee_traj.append((wp.copy(), mj_data.qpos[:7].copy()))
            print(f"  [sim ] wp {i+1}/{len(waypoints)}: "
                  f"tee=({tee[0]:+.3f}, {tee[1]:+.3f}, {np.degrees(tee[2]):+.1f}°)")
            if not viewer.is_running():
                break
    wps   = np.asarray([e[0] for e in ee_traj])
    joint = np.asarray([e[1] for e in ee_traj])
    return np.asarray(tee_traj), wps, joint


def sample_tee_optitrack(client, n_frames=10) -> Optional[np.ndarray]:
    """Average tee pose over n_frames from Optitrack."""
    poses = []
    for frame in client.MoCap(timeout=None):
        rbs = [rb for rb in frame.rigid_body_data.rigid_bodies if rb.tracking]
        if not rbs:
            continue
        xy, yaw = tee_pose(rbs[0])
        poses.append([xy[0], xy[1], yaw])
        if len(poses) >= n_frames:
            break
    if not poses:
        return None
    p = np.mean(poses, axis=0)
    return p


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


def exec_real(planner, client, waypoints, settle_s=1.0, n_frames=10):
    tee_traj, ee_traj = [], []
    for i, wp in enumerate(waypoints):
        label = f"wp {i+1}/{len(waypoints)}"
        time.sleep(1.0)
        try:
            ack = send_with_retry(planner, Command(xy=wp.tolist()), label)
        except RuntimeError as e:
            print(f"  [real] {label}: {e}; aborting.")
            for _ in range(i, len(waypoints)):
                tee_traj.append(np.full(3, np.nan))
                ee_traj.append(np.full(2, np.nan))
            break
        if not ack.get("accepted", False):
            tee_traj.append(np.full(3, np.nan))
            ee_traj.append(np.full(2, np.nan))
            continue

        time.sleep(settle_s)
        pose = sample_tee_optitrack(client, n_frames)
        if pose is None:
            pose = np.full(3, np.nan)
        else:
            print(f"  [real] {label}: "
                  f"tee=({pose[0]:+.3f}, {pose[1]:+.3f}, {np.degrees(pose[2]):+.1f}°)")
        tee_traj.append(pose)
        ee_traj.append(wp.copy())

    return np.asarray(tee_traj), np.asarray(ee_traj)


def review_mj(mj_model, mj_data, waypoints, sim_tee, real_tee, joint_q, dwell_s=1.5):
    """Loop through waypoints in MuJoCo viewer: physics tee=sim, tracker=real."""
    tracker_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "t_tracker")
    mocap_id   = mj_model.body_mocapid[tracker_id]
    from scipy.spatial.transform import Rotation

    n = len(waypoints)
    print("[review] looping — close viewer to exit")
    with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
        while viewer.is_running():
            for i in range(n):
                if not viewer.is_running():
                    break
                s, r = sim_tee[i], real_tee[i]

                # Robot joints
                mj_data.qpos[:7] = joint_q[i]

                # Sim tee
                if np.all(np.isfinite(s)):
                    mj_data.qpos[T_QPOS_OFFSET]     = s[0]
                    mj_data.qpos[T_QPOS_OFFSET + 1] = s[1]
                    mj_data.qpos[T_QPOS_OFFSET + 2] = s[2]

                # Real tee (tracker ghost)
                if np.all(np.isfinite(r)):
                    c, sv = np.cos(r[2]), np.sin(r[2])
                    quat = Rotation.from_matrix(
                        [[c,-sv,0],[sv,c,0],[0,0,1]]).as_quat()
                    mj_data.mocap_pos[mocap_id]  = [r[0], r[1], 0.020]
                    mj_data.mocap_quat[mocap_id] = [quat[3], quat[0], quat[1], quat[2]]

                mujoco.mj_forward(mj_model, mj_data)
                viewer.sync()
                time.sleep(dwell_s)


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server",       default="127.0.0.1")
    parser.add_argument("--local",        default="127.0.0.1")
    parser.add_argument("--server-host",  default="localhost")
    parser.add_argument("--cmd-port",     type=int, default=5555)
    parser.add_argument("--sweep-x",      type=float, default=SWEEP_X)
    parser.add_argument("--sweep-y-lo",   type=float, default=SWEEP_Y_LO)
    parser.add_argument("--sweep-y-hi",   type=float, default=SWEEP_Y_HI)
    parser.add_argument("--n",            type=int,   default=SWEEP_N_WP)
    parser.add_argument("--steps",        type=int,   default=500)
    parser.add_argument("--settle",       type=float, default=1.5)
    parser.add_argument("--cache",        default=None)
    args = parser.parse_args()

    mj_model  = mujoco.MjModel.from_xml_path(SCENE_XML_PATH)
    mj_data   = mujoco.MjData(mj_model)
    cache_path = args.cache or os.path.join(os.path.dirname(__file__), "ik_cache.pkl")
    ik_cache   = get_or_build_cache(path=cache_path)
    waypoints  = make_sweep_waypoints(args.sweep_x, args.sweep_y_lo, args.sweep_y_hi, args.n)

    params = NatNetParams(server_address=args.server, local_ip_address=args.local,
                          use_multicast=False)
    planner = PlannerClient(
        command_addr=f"tcp://{args.server_host}:{args.cmd_port}",
        response_addr="tcp://localhost:5556",
        conflate=False, command_timeout_ms=5000,
    )

    print(f"Connecting to Motive at {args.server}...")
    with NatNetClient(params) as client:
        if client is None:
            print("Failed to connect."); return

        print(f"[sysid] {len(waypoints)} waypoints, x={args.sweep_x:.3f}, "
              f"y={args.sweep_y_lo:+.3f}→{args.sweep_y_hi:+.3f}")

        print("Seeding MuJoCo from Optitrack...")
        init_pose = sample_tee_optitrack(client, n_frames=30)
        if init_pose is not None:
            mj_data.qpos[T_QPOS_OFFSET]     = init_pose[0]
            mj_data.qpos[T_QPOS_OFFSET + 1] = init_pose[1]
            mj_data.qpos[T_QPOS_OFFSET + 2] = init_pose[2]
            mujoco.mj_forward(mj_model, mj_data)
            print(f"  tee=({init_pose[0]:+.3f}, {init_pose[1]:+.3f}, {np.degrees(init_pose[2]):+.1f}°)")

        print("\n=== Sim sweep ===")
        sim_tee, _, joint_q = exec_sim(mj_model, mj_data, ik_cache, waypoints, args.steps)

        print("\n=== Real sweep ===")
        real_tee, _ = exec_real(planner, client, waypoints, settle_s=args.settle)

        # sim_tee[i+1] corresponds to real_tee[i]
        # print_comparison(waypoints[:-2], sim_tee[2:], real_tee[:-2])
        # review_mj(mj_model, mj_data, waypoints[:-2], sim_tee[2:], real_tee[:-2], joint_q[2:])
        print_comparison(waypoints, sim_tee, real_tee)
        review_mj(mj_model, mj_data, waypoints, sim_tee, real_tee, joint_q)

    planner.close()


if __name__ == "__main__":
    main()
