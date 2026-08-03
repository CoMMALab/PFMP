"""
push_tee_optitrack.py — Motive/OptiTrack perception for the push-T task.

Usage:
    python examples/fr3/push_tee_optitrack.py --mocap-ip 192.168.1.100
    python examples/fr3/push_tee_optitrack.py --mocap-ip 192.168.1.100 --mocap
"""

import argparse
import sys
import os
import threading
import time
from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation

from new_natnet_client.Client import NatNetClient, NatNetParams
from geom_tee import OPTI_TO_BODY
from env import load_env

sys.path.insert(0, os.path.dirname(__file__))

T_OPTI_TO_WORLD = np.array([
    [1., 0., 0.,  0.44 ],
    [0., 1., 0.,  -0.01 ],
    [0., 0., 1.,  0.00 ],
    [0., 0., 0.,  1.    ],
])

def opti_to_mjx(pos, quat_xyzw):
    """Remap Motive pos+quat to MuJoCo coordinate frame (+x,+y → -x,-y)."""
    p = np.array([-pos[0], -pos[1], pos[2]])
    R = Rotation.from_quat(quat_xyzw).as_matrix()
    R_fix = np.diag([-1., -1., 1.])
    R_new = R_fix @ R @ R_fix  # conjugate to keep proper rotation
    T = np.eye(4)
    T[:3, :3] = R_new
    T[:3, 3]  = p
    return T


def rb_to_tee_world(rb, T_opti_to_world):
    """Convert Optitrack rigid body to full 4x4 tee pose in world frame."""
    T_rb = opti_to_mjx([rb.pos.x, rb.pos.y, rb.pos.z],
                        [rb.rot.x, rb.rot.y, rb.rot.z, rb.rot.w])
    return T_opti_to_world @ T_rb @ np.linalg.inv(OPTI_TO_BODY)


def tee_pose(rb):
    """xy and yaw of the tee in world frame.

    Assumes the Motive ground plane is set upright, so the rigid body frame is not
    rolled and the yaw reads directly. If --mocap shows roll ~180 deg, fix the
    ground plane in Motive rather than compensating here.
    """
    T = rb_to_tee_world(rb, T_OPTI_TO_WORLD)
    return T[:2, 3], np.arctan2(T[1, 0], T[0, 0])


@dataclass
class LiveTee:
    """Latest tee pose seen by OptiTrack. `track_tee` (its own thread) writes
    this; planning/render loops read it — always under the caller's lock, since
    both sides touch it from different threads."""
    xy: np.ndarray
    yaw: float
    seen: bool = False
    stamp: float = 0.0

    def is_stale(self, max_age: float) -> bool:
        """True if we've never seen the tee, or the last sighting is older
        than `max_age` seconds."""
        return (not self.seen) or (time.time() - self.stamp > max_age)


def track_tee(client, live: LiveTee, lock: threading.Lock, stop: threading.Event):
    """Sole OptiTrack reader: publishes each frame's tee pose into `live`
    until `stop` is set. Runs on its own thread; `lock` guards `live` against
    the planning/render loops reading it concurrently."""
    for frame in client.MoCap(timeout=None):
        if stop.is_set():
            return
        rbs = [rb for rb in frame.rigid_body_data.rigid_bodies if rb.tracking]
        if not rbs:
            with lock:
                live.seen = False
            continue
        xy, yaw = tee_pose(rbs[0])
        with lock:
            live.xy = xy
            live.yaw = float(yaw)
            live.seen = True
            live.stamp = time.time()


def stream_loop(client, args):
    if args.mocap:
        import mujoco
        import mujoco.viewer

        xml = os.path.join(os.path.dirname(__file__),
                           "../models/mjx_t_push.xml")
        mj_model = mujoco.MjModel.from_xml_path(xml)
        mj_data  = mujoco.MjData(mj_model)
        tracker_id  = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "t_tracker")
        mocap_id    = mj_model.body_mocapid[tracker_id]
        tracker_geom_ids = [
            mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, "tracker_stem"),
            mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, "tracker_bar"),
        ]
        TRACKER_RGBA = np.array([0., 0.6, 1., 0.3])

        def set_tracker_visible(visible):
            alpha = TRACKER_RGBA[3] if visible else 0.0
            for gid in tracker_geom_ids:
                mj_model.geom_rgba[gid] = [*TRACKER_RGBA[:3], alpha]

        with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
            for frame in client.MoCap(timeout=None):
                rbs = [rb for rb in frame.rigid_body_data.rigid_bodies if rb.tracking]
                if not rbs:
                    set_tracker_visible(False)
                    viewer.sync()
                    continue
                set_tracker_visible(True)
                T = rb_to_tee_world(rbs[0], T_OPTI_TO_WORLD)
                quat_xyzw = Rotation.from_matrix(T[:3, :3]).as_quat()
                yaw = np.arctan2(T[1, 0], T[0, 0])
                # Full orientation alongside the extracted yaw: if roll/pitch are
                # ~0 the two agree and any constant offset is a genuine frame
                # offset; if they sit near +/-180 or wander, atan2(R[1,0],R[0,0])
                # is reading a tilted frame and the yaw is not what you see.
                rpy = Rotation.from_matrix(T[:3, :3]).as_euler("xyz", degrees=True)
                print(f"\rxy=({T[0,3]:+.3f}, {T[1,3]:+.3f})  yaw={np.degrees(yaw):+7.1f}deg"
                      f"   rpy=({rpy[0]:+7.1f}, {rpy[1]:+7.1f}, {rpy[2]:+7.1f})",
                      end="", flush=True)
                mj_data.mocap_pos[mocap_id]  = T[:3, 3]
                mj_data.mocap_quat[mocap_id] = [quat_xyzw[3], quat_xyzw[0],
                                                 quat_xyzw[1], quat_xyzw[2]]
                mujoco.mj_forward(mj_model, mj_data)
                viewer.sync()
                if not viewer.is_running():
                    break
    elif args.inspect_rb:
        print("Printing marker positions in RB frame (Ctrl-C to stop)...\n")
        for frame in client.MoCap(timeout=None):
            rbs = [rb for rb in frame.rigid_body_data.rigid_bodies if rb.tracking]
            markers = frame.labeled_marker_data.markers
            if not rbs or not markers:
                continue
            rb = rbs[0]
            T_rb = opti_to_mjx([rb.pos.x, rb.pos.y, rb.pos.z],
                                [rb.rot.x, rb.rot.y, rb.rot.z, rb.rot.w])
            R_rb = T_rb[:3, :3]
            p_rb = T_rb[:3, 3]
            print(f"RB pos: ({p_rb[0]:.4f}, {p_rb[1]:.4f}, {p_rb[2]:.4f})")
            for m in markers:
                p_w_raw = np.array([m.pos.x, m.pos.y, m.pos.z])
                p_w = opti_to_mjx(p_w_raw, [0, 0, 0, 1])[:3, 3]
                p_rb_frame = R_rb.T @ (p_w - p_rb)
                print(f"  id={m.identifier}  local=({p_rb_frame[0]:+.4f}, {p_rb_frame[1]:+.4f}, {p_rb_frame[2]:+.4f})")
            print()
            break  # one frame is enough
    else:
        for frame in client.MoCap(timeout=None):
            t = frame.suffix_data.timestamp
            for rb in frame.rigid_body_data.rigid_bodies:
                if not rb.tracking:
                    continue
                p, q = rb.pos, rb.rot
                print(f"t={t:.3f}  id={rb.identifier}  "
                      f"pos=({p.x:7.4f}, {p.y:7.4f}, {p.z:7.4f})  "
                      f"quat=({q.x:.3f}, {q.y:.3f}, {q.z:.3f}, {q.w:.3f})  "
                      f"err={rb.err:.4f}")


def main():
    env = load_env()
    parser = argparse.ArgumentParser()
    parser.add_argument("--mocap-ip", default=env["mocap_ip"])
    parser.add_argument("--local-ip", default=env["local_ip"])
    parser.add_argument("--mocap",      action="store_true",
                        help="Drive MuJoCo tee body from Optitrack pose")
    parser.add_argument("--inspect-rb", action="store_true",
                        help="Print marker positions in RB frame (one frame)")
    args = parser.parse_args()

    params = NatNetParams(server_address=args.mocap_ip, local_ip_address=args.local_ip,
                          use_multicast=False)

    print(f"Connecting to Motive at {args.mocap_ip}...")
    with NatNetClient(params) as client:
        if client is None:
            print("Failed to connect."); return
        print("Connected.\n")
        stream_loop(client, args)


if __name__ == "__main__":
    main()
