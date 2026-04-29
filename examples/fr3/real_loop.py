"""
real_loop.py — Minimal real-robot execution loop with EKF perception.

Connects to the Franka via franky, starts cameras, runs the EKF perception
loop, and accepts simple XY waypoint commands.  No planner — just the
infrastructure to verify perception ↔ execution integration.

Usage:
    python examples/fr3/real_loop.py --ip 172.16.0.2
    python examples/fr3/real_loop.py --dry-run          # no robot
"""

import argparse
import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "perception"))

from camera import WorldFrame, init_cameras, get_rgbd_frames, get_depth_scale, SERIAL_NUMS
from geom_tee import THICKNESS, TEE_TAG_SIZE_M
from ekf import EKF, PoseDetection
from april_detector import detect_tee_tag
from template_detector import segment_T, template_match_T, segmentation_median_world_xy

sys.path.insert(0, os.path.dirname(__file__))
from ik_cache import ik_from_cache, get_or_build_cache, HOME_Q, EE_HEIGHT

ROBOT_IP      = "172.16.0.3"
VELOCITY_SCALE = 0.08
MAX_JUMP_DEG   = 20.0


def ee_xy_from_robot(robot) -> np.ndarray:
    pos = np.array(robot.state.O_T_EE)[12:15]
    return pos[:2]


def check_jump(q_prev: np.ndarray, q_next: np.ndarray) -> bool:
    diff = np.abs((q_next - q_prev + np.pi) % (2 * np.pi) - np.pi)
    worst = float(np.degrees(diff.max()))
    if worst > MAX_JUMP_DEG:
        print(f"[WARN] joint jump {worst:.1f}° > {MAX_JUMP_DEG}° — skipping")
        return False
    return True


def perception_step(wf, pipe, K_top, dist_top, serial_top, depth_scale, ekf):
    """One perception frame: segment, detect tag, template match, update EKF."""
    color, depth = get_rgbd_frames(pipe, "top")

    mask = segment_T(
        frame=wf, color=color, depth=depth,
        depth_scale=depth_scale, cam="top",
    )

    detections = []

    tag_det = detect_tee_tag(
        color=color, depth=depth, K=K_top, dist=dist_top,
        wf=wf, serial=serial_top, depth_scale=depth_scale,
        name="tag", R=np.diag([1e-4, 1e-4, 1e-3]), gated=False,
    )
    if tag_det:
        detections.append(tag_det)

    if ekf.is_lost:
        seed_xy = segmentation_median_world_xy(
            mask=mask, depth=depth, K=K_top,
            T_c2w=wf.t_cams[serial_top], depth_scale=depth_scale,
        )
        if seed_xy is not None:
            ekf = ekf.reset(seed_xy)
            print(f"[EKF] reset to seed xy={seed_xy}")

    best_xy, best_yaw, best_iou, _ = template_match_T(
        mask=mask, frame=wf, serial=serial_top, K=K_top,
        body_xy=ekf.mean[:2], body_yaw=ekf.mean[2], body_z=THICKNESS / 2,
    )
    if best_iou >= 0.25:
        detections.append(PoseDetection(
            name="template",
            pose=np.array([best_xy[0], best_xy[1], best_yaw]),
            R=np.diag([1e-4, 1e-4, 1e-3]),
            gated=False,
        ))

    ekf = ekf.update(detections)
    return ekf, mask, color


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip",      default=ROBOT_IP)
    parser.add_argument("--speed",   type=float, default=VELOCITY_SCALE)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--wf",      default="wf.npz")
    parser.add_argument("--cache",   default=None)
    args = parser.parse_args()

    # ── World frame & IK cache ────────────────────────────────────────────────
    wf = WorldFrame.load(args.wf)
    cache_path = args.cache or os.path.join(os.path.dirname(__file__), "..", "ik_cache.pkl")
    ik_cache   = get_or_build_cache(path=cache_path)

    # ── Cameras ───────────────────────────────────────────────────────────────
    pipelines, intrinsics = init_cameras(["top"])
    pipe      = pipelines["top"]
    K_top, dist_top = intrinsics["top"]
    serial_top      = SERIAL_NUMS["top"]
    depth_scale     = get_depth_scale(pipe)

    print("Warming up camera…")
    for _ in range(60):
        get_rgbd_frames(pipe, "top")

    # ── Robot ─────────────────────────────────────────────────────────────────
    robot = None
    if not args.dry_run:
        import franky
        print(f"Connecting to {args.ip}…")
        robot = franky.Robot(args.ip)
        robot.relative_dynamics_factor = args.speed
        robot.recover_from_errors()
        robot.move(franky.JointMotion(HOME_Q.tolist()))
        print("At home.")

    # ── EKF init ──────────────────────────────────────────────────────────────
    ekf = EKF.init(wf)

    q_prev = np.array(HOME_Q)

    print("\nLoop running. Commands: xy <x> <y>  |  q (quit)")
    import cv2

    try:
        while True:
            ekf, mask, color = perception_step(
                wf, pipe, K_top, dist_top, serial_top, depth_scale, ekf
            )

            print(f"[EKF] xy=({ekf.mean[0]:.3f}, {ekf.mean[1]:.3f})  "
                  f"yaw={np.degrees(ekf.mean[2]):.1f}°  "
                  f"sigma_xy=({np.sqrt(ekf.cov[0,0]):.3f}, {np.sqrt(ekf.cov[1,1]):.3f})")

            overlay = color.copy()
            overlay[mask] = (overlay[mask] * 0.5 + np.array([0, 255, 0]) * 0.5).astype(np.uint8)
            cv2.imshow("perception", overlay)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

            # Non-blocking command input
            import select
            if select.select([sys.stdin], [], [], 0.0)[0]:
                line = sys.stdin.readline().strip()
                parts = line.split()
                if parts and parts[0] == "xy" and len(parts) == 3:
                    target_xy = np.array([float(parts[1]), float(parts[2])])
                    import jax.numpy as jnp
                    target_q = np.array(ik_from_cache(jnp.array(target_xy), ik_cache))
                    if robot is not None and check_jump(q_prev, target_q):
                        robot.move(franky.JointMotion(target_q.tolist()))
                        q_prev = target_q
                        print(f"Moved to xy={target_xy}")

    finally:
        pipe.stop()
        cv2.destroyAllWindows()
        if robot is not None:
            robot.move(franky.JointMotion(HOME_Q.tolist()))
            print("Returned to home.")


if __name__ == "__main__":
    main()
