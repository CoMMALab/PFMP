"""
real_loop.py — Minimal real-robot execution loop with EKF perception.

Connects to the Franka via franky, starts cameras, runs the EKF perception
loop, and accepts simple XY waypoint commands. No planner — just the
infrastructure to verify perception ↔ execution integration.

Usage:
    python examples/fr3/real_loop.py --ip 172.16.0.2
    python examples/fr3/real_loop.py --dry-run
"""

import os
os.environ["JAX_PLATFORMS"] = "cpu"

import argparse
import select
import sys
import threading
import time
from typing import NamedTuple, Optional

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "perception"))

from camera import WorldFrame, init_cameras, get_rgbd_frames, get_depth_scale, SERIAL_NUMS
from geom_tee import THICKNESS, TEE_TAG_SIZE_M
from ekf import EKF, PoseDetection
from april_detector import detect_tee_tag
from template_detector import segment_T, template_match_T, segmentation_median_world_xy, project
from main import project_world_xy, draw_pose_marker, CYAN

sys.path.insert(0, os.path.dirname(__file__))
from ik_cache import ik_from_cache, get_or_build_cache, HOME_Q, EE_HEIGHT


ROBOT_IP = "172.16.0.3"
VELOCITY_SCALE = 0.05
MAX_JUMP_DEG = 20.0
XY_STEP_M = 0.025


class Snapshot(NamedTuple):
    color: Optional[np.ndarray]
    depth: Optional[np.ndarray]
    seq: int
    mask: Optional[np.ndarray]
    ekf_mean: Optional[np.ndarray]
    path_xy: Optional[np.ndarray]


class SharedSnapshot:
    def __init__(self):
        self.lock = threading.Lock()
        self.snap = Snapshot(
            color=None,
            depth=None,
            seq=0,
            mask=None,
            ekf_mean=None,
            path_xy=None,
        )

    def update_frame(self, color: np.ndarray, depth: np.ndarray):
        with self.lock:
            self.snap = self.snap._replace(
                color=color.copy(),
                depth=depth.copy(),
                seq=self.snap.seq + 1,
            )

    def update_perception(
        self,
        mask: Optional[np.ndarray],
        ekf_mean: Optional[np.ndarray],
    ):
        with self.lock:
            self.snap = self.snap._replace(
                mask=None if mask is None else mask.copy(),
                ekf_mean=None if ekf_mean is None else np.asarray(ekf_mean, dtype=float).copy(),
            )

    def update_path(self, path_xy: Optional[np.ndarray]):
        with self.lock:
            self.snap = self.snap._replace(
                path_xy=None if path_xy is None else np.asarray(path_xy, dtype=float).copy()
            )

    def get(self) -> Snapshot:
        with self.lock:
            return Snapshot(
                color=None if self.snap.color is None else self.snap.color.copy(),
                depth=None if self.snap.depth is None else self.snap.depth.copy(),
                seq=self.snap.seq,
                mask=None if self.snap.mask is None else self.snap.mask.copy(),
                ekf_mean=None if self.snap.ekf_mean is None else self.snap.ekf_mean.copy(),
                path_xy=None if self.snap.path_xy is None else self.snap.path_xy.copy(),
            )


class CameraVisConfig(NamedTuple):
    wf: WorldFrame
    pipe: object
    serial_top: str
    K_top: np.ndarray


def camera_vis_loop(
    cfg: CameraVisConfig,
    shared: SharedSnapshot,
    stop_event: threading.Event,
):
    import cv2

    while not stop_event.is_set():
        color, depth = get_rgbd_frames(cfg.pipe, "top")
        shared.update_frame(color, depth)

        snap = shared.get()
        overlay = color.copy()

        if snap.mask is not None and snap.mask.shape[:2] == overlay.shape[:2]:
            overlay[snap.mask] = (
                overlay[snap.mask] * 0.5 + np.array([0, 255, 0]) * 0.5
            ).astype(np.uint8)

        if snap.path_xy is not None and len(snap.path_xy) > 0:
            T_w2c = np.linalg.inv(cfg.wf.t_cams[cfg.serial_top])

            pix = project(
                verts_w=snap.path_xy,
                body_z=EE_HEIGHT,
                T_w2c=T_w2c,
                K=cfg.K_top,
            )

            if len(pix) >= 2:
                cv2.polylines(
                    overlay,
                    [pix.reshape(-1, 1, 2)],
                    isClosed=False,
                    color=(0, 0, 255),
                    thickness=2,
                )

            for i, (u, v) in enumerate(pix):
                cv2.circle(
                    overlay,
                    (int(u), int(v)),
                    6 if i == 0 else 4,
                    (0, 255, 0) if i == 0 else (255, 0, 0),  # first green, rest blue
                    -1,
                )

        if snap.ekf_mean is not None:
            T_w2c = np.linalg.inv(cfg.wf.t_cams[cfg.serial_top])

            draw_pose_marker(
                image=overlay,
                xy=snap.ekf_mean[:2],
                yaw=float(snap.ekf_mean[2]),
                z_body=THICKNESS / 2,
                T_w2c=T_w2c,
                K=cfg.K_top,
                color=CYAN,
                outline_thickness=3,
                center_radius=7,
            )

        cv2.imshow("perception", overlay)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            stop_event.set()
            break


def check_jump(q_prev: np.ndarray, q_next: np.ndarray) -> bool:
    diff = np.abs((q_next - q_prev + np.pi) % (2 * np.pi) - np.pi)
    worst = float(np.degrees(diff.max()))
    if worst > MAX_JUMP_DEG:
        print(f"[WARN] joint jump {worst:.1f}° > {MAX_JUMP_DEG}° — skipping")
        return False
    return True


def perception_step_from_frame(
    wf,
    color,
    depth,
    K_top,
    dist_top,
    serial_top,
    depth_scale,
    ekf,
):
    """One perception update using an already-captured frame."""
    mask = segment_T(
        frame=wf,
        color=color,
        depth=depth,
        depth_scale=depth_scale,
        cam="top",
    )

    detections = []

    tag_det = detect_tee_tag(
        color=color,
        depth=depth,
        K=K_top,
        dist=dist_top,
        wf=wf,
        serial=serial_top,
        depth_scale=depth_scale,
        name="tag",
        R=np.diag([1e-4, 1e-4, 1e-3]),
        gated=False,
    )
    if tag_det:
        detections.append(tag_det)

    if ekf.is_lost:
        seed_xy = segmentation_median_world_xy(
            mask=mask,
            depth=depth,
            K=K_top,
            T_c2w=wf.t_cams[serial_top],
            depth_scale=depth_scale,
        )
        if seed_xy is not None:
            ekf = ekf.reset(seed_xy)
            print(f"[EKF] reset to seed xy={seed_xy}")

    best_xy, best_yaw, best_iou, _ = template_match_T(
        mask=mask,
        frame=wf,
        serial=serial_top,
        K=K_top,
        body_xy=ekf.mean[:2],
        body_yaw=ekf.mean[2],
        body_z=THICKNESS / 2,
    )

    if best_iou >= 0.25:
        detections.append(PoseDetection(
            name="template",
            pose=np.array([best_xy[0], best_xy[1], best_yaw]),
            R=np.diag([1e-4, 1e-4, 1e-3]),
            gated=False,
        ))

    ekf = ekf.update(detections)
    return ekf, mask

def robot_current_xy(robot) -> np.ndarray:
    pose = robot.current_cartesian_state.pose.end_effector_pose
    return np.array([
        float(pose.translation[0]),
        float(pose.translation[1]),
    ])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", default=ROBOT_IP)
    parser.add_argument("--speed", type=float, default=VELOCITY_SCALE)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--wf", default="wf.npz")
    parser.add_argument("--cache", default=None)
    parser.add_argument("--xy-step", type=float, default=XY_STEP_M)  # unused with CartesianMotion

    # Default behavior: autonomous random target sampling.
    # Manual behavior: wait for stdin commands: xy <x> <y>, home, q.
    parser.add_argument("--manual", action="store_true")

    args = parser.parse_args()

    # ── World frame & IK cache ────────────────────────────────────────────────
    wf = WorldFrame.load(args.wf)
    cache_path = args.cache or os.path.join(
        os.path.dirname(__file__),
        "..",
        "ik_cache.pkl",
    )
    ik_cache = get_or_build_cache(path=cache_path)

    # ── Cameras ───────────────────────────────────────────────────────────────
    pipelines, intrinsics = init_cameras(["top"])
    pipe = pipelines["top"]
    K_top, dist_top = intrinsics["top"]
    serial_top = SERIAL_NUMS["top"]
    depth_scale = get_depth_scale(pipe)

    print("Warming up camera…")
    for _ in range(60):
        get_rgbd_frames(pipe, "top")

    # ── Shared camera/display thread ──────────────────────────────────────────
    shared = SharedSnapshot()
    stop_event = threading.Event()

    vis_cfg = CameraVisConfig(
        wf=wf,
        pipe=pipe,
        serial_top=serial_top,
        K_top=K_top,
    )

    vis_thread = threading.Thread(
        target=camera_vis_loop,
        args=(vis_cfg, shared, stop_event),
        daemon=True,
    )
    vis_thread.start()

    # ── Robot ─────────────────────────────────────────────────────────────────
    robot = None
    q_prev = np.array(HOME_Q)
    current_xy = None  # None means: not on the pushing plane yet

    if not args.dry_run:
        import franky

        print(f"Connecting to {args.ip}…")
        robot = franky.Robot(args.ip)
        robot.relative_dynamics_factor = args.speed
        robot.recover_from_errors()
        robot.move(franky.JointMotion(HOME_Q.tolist()))
        q_prev = np.array(robot.current_joint_state.position, dtype=float)
        print("At home. Not on pushing plane.")

    # ── EKF init ──────────────────────────────────────────────────────────────
    ekf = EKF.init(wf)
    last_ekf_seq = -1

    rng = np.random.default_rng()

    if args.manual:
        print("\nLoop running in MANUAL mode.")
        print("Commands: xy <x> <y>  |  home  |  q")
    else:
        print("\nLoop running in AUTO random-target mode.")
        print("Commands: q")

    def go_home():
        nonlocal current_xy, q_prev

        if robot is not None:
            robot.recover_from_errors()
            robot.move(franky.JointMotion(HOME_Q.tolist()))
            q_prev = np.array(robot.current_joint_state.position, dtype=float)
        else:
            q_prev = np.array(HOME_Q)

        current_xy = None
        shared.update_path(None)
        print("Returned to home. Exited pushing plane.")

    def execute_xy(target_xy: np.ndarray):
        nonlocal current_xy, q_prev

        import jax.numpy as jnp

        target_xy = np.asarray(target_xy, dtype=float)

        if current_xy is None:
            # From HOME / non-pushing pose: descend directly to the requested xy
            # point on the pushing plane. Do not interpolate from HOME's
            # projected xy.
            q_target = np.array(ik_from_cache(jnp.array(target_xy), ik_cache))

            shared.update_path(np.array([target_xy]))

            if robot is not None:
                robot.recover_from_errors()
                robot.move(franky.JointMotion(q_target.tolist()))
                q_prev = np.array(robot.current_joint_state.position, dtype=float)
                current_xy = robot_current_xy(robot)
            else:
                q_prev = q_target
                current_xy = target_xy.copy()

            print(f"Entered pushing plane at xy={current_xy}, commanded={target_xy}")
            return

        # Already on the pushing plane: move linearly in Cartesian space.
        path_xy = np.vstack([current_xy, target_xy])
        shared.update_path(path_xy)

        if robot is not None:
            delta_xy = target_xy - current_xy

            while True:
                try:
                    robot.move(franky.CartesianMotion(
                        franky.Affine([
                            float(delta_xy[0]),
                            float(-delta_xy[1]),  # franky expects negated y coord
                            0.0,
                        ]),
                        franky.ReferenceType.Relative,
                    ))
                    break
                except Exception as e:
                    robot.recover_from_errors()
                    print(f"[WARN] motion attempt failed: {e}")

            q_prev = np.array(robot.current_joint_state.position, dtype=float)
            current_xy = robot_current_xy(robot)
        else:
            q_prev = np.array(ik_from_cache(jnp.array(target_xy), ik_cache))
            current_xy = target_xy.copy()

        print(f"Moved linearly to xy={current_xy}, commanded={target_xy}")

    try:
        while not stop_event.is_set():
            snap = shared.get()

            if (
                snap.color is not None
                and snap.depth is not None
                and snap.seq != last_ekf_seq
            ):
                ekf, mask = perception_step_from_frame(
                    wf=wf,
                    color=snap.color,
                    depth=snap.depth,
                    K_top=K_top,
                    dist_top=dist_top,
                    serial_top=serial_top,
                    depth_scale=depth_scale,
                    ekf=ekf,
                )

                shared.update_perception(mask=mask, ekf_mean=ekf.mean.copy())
                last_ekf_seq = snap.seq

            if args.manual:
                # Manual mode: consume stdin only when user typed a command.
                if select.select([sys.stdin], [], [], 0.0)[0]:
                    line = sys.stdin.readline().strip()
                    parts = line.split()

                    if not parts:
                        continue

                    if parts[0] == "q":
                        stop_event.set()
                        break

                    if parts[0] == "home":
                        go_home()
                        continue

                    if parts[0] == "xy" and len(parts) == 3:
                        target_xy = np.array([float(parts[1]), float(parts[2])])
                        execute_xy(target_xy)
                        continue

                    print(f"[WARN] unknown command: {line}")

            else:
                # Auto mode: preserve your current behavior.
                # It continuously samples random xy targets and executes them.
                target_xy = rng.normal([0.4, 0.0], 0.025, size=2)
                execute_xy(target_xy)

            time.sleep(0.001)

    finally:
        stop_event.set()
        vis_thread.join(timeout=1.0)

        try:
            pipe.stop()
        except Exception as e:
            print(f"[WARN] could not stop camera pipe: {e}")

        try:
            import cv2
            cv2.destroyAllWindows()
        except Exception:
            pass

        if robot is not None:
            try:
                robot.recover_from_errors()
                robot.move(franky.JointMotion(HOME_Q.tolist()))
                current_xy = None
                shared.update_path(None)
                print("Returned to home. Exited pushing plane.")
            except Exception as e:
                print(f"[WARN] could not return to home: {e}")

if __name__ == "__main__":
    main()