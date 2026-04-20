import argparse
import time
import cv2
import numpy as np

from camera import (
    TEE_TAG_ID,
    WorldFrame,
    detect_apriltags,
    get_rgbd_frames,
    get_tag_transform_by_id,
    init_cameras,
)
from utils import (
    average_transforms,
    enforce_world_z_yaw_only,
    fuse_yaw_only_transforms_world,
    rotation_matrix_to_rpy_zyx_deg,
)
from viz import draw_tee_wireframe_on_image, plot_tee_world

import matplotlib
matplotlib.use('TkAgg')


"""
Localize the tee in the world frame.
"""
def localize_tee(args):
    wf = WorldFrame.load("wf.npz")
    if not wf.is_anchored():
        raise RuntimeError(
            "wf.npz is unanchored (t_aruco = I). Run anchor.py to produce a "
            "robot/world-anchored wf.npz before localizing the tee."
        )
    # Saved world frame stores camera-to-world transforms.
    cam_to_world = {
        "top": wf.t_cam1,
        "corner": wf.t_cam2,
        "side": wf.t_cam3,
    }

    pipelines, intrinsics = init_cameras(args.cams)
    try:
        tee_tag_size_m = args.tee_tag_size_m
        print(f"[TEE] using tag id={TEE_TAG_ID} size={tee_tag_size_m:.4f} m")
        time.sleep(1.0)  # warm up sensors before one-shot capture
        tee_positions_world = []
        tee_transforms_world = []
        raw_tee_world = []
        overlay_panels = []
        panel_h, panel_w = 480, 640

        for cam in args.cams:
            frames = get_rgbd_frames(pipelines[cam], cam, retries=4, timeout_ms=10000)
            if frames is None:
                print(f"[WARN] {cam}: no RGBD frame, skipping Tee localization for this camera")
                color_image = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
                cv2.putText(
                    color_image,
                    f"{cam} (no frame)",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )
                overlay_panels.append(color_image)
                continue

            color_image, _ = frames
            K, dist = intrinsics[cam]
            detections, _, debug = detect_apriltags(color_image, K, dist, tee_tag_size_m, staged=True, return_debug=True)
            tee_detections = [det for det in detections if det[0] == TEE_TAG_ID]

            if tee_detections:
                cv2.aruco.drawDetectedMarkers(
                    color_image,
                    [det[2] for det in tee_detections],
                    np.array([TEE_TAG_ID] * len(tee_detections)),
                )
                for _, t_tee_to_cam_det, _, rvec, tvec in tee_detections:
                    cv2.drawFrameAxes(color_image, K, dist, rvec, tvec, tee_tag_size_m * 0.6)
                    draw_tee_wireframe_on_image(color_image, K, dist, t_tee_to_cam_det)

            t_tee_to_cam = get_tag_transform_by_id(tee_detections, TEE_TAG_ID)
            if t_tee_to_cam is None:
                print(f"[WARN] {cam}: Tee tag id={TEE_TAG_ID} not detected")
                cv2.putText(
                    color_image,
                    f"{cam} | Tee tag {TEE_TAG_ID}: not detected",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )
                overlay_panels.append(color_image)
                continue

            # Tee origin in world frame: p_world = T_cam_to_world @ T_tee_to_cam @ [0,0,0,1].
            t_tee_to_world = cam_to_world[cam] @ t_tee_to_cam
            if args.yaw_only:
                t_tee_to_world = enforce_world_z_yaw_only(t_tee_to_world)
            p_tee_world_h = t_tee_to_world @ np.array([0.0, 0.0, 0.0, 1.0])
            p_tee_world = p_tee_world_h[:3]
            tee_positions_world.append(p_tee_world)
            tee_transforms_world.append(t_tee_to_world)
            raw_tee_world.append((cam, t_tee_to_world))
            roll_deg, pitch_deg, yaw_deg = rotation_matrix_to_rpy_zyx_deg(t_tee_to_world[:3, :3])
            cv2.putText(
                color_image,
                f"{cam} | Tee tag {TEE_TAG_ID}: detected ({len(tee_detections)})",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                color_image,
                f"world xyz: {p_tee_world[0]: .3f}, {p_tee_world[1]: .3f}, {p_tee_world[2]: .3f}",
                (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 0),
                2,
                cv2.LINE_AA,
            )
            overlay_panels.append(color_image)
            print(
                f"[TEE][world] {cam:6s} -> x={p_tee_world[0]: .4f}, "
                f"y={p_tee_world[1]: .4f}, z={p_tee_world[2]: .4f}"
            )
            print(
                f"[TEE][world] {cam:6s} -> roll={roll_deg: .1f} deg, "
                f"pitch={pitch_deg: .1f} deg, yaw={yaw_deg: .1f} deg (ZYX)"
            )

        if tee_positions_world:
            mean_pos = np.mean(np.stack(tee_positions_world), axis=0)
            mean_T = (
                fuse_yaw_only_transforms_world(tee_transforms_world)
                if args.yaw_only
                else average_transforms(tee_transforms_world)
            )
            print(
                f"[TEE][world] mean  -> x={mean_pos[0]: .4f}, "
                f"y={mean_pos[1]: .4f}, z={mean_pos[2]: .4f}"
            )
            if mean_T is not None:
                m_roll, m_pitch, m_yaw = rotation_matrix_to_rpy_zyx_deg(mean_T[:3, :3])
                print(
                    f"[TEE][world] mean  -> roll={m_roll: .1f} deg, "
                    f"pitch={m_pitch: .1f} deg, yaw={m_yaw: .1f} deg (ZYX)"
                )
                if overlay_panels:
                    canvas = np.hstack(overlay_panels)
                    cv2.imshow("Tee Localization Overlays (top | corner | side)", canvas)
                    print("Tee overlays shown. Press any key to continue to 3D plot.")
                    cv2.waitKey(0)
                    cv2.destroyWindow("Tee Localization Overlays (top | corner | side)")
                print("[TEE][world] plotting raw Tee poses and fused mean pose")
                plot_tee_world(wf, raw_tee_world, mean_T)
        else:
            print("[TEE] No valid Tee detections from selected cameras.")
    finally:
        for pipe in pipelines.values():
            pipe.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cams", nargs="+", default=["top", "corner", "side"])
    parser.add_argument("--tee_tag_size_m", type=float, default=0.0445)
    parser.add_argument(
        "--yaw-only",
        action="store_true",
        help="Constrain tee orientation to world Z yaw only (default: off).",
    )
    args = parser.parse_args()
    localize_tee(args)