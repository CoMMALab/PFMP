import argparse
import time
import cv2
import numpy as np

from camera import (
    TEE_TAG_ID,
    WorldFrame,
    SERIAL_NUMS_INV,
    detect_apriltags,
    get_depth_scale,
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
from viz import draw_tee_wireframe_on_image, plot_tee_world_with_backend

import matplotlib
matplotlib.use('TkAgg')


"""
Localize the tee in the world frame.
"""
def _depth_to_colormap(depth_image: np.ndarray) -> np.ndarray:
    depth = depth_image.astype(np.float32)
    valid = depth > 0
    if not np.any(valid):
        return np.zeros((*depth_image.shape, 3), dtype=np.uint8)
    vis = np.zeros_like(depth, dtype=np.uint8)
    valid_depth = depth[valid]
    d_min = float(np.percentile(valid_depth, 2))
    d_max = float(np.percentile(valid_depth, 98))
    if d_max <= d_min:
        d_max = d_min + 1.0
    vis[valid] = np.clip((depth[valid] - d_min) * 255.0 / (d_max - d_min), 0, 255).astype(np.uint8)
    return cv2.applyColorMap(vis, cv2.COLORMAP_TURBO)


def _depth_to_world_points(
    depth_image: np.ndarray,
    K: np.ndarray,
    t_cam_to_world: np.ndarray,
    depth_scale: float,
    stride: int,
    min_depth_m: float,
    max_depth_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    depth_m = depth_image.astype(np.float64) * float(depth_scale)
    h, w = depth_m.shape
    ys = np.arange(0, h, max(1, int(stride)), dtype=np.int32)
    xs = np.arange(0, w, max(1, int(stride)), dtype=np.int32)
    u, v = np.meshgrid(xs, ys)
    z = depth_m[v, u]
    valid = (z > min_depth_m) & (z < max_depth_m)
    if not np.any(valid):
        return np.empty((0, 3), dtype=np.float64), np.empty((0,), dtype=np.float64)

    u = u[valid].astype(np.float64)
    v = v[valid].astype(np.float64)
    z = z[valid]

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    x = (u - cx) / fx * z
    y = (v - cy) / fy * z
    pts_cam = np.column_stack([x, y, z])
    pts_cam_h = np.column_stack([pts_cam, np.ones(len(pts_cam), dtype=np.float64)])
    pts_world = (t_cam_to_world @ pts_cam_h.T).T[:, :3]
    return pts_world, z


def localize_tee(args):
    wf = WorldFrame.load("wf.npz")
    if not wf.is_anchored():
        raise RuntimeError(
            "wf.npz is unanchored (t_aruco = I). Run anchor.py to produce a "
            "robot/world-anchored wf.npz before localizing the tee."
        )
    # Saved world frame stores camera-to-world transforms.
    cam_to_world = {SERIAL_NUMS_INV[s]: T for s, T in wf.t_cams.items()}
    missing_pc_cams = [cam for cam in args.pc_cams if cam not in args.cams]
    if missing_pc_cams:
        print(
            f"[TEE][pc][WARN] requested pc cams not in --cams and will be ignored: {missing_pc_cams}"
        )

    pipelines, intrinsics = init_cameras(args.cams)
    depth_scales = {}
    if args.viz_tool == "o3d":
        for cam in args.cams:
            if args.depth_scale is None:
                depth_scales[cam] = get_depth_scale(pipelines[cam])
            else:
                depth_scales[cam] = float(args.depth_scale)
            print(f"[TEE][pc] {cam}: depth_scale={depth_scales[cam]:.8f} m/unit")
    try:
        tee_tag_size_m = args.tee_tag_size_m
        print(f"[TEE] using tag id={TEE_TAG_ID} size={tee_tag_size_m:.4f} m")
        time.sleep(1.0)  # warm up sensors before one-shot capture
        tee_positions_world = []
        tee_transforms_world = []
        raw_tee_world = []
        rgb_panels = []
        depth_panels = []
        panel_h, panel_w = 480, 640
        depth_points_world_by_cam = {}
        depth_values_by_cam = {}
        pc_colors = {
            "top": (0.9, 0.25, 0.25),
            "corner": (0.25, 0.9, 0.25),
            "side": (0.25, 0.45, 0.95),
        }

        for cam in args.cams:
            frames = get_rgbd_frames(pipelines[cam], cam, retries=4, timeout_ms=10000)
            if frames is None:
                print(f"[WARN] {cam}: no RGBD frame, skipping Tee localization for this camera")
                color_image = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
                depth_vis = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
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
                cv2.putText(
                    depth_vis,
                    f"{cam} depth (no frame)",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )
                rgb_panels.append(color_image)
                depth_panels.append(depth_vis)
                continue

            color_image, depth_image = frames
            depth_vis = _depth_to_colormap(depth_image)
            K, dist = intrinsics[cam]
            if args.viz_tool == "o3d":
                pts_world, z_m = _depth_to_world_points(
                    depth_image=depth_image,
                    K=K,
                    t_cam_to_world=cam_to_world[cam],
                    depth_scale=depth_scales[cam],
                    stride=args.pc_stride,
                    min_depth_m=args.pc_min_depth_m,
                    max_depth_m=args.pc_max_depth_m,
                )
                depth_points_world_by_cam[cam] = pts_world
                depth_values_by_cam[cam] = z_m
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
                cv2.putText(
                    depth_vis,
                    f"{cam} depth",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                rgb_panels.append(color_image)
                depth_panels.append(depth_vis)
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
            cv2.putText(
                depth_vis,
                f"{cam} depth",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            rgb_panels.append(color_image)
            depth_panels.append(depth_vis)
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
                if rgb_panels and depth_panels:
                    canvas = np.vstack((np.hstack(rgb_panels), np.hstack(depth_panels)))
                    cv2.imshow("Tee Localization Overlays (RGB top row, Depth bottom row)", canvas)
                    print("RGB/depth overlays shown. Press any key to continue to 3D plot.")
                    cv2.waitKey(0)
                    cv2.destroyWindow("Tee Localization Overlays (RGB top row, Depth bottom row)")
                print(f"[TEE][world] plotting raw Tee poses and fused mean pose ({args.viz_tool})")
                world_point_clouds = None
                if args.viz_tool == "o3d":
                    world_point_clouds = []
                    total_points = 0
                    for cam_name in args.pc_cams:
                        pts = depth_points_world_by_cam.get(cam_name)
                        z_vals = depth_values_by_cam.get(cam_name)
                        if pts is None or z_vals is None or pts.size == 0:
                            continue
                        total_points += len(pts)
                        # Depth gradient colors (Turbo colormap), blended with camera tint.
                        z_norm = (z_vals - args.pc_min_depth_m) / max(
                            1e-6, args.pc_max_depth_m - args.pc_min_depth_m
                        )
                        z_norm = np.clip(z_norm, 0.0, 1.0)
                        depth_u8 = np.round(z_norm * 255.0).astype(np.uint8).reshape(-1, 1)
                        depth_bgr = cv2.applyColorMap(depth_u8, cv2.COLORMAP_TURBO).reshape(-1, 3)
                        depth_rgb = depth_bgr[:, ::-1].astype(np.float64) / 255.0
                        cam_rgb = np.array(pc_colors.get(cam_name, (0.7, 0.7, 0.7)), dtype=np.float64)
                        blended_rgb = np.clip(0.55 * depth_rgb + 0.45 * cam_rgb[None, :], 0.0, 1.0)
                        world_point_clouds.append(
                            (cam_name, pts, blended_rgb)
                        )
                    print(
                        f"[TEE][world] fused depth cloud points: {total_points} "
                        f"from cams={args.pc_cams}"
                    )
                plot_tee_world_with_backend(
                    wf,
                    raw_tee_world,
                    mean_T,
                    viz_tool=args.viz_tool,
                    world_point_clouds=world_point_clouds,
                )
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
    parser.add_argument(
        "--viz-tool",
        choices=["mpl", "o3d"],
        default="mpl",
        help="3D visualization backend (default: mpl).",
    )
    parser.add_argument(
        "--depth-scale",
        type=float,
        default=None,
        help=(
            "Override depth conversion scale (meters per raw unit). "
            "Default: auto-detect exact per-camera scale from RealSense sensor."
        ),
    )
    parser.add_argument(
        "--pc-cams",
        nargs="+",
        choices=["top", "corner", "side"],
        default=["top", "corner", "side"],
        help="Which camera point clouds to include in O3D scene.",
    )
    parser.add_argument(
        "--pc-stride",
        type=int,
        default=4,
        help="Subsample stride for depth point cloud (default: 4).",
    )
    parser.add_argument(
        "--pc-min-depth-m",
        type=float,
        default=0.1,
        help="Minimum valid depth for point cloud in meters (default: 0.1).",
    )
    parser.add_argument(
        "--pc-max-depth-m",
        type=float,
        default=3.0,
        help="Maximum valid depth for point cloud in meters (default: 3.0).",
    )
    args = parser.parse_args()
    localize_tee(args)