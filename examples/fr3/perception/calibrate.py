import time
import argparse
import numpy as np
import cv2
from camera import *
from viz import print_world_coordinates, plot_worldframe

"""
Compute anchored per-camera extrinsics from a single multi-view AprilTag observation.

Detects the table AprilTag from all cameras, Procrustes-registers depth-backprojected
corners to the known tag geometry in robot frame (T_ARUCO). Outputs an anchored
WorldFrame (wf.npz) ready for use — no separate anchor step needed.
"""

# Camera label -> WorldFrame slot index (cam1=top, cam2=corner, cam3=side).
CAM_TO_IDX = {"top": 1, "corner": 2, "side": 3}
DEPTH_SCALE_M = 1e-3   # RealSense: raw depth units → metres

def fit_ransac(
    wf: WorldFrame,
    depth: np.ndarray,
    cam: str,
    min_inliers: int = 500,
    n_iters: int = 100,
    thresh_m: float = 0.005,
) -> "tuple[np.ndarray, float] | None":
    """Fit a plane to the table surface in camera frame via RANSAC.

    Returns (normal, d) where normal·x = d defines the plane, or None if
    fewer than min_inliers agree on any candidate.
    """
    K = wf._K_cam(CAM_TO_IDX[cam])
    H, W = depth.shape
    d = depth.astype(np.float64) * DEPTH_SCALE_M
    valid = (d > 0.01) & np.isfinite(d)

    ys, xs = np.mgrid[0:H, 0:W]
    pts = np.stack([
        (xs - K[0, 2]) / K[0, 0] * d,
        (ys - K[1, 2]) / K[1, 1] * d,
        d,
    ], axis=-1)[valid]   # (N, 3) in camera frame

    N = len(pts)
    if N < min_inliers:
        return None

    rng = np.random.default_rng(0)
    best_normal, best_d, best_count = None, 0.0, 0

    for _ in range(n_iters):
        idx = rng.choice(N, 3, replace=False)
        p0, p1, p2 = pts[idx]
        n = np.cross(p1 - p0, p2 - p0)
        norm = np.linalg.norm(n)
        if norm < 1e-9:
            continue
        n = n / norm
        dval = float(n @ p0)
        residuals = np.abs(pts @ n - dval)
        count = int((residuals < thresh_m).sum())
        if count > best_count:
            best_normal, best_d, best_count = n, dval, count

    if best_count < min_inliers:
        return None

    # Least-squares refit on inliers (subsample to keep SVD tractable)
    inliers = pts[np.abs(pts @ best_normal - best_d) < thresh_m]
    if len(inliers) > 10_000:
        inliers = inliers[rng.choice(len(inliers), 10_000, replace=False)]
    _, _, Vt = np.linalg.svd(inliers - inliers.mean(axis=0))
    normal = Vt[-1]
    if normal @ best_normal < 0:
        normal = -normal
    d_fit = float(normal @ inliers.mean(axis=0))
    print(f"[RANSAC] {cam}: {best_count} inliers, normal={normal}, d={d_fit:.4f}m")
    return normal, d_fit

def calibrate(args):
    tag_configs = {
        "table": (TABLE_TAG_ID, args.table_tag_size_m),
        "tee":   (TEE_TAG_ID,   args.tee_tag_size_m),
    }
    if args.tag not in tag_configs:
        raise ValueError(f"Unknown --tag value: {args.tag}")
    tag_id, tag_size_m = tag_configs[args.tag]
    print(f"[CALIB] using {args.tag} tag id={tag_id} size={tag_size_m:.4f} m")

    pipelines, intrinsics = init_cameras(args.cams)
    time.sleep(1)

    # 1. Detect the AprilTag from each camera, accumulate into AprilDetector
    detector = AprilDetector(tag_id=tag_id, tag_size_m=tag_size_m)
    color_images = {}
    depth_images = {}
    try:
        for cam in args.cams:
            K, dist = intrinsics[cam]
            serial = SERIAL_NUMS[cam]
            ok = False
            for attempt in range(60):
                rgbd = get_rgbd_frames(pipelines[cam], cam, retries=4, timeout_ms=10000)
                ok, detector = detector.detect(serial, rgbd, K, dist)
                if ok:
                    break
                print(f"[{cam}] attempt {attempt + 1}/60: tag not detected, retrying...")
            if not ok:
                raise RuntimeError(f"[{cam}] {args.tag} tag id={tag_id} not detected after 60 attempts")
            print(f"[{cam}] detected, corners_3d:\n{detector.corners[serial]}")
            color_images[cam] = rgbd[0].copy()
            depth_images[cam] = rgbd[1].copy()
    finally:
        for pipe in pipelines.values():
            pipe.stop()

    # 2. Build anchored WorldFrame directly via Procrustes registration
    wf = WorldFrame.anchor(detector, T_ARUCO)

    # 3. Fit table plane per camera via RANSAC (in camera frame)
    planes = {}
    for cam in args.cams:
        result = fit_ransac(wf, depth_images[cam], cam)
        if result is not None:
            planes[cam] = result

    # 4. Recover 2D keypoints from stored 3D camera-frame corners for visualization
    keypoints = {}
    keypoint_depths = {}
    for cam in args.cams:
        serial = SERIAL_NUMS[cam]
        pts_cam = detector.corners[serial]           # (4, 3)
        K = detector.K[serial]
        keypoints[cam] = np.column_stack([
            pts_cam[:, 0] / pts_cam[:, 2] * K[0, 0] + K[0, 2],
            pts_cam[:, 1] / pts_cam[:, 2] * K[1, 1] + K[1, 2],
        ])
        keypoint_depths[cam] = pts_cam[:, 2]

    # Unproject each src camera's tag corners into world, then project into every dst camera.
    reproj_keypoints = {dst: {} for dst in args.cams}
    for src in args.cams:
        for dst in args.cams:
            if src == dst:
                continue
            reproj_keypoints[dst][src] = wf.reproject(
                keypoints[src], keypoint_depths[src], CAM_TO_IDX[src], CAM_TO_IDX[dst],
            )

    # L2 distance (pixels) between observed and reprojected corners for each src->dst pair.
    print("[CALIB] pairwise reprojection errors (Procrustes):")
    for dst in args.cams:
        for src in args.cams:
            if src == dst:
                continue
            diff = keypoints[dst] - reproj_keypoints[dst][src]
            print(f"  {src}_to_{dst}: {np.linalg.norm(diff):.3f}")

    return wf, keypoints, reproj_keypoints, color_images, planes

def visualize_keypoints(wf: WorldFrame, keypoints: dict, reproj_keypoints: dict, color_images: dict):
    def draw_box(img, pts, color, label):
        pts = np.round(pts).astype(int)
        pts = pts.reshape(-1, 1, 2)
        cv2.polylines(img, [pts], isClosed=True, color=color, thickness=2)
        anchor = tuple(pts[0, 0])
        cv2.putText(
            img,
            label,
            (anchor[0] + 5, anchor[1] - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )

    src_colors = {
        "top": (0, 0, 255),      # red
        "corner": (255, 0, 0),   # blue
        "side": (0, 165, 255),   # orange
    }
    detected_color = (0, 255, 0)  # green
    order = ["top", "corner", "side"]

    vis_panels = []
    for dst in order:
        if dst not in color_images:
            continue
        img = color_images[dst].copy()
        draw_box(img, keypoints[dst], detected_color, f"{dst} detected")
        for src in order:
            if src == dst or src not in reproj_keypoints.get(dst, {}):
                continue
            draw_box(img, reproj_keypoints[dst][src], src_colors[src], f"{src}->{dst} reproj")
        vis_panels.append(img)

    if not vis_panels:
        print("[CALIB] no visualization panels to show")
        return

    h = min(panel.shape[0] for panel in vis_panels)
    panels = [cv2.resize(panel, (int(panel.shape[1] * h / panel.shape[0]), h)) for panel in vis_panels]
    canvas = np.hstack(panels)
    cv2.imshow("Calibration Keypoints: detected (green) + reprojected", canvas)
    print("Calibration reprojection view shown. Press any key to close.")
    cv2.waitKey(0)
    cv2.destroyWindow("Calibration Keypoints: detected (green) + reprojected")

def visualize_planes_meshcat(wf: WorldFrame, planes: dict):
    """Show camera frames and fitted table planes in meshcat."""
    import meshcat
    import meshcat.geometry as g

    vis = meshcat.Visualizer()
    print(f"[CALIB] meshcat: {vis.url()}")

    cam_colors = {"top": 0xff0000, "corner": 0x00cc00, "side": 0x0000ff}
    cam_to_t = {"top": wf.t_cam1, "corner": wf.t_cam2, "side": wf.t_cam3}

    for cam, T in cam_to_t.items():
        vis[f"cam/{cam}"].set_object(
            g.triad(scale=0.05),
        )
        vis[f"cam/{cam}"].set_transform(T)

    # Table frame (T_ARUCO)
    vis["aruco"].set_object(g.triad(scale=0.1))
    vis["aruco"].set_transform(wf.t_aruco)

    # Fitted planes: draw a square patch centred at the closest point on the plane,
    # oriented so its normal matches the fitted normal, transformed to world frame.
    for cam, (normal_cam, d_cam) in planes.items():
        T_cam = cam_to_t[cam]
        # Centre of plane patch in camera frame: closest point to origin
        centre_cam = normal_cam * d_cam
        centre_world = (T_cam @ np.append(centre_cam, 1.0))[:3]

        # Rotate plane normal (camera frame) to world frame
        normal_world = T_cam[:3, :3] @ normal_cam
        normal_world /= np.linalg.norm(normal_world)

        # Build a transform that puts the xy-plane patch at centre_world, normal along z
        z = normal_world
        x = np.array([1.0, 0.0, 0.0]) if abs(z[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        x -= x.dot(z) * z; x /= np.linalg.norm(x)
        y = np.cross(z, x)
        R_patch = np.eye(4)
        R_patch[:3, 0] = x; R_patch[:3, 1] = y; R_patch[:3, 2] = z
        R_patch[:3, 3] = centre_world

        vis[f"plane/{cam}"].set_object(
            g.Box([0.4, 0.4, 0.002]),
            g.MeshLambertMaterial(color=cam_colors[cam], opacity=0.4, transparent=True),
        )
        vis[f"plane/{cam}"].set_transform(R_patch)

    input("Open browser at the printed URL. Press Enter to exit.")


def main(args):
    if args.only_plot:
        wf = WorldFrame.load(args.output_path)
        print_world_coordinates(wf)
        plot_worldframe(wf, viz_tool="mpl")
        return

    wf, keypoints, reproj_keypoints, color_images, planes = calibrate(args)
    print("[CALIB] saving anchored WorldFrame (world = robot frame via T_ARUCO).")
    print_world_coordinates(wf)

    if not args.no_save:
        wf.save(args.output_path)
        print(f"[CALIB] wrote {args.output_path}")

    if not args.no_plot:
        plot_worldframe(wf, viz_tool=args.viz_tool)
        visualize_keypoints(wf, keypoints, reproj_keypoints, color_images)
        visualize_planes_meshcat(wf, planes)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Compute anchored multi-camera extrinsics from a single AprilTag observation. "
            "Outputs a robot-frame wf.npz directly (no separate anchor step needed)."
        )
    )
    parser.add_argument("--cams", nargs="+", default=["top", "corner", "side"])
    parser.add_argument(
        "--tag", choices=["table", "tee"], default="table",
        help="Which AprilTag to calibrate against. Default: table.",
    )
    parser.add_argument("--table-tag-size-m", type=float, default=0.1651)
    parser.add_argument("--tee-tag-size-m", type=float, default=0.0445)
    parser.add_argument("--only-plot", action="store_true",
        help="Skip calibration; load --output-path and plot it.")
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument(
        "--viz-tool",
        choices=["mpl", "o3d"],
        default="mpl",
        help="3D visualization backend for world-frame plot (default: mpl).",
    )
    parser.add_argument("--output-path", type=str, default="wf.npz")
    args = parser.parse_args()
    main(args)
