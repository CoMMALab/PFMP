import time
import argparse
import numpy as np
import cv2
from camera import *
from viz import print_world_coordinates
from scipy.optimize import least_squares


"""
Compute anchored per-camera extrinsics from a single multi-view AprilTag observation.

Detects the table AprilTag from all cameras, Procrustes-registers depth-backprojected
corners to the known tag geometry in robot frame (T_ARUCO). Outputs an anchored
WorldFrame (wf.npz) ready for use — no separate anchor step needed.
"""

CAM_ORDER = ["top", "corner", "side"]  # matches WorldFrame cam1/cam2/cam3

def fit_ransac(
    wf: WorldFrame,
    depth: np.ndarray,
    cam: str,
    corners_cam: np.ndarray,       # (4, 3) PnP corners in camera frame
    depth_scale: float,
    radius_m: float = 0.4,
    min_inliers: int = 500,
    n_iters: int = 100,
    thresh_m: float = 0.005,
) -> "tuple[np.ndarray, float] | None":
    """Fit a plane to the table surface in camera frame via RANSAC.

    Only considers depth points within radius_m of the tag center (mean of corners),
    discarding background clutter. Returns (normal, d) where normal·x = d, or None.
    """
    K = wf._K_cam(CAM_ORDER.index(cam) + 1)
    H, W = depth.shape
    d = depth.astype(np.float64) * depth_scale
    valid = (d > 0.01) & np.isfinite(d)

    ys, xs = np.mgrid[0:H, 0:W]
    pts = np.stack([
        (xs - K[0, 2]) / K[0, 0] * d,
        (ys - K[1, 2]) / K[1, 1] * d,
        d,
    ], axis=-1)[valid]   # (N, 3) in camera frame

    tag_center = corners_cam.mean(axis=0)
    pts = pts[np.linalg.norm(pts - tag_center, axis=1) < radius_m]

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
    inlier_mask = np.abs(pts @ normal - d_fit) < thresh_m
    print(f"[RANSAC] {cam}: {best_count} inliers, normal={normal}, d={d_fit:.4f}m")
    return normal, d_fit, pts, inlier_mask   # pts in camera frame

def reprojection_loss(wf: WorldFrame, world_pts: np.ndarray, keypoints: dict) -> np.ndarray:
    """Reprojection residuals for tag corners across all cameras.

    Returns (24,): 3 cams × 4 corners × 2 (u, v).
    """
    cam_order = CAM_ORDER
    res = []
    for i, cam in enumerate(cam_order):
        K = wf._K_cam(i + 1)
        T_world_to_cam = np.linalg.inv(wf._t_cam(i + 1))
        pts_cam = (world_pts @ T_world_to_cam[:3, :3].T) + T_world_to_cam[:3, 3]
        u = pts_cam[:, 0] / pts_cam[:, 2] * K[0, 0] + K[0, 2]
        v = pts_cam[:, 1] / pts_cam[:, 2] * K[1, 1] + K[1, 2]
        res.append((np.column_stack([u, v]) - keypoints[cam]).ravel())
    return np.concatenate(res)


def planar_loss(wf: WorldFrame, planes: dict) -> np.ndarray:
    """Planar residuals for RANSAC inliers across all cameras.

    table_normal and table_d are implicit in wf.t_aruco.
    Returns (N_inliers_total,): one signed distance residual per inlier point.
    """
    table_normal = wf.t_aruco[:3, 2]
    table_d = table_normal @ wf.t_aruco[:3, 3]
    cam_to_t = {"top": wf.t_cam1, "corner": wf.t_cam2, "side": wf.t_cam3}
    res = []
    for cam, (_, _, pts_cam, inlier_mask) in planes.items():
        inlier_pts = pts_cam[inlier_mask]
        pts_h = np.column_stack([inlier_pts, np.ones(len(inlier_pts))])
        pts_world = (cam_to_t[cam] @ pts_h.T).T[:, :3]
        res.append(pts_world @ table_normal - table_d)
    return np.concatenate(res)


def bundle_adjust(wf: WorldFrame, world_pts: np.ndarray, keypoints: dict, planes: dict) -> WorldFrame:
    """Refine camera extrinsics via joint least squares over image reprojection + planar residuals.

    Params: 18 = 3 cameras × (rvec[3], tvec[3]) world→cam.
    Residuals: reprojection_loss (24,) + planar_loss (N_inliers,).
    """
    n_planar = sum(mask.sum() for _, _, _, mask in planes.values())
    planar_weight = float(np.sqrt(24.0 / max(n_planar, 1)))

    def residuals(x):
        wf_x = wf.update(x)
        return np.concatenate([
            reprojection_loss(wf_x, world_pts, keypoints),
            planar_weight * planar_loss(wf_x, planes),
        ])

    # Initialise x from current extrinsics
    x0 = np.zeros(18)
    for i in range(3):
        T_world_to_cam = np.linalg.inv(wf._t_cam(i + 1))
        rvec, _ = cv2.Rodrigues(T_world_to_cam[:3, :3])
        x0[6*i : 6*i+3] = rvec.ravel()
        x0[6*i+3 : 6*i+6] = T_world_to_cam[:3, 3]

    r0 = residuals(x0)
    print(f"[BA] start: reproj={np.linalg.norm(r0[:24]):.3f}px  planar={np.linalg.norm(r0[24:]):.4f}m")

    result = least_squares(residuals, x0, method="trf", verbose=0)

    r1 = result.x
    rf = residuals(r1)
    print(f"[BA]   end: reproj={np.linalg.norm(rf[:24]):.3f}px  planar={np.linalg.norm(rf[24:]):.4f}m  success={result.success}")

    return wf.update(result.x)


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
    depth_scales = {}
    try:
        for cam in args.cams:
            K, dist = intrinsics[cam]
            serial = SERIAL_NUMS[cam]
            depth_scales[cam] = get_depth_scale(pipelines[cam])
            print(f"[{cam}] depth scale: {depth_scales[cam]:.6f} m/unit")
            ok = False
            for attempt in range(60):
                rgbd = get_rgbd_frames(pipelines[cam], cam, retries=4, timeout_ms=10000)
                ok, detector = detector.detect(serial, rgbd, K, dist, depth_scale=depth_scales[cam])
                if ok:
                    break
                print(f"[{cam}] attempt {attempt + 1}/60: tag not detected, retrying...")
            if not ok:
                raise RuntimeError(f"[{cam}] {args.tag} tag id={tag_id} not detected after 60 attempts")

            # Average depth over several frames to reduce per-frame noise
            n_depth_frames = 10
            depth_acc = rgbd[1].astype(np.float64)
            for _ in range(n_depth_frames - 1):
                frame = get_rgbd_frames(pipelines[cam], cam, retries=4, timeout_ms=10000)
                if frame is not None:
                    depth_acc += frame[1].astype(np.float64)
            depth_avg = (depth_acc / n_depth_frames).astype(np.uint16)

            print(f"[{cam}] detected, corners_3d:\n{detector.corners[serial]}")
            color_images[cam] = rgbd[0].copy()
            depth_images[cam] = depth_avg
    finally:
        for pipe in pipelines.values():
            pipe.stop()

    # 2. Build anchored WorldFrame directly via Procrustes registration
    wf = WorldFrame.anchor(detector, T_ARUCO)

    # 3. Fit table plane per camera via RANSAC (in camera frame)
    planes = {}
    for cam in args.cams:
        corners_cam = detector.corners[SERIAL_NUMS[cam]]
        result = fit_ransac(wf, depth_images[cam], cam, corners_cam, depth_scale=depth_scales[cam])
        if result is not None:
            planes[cam] = result  # (normal, d, pts_cam, inlier_mask)

    # 4. Recover 2D keypoints and world_pts for BA and visualization
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

    s = tag_size_m / 2.0
    world_pts = (T_ARUCO @ np.column_stack([
        [[-s, s, 0.], [s, s, 0.], [s, -s, 0.], [-s, -s, 0.]],
        np.ones(4),
    ]).T).T[:, :3]

    # 5. Refine extrinsics via joint BA over reprojection + planar residuals
    if planes:
        wf = bundle_adjust(wf, world_pts, keypoints, planes)

    # Unproject each src camera's tag corners into world, then project into every dst camera.
    reproj_keypoints = {dst: {} for dst in args.cams}
    for src in args.cams:
        for dst in args.cams:
            if src == dst:
                continue
            reproj_keypoints[dst][src] = wf.reproject(
                keypoints[src], keypoint_depths[src], CAM_ORDER.index(src) + 1, CAM_ORDER.index(dst) + 1,
            )

    # L2 distance (pixels) between observed and reprojected corners for each src->dst pair.
    print("[CALIB] pairwise reprojection errors (post-BA):")
    for dst in args.cams:
        for src in args.cams:
            if src == dst:
                continue
            diff = keypoints[dst] - reproj_keypoints[dst][src]
            print(f"  {src}_to_{dst}: {np.linalg.norm(diff):.3f}")

    return wf, keypoints, reproj_keypoints, color_images, planes

def visualize_keypoints(keypoints: dict, reproj_keypoints: dict, color_images: dict):
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
    order = CAM_ORDER

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

    # Per camera: fitted plane patch + point cloud (inliers bright, outliers dim).
    for cam, (normal_cam, d_cam, pts_cam, inlier_mask) in planes.items():
        T_cam = cam_to_t[cam]
        hex_color = cam_colors[cam]
        rgb_bright = np.array([(hex_color >> 16) & 0xff,
                                (hex_color >>  8) & 0xff,
                                 hex_color        & 0xff], dtype=np.float32) / 255.0
        rgb_dim = rgb_bright * 0.3

        # Transform points to world frame
        pts_h = np.column_stack([pts_cam, np.ones(len(pts_cam))])
        pts_world = (T_cam @ pts_h.T).T[:, :3].astype(np.float32)

        # Anchor plane patch at the known tag origin in world frame
        centre_world = wf.t_aruco[:3, 3]

        colors = np.where(inlier_mask[:, None], rgb_bright, rgb_dim).astype(np.float32)

        # Subsample for meshcat performance
        if len(pts_world) > 20_000:
            idx = np.random.default_rng(0).choice(len(pts_world), 20_000, replace=False)
            pts_world, colors = pts_world[idx], colors[idx]

        vis[f"pts/{cam}"].set_object(g.PointCloud(pts_world.T, colors.T, size=0.003))
        normal_world = T_cam[:3, :3] @ normal_cam
        normal_world /= np.linalg.norm(normal_world)

        z = normal_world
        x = np.array([1.0, 0.0, 0.0]) if abs(z[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        x -= x.dot(z) * z; x /= np.linalg.norm(x)
        y = np.cross(z, x)
        R_patch = np.eye(4)
        R_patch[:3, 0] = x; R_patch[:3, 1] = y; R_patch[:3, 2] = z
        R_patch[:3, 3] = centre_world

        vis[f"plane/{cam}"].set_object(
            g.Box([0.4, 0.4, 0.002]),
            g.MeshLambertMaterial(color=hex_color, opacity=0.4, transparent=True),
        )
        vis[f"plane/{cam}"].set_transform(R_patch)

    input("Open browser at the printed URL. Press Enter to exit.")


def main(args):
    if args.only_plot:
        wf = WorldFrame.load(args.output_path)
        print_world_coordinates(wf)
        visualize_planes_meshcat(wf, {})
        return

    wf, keypoints, reproj_keypoints, color_images, planes = calibrate(args)
    print("[CALIB] saving anchored WorldFrame (world = robot frame via T_ARUCO).")
    print_world_coordinates(wf)

    if not args.no_save:
        wf.save(args.output_path)
        print(f"[CALIB] wrote {args.output_path}")

    if not args.no_plot:
        visualize_keypoints(keypoints, reproj_keypoints, color_images)
        visualize_planes_meshcat(wf, planes)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Compute anchored multi-camera extrinsics from a single AprilTag observation. "
            "Outputs a robot-frame wf.npz directly (no separate anchor step needed)."
        )
    )
    parser.add_argument("--cams", nargs="+", default=CAM_ORDER)
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
