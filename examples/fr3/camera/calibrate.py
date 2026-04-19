import time
import argparse
import numpy as np
import cv2
from scipy.optimize import least_squares
from camera import *

"""
Create world frame from april detections and intrinsics.
"""
def _table_tag_world_corners(tag_size_m: float) -> np.ndarray:
    """Return 4 table-tag corners in world frame using T_ARUCO (tag->world)."""
    s = tag_size_m / 2.0
    corners_tag = np.array(
        [[-s, s, 0.0], [s, s, 0.0], [s, -s, 0.0], [-s, -s, 0.0]],
        dtype=np.float64,
    )
    corners_tag_h = np.column_stack([corners_tag, np.ones(4, dtype=np.float64)])
    corners_world = (T_ARUCO @ corners_tag_h.T).T[:, :3]
    return corners_world

def _corner_depths_from_tag_pose(t_tag_to_cam: np.ndarray, tag_size_m: float) -> np.ndarray:
    """Return metric z-depths (meters) of the 4 AprilTag corners in camera frame."""
    s = tag_size_m / 2.0
    obj_pts_tag = np.array(
        [[-s, s, 0.0], [s, s, 0.0], [s, -s, 0.0], [-s, -s, 0.0]],
        dtype=np.float64,
    )  # same corner ordering used in detect_apriltags/solvePnP
    R = t_tag_to_cam[:3, :3]
    t = t_tag_to_cam[:3, 3]
    pts_cam = (R @ obj_pts_tag.T).T + t[None, :]
    return pts_cam[:, 2]

def _compute_reproj_keypoints(
    wf: WorldFrame, cams: list[str], keypoints: dict, keypoint_depths: dict
) -> dict:
    cam_to_idx = {"top": 1, "corner": 2, "side": 3}
    reproj_keypoints = {dst: {} for dst in cams}
    for src in cams:
        for dst in cams:
            if src == dst:
                continue
            reproj_keypoints[dst][src] = wf.reproject(
                keypoints[src],
                keypoint_depths[src],
                cam_to_idx[src],
                cam_to_idx[dst],
            )
    return reproj_keypoints

def _print_reprojection_errors(cams: list[str], keypoints: dict, reproj_keypoints: dict):
    print("[CALIB] reprojection pixel errors (L2 over 4 corners):")
    for dst in cams:
        for src in cams:
            if src == dst:
                continue
            diff = keypoints[dst].astype(np.float64) - reproj_keypoints[dst][src]
            print(f"  {src}_to_{dst}: {np.linalg.norm(diff):.3f}")

def _project_world_points_to_cam_pixels(world_pts: np.ndarray, K: np.ndarray, t_cam_to_world: np.ndarray) -> np.ndarray:
    t_world_to_cam = np.linalg.inv(t_cam_to_world)
    R = t_world_to_cam[:3, :3]
    t = t_world_to_cam[:3, 3]
    pts_cam = (R @ world_pts.T).T + t[None, :]
    u = pts_cam[:, 0] / pts_cam[:, 2] * K[0, 0] + K[0, 2]
    v = pts_cam[:, 1] / pts_cam[:, 2] * K[1, 1] + K[1, 2]
    return np.column_stack([u, v])

def fine_tune_world_frame(
    wf: WorldFrame,
    keypoints: dict,
    tag_size_m: float,
    iters: int = 400,
    huber_delta_px: float = 2.0,
) -> WorldFrame:
    """Bundle adjustment on camera extrinsics against the *known* 3D tag corners.

    The table AprilTag is the calibration target: its 4 corners in world frame are
    fully determined by `T_ARUCO` (tag->world) and the physical `tag_size_m`. We
    treat those world points as fixed ground truth and optimize each camera's
    world->cam extrinsic so that projecting the 3D corners reproduces the observed
    pixel detections in every camera.

    Params: 18 = 3 cameras x (rvec[3], tvec[3]) for world->cam.
    Residuals: 24 = 3 cameras x 4 corners x 2 (u, v).
    """
    cam_order = ["top", "corner", "side"]
    Ks = [wf.K_cam1, wf.K_cam2, wf.K_cam3]
    for T in (wf.t_cam1, wf.t_cam2, wf.t_cam3):
        det = float(np.linalg.det(T[:3, :3]))
        assert np.isclose(det, 1.0, atol=1e-6), (
            f"Camera extrinsic is not a proper rotation (det={det:.6f}); "
            "check T_ARUCO and detection pipeline for reflections."
        )
    t_world_to_cam_init = [
        np.linalg.inv(wf.t_cam1),
        np.linalg.inv(wf.t_cam2),
        np.linalg.inv(wf.t_cam3),
    ]

    # Ground-truth 3D structure: fixed corners of the table tag in world frame.
    world_pts = _table_tag_world_corners(tag_size_m)  # (4, 3)

    x0 = np.zeros(18, dtype=np.float64)
    for i in range(3):
        rvec, _ = cv2.Rodrigues(t_world_to_cam_init[i][:3, :3])
        x0[6 * i: 6 * i + 3] = rvec.ravel()
        x0[6 * i + 3: 6 * i + 6] = t_world_to_cam_init[i][:3, 3]

    obs = {cam: keypoints[cam].astype(np.float64) for cam in cam_order}

    def residuals(x: np.ndarray) -> np.ndarray:
        res = []
        for i, cam in enumerate(cam_order):
            R, _ = cv2.Rodrigues(x[6 * i: 6 * i + 3].reshape(3, 1))
            t = x[6 * i + 3: 6 * i + 6]
            pts_cam = world_pts @ R.T + t  # (4, 3)
            u = pts_cam[:, 0] / pts_cam[:, 2] * Ks[i][0, 0] + Ks[i][0, 2]
            v = pts_cam[:, 1] / pts_cam[:, 2] * Ks[i][1, 1] + Ks[i][1, 2]
            res.append((np.column_stack([u, v]) - obs[cam]).ravel())
        return np.concatenate(res)

    def sum_sq_px(x: np.ndarray) -> float:
        r = residuals(x)
        return float(np.sum(r ** 2))

    obj_before = sum_sq_px(x0)
    print(
        f"[CALIB] BA: residuals=24 (3 cams x 4 corners x uv), params=18, "
        f"start sum-sq pixel error={obj_before:.3f}"
    )
    result = least_squares(
        residuals,
        x0,
        method="trf",
        loss="huber",
        f_scale=float(huber_delta_px),
        max_nfev=max(2000, int(iters)),
        verbose=0,
    )

    x_opt = result.x
    t_cam_to_world_opt = []
    for i in range(3):
        R, _ = cv2.Rodrigues(x_opt[6 * i: 6 * i + 3].reshape(3, 1))
        t = x_opt[6 * i + 3: 6 * i + 6]
        t_world_to_cam = np.eye(4, dtype=np.float64)
        t_world_to_cam[:3, :3] = R
        t_world_to_cam[:3, 3] = t
        t_cam_to_world_opt.append(np.linalg.inv(t_world_to_cam))

    obj_after = sum_sq_px(x_opt)
    print(
        f"[CALIB] BA: end sum-sq pixel error={obj_after:.3f} "
        f"(delta={obj_before - obj_after:.3f}, success={result.success})"
    )
    if obj_after >= obj_before:
        print("[CALIB] BA did not improve reprojection; keeping initial world frame")
        return wf

    return WorldFrame(
        t_aruco=wf.t_aruco,
        K_cam1=wf.K_cam1,
        K_cam2=wf.K_cam2,
        K_cam3=wf.K_cam3,
        t_cam1=t_cam_to_world_opt[0],
        t_cam2=t_cam_to_world_opt[1],
        t_cam3=t_cam_to_world_opt[2],
        t_pose=wf.t_pose,
    )

def _pairwise_objective_px2(cams: list[str], keypoints: dict, reproj_keypoints: dict) -> float:
    total = 0.0
    for dst in cams:
        for src in cams:
            if src == dst:
                continue
            diff = keypoints[dst].astype(np.float64) - reproj_keypoints[dst][src]
            total += float(np.sum(diff ** 2))
    return total

def calibrate(args):
    # 1. build WorldFrame with april detections (wf = build_world_frame(april_dets, T_ARUCO))
    table_tag_size_m = args.table_tag_size_m
    print(f"[CALIB] using tag id={TABLE_TAG_ID} size={table_tag_size_m:.4f} m")
    pipelines, intrinsics = init_cameras(args.cams)
    time.sleep(1) # give cameras a brief moment to start producing stable frames
    april_dets = {}
    keypoints = {}
    keypoint_depths = {}
    color_images = {}
    assert len(args.cams) == 3, "Only 3 cameras are supported for now"
    try:
        for cam in args.cams:
            K, dist = intrinsics[cam]
            print(f"[{cam}] K: {K}")

            # get all necessary data for calibration =====================
            color_image, _ = get_rgbd_frames(pipelines[cam], cam, retries=4, timeout_ms=10000)
            detections, cam_corners = detect_apriltags(color_image, K, dist, table_tag_size_m)

            t_table_to_cam = get_tag_transform_by_id(detections, TABLE_TAG_ID)
            if t_table_to_cam is None:
                raise RuntimeError(f"[{cam}] Table tag id={TABLE_TAG_ID} not detected during calibration")
            april_dets[cam] = t_table_to_cam

            # keypoints (4 corners) from detected table tag.
            if len(cam_corners) == 0:
                raise RuntimeError(f"[{cam}] No AprilTag corner set returned by detector")
            # Keep subpixel precision for BA reprojection error.
            cam_corners = cam_corners[0][0].astype(np.float64)  # (4, 2)
            keypoints[cam] = cam_corners
            # Use pose-derived metric corner depths (meters) to match reproject() contract.
            keypoint_depths[cam] = _corner_depths_from_tag_pose(t_table_to_cam, table_tag_size_m)
            color_images[cam] = color_image.copy()
    finally:
        for pipe in pipelines.values():
            pipe.stop()

    print(f"[CALIB] keypoints (april tag corners)====")
    for cam in keypoints:
        print(f"[{cam}] {keypoints[cam]}")
    print("================================")

    april_dets = AprilDetections(
        cam1=april_dets["top"],
        cam2=april_dets["corner"],
        cam3=april_dets["side"],
    )
    intrinsics = Intrinsics(
        cam1=intrinsics["top"][0],
        cam2=intrinsics["corner"][0],
        cam3=intrinsics["side"][0],
    )
    wf = WorldFrame.build_world_frame(april_dets, T_ARUCO, intrinsics)

    # Report baseline quality before any refinement.
    reproj_keypoints_initial = _compute_reproj_keypoints(wf, args.cams, keypoints, keypoint_depths)
    print("[CALIB] pairwise reprojection errors before fine-tune:")
    _print_reprojection_errors(args.cams, keypoints, reproj_keypoints_initial)
    pairwise_obj_initial = _pairwise_objective_px2(args.cams, keypoints, reproj_keypoints_initial)

    if not args.no_fine_tune:
        wf = fine_tune_world_frame(
            wf,
            keypoints,
            table_tag_size_m,
            iters=args.fine_tune_iters,
        )

    reproj_keypoints = _compute_reproj_keypoints(wf, args.cams, keypoints, keypoint_depths)
    pairwise_obj_final = _pairwise_objective_px2(args.cams, keypoints, reproj_keypoints)
    print(f"[CALIB] pairwise objective: before={pairwise_obj_initial:.3f}, after={pairwise_obj_final:.3f}")
    if pairwise_obj_final > pairwise_obj_initial:
        print("[CALIB] WARNING: pairwise objective worsened after BA")
    print("[CALIB] pairwise reprojection errors after fine-tune:")

    print(f"[CALIB] reprojected keypoints (april tag corners)====")
    for dst in args.cams:
        for src in args.cams:
            if src == dst:
                continue
            print(f"{src}_to_{dst}: {reproj_keypoints[dst][src]}")
    print("================================")

    _print_reprojection_errors(args.cams, keypoints, reproj_keypoints)

    return wf, keypoints, reproj_keypoints, color_images

# def fine_tune(wf: WorldFrame, args):
#     return wf

# def fine_tune(wf: WorldFrame, args):
#     # 1. get keypoints from all cameras
#     keypoints = []
#     for cam in args.cams:
#         color_image, _ = get_rgbd_frames(pipelines[cam], cam, retries=4, timeout_ms=10000)
#         keypoints.append(detect_keypoints(color_image))

#     # 2. match keypoints between cameras
#     matches = match_keypoints(keypoints)

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

def main(args):
    wf, keypoints, reproj_keypoints, color_images = calibrate(args)
    wf.print_world_coordinates()

    if not args.no_save:
        wf.save(args.output_path)

    if not args.no_plot:
        wf.plot()

        visualize_keypoints(wf, keypoints, reproj_keypoints, color_images)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cams", nargs="+", default=["top", "corner", "side"])
    parser.add_argument("--table_tag_size_m", type=float, default=0.1651)
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--no-fine-tune", action="store_true")
    parser.add_argument("--fine-tune-iters", type=int, default=4000)
    parser.add_argument("--output-path", type=str, default="wf.npz")
    args = parser.parse_args()
    main(args)
