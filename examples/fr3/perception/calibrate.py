import time
import argparse
import numpy as np
import cv2
from scipy.optimize import least_squares
from camera import *
from viz import print_world_coordinates, plot_worldframe

import matplotlib
matplotlib.use('TkAgg')

"""
Compute per-camera extrinsics from a single multi-view AprilTag observation.

The world frame produced here is the reference camera's frame at calibration
time (so `t_cam_ref ≈ I`). The output is an UNANCHORED `WorldFrame` (sentinel
`t_aruco = I`). Run `anchor.py` afterwards to express everything in the robot
base/world frame.
"""

# Camera label -> WorldFrame slot index (cam1=top, cam2=corner, cam3=side).
CAM_TO_IDX = {"top": 1, "corner": 2, "side": 3}

def _tag_corners_local(tag_size_m: float) -> np.ndarray:
    """Return the 4 AprilTag corners in tag-local frame, matching solvePnP order."""
    s = tag_size_m / 2.0
    return np.array(
        [[-s, s, 0.0], [s, s, 0.0], [s, -s, 0.0], [-s, -s, 0.0]],
        dtype=np.float64,
    )

def _tag_corners_in_frame(t_tag_to_frame: np.ndarray, tag_size_m: float) -> np.ndarray:
    """Lift the 4 tag corners into a target frame via t_tag_to_frame (4x4)."""
    corners_tag = _tag_corners_local(tag_size_m)
    corners_h = np.column_stack([corners_tag, np.ones(4, dtype=np.float64)])
    return (t_tag_to_frame @ corners_h.T).T[:, :3]

def _corner_depths_from_tag_pose(t_tag_to_cam: np.ndarray, tag_size_m: float) -> np.ndarray:
    """Return metric z-depths (meters) of the 4 AprilTag corners in camera frame."""
    R = t_tag_to_cam[:3, :3]
    t = t_tag_to_cam[:3, 3]
    pts_cam = _tag_corners_local(tag_size_m) @ R.T + t
    return pts_cam[:, 2]

def _compute_reproj_keypoints(
    wf: WorldFrame, cams: list[str], keypoints: dict, keypoint_depths: dict
) -> dict:
    reproj_keypoints = {dst: {} for dst in cams}
    for src in cams:
        for dst in cams:
            if src == dst:
                continue
            reproj_keypoints[dst][src] = wf.reproject(
                keypoints[src],
                keypoint_depths[src],
                CAM_TO_IDX[src],
                CAM_TO_IDX[dst],
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

def _pairwise_objective_px2(cams: list[str], keypoints: dict, reproj_keypoints: dict) -> float:
    total = 0.0
    for dst in cams:
        for src in cams:
            if src == dst:
                continue
            diff = keypoints[dst].astype(np.float64) - reproj_keypoints[dst][src]
            total += float(np.sum(diff ** 2))
    return total

def fine_tune_world_frame(
    wf: WorldFrame,
    keypoints: dict,
    world_pts: np.ndarray,
    iters: int = 400,
    huber_delta_px: float = 2.0,
) -> WorldFrame:
    """Bundle adjustment on camera extrinsics against fixed 3D tag corners.

    `world_pts` (4, 3) are the tag corners expressed in the current world frame
    of `wf`. They are treated as ground-truth structure; only the 3 cameras'
    extrinsics are optimized. Reprojection error against the observed pixel
    detections in every camera is minimized.

    Params: 18 = 3 cameras x (rvec[3], tvec[3]) for world->cam.
    Residuals: 24 = 3 cameras x 4 corners x 2 (u, v).
    """
    cam_order = ["top", "corner", "side"]
    Ks = [wf.K_cam1, wf.K_cam2, wf.K_cam3]
    for T in (wf.t_cam1, wf.t_cam2, wf.t_cam3):
        det = float(np.linalg.det(T[:3, :3]))
        assert np.isclose(det, 1.0, atol=1e-6), (
            f"Camera extrinsic is not a proper rotation (det={det:.6f})."
        )
    t_world_to_cam_init = [
        np.linalg.inv(wf.t_cam1),
        np.linalg.inv(wf.t_cam2),
        np.linalg.inv(wf.t_cam3),
    ]

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
        print("[CALIB] BA did not improve reprojection; keeping initial extrinsics")
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

def _resolve_tag(args) -> tuple[int, float, str]:
    """Map --tag flag to (tag_id, tag_size_m, label)."""
    if args.tag == "table":
        return TABLE_TAG_ID, args.table_tag_size_m, "table"
    if args.tag == "tee":
        return TEE_TAG_ID, args.tee_tag_size_m, "tee"
    raise ValueError(f"Unknown --tag value: {args.tag}")

def _build_extrinsics_from_ref_cam(
    cams: list[str],
    ref_cam: str,
    t_tag_to_cam: dict,
    intrinsics: dict,
) -> WorldFrame:
    """Build an unanchored WorldFrame with world = ref-cam frame.

    t_camN_in_world = inv(t_world_to_camN) where
        t_world_to_camN = T_camRef_to_camN = T_tag_to_camN @ inv(T_tag_to_camRef)
    so t_camRef_in_world = identity by construction.
    """
    inv_ref = np.linalg.inv(t_tag_to_cam[ref_cam])
    t_cam_in_world = {}
    for cam in cams:
        t_world_to_cam = t_tag_to_cam[cam] @ inv_ref
        t_cam_in_world[cam] = np.linalg.inv(t_world_to_cam)

    # Slot mapping: top→cam1, corner→cam2, side→cam3.
    return WorldFrame(
        t_aruco=np.eye(4, dtype=np.float64),  # sentinel: unanchored
        K_cam1=intrinsics["top"][0],
        K_cam2=intrinsics["corner"][0],
        K_cam3=intrinsics["side"][0],
        t_cam1=t_cam_in_world["top"],
        t_cam2=t_cam_in_world["corner"],
        t_cam3=t_cam_in_world["side"],
    )

def calibrate(args):
    tag_id, tag_size_m, tag_label = _resolve_tag(args)
    print(f"[CALIB] using {tag_label} tag id={tag_id} size={tag_size_m:.4f} m")
    print(f"[CALIB] reference camera = {args.ref_cam} (world frame = its pose)")
    if args.ref_cam not in args.cams:
        raise ValueError(f"--ref-cam {args.ref_cam!r} not in --cams {args.cams}")

    pipelines, intrinsics = init_cameras(args.cams)
    time.sleep(1)  # give cameras a brief moment to start producing stable frames
    t_tag_to_cam = {}
    keypoints = {}
    keypoint_depths = {}
    color_images = {}
    assert len(args.cams) == 3, "Only 3 cameras are supported for now"
    try:
        for cam in args.cams:
            K, dist = intrinsics[cam]
            print(f"[{cam}] K: {K}")

            color_image, _ = get_rgbd_frames(pipelines[cam], cam, retries=4, timeout_ms=10000)
            detections, cam_corners = detect_apriltags(color_image, K, dist, tag_size_m)

            t = get_tag_transform_by_id(detections, tag_id)
            if t is None:
                raise RuntimeError(
                    f"[{cam}] {tag_label} tag id={tag_id} not detected during calibration"
                )
            t_tag_to_cam[cam] = t

            corners_px = get_tag_corners_by_id(detections, tag_id)
            if corners_px is None:
                raise RuntimeError(f"[{cam}] No corner set returned for tag id={tag_id}")
            keypoints[cam] = corners_px  # subpixel, (4, 2) float64
            keypoint_depths[cam] = _corner_depths_from_tag_pose(t, tag_size_m)
            color_images[cam] = color_image.copy()
    finally:
        for pipe in pipelines.values():
            pipe.stop()

    print(f"[CALIB] keypoints ({tag_label} tag corners)====")
    for cam in args.cams:
        print(f"[{cam}] {keypoints[cam]}")
    print("================================")

    wf = _build_extrinsics_from_ref_cam(args.cams, args.ref_cam, t_tag_to_cam, intrinsics)
    # Fixed 3D structure for BA: tag corners expressed in the ref-cam world.
    world_pts = _tag_corners_in_frame(t_tag_to_cam[args.ref_cam], tag_size_m)

    reproj_keypoints_initial = _compute_reproj_keypoints(wf, args.cams, keypoints, keypoint_depths)
    print("[CALIB] pairwise reprojection errors before fine-tune:")
    _print_reprojection_errors(args.cams, keypoints, reproj_keypoints_initial)
    pairwise_obj_initial = _pairwise_objective_px2(args.cams, keypoints, reproj_keypoints_initial)

    if not args.no_fine_tune:
        wf = fine_tune_world_frame(
            wf,
            keypoints,
            world_pts,
            iters=args.fine_tune_iters,
        )

    reproj_keypoints = _compute_reproj_keypoints(wf, args.cams, keypoints, keypoint_depths)
    pairwise_obj_final = _pairwise_objective_px2(args.cams, keypoints, reproj_keypoints)
    print(f"[CALIB] pairwise objective: before={pairwise_obj_initial:.3f}, after={pairwise_obj_final:.3f}")
    if pairwise_obj_final > pairwise_obj_initial:
        print("[CALIB] WARNING: pairwise objective worsened after BA")
    print("[CALIB] pairwise reprojection errors after fine-tune:")

    print(f"[CALIB] reprojected keypoints ({tag_label} tag corners)====")
    for dst in args.cams:
        for src in args.cams:
            if src == dst:
                continue
            print(f"{src}_to_{dst}: {reproj_keypoints[dst][src]}")
    print("================================")

    _print_reprojection_errors(args.cams, keypoints, reproj_keypoints)

    return wf, keypoints, reproj_keypoints, color_images

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
    print(
        f"[CALIB] saving UNANCHORED extrinsics (world = '{args.ref_cam}' camera frame). "
        f"Run anchor.py to produce a robot/world wf.npz."
    )
    print_world_coordinates(wf)

    if not args.no_save:
        wf.save(args.output_path)
        print(f"[CALIB] wrote {args.output_path}")

    if not args.no_plot:
        plot_worldframe(wf)
        visualize_keypoints(wf, keypoints, reproj_keypoints, color_images)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Compute multi-camera extrinsics from a single AprilTag observation. "
            "Output is an unanchored WorldFrame (world = reference camera). "
            "Run anchor.py afterwards to get a robot-frame wf.npz."
        )
    )
    parser.add_argument("--cams", nargs="+", default=["top", "corner", "side"])
    parser.add_argument(
        "--tag", choices=["table", "tee"], default="table",
        help="Which AprilTag to calibrate against. Default: table.",
    )
    parser.add_argument(
        "--ref-cam", default="top",
        help="Reference camera; defines the world frame of the unanchored output.",
    )
    parser.add_argument("--table_tag_size_m", type=float, default=0.1651)
    parser.add_argument("--tee_tag_size_m", type=float, default=0.0445)
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--no-fine-tune", action="store_true")
    parser.add_argument("--fine-tune-iters", type=int, default=4000)
    parser.add_argument("--output-path", type=str, default="extrinsics.npz")
    args = parser.parse_args()
    main(args)
