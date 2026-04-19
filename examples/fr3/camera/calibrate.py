import time
import argparse
import numpy as np
import cv2
from camera import *

"""
Create world frame from april detections and intrinsics.
"""
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
    cam_to_idx = {"top": 1, "corner": 2, "side": 3}
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
            cam_corners = cam_corners[0][0].astype(int)  # (4, 2)
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

    # Reproject each camera's 4 corners into every other camera.
    reproj_keypoints = {dst: {} for dst in args.cams}
    for src in args.cams:
        for dst in args.cams:
            if src == dst:
                continue
            reproj_keypoints[dst][src] = wf.reproject(
                keypoints[src],
                keypoint_depths[src],
                cam_to_idx[src],
                cam_to_idx[dst],
            )

    print(f"[CALIB] reprojected keypoints (april tag corners)====")
    for dst in args.cams:
        for src in args.cams:
            if src == dst:
                continue
            print(f"{src}_to_{dst}: {reproj_keypoints[dst][src]}")
    print("================================")

    print("[CALIB] reprojection pixel errors (L2 over 4 corners):")
    for dst in args.cams:
        for src in args.cams:
            if src == dst:
                continue
            diff = keypoints[dst].astype(np.float64) - reproj_keypoints[dst][src]
            print(f"  {src}_to_{dst}: {np.linalg.norm(diff):.3f}")

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

    if not args.no_plot:
        wf.plot()

        visualize_keypoints(wf, keypoints, reproj_keypoints, color_images)

    # 3. serialize calibrated WorldFrame
    if not args.no_save:
        wf.save(args.output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cams", nargs="+", default=["top", "corner", "side"])
    parser.add_argument("--table_tag_size_m", type=float, default=0.1651)
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--output-path", type=str, default="wf.npz")
    args = parser.parse_args()
    main(args)
