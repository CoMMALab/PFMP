import time
import argparse
import numpy as np
import cv2
from camera import *

"""
Create world frame from april detections and intrinsics.
"""
def calibrate(args):
    # 1. build WorldFrame with april detections (wf = build_world_frame(april_dets, T_ARUCO))
    table_tag_size_m = args.table_tag_size_m
    print(f"[CALIB] using tag id={TABLE_TAG_ID} size={table_tag_size_m:.4f} m")
    pipelines, intrinsics = init_cameras(args.cams)
    time.sleep(1) # give cameras a brief moment to start producing stable frames
    april_dets = {}
    keypoints = {}
    keypoint_depths = {}
    assert len(args.cams) == 3, "Only 3 cameras are supported for now"
    for cam in args.cams:
        K, dist = intrinsics[cam]
        print(f"[{cam}] K: {K}")

        # get all necessary data for calibration =====================
        color_image, depth_image = get_rgbd_frames(pipelines[cam], cam, retries=4, timeout_ms=10000)
        detections, cam_corners = detect_apriltags(color_image, K, dist, table_tag_size_m)


        t_table_to_cam = get_tag_transform_by_id(detections, TABLE_TAG_ID)
        if t_table_to_cam is None:
            raise RuntimeError(f"[{cam}] Table tag id={TABLE_TAG_ID} not detected during calibration")
        april_dets[cam] = t_table_to_cam

        # keypoint (only one april tag during calibration)
        cam_corners = cam_corners[0][0].astype(int) # np array of shape (4,2)    
        keypoints[cam] = cam_corners
        keypoint_depths[cam] = depth_image[cam_corners[:, 1], cam_corners[:, 0]] # np array of shape (4,)

    print(f"[CALIB] keypoints (april tag corners): {keypoints}")
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

    top_to_corner = wf.reproject(keypoints["top"], keypoint_depths["top"], 1, 2)
    corner_to_side = wf.reproject(keypoints["corner"], keypoint_depths["corner"], 2, 3)
    side_to_top = wf.reproject(keypoints["side"], keypoint_depths["side"], 3, 1)

    corner_diff = keypoints["corner"] - top_to_corner
    side_diff = keypoints["side"] - corner_to_side
    top_diff = keypoints["top"] - side_to_top

    print(f"[CALIB] corner_diff: {np.linalg.norm(corner_diff)}")
    print(f"[CALIB] side_diff: {np.linalg.norm(side_diff)}")
    print(f"[CALIB] top_diff: {np.linalg.norm(top_diff)}")

    return wf

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

def main(args):
    wf = calibrate(args)
    wf.print_world_coordinates()

    if not args.no_plot:
        wf.plot()

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
