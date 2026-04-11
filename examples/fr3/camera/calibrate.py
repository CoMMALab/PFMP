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
    assert len(args.cams) == 3, "Only 3 cameras are supported for now"
    for cam in args.cams:
        K, dist = intrinsics[cam]
        print(f"[{cam}] K: {K}")
        color_image, _ = get_rgbd_frames(pipelines[cam], cam, retries=4, timeout_ms=10000)
        detections = detect_apriltags(color_image, K, dist, table_tag_size_m)
        t_table_to_cam = get_tag_transform_by_id(detections, TABLE_TAG_ID)
        if t_table_to_cam is None:
            raise RuntimeError(f"[{cam}] Table tag id={TABLE_TAG_ID} not detected during calibration")
        april_dets[cam] = t_table_to_cam
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

    calibrate(args)
