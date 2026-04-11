import time
import numpy as np
import cv2
from camera import *


def get_camera_panel(pipe, cam_name, K, dist, tag_size_m, retries=5, timeout_ms=5000):
    frames = get_rgbd_frames(pipe, cam_name, retries, timeout_ms, colorize_depth=True)
    if frames is None:
        return None, None
    color_image, depth_image = frames
    detections, debug = detect_apriltags(color_image, K, dist, tag_size_m, staged=True, return_debug=True)

    print(f"\n[{cam_name}] detected {len(detections)} AprilTag(s)")
    for tag_id, T, _, _, _ in detections:
        print(f"[{cam_name}] tag {tag_id} T_tag_to_cam =")
        print(np.array2string(T, precision=6, suppress_small=True))

    if detections:
        cv2.aruco.drawDetectedMarkers(color_image, [d[2] for d in detections], np.array([d[0] for d in detections]))
        for _, _, _, rvec, tvec in detections:
            cv2.drawFrameAxes(color_image, K, dist, rvec, tvec, tag_size_m * 0.6)
    cv2.putText(
        color_image,
        f"staged detect: stage {debug['stage']}",
        (10, color_image.shape[0] - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        color_image,
        f"{cam_name} RGB",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        depth_image,
        f"{cam_name} Depth",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )

    gray_vis = cv2.cvtColor(debug["gray"], cv2.COLOR_GRAY2BGR)
    enh_vis = cv2.cvtColor(debug["enhanced"], cv2.COLOR_GRAY2BGR)
    cv2.putText(gray_vis, f"{cam_name} before", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(enh_vis, f"{cam_name} after", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2, cv2.LINE_AA)
    preprocess_panel = np.hstack((gray_vis, enh_vis))

    return np.hstack((color_image, depth_image)), preprocess_panel



def main(args):
    pipelines, intrinsics = init_cameras(args.cams)
    try:
        print("Capturing one frame per selected camera...")
        # Give sensors a brief moment to start producing stable frames.
        time.sleep(0.5)
        rows = []
        preprocess_rows = []
        for cam in args.cams:
            K, dist = intrinsics[cam]
            panel, preprocess_panel = get_camera_panel(
                pipelines[cam],
                cam,
                K,
                dist,
                args.table_tag_size_m,
                retries=4,
                timeout_ms=40000,
            )
            if panel is not None:
                rows.append(panel)
            if preprocess_panel is not None:
                preprocess_rows.append(preprocess_panel)

        if rows:
            canvas = np.vstack(rows)
            cv2.imshow("RealSense Multi-Camera Snapshot (RGB + Depth)", canvas)
            if preprocess_rows:
                pre_canvas = np.vstack(preprocess_rows)
                cv2.imshow("AprilTag Preprocessing (Before | After)", pre_canvas)
            print("Snapshot shown. Press any key to close.")
            cv2.waitKey(0)
        else:
            print("No frames captured from selected cameras.")
    finally:
        for pipe in pipelines.values():
            pipe.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cams", nargs="+", default=["top", "corner", "side"])
    parser.add_argument("--table_tag_size_m", type=float, default=0.1651)
    args = parser.parse_args()
    main(args)

