import pyrealsense2 as rs
import numpy as np
import cv2
import argparse
import time

SERIAL_NUMS = {
                "top": "234322070408",
                "corner": "234322070144",
                "side": "234322070341"
            }

# AprilTag setup (OpenCV ArUco AprilTag dictionary)
APRILTAG_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
APRILTAG_PARAMS = cv2.aruco.DetectorParameters()
APRILTAG_DETECTOR = cv2.aruco.ArucoDetector(APRILTAG_DICT, APRILTAG_PARAMS)

def start_pipeline(serial):
    # Start pipeline for one camera.
    print(f"Starting camera: {serial}")
    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_device(serial)

    # Safe baseline for D4xx / many RealSense setups
    cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    profile = pipe.start(cfg)
    intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    K = np.array(
        [[intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy], [0, 0, 1]],
        dtype=np.float64,
    )
    dist = np.array(intr.coeffs, dtype=np.float64)
    print(f"Camera started: {serial}")
    return pipe, K, dist

def pose_to_4x4(rvec, tvec):
    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = tvec.flatten()
    return T

def detect_apriltags(color_image, K, dist, tag_size_m):
    gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = APRILTAG_DETECTOR.detectMarkers(gray)
    if ids is None:
        return []

    s = tag_size_m / 2.0
    obj_points = np.array(
        [[-s, s, 0], [s, s, 0], [s, -s, 0], [-s, -s, 0]],
        dtype=np.float64,
    )
    detections = []
    for i, tag_id in enumerate(ids.flatten()):
        img_points = corners[i][0].astype(np.float64)
        ok, rvec, tvec = cv2.solvePnP(
            obj_points, img_points, K, dist, flags=cv2.SOLVEPNP_IPPE_SQUARE
        )
        if not ok:
            continue
        detections.append((int(tag_id), pose_to_4x4(rvec, tvec), corners[i], rvec, tvec))

    return detections

def get_camera_panel(pipe, cam_name, K, dist, tag_size_m, retries=5, timeout_ms=5000):
    frames = None
    for attempt in range(1, retries + 1):
        try:
            frames = pipe.wait_for_frames(timeout_ms=timeout_ms)
            break
        except RuntimeError:
            print(f"[WARN] {cam_name}: frame timeout (attempt {attempt}/{retries})")

    if frames is None:
        print(f"[WARN] {cam_name}: failed to capture frame after {retries} attempts")
        return None

    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()

    if not depth_frame or not color_frame:
        return None

    color_image = np.asanyarray(color_frame.get_data())
    depth_image = np.asanyarray(rs.colorizer().colorize(depth_frame).get_data())
    detections = detect_apriltags(color_image, K, dist, tag_size_m)

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

    return np.hstack((color_image, depth_image))

def main(args):
    pipelines = {}
    intrinsics = {}
    try:
        for cam in args.cams:
            pipe, K, dist = start_pipeline(SERIAL_NUMS[cam])
            pipelines[cam] = pipe
            intrinsics[cam] = (K, dist)

        print("Capturing one frame per selected camera...")
        # Give sensors a brief moment to start producing stable frames.
        time.sleep(0.5)
        rows = []
        for cam in args.cams:
            K, dist = intrinsics[cam]
            panel = get_camera_panel(
                pipelines[cam],
                cam,
                K,
                dist,
                args.tag_size_m,
                retries=4,
                timeout_ms=4000,
            )
            if panel is not None:
                rows.append(panel)

        if rows:
            canvas = np.vstack(rows)
            cv2.imshow("RealSense Multi-Camera Snapshot (RGB + Depth)", canvas)
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
    parser.add_argument(
        "--tag-size-m",
        type=float,
        default=0.1651,
        help="AprilTag edge length in meters for pose estimation.",
    )
    args = parser.parse_args()
    main(args)



