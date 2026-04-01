"""Multi-RealSense AprilTag detection + extrinsic calibration.

Usage:
  python multicamera.py              # live view with detection
  python multicamera.py --calibrate  # collect samples & save extrinsics
"""

import argparse
import json
import numpy as np
import cv2

try:
    import pyrealsense2 as rs
except ImportError:
    raise RuntimeError("pyrealsense2 not installed")

# AprilTag setup
MARKER_ID = 0
MARKER_SIZE = 0.1651
DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
PARAMS = cv2.aruco.DetectorParameters()
PARAMS.adaptiveThreshWinSizeMin = 3
PARAMS.adaptiveThreshWinSizeMax = 30
PARAMS.adaptiveThreshWinSizeStep = 5
PARAMS.minMarkerPerimeterRate = 0.01
PARAMS.maxMarkerPerimeterRate = 4.0
PARAMS.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
PARAMS.cornerRefinementWinSize = 5
DETECTOR = cv2.aruco.ArucoDetector(DICT, PARAMS)


def find_realsense_devices():
    ctx = rs.context()
    return [{"serial": d.get_info(rs.camera_info.serial_number),
             "name": d.get_info(rs.camera_info.name)} for d in ctx.query_devices()]


def start_pipeline(serial):
    """Start a pipeline, return (pipeline, K, dist)."""
    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_device(serial)
    cfg.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
    profile = pipe.start(cfg)
    intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    K = np.array([[intr.fx, 0, intr.ppx],
                  [0, intr.fy, intr.ppy],
                  [0, 0, 1]], dtype=np.float64)
    dist = np.array(intr.coeffs, dtype=np.float64)
    print(f"  [{serial}] fx={intr.fx:.1f} fy={intr.fy:.1f} cx={intr.ppx:.1f} cy={intr.ppy:.1f}")
    return pipe, K, dist


def detect_marker(frame, K, dist):
    """Detect AprilTag. Returns (rvec, tvec) or (None, None)."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (0, 0), 3)
    gray = cv2.addWeighted(gray, 1.5, blur, -0.5, 0)
    corners, ids, _ = DETECTOR.detectMarkers(gray)

    if ids is None or MARKER_ID not in ids:
        return None, None, None

    idx = list(ids.flatten()).index(MARKER_ID)
    cpx = corners[idx][0]
    s = MARKER_SIZE / 2
    obj = np.array([[-s, s, 0], [s, s, 0], [s, -s, 0], [-s, -s, 0]], dtype=np.float64)
    ok, rvec, tvec = cv2.solvePnP(obj, cpx.astype(np.float64), K, dist,
                                   flags=cv2.SOLVEPNP_IPPE_SQUARE)
    if not ok:
        return None, None, corners
    return rvec, tvec, corners


def pose_to_4x4(rvec, tvec):
    """Convert rvec/tvec to a 4x4 homogeneous transform (marker-to-camera)."""
    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = tvec.flatten()
    return T


def draw_detection(frame, rvec, tvec, corners, K, dist, cam_label):
    """Annotate frame with detection results."""
    if rvec is not None:
        cv2.aruco.drawDetectedMarkers(frame, corners)
        cv2.drawFrameAxes(frame, K, dist, rvec, tvec, MARKER_SIZE)
        tx, ty, tz = tvec.flatten()
        R, _ = cv2.Rodrigues(rvec)
        yaw = np.rad2deg(np.arctan2(R[1, 0], R[0, 0]))
        cv2.putText(frame, f"t=({tx:.3f},{ty:.3f},{tz:.3f}) yaw={yaw:.0f}",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(frame, f"{cam_label} DETECTED", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    else:
        cv2.putText(frame, f"{cam_label} ---", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    return frame


def compute_relative_poses(poses):
    """Given dict {cam_idx: 4x4 marker-to-cam}, compute pairwise cam-to-cam transforms.

    Returns dict of (i, j) -> 4x4 transform from cam_i to cam_j.
    Convention: a point in cam_i frame is mapped to cam_j frame by T @ p.
    """
    relative = {}
    indices = sorted(poses.keys())
    for i in indices:
        for j in indices:
            if i >= j:
                continue
            # T_marker_to_i and T_marker_to_j
            # T_i_to_j = T_marker_to_j @ inv(T_marker_to_i)
            T_ij = poses[j] @ np.linalg.inv(poses[i])
            relative[(i, j)] = T_ij
    return relative


def format_transform(T):
    """Pretty-print a 4x4 transform as translation + rotation."""
    t = T[:3, 3]
    R = T[:3, :3]
    # Decompose to axis-angle for compact display
    rvec, _ = cv2.Rodrigues(R)
    angle = np.linalg.norm(rvec)
    axis = rvec.flatten() / angle if angle > 1e-6 else np.array([0, 0, 1])
    return (f"t=({t[0]:+.4f}, {t[1]:+.4f}, {t[2]:+.4f})  "
            f"rot={np.rad2deg(angle):.1f} deg  "
            f"axis=({axis[0]:+.2f},{axis[1]:+.2f},{axis[2]:+.2f})  "
            f"dist={np.linalg.norm(t):.4f}m")


def run_live(cams, devices):
    """Live view with detection overlay."""
    print(f"\nShowing {len(cams)} camera feeds. Press Q to quit.")
    while True:
        tiles = []
        for i, cam in enumerate(cams):
            fs = cam["pipe"].wait_for_frames()
            color = fs.get_color_frame()
            if not color:
                continue
            img = np.asanyarray(color.get_data())
            rvec, tvec, corners = detect_marker(img, cam["K"], cam["dist"])
            label = f"cam{i} [{cam['serial'][-4:]}]"
            img = draw_detection(img, rvec, tvec, corners, cam["K"], cam["dist"], label)
            tiles.append(img)

        if tiles:
            h = 360
            resized = [cv2.resize(f, (int(f.shape[1] * h / f.shape[0]), h)) for f in tiles]
            cv2.imshow("RealSense Cameras", np.hstack(resized))

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


def run_calibrate(cams, devices, num_samples=30, outfile="camera_extrinsics.json"):
    """Collect samples where all cameras see the tag, average relative poses, save."""
    n = len(cams)
    print(f"\nExtrinsic calibration: collecting {num_samples} samples where ALL {n} cameras detect the tag.")
    print("Hold the AprilTag visible to all cameras. Press Q to abort.\n")

    # Accumulate per-pair transforms
    samples = {(i, j): [] for i in range(n) for j in range(i + 1, n)}
    count = 0

    while count < num_samples:
        # Grab frames from all cameras
        poses = {}
        tiles = []
        for i, cam in enumerate(cams):
            fs = cam["pipe"].wait_for_frames()
            color = fs.get_color_frame()
            if not color:
                continue
            img = np.asanyarray(color.get_data())
            rvec, tvec, corners = detect_marker(img, cam["K"], cam["dist"])
            label = f"cam{i} [{cam['serial'][-4:]}]"
            img = draw_detection(img, rvec, tvec, corners, cam["K"], cam["dist"], label)
            tiles.append(img)
            if rvec is not None:
                poses[i] = pose_to_4x4(rvec, tvec)

        # Show mosaic
        if tiles:
            h = 360
            resized = [cv2.resize(f, (int(f.shape[1] * h / f.shape[0]), h)) for f in tiles]
            mosaic = np.hstack(resized)
            status = f"Samples: {count}/{num_samples}  Detecting: {len(poses)}/{n} cams"
            cv2.putText(mosaic, status, (10, mosaic.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imshow("Calibration", mosaic)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("Aborted.")
            return

        # Only count if ALL cameras detected
        if len(poses) == n:
            rel = compute_relative_poses(poses)
            for key, T in rel.items():
                samples[key].append(T)
            count += 1
            if count % 5 == 0:
                print(f"  {count}/{num_samples} samples collected")

    # Average the transforms (average translations, average rotations via mean rvec)
    print(f"\nCollected {num_samples} samples. Computing mean extrinsics...\n")
    result = {"ref_camera": 0, "serials": [c["serial"] for c in cams], "pairs": {}}

    for (i, j), T_list in samples.items():
        # Mean translation
        translations = np.array([T[:3, 3] for T in T_list])
        mean_t = translations.mean(axis=0)
        std_t = translations.std(axis=0)

        # Mean rotation via SVD (robust near ±π where rvec averaging fails)
        Rs = np.array([T[:3, :3] for T in T_list])
        U, _, Vt = np.linalg.svd(Rs.mean(axis=0))
        mean_R = U @ Vt
        if np.linalg.det(mean_R) < 0:
            Vt[-1] *= -1
            mean_R = U @ Vt
        rvecs = np.array([cv2.Rodrigues(T[:3, :3])[0].flatten() for T in T_list])
        mean_rvec = cv2.Rodrigues(mean_R)[0].flatten()
        std_rvec = rvecs.std(axis=0)

        mean_T = np.eye(4)
        mean_T[:3, :3] = mean_R
        mean_T[:3, 3] = mean_t

        pair_key = f"{i}_to_{j}"
        result["pairs"][pair_key] = {
            "transform_4x4": mean_T.tolist(),
            "translation_mean": mean_t.tolist(),
            "translation_std": std_t.tolist(),
            "rvec_mean": mean_rvec.tolist(),
            "rvec_std": std_rvec.tolist(),
        }

        print(f"  cam{i} -> cam{j}: {format_transform(mean_T)}")
        print(f"    t_std=({std_t[0]:.4f}, {std_t[1]:.4f}, {std_t[2]:.4f})")

    # Also store cam0-relative transforms for convenience
    result["cam_to_ref"] = {}
    for i in range(n):
        if i == 0:
            result["cam_to_ref"][str(i)] = np.eye(4).tolist()
        else:
            pair_key = f"0_to_{i}"
            if pair_key in result["pairs"]:
                # This is T_cam0_to_cami, we want T_cami_to_cam0 = inv(T_cam0_to_cami)
                T = np.array(result["pairs"][pair_key]["transform_4x4"])
                result["cam_to_ref"][str(i)] = np.linalg.inv(T).tolist()

    # Save intrinsics too
    result["intrinsics"] = {}
    for i, cam in enumerate(cams):
        result["intrinsics"][str(i)] = {
            "serial": cam["serial"],
            "K": cam["K"].tolist(),
            "dist": cam["dist"].tolist(),
        }

    with open(outfile, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved extrinsics to {outfile}")


def main():
    parser = argparse.ArgumentParser(description="Multi-RealSense detection & extrinsic calibration")
    parser.add_argument("--calibrate", action="store_true", help="Run extrinsic calibration")
    parser.add_argument("-n", type=int, default=30, help="Number of calibration samples")
    parser.add_argument("-o", default="camera_extrinsics.json", help="Output file for extrinsics")
    args = parser.parse_args()

    devices = find_realsense_devices()
    if not devices:
        print("No RealSense devices found.")
        return

    print(f"Found {len(devices)} RealSense device(s):")
    for i, d in enumerate(devices):
        print(f"  [{i}] {d['name']}  serial={d['serial']}")

    print("\nStarting pipelines...")
    cams = []
    for d in devices:
        pipe, K, dist = start_pipeline(d["serial"])
        cams.append({"pipe": pipe, "K": K, "dist": dist, "serial": d["serial"]})

    try:
        if args.calibrate:
            run_calibrate(cams, devices, num_samples=args.n, outfile=args.o)
        else:
            run_live(cams, devices)
    finally:
        for cam in cams:
            cam["pipe"].stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
