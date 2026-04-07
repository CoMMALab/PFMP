from typing import NamedTuple, Optional
import pyrealsense2 as rs
import numpy as np
import cv2
import argparse
import time
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

SERIAL_NUMS = {
                "top": "234322070408",
                "corner": "234322070144",
                "side": "234322070341",
            }

APRILTAG_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
APRILTAG_PARAMS = cv2.aruco.DetectorParameters()
APRILTAG_DETECTOR = cv2.aruco.ArucoDetector(APRILTAG_DICT, APRILTAG_PARAMS)

TABLE_TAG_ID = 0
TEE_TAG_ID = 2

class AprilDetections(NamedTuple):
    cam1: np.array
    cam2: np.array
    cam3: np.array

# tag_to_world (T_R^A)
T_ARUCO = np.array([
    [-1.0,  0.0,  0.0,  0.379],
    [0.0,  1.0,  0.0, -0.064],
    [0.0,  0.0,  1.0,  0.012],
    [0.0,  0.0,  0.0,  1.0],
])

class WorldFrame(NamedTuple):
    t_aruco: np.array
    t_cam1: np.array
    t_cam2: np.array
    t_cam3: np.array
    t_pose: Optional[np.array] = None

    @staticmethod
    def build_world_frame(dets: AprilDetections, t_aruco: np.array) -> "Self":
        # T_world_cam = T_world_aruco @ inv(T_cam_aruco)
        t_cam1 = t_aruco @ np.linalg.inv(dets.cam1)
        t_cam2 = t_aruco @ np.linalg.inv(dets.cam2)
        t_cam3 = t_aruco @ np.linalg.inv(dets.cam3)

        return WorldFrame(t_aruco=t_aruco, t_cam1=t_cam1, t_cam2=t_cam2, t_cam3=t_cam3)

    def save(self, path: str):
        arrays = dict(t_aruco=self.t_aruco, t_cam1=self.t_cam1, t_cam2=self.t_cam2, t_cam3=self.t_cam3)
        if self.t_pose is not None:
            arrays["t_pose"] = self.t_pose
        np.savez(path, **arrays)

    @staticmethod
    def load(path: str) -> "WorldFrame":
        data = np.load(path)
        return WorldFrame(
            t_aruco=data["t_aruco"],
            t_cam1=data["t_cam1"],
            t_cam2=data["t_cam2"],
            t_cam3=data["t_cam3"],
            t_pose=data["t_pose"] if "t_pose" in data else None,
        )

    def into(self, p: np.array) -> np.array:
        """Transform a homogeneous (4,) point from each camera frame into world frame.
        Returns (3, 3) world-space xyz for cam1, cam2, cam3.
        """
        return np.stack([
            (self.t_cam1 @ p)[:3],
            (self.t_cam2 @ p)[:3],
            (self.t_cam3 @ p)[:3],
        ])

    def plot(self, ps: Optional[np.array] = None):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        ax.set_xlim(-0.5, 0.5)
        ax.set_ylim(-0.5, 0.5)
        ax.set_zlim(-0.5, 0.5)

        def draw_vec(T, label, color, axis_scale=0.05):
            pos = T[:3, 3]
            ax.quiver(0, 0, 0, *pos, color=color, normalize=False)
            ax.text(*pos, label, fontsize=8)
            for i, c in enumerate(["r", "g", "b"]):
                ax.quiver(*pos, *T[:3, i], color=c, length=axis_scale, normalize=False)

        ax.scatter([0], [0], [0], c="k", s=40)
        draw_vec(self.t_aruco, "aruco", "k")
        draw_vec(self.t_cam1,  "cam1",  "r")
        draw_vec(self.t_cam2,  "cam2",  "g")
        draw_vec(self.t_cam3,  "cam3",  "b")
        if self.t_pose is not None:
            draw_vec(self.t_pose, "pose", "m")

        if ps is not None:
            ax.scatter(ps[:, 0], ps[:, 1], ps[:, 2], c="k", s=10)

        ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
        plt.tight_layout()
        plt.show()

    def print_world_coordinates(self):
        """Print key object coordinates in world frame (meters)."""
        objects = [
            ("robot/world", np.array([0.0, 0.0, 0.0], dtype=np.float64)),
            ("aruco", self.t_aruco[:3, 3]),
            ("cam1", self.t_cam1[:3, 3]),
            ("cam2", self.t_cam2[:3, 3]),
            ("cam3", self.t_cam3[:3, 3]),
        ]
        print("\nWorld-frame coordinates (meters):")
        for name, p in objects:
            print(f"  {name:12s} -> x={p[0]: .4f}, y={p[1]: .4f}, z={p[2]: .4f}")

def test_world():
    def make_cam(tx, ty, tz, angle_z):
        c, s = np.cos(angle_z), np.sin(angle_z)
        return np.array([
            [ c, -s, 0, tx],
            [ s,  c, 0, ty],
            [ 0,  0, 1, tz],
            [ 0,  0, 0,  1],
        ], dtype=float)

    fake_cam1 = make_cam(  0.0, -0.25, 0.25, 0.0)
    fake_cam2 = make_cam(-0.15,  0.0 ,  0.5, 0.0)
    fake_cam3 = make_cam(  0.0,  0.25, 0.25, 0.0)

    dets = AprilDetections(cam1=fake_cam1, cam2=fake_cam2, cam3=fake_cam3)
    wf = WorldFrame.build_world_frame(dets, T_ARUCO)
    print("cam1 world:\n", wf.t_cam1)
    print("cam2 world:\n", wf.t_cam2)
    print("cam3 world:\n", wf.t_cam3)
    wf.plot()

def start_pipeline(serial):
    # Start pipeline for one camera.
    print(f"Starting camera: {serial}")
    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_device(serial)

    # Safe baseline for D4xx / many RealSense setups
    cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 15)
    cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 15)

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

def get_rgbd_frames(pipe, cam_name, retries=5, timeout_ms=5000, colorize_depth=False):
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
    if colorize_depth:
        depth_image = np.asanyarray(rs.colorizer().colorize(depth_frame).get_data())
    else:
        depth_image = np.asanyarray(depth_frame.get_data())
    return color_image, depth_image

def get_camera_panel(pipe, cam_name, K, dist, tag_size_m, retries=5, timeout_ms=5000):
    color_image, depth_image = get_rgbd_frames(pipe, cam_name, retries, timeout_ms, colorize_depth=True)
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

def init_cameras(cams: list[str]):
    pipelines = {}
    intrinsics = {}
    for cam in cams:
        pipe, K, dist = start_pipeline(SERIAL_NUMS[cam])
        pipelines[cam] = pipe
        intrinsics[cam] = (K, dist)
    return pipelines, intrinsics

def get_tag_transform_by_id(detections, tag_id):
    """Return 4x4 T_tag_to_cam for a specific tag id, or None if missing."""
    for det_id, T, _, _, _ in detections:
        if det_id == tag_id:
            return T
    return None

def localize_tee(args):
    wf = WorldFrame.load("wf.npz")
    # Saved world frame stores camera-to-world transforms.
    cam_to_world = {
        "top": wf.t_cam1,
        "corner": wf.t_cam2,
        "side": wf.t_cam3,
    }

    pipelines, intrinsics = init_cameras(args.cams)
    try:
        tee_tag_size_m = args.tee_tag_size_m
        print(f"[TEE] using tag id={TEE_TAG_ID} size={tee_tag_size_m:.4f} m")
        time.sleep(1.0)  # warm up sensors before one-shot capture
        tee_positions_world = []

        for cam in args.cams:
            frames = get_rgbd_frames(pipelines[cam], cam, retries=4, timeout_ms=10000)
            if frames is None:
                print(f"[WARN] {cam}: no RGBD frame, skipping Tee localization for this camera")
                continue

            color_image, _ = frames
            K, dist = intrinsics[cam]
            detections = detect_apriltags(color_image, K, dist, tee_tag_size_m)
            t_tee_to_cam = get_tag_transform_by_id(detections, TEE_TAG_ID)
            if t_tee_to_cam is None:
                print(f"[WARN] {cam}: Tee tag id={TEE_TAG_ID} not detected")
                continue

            # Tee origin in world frame: p_world = T_cam_to_world @ T_tee_to_cam @ [0,0,0,1].
            p_tee_world_h = cam_to_world[cam] @ (t_tee_to_cam @ np.array([0.0, 0.0, 0.0, 1.0]))
            p_tee_world = p_tee_world_h[:3]
            tee_positions_world.append(p_tee_world)
            print(
                f"[TEE][world] {cam:6s} -> x={p_tee_world[0]: .4f}, "
                f"y={p_tee_world[1]: .4f}, z={p_tee_world[2]: .4f}"
            )

        if tee_positions_world:
            mean_pos = np.mean(np.stack(tee_positions_world), axis=0)
            print(
                f"[TEE][world] mean  -> x={mean_pos[0]: .4f}, "
                f"y={mean_pos[1]: .4f}, z={mean_pos[2]: .4f}"
            )
        else:
            print("[TEE] No valid Tee detections from selected cameras.")
    finally:
        for pipe in pipelines.values():
            pipe.stop()

def main(args):
    pipelines, intrinsics = init_cameras(args.cams)
    try:
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
                args.table_tag_size_m,
                retries=4,
                timeout_ms=40000,
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
    parser.add_argument("--calibrate", action="store_true")
    parser.add_argument("--cams", nargs="+", default=["top", "corner", "side"])
    parser.add_argument("--tee", action="store_true")
    parser.add_argument(
        "--table-tag-size-m",
        type=float,
        default=0.1651,
        help="Table tag (id=0) edge length in meters.",
    )
    parser.add_argument(
        "--tee-tag-size-m",
        type=float,
        default=0.0445,
        help="Tee tag (id=2) edge length in meters.",
    )
    args = parser.parse_args()
    if args.calibrate:
        # test_world()
        # TODO

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
        wf = WorldFrame.build_world_frame(april_dets, T_ARUCO)
        wf.print_world_coordinates()

        wf.plot()

        # 3. serialize calibrated WorldFrame
        wf.save("wf.npz")

    elif args.tee:
        # localize the tee
        localize_tee(args)

    else:
        main(args)
    