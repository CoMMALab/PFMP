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
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

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

INCH_TO_M = 0.0254
TEE_WIDTH_M = 8.0 * INCH_TO_M
TEE_LENGTH_M = 8.0 * INCH_TO_M
TEE_THICKNESS_M = 2.0 * INCH_TO_M

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

def _pose_from_detections(corners, ids, K, dist, tag_size_m):
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

def _staged_preprocess_images(color_image):
    gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    blur = cv2.GaussianBlur(enhanced, (0, 0), 1.0)
    enhanced = cv2.addWeighted(enhanced, 1.35, blur, -0.35, 0.0)
    return gray, enhanced

def detect_apriltags(color_image, K, dist, tag_size_m, staged=True, return_debug=False):
    gray, enhanced = _staged_preprocess_images(color_image)
    debug = {"gray": gray, "enhanced": enhanced, "stage": 0}

    # Stage 1: raw grayscale.
    corners, ids, _ = APRILTAG_DETECTOR.detectMarkers(gray)
    detections = _pose_from_detections(corners, ids, K, dist, tag_size_m)
    if detections:
        debug["stage"] = 1
        return (detections, debug) if return_debug else detections
    if not staged:
        return ([], debug) if return_debug else []

    # Stage 2: CLAHE + mild unsharp mask.
    corners, ids, _ = APRILTAG_DETECTOR.detectMarkers(enhanced)
    detections = _pose_from_detections(corners, ids, K, dist, tag_size_m)
    if detections:
        debug["stage"] = 2
        return (detections, debug) if return_debug else detections

    # Stage 3: upscale fallback for small/far tags.
    scale = 1.5
    up = cv2.resize(enhanced, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    corners, ids, _ = APRILTAG_DETECTOR.detectMarkers(up)
    if ids is None:
        debug["stage"] = 3
        return ([], debug) if return_debug else []
    for c in corners:
        c[0] /= scale
    detections = _pose_from_detections(corners, ids, K, dist, tag_size_m)
    debug["stage"] = 3
    return (detections, debug) if return_debug else detections

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

def average_transforms(transforms):
    """Average 4x4 rigid transforms (mean t, SVD-mean R)."""
    if not transforms:
        return None
    translations = np.array([T[:3, 3] for T in transforms], dtype=np.float64)
    mean_t = translations.mean(axis=0)

    Rs = np.array([T[:3, :3] for T in transforms], dtype=np.float64)
    U, _, Vt = np.linalg.svd(Rs.mean(axis=0))
    mean_R = U @ Vt
    if np.linalg.det(mean_R) < 0:
        Vt[-1] *= -1
        mean_R = U @ Vt

    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = mean_R
    T[:3, 3] = mean_t
    return T

def rotation_matrix_to_rpy_zyx_deg(R):
    """Return roll/pitch/yaw in degrees using ZYX convention."""
    sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    singular = sy < 1e-6
    if not singular:
        roll = np.arctan2(R[2, 1], R[2, 2])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = np.arctan2(R[1, 0], R[0, 0])
    else:
        roll = np.arctan2(-R[1, 2], R[1, 1])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = 0.0
    return np.rad2deg(np.array([roll, pitch, yaw], dtype=np.float64))

def enforce_world_z_yaw_only(T):
    """Project a pose to yaw-only rotation about +world-Z, preserving translation."""
    yaw = np.arctan2(T[1, 0], T[0, 0])
    c, s = np.cos(yaw), np.sin(yaw)
    T_proj = np.eye(4, dtype=np.float64)
    T_proj[:3, :3] = np.array(
        [
            [c, -s, 0.0],
            [s,  c, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    T_proj[:3, 3] = T[:3, 3]
    return T_proj

def fuse_yaw_only_transforms_world(transforms):
    """Fuse yaw-only poses with circular-mean yaw and mean translation."""
    if not transforms:
        return None
    translations = np.array([T[:3, 3] for T in transforms], dtype=np.float64)
    mean_t = translations.mean(axis=0)
    yaws = np.array([np.arctan2(T[1, 0], T[0, 0]) for T in transforms], dtype=np.float64)
    mean_yaw = np.arctan2(np.sin(yaws).mean(), np.cos(yaws).mean())
    c, s = np.cos(mean_yaw), np.sin(mean_yaw)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    T[:3, 3] = mean_t
    return T

def plot_tee_world(wf: WorldFrame, raw_tee_world, mean_tee_world: np.array):
    """Plot raw and average Tee poses in world frame with robot/world origin."""
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    def draw_frame(T, label, axis_scale=0.07):
        pos = T[:3, 3]
        ax.text(*pos, label, fontsize=9)
        for i, c in enumerate(["r", "g", "b"]):
            axis_vec = T[:3, i] * axis_scale
            ax.quiver(pos[0], pos[1], pos[2], axis_vec[0], axis_vec[1], axis_vec[2], color=c, normalize=False)

    def box_faces_local(xmin, xmax, ymin, ymax, zmin, zmax):
        verts = np.array(
            [
                [xmin, ymin, zmin],
                [xmax, ymin, zmin],
                [xmax, ymax, zmin],
                [xmin, ymax, zmin],
                [xmin, ymin, zmax],
                [xmax, ymin, zmax],
                [xmax, ymax, zmax],
                [xmin, ymax, zmax],
            ],
            dtype=np.float64,
        )
        faces = [
            [0, 1, 2, 3],  # bottom
            [4, 5, 6, 7],  # top
            [0, 1, 5, 4],
            [1, 2, 6, 5],
            [2, 3, 7, 6],
            [3, 0, 4, 7],
        ]
        return verts, faces

    def transform_points(T, points):
        pts_h = np.hstack((points, np.ones((points.shape[0], 1), dtype=np.float64)))
        return (T @ pts_h.T).T[:, :3]

    def draw_t_shape(T, color, alpha, label):
        # Visualization-only flip: rotate Tee 180 deg about its local Z (yaw).
        yaw_flip = np.eye(4, dtype=np.float64)
        yaw_flip[:3, :3] = np.array(
            [
                [-1.0,  0.0, 0.0],
                [ 0.0, -1.0, 0.0],
                [ 0.0,  0.0, 1.0],
            ],
            dtype=np.float64,
        )
        T_draw = T @ yaw_flip

        half_w = TEE_WIDTH_M / 2.0
        half_t = TEE_THICKNESS_M / 2.0
        stem_half_w = TEE_THICKNESS_M / 2.0
        stem_len = TEE_LENGTH_M - TEE_THICKNESS_M  # total length stays 8in with 2in top thickness

        # Tag origin is at the junction where top and stem meet.
        # Top bar extends in +Y; stem extends in -Y.
        top_verts, top_faces = box_faces_local(
            -half_w, half_w, 0.0, TEE_THICKNESS_M, -half_t, half_t
        )
        stem_verts, stem_faces = box_faces_local(
            -stem_half_w, stem_half_w, -stem_len, 0.0, -half_t, half_t
        )

        for verts, faces in [(top_verts, top_faces), (stem_verts, stem_faces)]:
            verts_w = transform_points(T_draw, verts)
            poly = Poly3DCollection(
                [[verts_w[i] for i in face] for face in faces],
                facecolors=color,
                edgecolors=color,
                linewidths=0.8,
                alpha=alpha,
            )
            ax.add_collection3d(poly)

        draw_frame(T_draw, label, axis_scale=0.05)

    # World/robot origin: only draw a single 3-axis frame (no point marker).
    draw_frame(np.eye(4, dtype=np.float64), "world")

    # Use consistent colors for each camera and its Tee estimate.
    raw_colors = {"top": "tab:red", "corner": "tab:green", "side": "tab:blue"}
    for cam_name, T in raw_tee_world:
        c = raw_colors.get(cam_name, "gray")
        draw_t_shape(T, c, alpha=0.18, label=f"tee_{cam_name}")

    # Average/fused Tee pose in world frame.
    draw_t_shape(mean_tee_world, "orange", alpha=0.35, label="tee_mean")

    # Optional context: camera centers (same color as corresponding Tee estimate).
    all_points = [np.zeros(3)]
    cam_entries = [("top", wf.t_cam1), ("corner", wf.t_cam2), ("side", wf.t_cam3)]
    for cam_name, T in cam_entries:
        color = raw_colors[cam_name]
        label = f"cam_{cam_name}"
        p = T[:3, 3]
        all_points.append(p)
        ax.scatter([p[0]], [p[1]], [p[2]], c=color, s=30)
        ax.text(p[0], p[1], p[2], label, fontsize=8)
    for _, T in raw_tee_world:
        all_points.append(T[:3, 3])
    all_points.append(mean_tee_world[:3, 3])

    pts = np.array(all_points, dtype=np.float64)
    max_abs = float(np.max(np.abs(pts))) if pts.size else 0.8
    lim = max(0.8, max_abs + 0.2)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)

    ax.set_title("Tee Poses in World Frame (Raw + Mean)")
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    plt.tight_layout()
    plt.show()

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
        tee_transforms_world = []
        raw_tee_world = []

        for cam in args.cams:
            frames = get_rgbd_frames(pipelines[cam], cam, retries=4, timeout_ms=10000)
            if frames is None:
                print(f"[WARN] {cam}: no RGBD frame, skipping Tee localization for this camera")
                continue

            color_image, _ = frames
            K, dist = intrinsics[cam]
            detections, debug = detect_apriltags(color_image, K, dist, tee_tag_size_m, staged=True, return_debug=True)
            t_tee_to_cam = get_tag_transform_by_id(detections, TEE_TAG_ID)
            if t_tee_to_cam is None:
                print(f"[WARN] {cam}: Tee tag id={TEE_TAG_ID} not detected")
                continue

            # Tee origin in world frame: p_world = T_cam_to_world @ T_tee_to_cam @ [0,0,0,1].
            t_tee_to_world = cam_to_world[cam] @ t_tee_to_cam
            # Enforce physical constraint: Tee rotates only about +world-Z.
            t_tee_to_world = enforce_world_z_yaw_only(t_tee_to_world)
            p_tee_world_h = t_tee_to_world @ np.array([0.0, 0.0, 0.0, 1.0])
            p_tee_world = p_tee_world_h[:3]
            tee_positions_world.append(p_tee_world)
            tee_transforms_world.append(t_tee_to_world)
            raw_tee_world.append((cam, t_tee_to_world))
            roll_deg, pitch_deg, yaw_deg = rotation_matrix_to_rpy_zyx_deg(t_tee_to_world[:3, :3])
            print(
                f"[TEE][world] {cam:6s} -> x={p_tee_world[0]: .4f}, "
                f"y={p_tee_world[1]: .4f}, z={p_tee_world[2]: .4f}"
            )
            print(
                f"[TEE][world] {cam:6s} -> roll={roll_deg: .1f} deg, "
                f"pitch={pitch_deg: .1f} deg, yaw={yaw_deg: .1f} deg (ZYX)"
            )

        if tee_positions_world:
            mean_pos = np.mean(np.stack(tee_positions_world), axis=0)
            mean_T = fuse_yaw_only_transforms_world(tee_transforms_world)
            print(
                f"[TEE][world] mean  -> x={mean_pos[0]: .4f}, "
                f"y={mean_pos[1]: .4f}, z={mean_pos[2]: .4f}"
            )
            if mean_T is not None:
                m_roll, m_pitch, m_yaw = rotation_matrix_to_rpy_zyx_deg(mean_T[:3, :3])
                print(
                    f"[TEE][world] mean  -> roll={m_roll: .1f} deg, "
                    f"pitch={m_pitch: .1f} deg, yaw={m_yaw: .1f} deg (ZYX, Z-only constrained)"
                )
                print("[TEE][world] plotting raw Tee poses and fused mean pose")
                plot_tee_world(wf, raw_tee_world, mean_T)
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
    