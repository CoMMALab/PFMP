"""
Camera utilities for RealSense cameras.
"""

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
                # "top": "234322070408", # old
                "top": "817512070808", #austin's replacement
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

class Intrinsics(NamedTuple):
    cam1: np.array
    cam2: np.array
    cam3: np.array

# tag_to_world (T_R^A): 180 deg about world Z, then translate to tag origin.
T_ARUCO = np.array([
    [-1.0,  0.0,  0.0,  0.379],
    [ 0.0, -1.0,  0.0, -0.064],
    [ 0.0,  0.0,  1.0,  0.012],
    [ 0.0,  0.0,  0.0,  1.0  ],
])
assert np.isclose(np.linalg.det(T_ARUCO[:3, :3]), 1.0), \
    "T_ARUCO rotation must be a proper rotation (det = +1); diag(-1,1,1) is a reflection."

INCH_TO_M = 0.0254
TEE_WIDTH_M = 8.0 * INCH_TO_M
TEE_LENGTH_M = 8.0 * INCH_TO_M
TEE_THICKNESS_M = 2.0 * INCH_TO_M

class WorldFrame(NamedTuple):
    t_aruco: np.array
    K_cam1: np.array
    K_cam2: np.array
    K_cam3: np.array
    t_cam1: np.array
    t_cam2: np.array
    t_cam3: np.array
    t_pose: Optional[np.array] = None

    @staticmethod
    def build_world_frame(dets: AprilDetections, t_aruco: np.array, intrs: Intrinsics) -> "Self":
        # T_world_cam = T_world_aruco @ inv(T_cam_aruco)
        t_cam1 = t_aruco @ np.linalg.inv(dets.cam1)
        t_cam2 = t_aruco @ np.linalg.inv(dets.cam2)
        t_cam3 = t_aruco @ np.linalg.inv(dets.cam3)

        return WorldFrame(t_aruco=t_aruco,
            K_cam1=intrs.cam1, K_cam2=intrs.cam2, K_cam3=intrs.cam3,
            t_cam1=t_cam1, t_cam2=t_cam2, t_cam3=t_cam3,
        )

    def save(self, path: str):
        arrays = dict(t_aruco=self.t_aruco,
            K_cam1=self.K_cam1, K_cam2=self.K_cam2, K_cam3=self.K_cam3,
            t_cam1=self.t_cam1, t_cam2=self.t_cam2, t_cam3=self.t_cam3,
        )
        if self.t_pose is not None:
            arrays["t_pose"] = self.t_pose
        np.savez(path, **arrays)

    @staticmethod
    def load(path: str) -> "WorldFrame":
        data = np.load(path)
        return WorldFrame(
            t_aruco=data["t_aruco"],
            K_cam1=data["K_cam1"],
            K_cam2=data["K_cam2"],
            K_cam3=data["K_cam3"],
            t_cam1=data["t_cam1"],
            t_cam2=data["t_cam2"],
            t_cam3=data["t_cam3"],
            t_pose=data["t_pose"] if "t_pose" in data else None,
        )

    def is_anchored(self) -> bool:
        """True if t_aruco is a non-identity transform (i.e. anchored to robot/world).

        calibrate.py writes t_aruco = I as a sentinel for "unanchored extrinsics
        only"; anchor.py writes t_aruco = T_ARUCO. Downstream consumers that
        depend on robot/world coordinates should refuse unanchored files.
        """
        return not np.allclose(self.t_aruco, np.eye(4), atol=1e-8)

    def _K_cam(self, cam: int) -> np.ndarray:
        return {1: self.K_cam1, 2: self.K_cam2, 3: self.K_cam3}[cam]

    def _t_cam(self, cam: int) -> np.ndarray:
        return {1: self.t_cam1, 2: self.t_cam2, 3: self.t_cam3}[cam]

    def reproject(
        self,
        pts_px: np.ndarray,  # (N, 2) pixel coords in source camera
        depths: np.ndarray,  # (N,)   metric depth per point (metres)
        src: int,            # source camera: 1, 2, or 3
        dst: int,            # destination camera: 1, 2, or 3
    ) -> np.ndarray:         # (N, 2) pixel coords in destination camera
        """Unproject pixel-space points from one camera to world, then project to another camera.

        1. Back-project to 3-D in source camera frame: X = K_src^{-1} @ [u,v,1] * depth
        2. Transform to world via T_cam_src_to_world.
        3. Transform to destination camera frame.
        4. Project with K_dst.
        """
        pts_px = np.asarray(pts_px, dtype=np.float64)
        depths = np.asarray(depths, dtype=np.float64).ravel()

        # Back-project: normalised image coords × depth
        K_src = self._K_cam(src)
        fx, fy = K_src[0, 0], K_src[1, 1]
        cx, cy = K_src[0, 2], K_src[1, 2]
        xs = (pts_px[:, 0] - cx) / fx * depths
        ys = (pts_px[:, 1] - cy) / fy * depths
        pts_src = np.column_stack([xs, ys, depths])          # (N, 3)

        # Source camera frame → world
        pts_src_h = np.column_stack([pts_src, np.ones(len(pts_src))])
        pts_world = (self._t_cam(src) @ pts_src_h.T).T[:, :3]   # (N, 3)

        # World → destination camera frame
        pts_world_h = np.column_stack([pts_world, np.ones(len(pts_world))])
        pts_dst = (np.linalg.inv(self._t_cam(dst)) @ pts_world_h.T).T[:, :3]  # (N, 3)

        # Perspective project
        K_dst = self._K_cam(dst)
        u = pts_dst[:, 0] / pts_dst[:, 2] * K_dst[0, 0] + K_dst[0, 2]
        v = pts_dst[:, 1] / pts_dst[:, 2] * K_dst[1, 1] + K_dst[1, 2]
        return np.column_stack([u, v])

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
        return (detections, corners, debug) if return_debug else (detections, corners)
    if not staged:
        return ([], [], debug) if return_debug else ([], [])

    # Stage 2: CLAHE + mild unsharp mask.
    corners, ids, _ = APRILTAG_DETECTOR.detectMarkers(enhanced)
    detections = _pose_from_detections(corners, ids, K, dist, tag_size_m)
    if detections:
        debug["stage"] = 2
        return (detections, corners, debug) if return_debug else (detections, corners)

    # Stage 3: upscale fallback for small/far tags.
    scale = 1.5
    up = cv2.resize(enhanced, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    corners, ids, _ = APRILTAG_DETECTOR.detectMarkers(up)
    if ids is None:
        debug["stage"] = 3
        return ([], [], debug) if return_debug else ([], [])
    for c in corners:
        c[0] /= scale
    detections = _pose_from_detections(corners, ids, K, dist, tag_size_m)
    debug["stage"] = 3

    return (detections, corners, debug) if return_debug else (detections, corners)


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

def get_tag_corners_by_id(detections, tag_id):
    """Return AprilTag corners (4, 2) for a specific tag id, or None if missing."""
    for det_id, _, corners, _, _ in detections:
        if det_id == tag_id:
            return corners[0].astype(np.float64)
    return None

def _wf_intrinsics_by_cam_name(wf: WorldFrame, cam_name: str) -> np.ndarray:
    return {"top": wf.K_cam1, "corner": wf.K_cam2, "side": wf.K_cam3}[cam_name]

def _wf_t_cam_by_cam_name(wf: WorldFrame, cam_name: str) -> np.ndarray:
    return {"top": wf.t_cam1, "corner": wf.t_cam2, "side": wf.t_cam3}[cam_name]

def multi_view_tag_pnp(
    wf: WorldFrame,
    observations,
    tag_size_m: float,
    yaw_only: bool = False,
    verbose: bool = False,
):
    """Estimate T_tag_to_world from multiple camera corner observations.

    Args:
        wf: WorldFrame containing per-camera intrinsics and cam->world extrinsics.
        observations: iterable of (cam_name, corners_px) where corners_px is (4, 2).
        tag_size_m: Physical tag size in meters.
        yaw_only: If True, project each estimate to yaw-only in world before fusing.
        verbose: If True, print per-camera diagnostics.
    """
    s = tag_size_m / 2.0
    obj_points = np.array(
        [[-s, s, 0.0], [s, s, 0.0], [s, -s, 0.0], [-s, -s, 0.0]],
        dtype=np.float64,
    )
    dist = np.zeros(5, dtype=np.float64)
    transforms_world = []

    for cam_name, corners_px in observations:
        if cam_name not in ("top", "corner", "side"):
            if verbose:
                print(f"[WARN] multi_view_tag_pnp: unknown camera '{cam_name}', skipping")
            continue

        img_points = np.asarray(corners_px, dtype=np.float64).reshape(-1, 2)
        if img_points.shape != (4, 2):
            if verbose:
                print(
                    f"[WARN] multi_view_tag_pnp: {cam_name} corners shape {img_points.shape}, "
                    "expected (4, 2); skipping"
                )
            continue

        K = _wf_intrinsics_by_cam_name(wf, cam_name)
        ok, rvec, tvec = cv2.solvePnP(
            obj_points,
            img_points,
            K,
            dist,
            flags=cv2.SOLVEPNP_IPPE_SQUARE,
        )
        if not ok:
            if verbose:
                print(f"[WARN] multi_view_tag_pnp: solvePnP failed for {cam_name}")
            continue

        t_tag_to_cam = pose_to_4x4(rvec, tvec)
        t_tag_to_world = _wf_t_cam_by_cam_name(wf, cam_name) @ t_tag_to_cam
        if yaw_only:
            t_tag_to_world = enforce_world_z_yaw_only(t_tag_to_world)
        transforms_world.append(t_tag_to_world)

        if verbose:
            print(f"[ANCHOR][PnP] {cam_name}: t={t_tag_to_world[:3, 3]}")

    if not transforms_world:
        return None

    fused = (
        fuse_yaw_only_transforms_world(transforms_world)
        if yaw_only
        else average_transforms(transforms_world)
    )
    if verbose and fused is not None:
        print(f"[ANCHOR][PnP] fused from {len(transforms_world)} camera(s): t={fused[:3, 3]}")
    return fused

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