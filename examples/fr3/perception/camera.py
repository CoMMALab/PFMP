"""
Camera utilities for RealSense cameras.
"""

from collections import OrderedDict
from typing import NamedTuple, Optional
import pyrealsense2 as rs
import numpy as np
import cv2
from .utils import (
    average_transforms,
    enforce_world_z_yaw_only,
    fuse_yaw_only_transforms_world,
    pose_to_4x4,
)

TABLE_TAG_ID = 0
TEE_TAG_ID = 2

# tag_to_world (T_R^A): 180 deg about world Z, then translate to tag origin.
T_ARUCO = np.array([
    [-1.0,  0.0,  0.0,  0.379],
    [ 0.0, -1.0,  0.0, -0.064],
    [ 0.0,  0.0,  1.0,  0.012],
    [ 0.0,  0.0,  0.0,  1.0  ],
])
assert np.isclose(np.linalg.det(T_ARUCO[:3, :3]), 1.0), \
    "T_ARUCO rotation must be a proper rotation (det = +1); diag(-1,1,1) is a reflection."

# Ordered name -> serial mapping; insertion order defines CAM_ORDER downstream.
SERIAL_NUMS: OrderedDict[str, str] = OrderedDict([
    # ("top", "234322070408"),  # old top camera
    # ("side",   "817512070808"),
    ("top",    "234322070341"),
    # ("corner", "234322070144"),
])
SERIAL_NUMS_INV: OrderedDict[str, str] = OrderedDict((v, k) for k, v in SERIAL_NUMS.items())


_APRILTAG_DICT   = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
_APRILTAG_PARAMS = cv2.aruco.DetectorParameters()
_BASE_DETECTOR   = cv2.aruco.ArucoDetector(_APRILTAG_DICT, _APRILTAG_PARAMS)


class AprilDetector(NamedTuple):
    tag_id: int
    tag_size_m: float
    corners: dict[str, np.ndarray] = {}  # serial -> (4, 3) corners in camera frame
    K:       dict[str, np.ndarray] = {}  # serial -> (3, 3) camera matrix
    dist:    dict[str, np.ndarray] = {}  # serial -> (5,) distortion coefficients

    def detect(self, serial: str, rgbd: tuple[np.ndarray, np.ndarray], K: np.ndarray, dist: np.ndarray, depth_scale: float = 1e-3) -> "tuple[bool, AprilDetector]":
        """Detect self.tag_id in rgbd and append result keyed by serial.

        Corners are stored as (4, 3) in camera frame, derived from PnP rather than
        raw depth at corner pixels (depth is unreliable at tag edges).
        Returns (True, updated_detector) on success, (False, self) if tag not found.
        """
        color, _ = rgbd
        gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
        raw_corners, ids, _ = _BASE_DETECTOR.detectMarkers(gray)
        if ids is not None:
            s = self.tag_size_m / 2.0
            obj_pts = np.array(
                [[-s, s, 0.0], [s, s, 0.0], [s, -s, 0.0], [-s, -s, 0.0]],
                dtype=np.float64,
            )
            for i, tid in enumerate(ids.flatten()):
                if tid == self.tag_id:
                    px = raw_corners[i][0].astype(np.float64)  # (4, 2)
                    ok, rvec, tvec = cv2.solvePnP(
                        obj_pts, px, K, dist, flags=cv2.SOLVEPNP_IPPE_SQUARE
                    )
                    if not ok:
                        return False, self
                    R, _ = cv2.Rodrigues(rvec)
                    # 3D corners in camera frame via PnP pose — more stable than raw depth
                    corners_3d = (obj_pts @ R.T) + tvec.ravel()  # (4, 3)
                    return True, self._replace(
                        corners={**self.corners, serial: corners_3d},
                        K={**self.K, serial: K},
                        dist={**self.dist, serial: dist},
                    )
        return False, self


class WorldFrame(NamedTuple):
    t_aruco:  np.ndarray
    Ks:       OrderedDict[str, np.ndarray]  # serial -> (3, 3) intrinsics
    t_cams:   OrderedDict[str, np.ndarray]  # serial -> (4, 4) cam-to-world
    t_pose:   Optional[np.ndarray] = None

    @staticmethod
    def anchor(det: AprilDetector, t_april_to_robot: np.ndarray) -> "WorldFrame":
        """Build an anchored WorldFrame from per-camera AprilTag detections.

        For each camera, Procrustes-registers the PnP corners (camera frame)
        against the known tag corners in robot frame to get T_cam_to_world.
        """
        s = det.tag_size_m / 2.0
        local_corners = np.array(
            [[-s, s, 0.0], [s, s, 0.0], [s, -s, 0.0], [-s, -s, 0.0]],
            dtype=np.float64,
        )
        local_h = np.column_stack([local_corners, np.ones(4)])
        pts_world = (t_april_to_robot @ local_h.T).T[:, :3]   # (4, 3) in robot frame

        t_cams, Ks = OrderedDict(), OrderedDict()
        for serial, pts_cam in det.corners.items():
            mu_w = pts_world.mean(axis=0)
            mu_c = pts_cam.mean(axis=0)
            H = (pts_cam - mu_c).T @ (pts_world - mu_w)
            U, _, Vt = np.linalg.svd(H)
            R = Vt.T @ U.T
            if np.linalg.det(R) < 0:
                Vt[-1] *= -1
                R = Vt.T @ U.T
            t = mu_w - R @ mu_c
            T = np.eye(4, dtype=np.float64)
            T[:3, :3] = R
            T[:3, 3] = t
            t_cams[serial] = T
            Ks[serial] = det.K[serial]

        return WorldFrame(t_aruco=t_april_to_robot, Ks=Ks, t_cams=t_cams)

    def update(self, x: np.ndarray) -> "WorldFrame":
        """Return a new WorldFrame with extrinsics from x (6*N,): N × (rvec[3], tvec[3]) world→cam.

        Camera order follows self.t_cams insertion order.
        """
        t_cams = OrderedDict()
        for i, serial in enumerate(self.t_cams):
            R, _ = cv2.Rodrigues(x[6*i : 6*i+3].reshape(3, 1))
            t = x[6*i+3 : 6*i+6]
            T_world_to_cam = np.eye(4, dtype=np.float64)
            T_world_to_cam[:3, :3] = R
            T_world_to_cam[:3, 3] = t
            t_cams[serial] = np.linalg.inv(T_world_to_cam)
        return self._replace(t_cams=t_cams)

    def save(self, path: str):
        arrays = {"t_aruco": self.t_aruco, "cam_names": np.array(list(self.t_cams.keys()))}
        for name in self.t_cams:
            arrays[f"K_{name}"] = self.Ks[name]
            arrays[f"t_{name}"] = self.t_cams[name]
        if self.t_pose is not None:
            arrays["t_pose"] = self.t_pose
        np.savez(path, **arrays)

    @staticmethod
    def load(path: str) -> "WorldFrame":
        data = np.load(path, allow_pickle=False)
        serials = list(data["cam_names"])
        Ks     = OrderedDict((s, data[f"K_{s}"]) for s in serials)
        t_cams = OrderedDict((s, data[f"t_{s}"]) for s in serials)
        return WorldFrame(
            t_aruco=data["t_aruco"],
            Ks=Ks,
            t_cams=t_cams,
            t_pose=data["t_pose"] if "t_pose" in data else None,
        )

    def is_anchored(self) -> bool:
        return not np.allclose(self.t_aruco, np.eye(4), atol=1e-8)

    def reproject(
        self,
        pts_px: np.ndarray,  # (N, 2) pixel coords in source camera
        depths: np.ndarray,  # (N,)   metric depth per point (metres)
        src: str,            # source camera name
        dst: str,            # destination camera name
    ) -> np.ndarray:         # (N, 2) pixel coords in destination camera
        """Unproject pixel-space points from one camera to world, then project to another camera."""
        pts_px = np.asarray(pts_px, dtype=np.float64)
        depths = np.asarray(depths, dtype=np.float64).ravel()

        serial_src, serial_dst = SERIAL_NUMS[src], SERIAL_NUMS[dst]
        K_src = self.Ks[serial_src]
        xs = (pts_px[:, 0] - K_src[0, 2]) / K_src[0, 0] * depths
        ys = (pts_px[:, 1] - K_src[1, 2]) / K_src[1, 1] * depths
        pts_src_h = np.column_stack([xs, ys, depths, np.ones(len(depths))])
        pts_world = (self.t_cams[serial_src] @ pts_src_h.T).T[:, :3]

        pts_world_h = np.column_stack([pts_world, np.ones(len(pts_world))])
        pts_dst = (np.linalg.inv(self.t_cams[serial_dst]) @ pts_world_h.T).T[:, :3]

        K_dst = self.Ks[serial_dst]
        u = pts_dst[:, 0] / pts_dst[:, 2] * K_dst[0, 0] + K_dst[0, 2]
        v = pts_dst[:, 1] / pts_dst[:, 2] * K_dst[1, 1] + K_dst[1, 2]
        return np.column_stack([u, v])


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

    frames = rs.align(rs.stream.color).process(frames)
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

def get_depth_scale(pipe) -> float:
    """Return metric scale (meters per raw depth unit) for an active pipeline."""
    profile = pipe.get_active_profile()
    sensor = profile.get_device().first_depth_sensor()
    return float(sensor.get_depth_scale())

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
    return wf.Ks[SERIAL_NUMS[cam_name]]

def _wf_t_cam_by_cam_name(wf: WorldFrame, cam_name: str) -> np.ndarray:
    return wf.t_cams[SERIAL_NUMS[cam_name]]

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
        if cam_name not in wf.t_cams:
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
