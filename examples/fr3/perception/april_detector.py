from __future__ import annotations

import numpy as np

from camera import WorldFrame, AprilDetector, TEE_TAG_ID
from ekf import EKF, PoseDetection
from geom_tee import TEE_TAG_SIZE_M, TAG_IN_BODY


def tag_pose_from_corners(corners_cam: np.ndarray) -> np.ndarray:
    center_cam = corners_cam.mean(axis=0)

    x_axis = corners_cam[1] - corners_cam[0]
    x_axis = x_axis / np.linalg.norm(x_axis)

    y_axis = corners_cam[0] - corners_cam[3]
    y_axis = y_axis / np.linalg.norm(y_axis)

    z_axis = np.cross(x_axis, y_axis)
    z_axis = z_axis / np.linalg.norm(z_axis)

    y_axis = np.cross(z_axis, x_axis)
    y_axis = y_axis / np.linalg.norm(y_axis)

    T_tag_to_cam = np.eye(4)
    T_tag_to_cam[:3, 0] = x_axis
    T_tag_to_cam[:3, 1] = y_axis
    T_tag_to_cam[:3, 2] = z_axis
    T_tag_to_cam[:3, 3] = center_cam

    return T_tag_to_cam


def pose_from_T_body(T_body: np.ndarray) -> np.ndarray:
    return np.array(
        [
            T_body[0, 3],
            T_body[1, 3],
            np.arctan2(T_body[1, 0], T_body[0, 0]),
        ],
        dtype=np.float64,
    )


def detect_tee_tag(
    color: np.ndarray,
    depth: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray,
    wf: WorldFrame,
    serial: str,
    depth_scale: float,
    name: str = "tag",
    R: np.ndarray | None = None,
    gated: bool = True,
) -> list[PoseDetection]:
    """
    Convert AprilDetector output into EKF PoseDetection(s).

    Returns [] if tag is not visible.
    """
    detector = AprilDetector(
        tag_id=TEE_TAG_ID,
        tag_size_m=TEE_TAG_SIZE_M,
    )

    ok, detector = detector.detect(
        serial=serial,
        rgbd=(color, depth),
        K=K,
        dist=dist,
        depth_scale=depth_scale,
    )

    if not ok:
        return None

    corners_cam = detector.corners[serial]
    T_tag_to_cam = tag_pose_from_corners(corners_cam)

    T_body = wf.t_cams[serial] @ T_tag_to_cam @ np.linalg.inv(TAG_IN_BODY)
    pose = pose_from_T_body(T_body)

    return PoseDetection(
        name=name,
        pose=pose,
        R=np.diag([1e-4, 1e-4, 1e-3]) if R is None else R,
        gated=gated,
    )