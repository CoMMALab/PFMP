import numpy as np
import cv2

# transforms
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

def pose_to_4x4(rvec, tvec):
    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = tvec.flatten()
    return T




# 