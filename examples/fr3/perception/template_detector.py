import numpy as np
from geom_tee import THICKNESS
from camera import WorldFrame, SERIAL_NUMS


def project(verts_w, body_z, T_w2c, K):
    """Project (N, 2) world xy verts at body_z into image pixel coords (N, 2) int."""
    import cv2
    pts3d = np.column_stack([verts_w, np.full(len(verts_w), body_z), np.ones(len(verts_w))])
    pts_cam = (T_w2c @ pts3d.T).T[:, :3]
    px = pts_cam[:, :2] / pts_cam[:, 2:3] * np.array([K[0,0], K[1,1]]) + np.array([K[0,2], K[1,2]])
    return px.astype(np.int32)


def template_match_T(
    mask: np.ndarray,
    frame: WorldFrame,
    serial: str,
    K: np.ndarray,
    body_xy: np.ndarray,
    body_yaw: float,
    body_z: float,
    dx: float = 0.015, dy: float = 0.015, dyaw: float = 0.2,
    n_samples: int = 200,
    seed: int = 0,
):
    """Monte Carlo search over (x, y, yaw) around initial estimate, scored by IoU.

    Samples x, y from uniform(-dx, dx) and yaw from von Mises centered on body_yaw.
    Returns (best_xy, best_yaw, best_iou, best_template_mask).
    """
    import cv2
    from geom_tee import verts_world
    T_w2c = np.linalg.inv(frame.t_cams[serial])
    H, W = mask.shape
    rng = np.random.default_rng(seed)

    xs   = body_xy[0] + rng.uniform(-dx,   dx,   n_samples)
    ys   = body_xy[1] + rng.uniform(-dy,   dy,   n_samples)
    yaws = rng.vonmises(body_yaw, kappa=1.0 / max(dyaw**2, 1e-6), size=n_samples)

    best_iou, best_xy, best_yaw, best_tmpl = -1.0, body_xy, body_yaw, None

    for x, y, yaw in zip(xs, ys, yaws):
        verts_px = project(verts_world(np.array([x, y]), yaw), body_z, T_w2c, K)
        tmpl = np.zeros((H, W), dtype=np.uint8)
        cv2.fillPoly(tmpl, [verts_px], 1)
        tmpl = tmpl.astype(bool)
        inter = (mask & tmpl).sum()
        union = (mask | tmpl).sum()
        iou = inter / union if union > 0 else 0.0
        if iou > best_iou:
            best_iou, best_xy, best_yaw, best_tmpl = iou, np.array([x, y]), yaw, tmpl

    return best_xy, best_yaw, best_iou, best_tmpl


# Orange T-block HSV range — tune to match physical block under lab lighting.
_T_HSV_LO = np.array([3,  50, 70], dtype=np.uint8)   # lower S from 80 -> 50
_T_HSV_HI = np.array([30, 255, 255], dtype=np.uint8) # widen H from 25 -> 30

_Z_MIN = -0.04
_Z_PAD = 0.025

_CLEAN_MASK = True

def segment_T(
    frame: WorldFrame,
    color: np.ndarray,
    depth: np.ndarray,
    depth_scale: float,
    cam: str = "top",
) -> np.ndarray:
    """Return a boolean mask (H, W) of pixels belonging to the T-block.

    Combines an orange HSV color filter with a depth z-filter in table frame.
    """
    import cv2
    hsv = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)
    color_mask = cv2.inRange(hsv, _T_HSV_LO, _T_HSV_HI).astype(bool)

    H, W = depth.shape
    d = depth.astype(np.float64) * depth_scale
    valid = (d > 0.01) & np.isfinite(d)

    serial = SERIAL_NUMS[cam]
    K = frame.Ks[serial]
    ys, xs = np.mgrid[0:H, 0:W]
    x_c = (xs - K[0, 2]) / K[0, 0] * d
    y_c = (ys - K[1, 2]) / K[1, 1] * d
    pts_cam = np.stack([x_c, y_c, d, np.ones_like(d)], axis=-1)  # (H, W, 4)

    pts_world = (frame.t_cams[serial] @ pts_cam.reshape(-1, 4).T).T  # (H*W, 4)
    t_world_to_table = np.linalg.inv(frame.t_aruco)
    pts_table = (t_world_to_table @ pts_world.T).T[:, 2].reshape(H, W)

    depth_mask = valid & (pts_table > _Z_MIN) & (pts_table < THICKNESS + _Z_PAD)
    mask = color_mask & depth_mask

    if _CLEAN_MASK:
        k = np.ones((5, 5), np.uint8)
        mask_u8 = mask.astype(np.uint8) * 255
        mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, k, iterations=1)
        mask = mask_u8.astype(bool)

    return mask

def segmentation_median_world_xy(
    mask: np.ndarray,
    depth: np.ndarray,
    K: np.ndarray,
    T_c2w: np.ndarray,
    depth_scale: float,
) -> np.ndarray | None:
    ys, xs = np.where(mask > 0)

    if len(xs) < 50:
        return None

    u = int(np.median(xs))
    v = int(np.median(ys))

    # Use local valid depth around the median pixel.
    r = 5
    patch = depth[
        max(0, v - r): min(depth.shape[0], v + r + 1),
        max(0, u - r): min(depth.shape[1], u + r + 1),
    ]

    valid = patch[patch > 0]
    if len(valid) == 0:
        return None

    z = float(np.median(valid)) * depth_scale

    x_c = (u - K[0, 2]) * z / K[0, 0]
    y_c = (v - K[1, 2]) * z / K[1, 1]

    p_c = np.array([x_c, y_c, z, 1.0])
    p_w = T_c2w @ p_c

    return p_w[:2]
