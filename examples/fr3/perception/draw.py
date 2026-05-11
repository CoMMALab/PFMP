"""Drawing utilities for T-block perception visualization."""

import cv2
import numpy as np
from typing import NamedTuple, Optional

from camera import WorldFrame
from ekf import EKF
from template_detector import project

# ── Colors (BGR) ──────────────────────────────────────────────────────────────

CYAN    = (255, 255,   0)
RED     = (  0,   0, 255)
BLUE    = (255,   0,   0)
GREEN   = (  0, 255,   0)
YELLOW  = (  0, 255, 255)
MAGENTA = (255,   0, 255)


# ── Primitives ────────────────────────────────────────────────────────────────

def draw_polyline(image, pts_px, color, thickness=2):
    cv2.polylines(image, [pts_px.reshape(-1, 1, 2)], isClosed=True,
                  color=color, thickness=thickness)


def project_world_xy(xy, z_body, T_w2c, K):
    return project(np.asarray(xy).reshape(-1, 2), z_body, T_w2c, K)


def draw_projected_outline(image, xy, yaw, z_body, T_w2c, K, color, thickness=2):
    from geom_tee import verts_world
    pts_px = project(verts_world(xy, yaw), z_body, T_w2c, K)
    draw_polyline(image, pts_px, color, thickness)
    return pts_px


def project_tag_corners(corners_cam, T_c2w, T_w2c, K):
    corners_h   = np.column_stack([corners_cam, np.ones(len(corners_cam))])
    corners_world = (T_c2w @ corners_h.T).T[:, :3]
    return project(corners_world[:, :2], float(corners_world[:, 2].mean()), T_w2c, K)


def draw_covariance_ellipse(image, mean_xy, cov_xy, z_body, T_w2c, K,
                             nsigmas=2.0, color=CYAN, thickness=2):
    vals, vecs = np.linalg.eigh(cov_xy)
    vals = np.maximum(vals, 0.0)
    order = np.argsort(vals)[::-1]
    vals, vecs = vals[order], vecs[:, order]
    angles = np.linspace(0.0, 2.0 * np.pi, 80)
    circle = np.stack([np.cos(angles), np.sin(angles)], axis=0)
    ellipse_xy = (mean_xy[:, None] + vecs @ np.diag(nsigmas * np.sqrt(vals)) @ circle).T
    ellipse_px = project(ellipse_xy, z_body, T_w2c, K)
    cv2.polylines(image, [ellipse_px.reshape(-1, 1, 2)], isClosed=True,
                  color=color, thickness=thickness)


def draw_pose_marker(image, xy, yaw, z_body, T_w2c, K, color,
                     outline_thickness=2, center_radius=6, heading_len=0.08):
    outline_px = draw_projected_outline(image, xy, yaw, z_body, T_w2c, K,
                                        color, outline_thickness)
    center_px  = project_world_xy(xy, z_body, T_w2c, K)[0]
    cv2.circle(image, tuple(center_px), center_radius, color, -1)
    heading_xy = np.array([xy, xy + heading_len * np.array([np.cos(yaw), np.sin(yaw)])])
    heading_px = project_world_xy(heading_xy, z_body, T_w2c, K)
    cv2.arrowedLine(image, tuple(heading_px[0]), tuple(heading_px[1]),
                    color=color, thickness=outline_thickness, tipLength=0.25)
    return outline_px


# ── Composite views ───────────────────────────────────────────────────────────

class TemplateDebug(NamedTuple):
    xy:     np.ndarray
    yaw:    float
    iou:    float
    z_body: float


def make_debug_view(color, mask, wf: WorldFrame, serial_top, K_top,
                    tag_detections, template_debug, ekf: EKF):
    from geom_tee import THICKNESS

    overlay = color.copy()
    overlay[mask] = (overlay[mask] * 0.5 + np.array([0, 255, 0]) * 0.5).astype(np.uint8)
    bw = (mask.astype(np.uint8) * 255)[:, :, None].repeat(3, axis=2)

    T_w2c  = np.linalg.inv(wf.t_cams[serial_top])
    ekf_z  = THICKNESS / 2
    ekf_xy, ekf_yaw = ekf.mean[:2], ekf.mean[2]

    draw_pose_marker(overlay, ekf_xy, ekf_yaw, ekf_z, T_w2c, K_top,
                     CYAN, outline_thickness=3, center_radius=8)
    draw_projected_outline(bw, ekf_xy, ekf_yaw, ekf_z, T_w2c, K_top,
                           CYAN, thickness=3)
    draw_covariance_ellipse(overlay, ekf_xy, ekf.cov[:2, :2], ekf_z,
                            T_w2c, K_top, color=CYAN)
    cv2.putText(overlay,
                f"EKF xy=({ekf_xy[0]:.3f},{ekf_xy[1]:.3f}) yaw={np.degrees(ekf_yaw):.1f} deg",
                (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, CYAN, 2, cv2.LINE_AA)

    for det in tag_detections:
        tag_xy, tag_yaw = det.pose[:2], det.pose[2]
        body_c = project_world_xy(tag_xy, ekf_z, T_w2c, K_top)[0]
        cv2.circle(overlay, tuple(body_c), 8, YELLOW, -1)
        cv2.putText(overlay, det.name, tuple(body_c + np.array([10, -10])),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, GREEN, 2, cv2.LINE_AA)
        draw_projected_outline(overlay, tag_xy, tag_yaw, ekf_z, T_w2c, K_top, GREEN)
        draw_projected_outline(bw,      tag_xy, tag_yaw, ekf_z, T_w2c, K_top, GREEN)

    if template_debug is not None:
        draw_projected_outline(overlay, template_debug.xy, template_debug.yaw,
                               template_debug.z_body, T_w2c, K_top, MAGENTA)
        cv2.putText(overlay, f"template IoU={template_debug.iou:.3f}",
                    (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, MAGENTA, 2, cv2.LINE_AA)

    return np.hstack([overlay, bw])
