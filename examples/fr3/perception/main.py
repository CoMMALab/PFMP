import sys, os
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from typing import Optional, NamedTuple
from camera import get_rgbd_frames, WorldFrame, SERIAL_NUMS
from ekf import PoseDetection, EKF
from geom_tee import THICKNESS
from april_detector import *
from template_detector import *


def init_acquisition(
    pipe,
    wf: WorldFrame,
    serial_top: str,
    K_top: np.ndarray,
    dist_top: np.ndarray,
    depth_scale: float,
    max_tag_frames: int = 50,
) -> list[PoseDetection]:
    # 1. Try AprilTag for a few frames.
    for i in range(max_tag_frames):
        color, depth = get_rgbd_frames(pipe, "top")

        top_detection = detect_tee_tag(
            color=color,
            depth=depth,
            K=K_top,
            dist=dist_top,
            wf=wf,
            serial=serial_top,
            depth_scale=depth_scale,
            name="tag_init",
            R=np.diag([1e-4, 1e-4, 1e-3]),
            gated=False,
        )

        if top_detection:
            print(f"[init] tag frame {i + 1}: {top_detection.pose}")
            return [top_detection]

    # 2. Fallback: one segmentation-seeded template search.
    color, depth = get_rgbd_frames(pipe, "top")

    mask = segment_T(
        frame=wf,
        color=color,
        depth=depth,
        depth_scale=depth_scale,
        cam="top",
    )

    seed_xy = segmentation_median_world_xy(
        mask=mask,
        depth=depth,
        K=K_top,
        T_c2w=wf.t_cams[serial_top],
        depth_scale=depth_scale,
    )

    if seed_xy is None:
        print("[init] no tag, no segmentation seed")
        return []

    best_xy, best_yaw, best_iou, _ = template_match_T(
        mask=mask,
        frame=wf,
        serial=serial_top,
        K=K_top,
        body_xy=seed_xy,
        body_yaw=0.0,
        body_z=THICKNESS / 2,
        dx=0.08,
        dy=0.08,
        dyaw=np.pi,
        nyaw=100,
    )

    if best_iou < 0.20:
        print(f"[init] template failed: IoU={best_iou:.3f}")
        return []

    pose = np.array([
        best_xy[0],
        best_xy[1],
        best_yaw,
    ], dtype=np.float64)

    q = max(best_iou, 0.05)

    print(f"[init] template: {pose}, IoU={best_iou:.3f}")

    return [PoseDetection(
        name="template_init",
        pose=pose,
        R=np.diag([
            2e-3 / q,
            2e-3 / q,
            3e-2 / q,
        ]),
        gated=False,
    )]

def draw_covariance_ellipse(
    image: np.ndarray,
    mean_xy: np.ndarray,
    cov_xy: np.ndarray,
    z_body: float,
    T_w2c: np.ndarray,
    K: np.ndarray,
    nsigmas: float = 2.0,
    color=(255, 255, 0),
    thickness: int = 2,
):
    """Draw a 2D covariance ellipse projected from world xy into image pixels."""
    import cv2

    vals, vecs = np.linalg.eigh(cov_xy)
    vals = np.maximum(vals, 0.0)

    # Largest eigenvalue first.
    order = np.argsort(vals)[::-1]
    vals = vals[order]
    vecs = vecs[:, order]

    angles = np.linspace(0.0, 2.0 * np.pi, 80)
    circle = np.stack([np.cos(angles), np.sin(angles)], axis=0)

    ellipse_xy = (
        mean_xy[:, None]
        + vecs @ np.diag(nsigmas * np.sqrt(vals)) @ circle
    ).T

    ellipse_px = project(ellipse_xy, z_body, T_w2c, K)

    cv2.polylines(
        image,
        [ellipse_px.reshape(-1, 1, 2)],
        isClosed=True,
        color=color,
        thickness=thickness,
    )

CYAN = (255, 255, 0)
RED = (0, 0, 255)
BLUE = (255, 0, 0)
GREEN = (0, 255, 0)
YELLOW = (0, 255, 255)
MAGENTA = (255, 0, 255)


def draw_polyline(
    image: np.ndarray,
    pts_px: np.ndarray,
    color: tuple[int, int, int],
    thickness: int = 2,
):
    import cv2

    cv2.polylines(
        image,
        [pts_px.reshape(-1, 1, 2)],
        isClosed=True,
        color=color,
        thickness=thickness,
    )


def draw_projected_outline(
    image: np.ndarray,
    xy: np.ndarray,
    yaw: float,
    z_body: float,
    T_w2c: np.ndarray,
    K: np.ndarray,
    color: tuple[int, int, int],
    thickness: int = 2,
):
    from geom_tee import verts_world

    pts_px = project(
        verts_world(xy, yaw),
        z_body,
        T_w2c,
        K,
    )

    draw_polyline(image, pts_px, color, thickness)
    return pts_px


def project_world_xy(
    xy: np.ndarray,
    z_body: float,
    T_w2c: np.ndarray,
    K: np.ndarray,
) -> np.ndarray:
    return project(
        np.asarray(xy).reshape(-1, 2),
        z_body,
        T_w2c,
        K,
    )


def project_tag_corners(
    corners_cam: np.ndarray,
    T_c2w: np.ndarray,
    T_w2c: np.ndarray,
    K: np.ndarray,
) -> np.ndarray:
    corners_cam_h = np.column_stack([
        corners_cam,
        np.ones(len(corners_cam)),
    ])

    corners_world = (T_c2w @ corners_cam_h.T).T[:, :3]

    return project(
        verts_w=corners_world[:, :2],
        body_z=float(corners_world[:, 2].mean()),
        T_w2c=T_w2c,
        K=K,
    )


def draw_pose_marker(
    image: np.ndarray,
    xy: np.ndarray,
    yaw: float,
    z_body: float,
    T_w2c: np.ndarray,
    K: np.ndarray,
    color: tuple[int, int, int],
    outline_thickness: int = 2,
    center_radius: int = 6,
    heading_len: float = 0.08,
):
    import cv2

    outline_px = draw_projected_outline(
        image=image,
        xy=xy,
        yaw=yaw,
        z_body=z_body,
        T_w2c=T_w2c,
        K=K,
        color=color,
        thickness=outline_thickness,
    )

    center_px = project_world_xy(xy, z_body, T_w2c, K)[0]

    cv2.circle(
        image,
        tuple(center_px),
        center_radius,
        color,
        -1,
    )

    heading_xy = np.array([
        xy,
        xy + heading_len * np.array([
            np.cos(yaw),
            np.sin(yaw),
        ]),
    ])

    heading_px = project_world_xy(heading_xy, z_body, T_w2c, K)

    cv2.arrowedLine(
        image,
        tuple(heading_px[0]),
        tuple(heading_px[1]),
        color=color,
        thickness=outline_thickness,
        tipLength=0.25,
    )

    return outline_px

class TemplateDebug(NamedTuple):
    xy: np.ndarray
    yaw: float
    iou: float
    z_body: float

def make_debug_view(
    color: np.ndarray,
    mask: np.ndarray,
    wf: WorldFrame,
    serial_top: str,
    K_top: np.ndarray,
    tag_detections: Optional[dict],
    template_debug: np.array,
    ekf: EKF,
):
    import cv2
    from geom_tee import THICKNESS

    overlay = color.copy()
    overlay[mask] = (
        overlay[mask] * 0.5 + np.array([0, 255, 0]) * 0.5
    ).astype(np.uint8)

    bw = (mask.astype(np.uint8) * 255)[:, :, None].repeat(3, axis=2)

    T_c2w = wf.t_cams[serial_top]
    T_w2c = np.linalg.inv(T_c2w)

    ekf_z = THICKNESS / 2

    # -------------------------------------------------------------------------
    # EKF render: fused estimate
    # -------------------------------------------------------------------------
    ekf_xy = ekf.mean[:2]
    ekf_yaw = ekf.mean[2]

    draw_pose_marker(
        image=overlay,
        xy=ekf_xy,
        yaw=ekf_yaw,
        z_body=ekf_z,
        T_w2c=T_w2c,
        K=K_top,
        color=CYAN,
        outline_thickness=3,
        center_radius=8,
    )

    draw_projected_outline(
        image=bw,
        xy=ekf_xy,
        yaw=ekf_yaw,
        z_body=ekf_z,
        T_w2c=T_w2c,
        K=K_top,
        color=CYAN,
        thickness=3,
    )

    draw_covariance_ellipse(
        image=overlay,
        mean_xy=ekf_xy,
        cov_xy=ekf.cov[:2, :2],
        z_body=ekf_z,
        T_w2c=T_w2c,
        K=K_top,
        nsigmas=2.0,
        color=CYAN,
        thickness=2,
    )

    cv2.putText(
        overlay,
        f"EKF xy=({ekf_xy[0]:.3f},{ekf_xy[1]:.3f}) yaw={np.degrees(ekf_yaw):.1f} deg",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        CYAN,
        2,
        cv2.LINE_AA,
    )

    # -------------------------------------------------------------------------
    # AprilTag debug render
    # -------------------------------------------------------------------------
    for det in tag_detections:
        tag_xy = det.pose[:2]
        tag_yaw = det.pose[2]
        tag_z = THICKNESS / 2

        # Tag-derived body origin.
        body_c = project_world_xy(tag_xy, tag_z, T_w2c, K_top)[0]
        cv2.circle(overlay, tuple(body_c), 8, YELLOW, -1)

        # Tag-derived T outline.
        draw_projected_outline(
            image=overlay,
            xy=tag_xy,
            yaw=tag_yaw,
            z_body=tag_z,
            T_w2c=T_w2c,
            K=K_top,
            color=GREEN,
            thickness=2,
        )

        draw_projected_outline(
            image=bw,
            xy=tag_xy,
            yaw=tag_yaw,
            z_body=tag_z,
            T_w2c=T_w2c,
            K=K_top,
            color=GREEN,
            thickness=2,
        )

    # -------------------------------------------------------------------------
    # Template debug render: draw actual fused template candidate
    # -------------------------------------------------------------------------
    if template_debug is not None:
        draw_projected_outline(
            image=overlay,
            xy=template_debug.xy,
            yaw=template_debug.yaw,
            z_body=template_debug.z_body,
            T_w2c=T_w2c,
            K=K_top,
            color=MAGENTA,
            thickness=2,
        )

        cv2.putText(
            overlay,
            f"template IoU={template_debug.iou:.3f}",
            (20, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            MAGENTA,
            2,
            cv2.LINE_AA,
        )

    return np.hstack([overlay, bw])



def main():
    import cv2
    from camera import (
        WorldFrame,
        init_cameras,
        get_rgbd_frames,
        get_depth_scale,
        SERIAL_NUMS, SERIAL_NUMS_INV
    )
    from geom_tee import TEE_TAG_SIZE_M

    wf = WorldFrame.load("wf.npz")
    pipelines, intrinsics = init_cameras(["top"])

    pipe = pipelines["top"]
    K_top, dist_top = intrinsics["top"]
    serial_top = SERIAL_NUMS["top"]
    depth_scale = get_depth_scale(pipe)

    try:
        init_detections = init_acquisition(
            pipe=pipe,
            wf=wf,
            serial_top=serial_top,
            K_top=K_top,
            dist_top=dist_top,
            depth_scale=depth_scale,
            max_tag_frames=50,
        )

        ekf = EKF.init(wf, init_detections)

        print(f"[EKF] initialized: mean={ekf.mean}")

        while True:
            color, depth = get_rgbd_frames(pipe, "top")

            mask = segment_T(
                frame=wf,
                color=color,
                depth=depth,
                depth_scale=depth_scale,
                cam="top",
            )

            detections = []

            for name, serial in SERIAL_NUMS.items():
                det = detect_tee_tag(
                    color=color,
                    depth=depth,
                    K=K_top,
                    dist=dist_top,
                    wf=wf,
                    serial=serial,
                    depth_scale=depth_scale,
                    name=name,
                    R=np.diag([1e-4, 1e-4, 1e-3]),
                    gated=False,
                )
                if det is not None:
                    detections.append(det)
                    print(f"appended {name}")
                    print(detections)

            best_xy, best_yaw, best_iou, _ = template_match_T(
                mask=mask,
                frame=wf,
                serial=serial_top,
                K=K_top,
                body_xy=ekf.mean[:2],
                body_yaw=ekf.mean[2],
                body_z=THICKNESS/2,
            )

            template_debug = None
            if best_iou >= 0.25:
                pose_template = np.array([
                    best_xy[0],
                    best_xy[1],
                    best_yaw,
                ], dtype=np.float64)

                q = max(best_iou, 0.05)
                detections.append(PoseDetection(
                    name="template",
                    pose=pose_template,
                    R=np.diag([1e-4, 1e-4, 1e-3]),
                    gated=False,
                ))
                print("appended template")
                print(detections)

                template_debug = TemplateDebug(
                    xy=np.array(best_xy, dtype=np.float64),
                    yaw=float(best_yaw),
                    iou=float(best_iou),
                    z_body=THICKNESS / 2,
                )

            print(detections)
            ekf = ekf.update(detections)

            view = make_debug_view(
                color=color,
                mask=mask,
                wf=wf,
                serial_top=serial_top,
                K_top=K_top,
                tag_detections=detections,
                template_debug=template_debug,
                ekf=ekf,
            )
            
            print("mean", ekf.mean)
            print("cov diag:", np.diag(ekf.cov))

            cv2.imshow("T segment", view)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        pipe.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()