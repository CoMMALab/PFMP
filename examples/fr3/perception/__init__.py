import numpy as np
from template_detector import segment_T, template_match_T
from camera import AprilDetector, WorldFrame, get_rgbd_frames
from ekf import EKF, TeeDetection
from draw import TemplateDebug
from geom_tee import TAG_IN_BODY, THICKNESS


def detection_from_tee_tag(
    detector: AprilDetector,
    wf: WorldFrame,
    serial: str,
    name: str = "tag",
    R: np.ndarray | None = None,
    gated: bool = True,
) -> TeeDetection:
    T_tag_to_cam = detector.pose(serial)
    T_body = wf.t_cams[serial] @ T_tag_to_cam @ np.linalg.inv(TAG_IN_BODY)
    T_pose = np.array([
        T_body[0, 3],
        T_body[1, 3],
        np.arctan2(T_body[1, 0], T_body[0, 0]),
    ], dtype=np.float64)
    return TeeDetection(
        name=name,
        pose=T_pose,
        R=np.diag([1e-4, 1e-4, 1e-3]) if R is None else R,
        gated=gated,
    )

def init_belief(
    pipe,
    detector: AprilDetector,
    wf: WorldFrame,
    serial_top: str,
    K_top: np.ndarray,
    dist_top: np.ndarray,
    depth_scale: float,
    max_tag_frames: int = 50,
) -> tuple[list[TeeDetection], AprilDetector]:
    # 1. Try AprilTag for a few frames.
    for i in range(max_tag_frames):
        color, depth = get_rgbd_frames(pipe, "top")

        ok, detector = detector.detect(
            serial=serial_top,
            rgbd=(color, depth),
            K=K_top,
            dist=dist_top,
            depth_scale=depth_scale,
        )

        if ok:
            top_detection = detection_from_tee_tag(
                detector=detector,
                wf=wf,
                serial=serial_top,
                name="tag_init",
                R=np.diag([1e-4, 1e-4, 1e-3]),
                gated=False,
            )
            print(f"[init] tag frame {i + 1}: {top_detection.pose}")
            return [top_detection], detector

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
        return [], detector

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
    )

    if best_iou < 0.20:
        print(f"[init] template failed: IoU={best_iou:.3f}")
        return [], detector

    pose = np.array([
        best_xy[0],
        best_xy[1],
        best_yaw,
    ], dtype=np.float64)

    q = max(best_iou, 0.05)

    print(f"[init] template: {pose}, IoU={best_iou:.3f}")

    return [TeeDetection(
        name="template_init",
        pose=pose,
        R=np.diag([
            2e-3 / q,
            2e-3 / q,
            3e-2 / q,
        ]),
        gated=False,
    )], detector


def update_belief(
    wf: WorldFrame,
    color: np.ndarray,
    depth: np.ndarray,
    K_top: np.ndarray,
    dist_top: np.ndarray,
    serial_top: str,
    depth_scale: float,
    ekf: EKF,
    tee_detector: AprilDetector,
) -> tuple[object, np.ndarray, AprilDetector, list[TeeDetection], TemplateDebug | None]:
    mask = segment_T(
        frame=wf,
        color=color,
        depth=depth,
        depth_scale=depth_scale,
        cam="top",
    )

    detections = []

    ok, tee_detector = tee_detector.detect(
        serial=serial_top,
        rgbd=(color, depth),
        K=K_top,
        dist=dist_top,
        depth_scale=depth_scale,
    )

    if ok:
        det = detection_from_tee_tag(
            detector=tee_detector,
            wf=wf,
            serial=serial_top,
            name="top",
            R=np.diag([1e-4, 1e-4, 1e-7]),
            gated=False,
        )
        detections.append(det)

    if ekf.is_lost:
        seed_xy = segmentation_median_world_xy(
            mask=mask,
            depth=depth,
            K=K_top,
            T_c2w=wf.t_cams[serial_top],
            depth_scale=depth_scale,
        )
        if seed_xy is not None:
            ekf = ekf.reset(seed_xy)
            print(f"[EKF] lost — reset to segmentation seed xy={seed_xy}")

    best_xy, best_yaw, best_iou, _ = template_match_T(
        mask=mask,
        frame=wf,
        serial=serial_top,
        K=K_top,
        body_xy=ekf.mean[:2],
        body_yaw=ekf.mean[2],
        body_z=THICKNESS / 2,
    )

    template_debug = None
    if best_iou >= 0.25:
        pose_template = np.array([
            best_xy[0],
            best_xy[1],
            best_yaw,
        ], dtype=np.float64)

        detections.append(TeeDetection(
            name="template",
            pose=pose_template,
            R=np.diag([1e-4, 1e-4, 1e-2]),
            gated=False,
        ))

        template_debug = TemplateDebug(
            xy=np.array(best_xy, dtype=np.float64),
            yaw=float(best_yaw),
            iou=float(best_iou),
            z_body=THICKNESS / 2,
        )

    ekf = ekf.update(detections)
    return ekf, mask, tee_detector, detections, template_debug