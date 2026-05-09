import sys, os
import numpy as np
import cv2
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from camera import get_rgbd_frames, WorldFrame, SERIAL_NUMS
from ekf import PoseDetection, EKF
from geom_tee import THICKNESS
from april_detector import *
from template_detector import *
from draw import TemplateDebug, make_debug_view
from camera import (
    WorldFrame,
    init_cameras,
    get_rgbd_frames,
    get_depth_scale,
    SERIAL_NUMS
)


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


def main():
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