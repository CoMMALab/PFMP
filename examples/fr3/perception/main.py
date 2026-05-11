import sys, os
import argparse
import numpy as np
import cv2
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ekf import TeeDetection, EKF
from geom_tee import THICKNESS, TEE_TAG_SIZE_M, TAG_IN_BODY
from template_detector import *
from draw import TemplateDebug, make_debug_view
from camera import (
    AprilDetector,
    TEE_TAG_ID,
    WorldFrame,
    init_cameras,
    get_rgbd_frames,
    get_depth_scale,
    SERIAL_NUMS
)
from perception import (
    detection_from_tee_tag,
    init_belief,
    update_belief
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--filter", choices=["ekf", "pf"], default="ekf")
    parser.add_argument("--wf", default="wf.npz")
    parser.add_argument("--video-out", default=None,
                        help="Optional path to save the displayed debug view as a video.")
    parser.add_argument("--video-fps", type=float, default=15.0)
    args = parser.parse_args()

    wf = WorldFrame.load(args.wf)
    pipelines, intrinsics = init_cameras(["top"])

    pipe = pipelines["top"]
    K_top, dist_top = intrinsics["top"]
    serial_top = SERIAL_NUMS["top"]
    depth_scale = get_depth_scale(pipe)
    tee_detector = AprilDetector(tag_id=TEE_TAG_ID, tag_size_m=TEE_TAG_SIZE_M)
    video_writer = None

    try:
        init_detections, tee_detector = init_belief(
            pipe=pipe,
            detector=tee_detector,
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

            ekf, mask, tee_detector, detections, template_debug = update_belief(
                wf=wf,
                color=color,
                depth=depth,
                K_top=K_top,
                dist_top=dist_top,
                serial_top=serial_top,
                depth_scale=depth_scale,
                ekf=ekf,
                tee_detector=tee_detector
            )

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

            if args.video_out is not None:
                if video_writer is None:
                    h, w = view.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    video_writer = cv2.VideoWriter(
                        args.video_out,
                        fourcc,
                        args.video_fps,
                        (w, h),
                    )
                    if not video_writer.isOpened():
                        raise RuntimeError(f"could not open video writer: {args.video_out}")
                    print(f"[video] recording debug view to {args.video_out}")
                video_writer.write(view)

            cv2.imshow("T segment", view)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        if video_writer is not None:
            video_writer.release()
            print(f"[video] saved {args.video_out}")
        pipe.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
