"""
push_tee_perception.py — isolated RealSense + T-block perception process.

This process owns the camera and perception stack, then publishes RGBD frames
and EKF beliefs over ZMQ.  Keeping it separate from push_tee_controller.py
prevents RealSense/OpenCV timing work from interfering with Franky motion.

Usage:
    python examples/fr3/push_tee_perception.py
    python examples/fr3/push_tee_perception.py --show
    python examples/fr3/push_tee_perception.py --bind tcp://*:5557
"""

import argparse
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "perception"))

from camera import (
    AprilDetector,
    TEE_TAG_ID,
    WorldFrame,
    init_cameras,
    get_depth_scale,
    get_rgbd_frames,
    SERIAL_NUMS,
)
from draw import make_debug_view
from ekf import EKF
from geom_tee import TEE_TAG_SIZE_M
from main import init_belief, update_belief

sys.path.insert(0, os.path.dirname(__file__))

from pub_sub.perception import PerceptionPublisher
from pub_sub.protocol import TeeBelief


def main():
    parser = argparse.ArgumentParser(description="Push-T perception publisher")
    parser.add_argument("--wf", default="wf.npz")
    parser.add_argument("--bind", default="tcp://*:5557")
    parser.add_argument("--show", action="store_true",
                        help="Show local OpenCV debug visualization")
    parser.add_argument("--max-init-tag-frames", type=int, default=50)
    parser.add_argument("--copy", action="store_true",
                        help="Ask ZMQ to copy RGBD buffers when publishing")
    args = parser.parse_args()

    wf = WorldFrame.load(args.wf)
    pipelines, intrinsics = init_cameras(["top"])
    pipe = pipelines["top"]
    K_top, dist_top = intrinsics["top"]
    serial_top = SERIAL_NUMS["top"]
    depth_scale = get_depth_scale(pipe)

    tee_detector = AprilDetector(tag_id=TEE_TAG_ID, tag_size_m=TEE_TAG_SIZE_M)
    publisher = PerceptionPublisher(bind_addr=args.bind)

    try:
        print("Warming up camera...")
        for _ in range(60):
            get_rgbd_frames(pipe, "top")

        init_detections, tee_detector = init_belief(
            pipe=pipe,
            detector=tee_detector,
            wf=wf,
            serial_top=serial_top,
            K_top=K_top,
            dist_top=dist_top,
            depth_scale=depth_scale,
            max_tag_frames=args.max_init_tag_frames,
        )

        belief_filter = EKF.init(wf, init_detections)

        print(f"[EKF] initialized: mean={belief_filter.mean}")
        print(f"[perception] publishing on {args.bind}")

        while True:
            color, depth = get_rgbd_frames(pipe, "top")
            belief_filter, mask, tee_detector, detections, template_debug = (
                update_belief(
                    wf=wf,
                    color=color,
                    depth=depth,
                    K_top=K_top,
                    dist_top=dist_top,
                    serial_top=serial_top,
                    depth_scale=depth_scale,
                    ekf=belief_filter,
                    tee_detector=tee_detector
                )
            )

            publisher.publish(
                rgb=color,
                depth=depth.astype(np.float32),
                beliefs=[
                    TeeBelief(
                        mean=belief_filter.mean.tolist(),
                        cov=belief_filter.cov.tolist(),
                    )
                ],
                timestamp=time.time(),
                copy=args.copy,
            )

            if args.show:
                view = make_debug_view(
                    color=color,
                    mask=mask,
                    wf=wf,
                    serial_top=serial_top,
                    K_top=K_top,
                    tag_detections=detections,
                    template_debug=template_debug,
                    ekf=belief_filter,
                )
                cv2.imshow("push_tee_perception", view)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    finally:
        publisher.close()
        try:
            pipe.stop()
        except Exception as e:
            print(f"[WARN] could not stop camera pipe: {e}")
        if args.show:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
