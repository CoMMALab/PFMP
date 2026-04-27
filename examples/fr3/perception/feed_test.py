import argparse
import cv2
import matplotlib.pyplot as plt
import numpy as np

from camera import (
    TEE_TAG_ID,
    WorldFrame,
    SERIAL_NUMS_INV,
    detect_apriltags,
    get_rgbd_frames,
    init_cameras,
)
from utils import (
    average_transforms,
    enforce_world_z_yaw_only,
    fuse_yaw_only_transforms_world,
)
from viz import update_tee_plot

def main(args) -> None:
    wf = WorldFrame.load(args.wf)
    if not wf.is_anchored():
        raise RuntimeError(
            f"{args.wf} is unanchored (t_aruco = I). Run anchor.py to produce a "
            "robot/world-anchored wf.npz before running the live feed."
        )
    cam_to_world = {SERIAL_NUMS_INV[s]: T for s, T in wf.t_cams.items()}

    cams = ["top", "corner", "side"]
    pipelines, intrinsics = init_cameras(cams)
    window_name = "Live Feed - top | corner | side"
    frame_h, frame_w = 480, 640
    plot_enabled = not args.no_plot

    fig = None
    ax = None
    if plot_enabled:
        plt.ion()
        fig = plt.figure("Live Tee 3D Pose")
        ax = fig.add_subplot(111, projection="3d")

    try:
        while True:
            panels = []
            raw_tee_world = []
            tee_transforms_world = []
            for cam in cams:
                frames = get_rgbd_frames(
                    pipelines[cam],
                    cam_name=cam,
                    retries=1,
                    timeout_ms=1000,
                    colorize_depth=False,
                )
                if frames is None:
                    color_image = np.zeros((frame_h, frame_w, 3), dtype=np.uint8)
                    cv2.putText(
                        color_image,
                        f"{cam} (no frame)",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        (0, 0, 255),
                        2,
                        cv2.LINE_AA,
                    )
                else:
                    color_image, _ = frames
                    K, dist = intrinsics[cam]
                    detections, _ = detect_apriltags(
                        color_image,
                        K,
                        dist,
                        args.tee_tag_size_m,
                        staged=True,
                    )
                    tee_detections = [det for det in detections if det[0] == TEE_TAG_ID]
                    if tee_detections:
                        cv2.aruco.drawDetectedMarkers(
                            color_image,
                            [det[2] for det in tee_detections],
                            np.array([TEE_TAG_ID] * len(tee_detections)),
                        )
                        for _, _, _, rvec, tvec in tee_detections:
                            cv2.drawFrameAxes(color_image, K, dist, rvec, tvec, args.tee_tag_size_m * 0.6)
                        t_tee_to_cam = tee_detections[0][1]
                        t_tee_to_world = cam_to_world[cam] @ t_tee_to_cam
                        if args.yaw_only:
                            t_tee_to_world = enforce_world_z_yaw_only(t_tee_to_world)
                        raw_tee_world.append((cam, t_tee_to_world))
                        tee_transforms_world.append(t_tee_to_world)

                    cv2.putText(
                        color_image,
                        cam,
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        (0, 255, 0),
                        2,
                        cv2.LINE_AA,
                    )
                    status_text = (
                        f"Tee tag {TEE_TAG_ID}: detected ({len(tee_detections)})"
                        if tee_detections
                        else f"Tee tag {TEE_TAG_ID}: not detected"
                    )
                    status_color = (0, 255, 0) if tee_detections else (0, 0, 255)
                    cv2.putText(
                        color_image,
                        status_text,
                        (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.75,
                        status_color,
                        2,
                        cv2.LINE_AA,
                    )
                    if tee_detections:
                        p_tee_world = raw_tee_world[-1][1][:3, 3]
                        cv2.putText(
                            color_image,
                            f"world xyz: {p_tee_world[0]: .3f}, {p_tee_world[1]: .3f}, {p_tee_world[2]: .3f}",
                            (20, 115),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.58,
                            (255, 255, 0),
                            2,
                            cv2.LINE_AA,
                        )
                panels.append(color_image)

            cv2.imshow(window_name, np.hstack(panels))
            if plot_enabled and ax is not None and fig is not None:
                mean_T = (
                    fuse_yaw_only_transforms_world(tee_transforms_world)
                    if args.yaw_only
                    else average_transforms(tee_transforms_world)
                )
                update_tee_plot(ax, wf, raw_tee_world, mean_T)
                fig.canvas.draw_idle()
                plt.pause(0.001)

            # Press q to quit.
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    except KeyboardInterrupt:
        pass
    finally:
        for pipe in pipelines.values():
            pipe.stop()
        cv2.destroyAllWindows()
        if fig is not None:
            plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Show live feeds and detect Tee AprilTag.")
    parser.add_argument(
        "--tee_tag_size_m",
        type=float,
        default=0.0445,
        help="Physical size of Tee AprilTag in meters.",
    )
    parser.add_argument(
        "--wf",
        default="wf.npz",
        help="Path to saved world frame npz (default: wf.npz).",
    )
    parser.add_argument(
        "--no_plot",
        action="store_true",
        help="Disable live 3D world plot.",
    )
    parser.add_argument(
        "--yaw-only",
        action="store_true",
        help="Constrain tee orientation to world Z yaw only (default: off).",
    )
    args = parser.parse_args()


    main(args)
