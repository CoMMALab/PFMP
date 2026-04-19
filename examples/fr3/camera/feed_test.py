import argparse
import cv2
import matplotlib.pyplot as plt
import numpy as np

from camera import (
    TEE_TAG_ID,
    TEE_LENGTH_M,
    TEE_THICKNESS_M,
    TEE_WIDTH_M,
    WorldFrame,
    detect_apriltags,
    enforce_world_z_yaw_only,
    fuse_yaw_only_transforms_world,
    get_rgbd_frames,
    init_cameras,
)
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from tee import draw_frame, draw_t_shape, box_faces_local, transform_points 

def update_tee_plot(ax, wf: WorldFrame, raw_tee_world, mean_tee_world):
    ax.cla()
    draw_frame(ax, np.eye(4, dtype=np.float64), "world")

    raw_colors = {"top": "tab:red", "corner": "tab:green", "side": "tab:blue"}
    all_points = [np.zeros(3)]
    cam_entries = [("top", wf.t_cam1), ("corner", wf.t_cam2), ("side", wf.t_cam3)]

    for cam_name, T in cam_entries:
        color = raw_colors[cam_name]
        p = T[:3, 3]
        all_points.append(p)
        ax.scatter([p[0]], [p[1]], [p[2]], c=color, s=30)
        ax.text(p[0], p[1], p[2], f"cam_{cam_name}", fontsize=8)

    for cam_name, T in raw_tee_world:
        draw_t_shape(ax, T, raw_colors.get(cam_name, "gray"), alpha=0.18, label=f"tee_{cam_name}")
        all_points.append(T[:3, 3])

    if mean_tee_world is not None:
        draw_t_shape(ax, mean_tee_world, "orange", alpha=0.35, label="tee_mean")
        all_points.append(mean_tee_world[:3, 3])

    pts = np.array(all_points, dtype=np.float64)
    max_abs = float(np.max(np.abs(pts))) if pts.size else 0.8
    lim = max(0.8, max_abs + 0.2)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)
    ax.set_title("Live Tee Poses in World Frame")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")


def main() -> None:
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
    args = parser.parse_args()

    wf = WorldFrame.load(args.wf)
    if not wf.is_anchored():
        raise RuntimeError(
            f"{args.wf} is unanchored (t_aruco = I). Run anchor.py to produce a "
            "robot/world-anchored wf.npz before running the live feed."
        )
    cam_to_world = {"top": wf.t_cam1, "corner": wf.t_cam2, "side": wf.t_cam3}

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
                mean_T = fuse_yaw_only_transforms_world(tee_transforms_world)
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
    main()
