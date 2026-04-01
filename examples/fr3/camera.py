from typing import NamedTuple, Optional
import pyrealsense2 as rs
import numpy as np
import cv2
import argparse
import time
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("TkAgg")
from mpl_toolkits.mplot3d import Axes3D

SERIAL_NUMS = {
                "top": "234322070408",
                "corner": "234322070144",
                "side": "234322070341"
            }

class AprilDetections(NamedTuple):
    cam1: np.array
    cam2: np.array
    cam3: np.array

T_ARUCO = np.array([
    [1.0,  0.0,  0.0,  0.379],
    [0.0, -1.0,  0.0, -0.064],
    [0.0,  0.0, -1.0,  0.012],
    [0.0,  0.0,  0.0,  1.0],
])

class WorldFrame(NamedTuple):
    t_aruco: np.array
    t_cam1: np.array
    t_cam2: np.array
    t_cam3: np.array
    t_pose: Optional[np.array] = None

    @staticmethod
    def build_world_frame(dets: AprilDetections, t_aruco: np.array) -> "Self":
        # T_world_cam = T_world_aruco @ inv(T_cam_aruco)
        t_cam1 = t_aruco @ np.linalg.inv(dets.cam1)
        t_cam2 = t_aruco @ np.linalg.inv(dets.cam2)
        t_cam3 = t_aruco @ np.linalg.inv(dets.cam3)
        return WorldFrame(t_aruco=t_aruco, t_cam1=t_cam1, t_cam2=t_cam2, t_cam3=t_cam3)

    def save(self, path: str):
        arrays = dict(t_aruco=self.t_aruco, t_cam1=self.t_cam1, t_cam2=self.t_cam2, t_cam3=self.t_cam3)
        if self.t_pose is not None:
            arrays["t_pose"] = self.t_pose
        np.savez(path, **arrays)

    @staticmethod
    def load(path: str) -> "WorldFrame":
        data = np.load(path)
        return WorldFrame(
            t_aruco=data["t_aruco"],
            t_cam1=data["t_cam1"],
            t_cam2=data["t_cam2"],
            t_cam3=data["t_cam3"],
            t_pose=data["t_pose"] if "t_pose" in data else None,
        )

    def into(self, p: np.array) -> np.array:
        """Transform a homogeneous (4,) point from each camera frame into world frame.
        Returns (3, 3) world-space xyz for cam1, cam2, cam3.
        """
        return np.stack([
            (self.t_cam1 @ p)[:3],
            (self.t_cam2 @ p)[:3],
            (self.t_cam3 @ p)[:3],
        ])

    def plot(self, ps: Optional[np.array] = None):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        ax.set_xlim(-0.5, 0.5)
        ax.set_ylim(-0.5, 0.5)
        ax.set_zlim(-0.5, 0.5)

        def draw_vec(T, label, color, axis_scale=0.05):
            pos = T[:3, 3]
            ax.quiver(0, 0, 0, *pos, color=color, normalize=False)
            ax.text(*pos, label, fontsize=8)
            for i, c in enumerate(["r", "g", "b"]):
                ax.quiver(*pos, *T[:3, i], color=c, length=axis_scale, normalize=False)

        ax.scatter([0], [0], [0], c="k", s=40)
        draw_vec(self.t_aruco, "aruco", "k")
        draw_vec(self.t_cam1,  "cam1",  "r")
        draw_vec(self.t_cam2,  "cam2",  "g")
        draw_vec(self.t_cam3,  "cam3",  "b")
        if self.t_pose is not None:
            draw_vec(self.t_pose, "pose", "m")

        if ps is not None:
            ax.scatter(ps[:, 0], ps[:, 1], ps[:, 2], c="k", s=10)

        ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
        plt.tight_layout()
        plt.show()

def test_world():
    def make_cam(tx, ty, tz, angle_z):
        c, s = np.cos(angle_z), np.sin(angle_z)
        return np.array([
            [ c, -s, 0, tx],
            [ s,  c, 0, ty],
            [ 0,  0, 1, tz],
            [ 0,  0, 0,  1],
        ], dtype=float)

    fake_cam1 = make_cam(  0.0, -0.25, 0.25, 0.0)
    fake_cam2 = make_cam(-0.15,  0.0 ,  0.5, 0.0)
    fake_cam3 = make_cam(  0.0,  0.25, 0.25, 0.0)

    dets = AprilDetections(cam1=fake_cam1, cam2=fake_cam2, cam3=fake_cam3)
    wf = WorldFrame.build_world_frame(dets, T_ARUCO)
    print("cam1 world:\n", wf.t_cam1)
    print("cam2 world:\n", wf.t_cam2)
    print("cam3 world:\n", wf.t_cam3)
    wf.plot()

def start_pipeline(serial):
    # Start pipeline for one camera.
    print(f"Starting camera: {serial}")
    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_device(serial)

    # Safe baseline for D4xx / many RealSense setups
    cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    pipe.start(cfg)
    print(f"Camera started: {serial}")
    return pipe

def get_camera_panel(pipe, cam_name, retries=3, timeout_ms=5000):
    frames = None
    for attempt in range(1, retries + 1):
        try:
            frames = pipe.wait_for_frames(timeout_ms=timeout_ms)
            break
        except RuntimeError:
            print(f"[WARN] {cam_name}: frame timeout (attempt {attempt}/{retries})")

    if frames is None:
        print(f"[WARN] {cam_name}: failed to capture frame after {retries} attempts")
        return None

    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()

    if not depth_frame or not color_frame:
        return None

    color_image = np.asanyarray(color_frame.get_data())
    depth_image = np.asanyarray(rs.colorizer().colorize(depth_frame).get_data())

    cv2.putText(
        color_image,
        f"{cam_name} RGB",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        depth_image,
        f"{cam_name} Depth",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )

    return np.hstack((color_image, depth_image))

def main(args):
    pipelines = {}
    try:
        for cam in args.cams:
            pipelines[cam] = start_pipeline(SERIAL_NUMS[cam])

        print("Capturing one frame per selected camera...")
        # Give sensors a brief moment to start producing stable frames.
        time.sleep(0.5)
        rows = []
        for cam in args.cams:
            panel = get_camera_panel(pipelines[cam], cam, retries=4, timeout_ms=4000)
            if panel is not None:
                rows.append(panel)

        if rows:
            canvas = np.vstack(rows)
            cv2.imshow("RealSense Multi-Camera Snapshot (RGB + Depth)", canvas)
            print("Snapshot shown. Press any key to close.")
            cv2.waitKey(0)
        else:
            print("No frames captured from selected cameras.")
    finally:
        for pipe in pipelines.values():
            pipe.stop()
        cv2.destroyAllWindows()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibrate", action="store_true")
    parser.add_argument("--cams", nargs="+", default=["top", "corner", "side"])
    args = parser.parse_args()
    if args.calibrate:
        test_world()
        # TODO
        # 1. build WorldFrame with april detections (wf = build_world_frame(april_dets, T_ARUCO))
        # 2. wf.plot() and save, update with visualization
        # 3. serialize calibrated WorldFrame

    main(args)
