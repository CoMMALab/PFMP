from typing import Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from tee import (
    TEE_CENTER_OFFSET_X_M,
    TEE_CENTER_OFFSET_Y_M,
    TEE_CENTER_OFFSET_Z_M,
    TEE_LENGTH_M,
    TEE_WIDTH_M,
    TEE_XY_THICKNESS_M,
    TEE_Z_THICKNESS_M,
)
from camera import WorldFrame


def plot_worldframe(wf: WorldFrame, ps: Optional[np.ndarray] = None):
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
    draw_vec(wf.t_aruco, "aruco", "k")
    draw_vec(wf.t_cam1, "cam1", "r")
    draw_vec(wf.t_cam2, "cam2", "g")
    draw_vec(wf.t_cam3, "cam3", "b")
    if wf.t_pose is not None:
        draw_vec(wf.t_pose, "pose", "m")

    if ps is not None:
        ax.scatter(ps[:, 0], ps[:, 1], ps[:, 2], c="k", s=10)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.tight_layout()
    plt.show()


def print_world_coordinates(wf: WorldFrame):
    """Print key object coordinates in world frame (meters)."""
    objects = [
        ("robot/world", np.array([0.0, 0.0, 0.0], dtype=np.float64)),
        ("aruco", wf.t_aruco[:3, 3]),
        ("cam1", wf.t_cam1[:3, 3]),
        ("cam2", wf.t_cam2[:3, 3]),
        ("cam3", wf.t_cam3[:3, 3]),
    ]
    print("\nWorld-frame coordinates (meters):")
    for name, p in objects:
        print(f"  {name:12s} -> x={p[0]: .4f}, y={p[1]: .4f}, z={p[2]: .4f}")

# TEE VIZ

def draw_frame(ax, T, label, axis_scale=0.07):
    pos = T[:3, 3]
    ax.text(*pos, label, fontsize=9)
    for i, c in enumerate(["r", "g", "b"]):
        axis_vec = T[:3, i] * axis_scale
        ax.quiver(pos[0], pos[1], pos[2], axis_vec[0], axis_vec[1], axis_vec[2], color=c, normalize=False)


def _box_faces_local(xmin, xmax, ymin, ymax, zmin, zmax):
    verts = np.array(
        [
            [xmin, ymin, zmin],
            [xmax, ymin, zmin],
            [xmax, ymax, zmin],
            [xmin, ymax, zmin],
            [xmin, ymin, zmax],
            [xmax, ymin, zmax],
            [xmax, ymax, zmax],
            [xmin, ymax, zmax],
        ],
        dtype=np.float64,
    )
    faces = [
        [0, 1, 2, 3],
        [4, 5, 6, 7],
        [0, 1, 5, 4],
        [1, 2, 6, 5],
        [2, 3, 7, 6],
        [3, 0, 4, 7],
    ]
    return verts, faces


def _box_faces():
    return _box_faces_local(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)[1]


def _transform_points(T, points):
    pts_h = np.hstack((points, np.ones((points.shape[0], 1), dtype=np.float64)))
    return (T @ pts_h.T).T[:, :3]


def draw_t_shape(ax, T, color, alpha, label):
    top_verts, stem_verts = _tee_box_vertices_local()
    top_faces = _box_faces()
    stem_faces = _box_faces()

    for verts, faces in [(top_verts, top_faces), (stem_verts, stem_faces)]:
        verts_w = _transform_points(T, verts)
        poly = Poly3DCollection(
            [[verts_w[i] for i in face] for face in faces],
            facecolors=color,
            edgecolors=color,
            linewidths=0.8,
            alpha=alpha,
        )
        ax.add_collection3d(poly)

    draw_frame(ax, T, label, axis_scale=0.05)


def plot_tee_world(wf: WorldFrame, raw_tee_world, mean_tee_world):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    update_tee_plot(ax, wf, raw_tee_world, mean_tee_world, title="Tee Poses in World Frame")
    plt.tight_layout()
    plt.show()


def update_tee_plot(
    ax,
    wf: WorldFrame,
    raw_tee_world,
    mean_tee_world,
    title: str = "Live Tee Poses in World Frame",
):
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
    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")


def _box_vertices_local(xmin, xmax, ymin, ymax, zmin, zmax) -> np.ndarray:
    return np.array(
        [
            [xmin, ymin, zmin],  # 0
            [xmax, ymin, zmin],  # 1
            [xmax, ymax, zmin],  # 2
            [xmin, ymax, zmin],  # 3
            [xmin, ymin, zmax],  # 4
            [xmax, ymin, zmax],  # 5
            [xmax, ymax, zmax],  # 6
            [xmin, ymax, zmax],  # 7
        ],
        dtype=np.float64,
    )


def _box_edges():
    return [
        (0, 1), (1, 2), (2, 3), (3, 0),  # bottom face
        (4, 5), (5, 6), (6, 7), (7, 4),  # top face
        (0, 4), (1, 5), (2, 6), (3, 7),  # vertical edges
    ]


def _tee_wireframe_local():
    top_verts, stem_verts = _tee_box_vertices_local()
    return [(top_verts, _box_edges()), (stem_verts, _box_edges())]


def _tee_box_vertices_local():
    half_w = TEE_WIDTH_M / 2.0
    stem_half_w = TEE_XY_THICKNESS_M / 2.0
    stem_len = TEE_LENGTH_M - TEE_XY_THICKNESS_M
    # Tag frame is on the upward-facing top surface, so tee volume sits below z=0.
    z_top = 0.0
    z_bottom = -TEE_Z_THICKNESS_M

    top_verts = _box_vertices_local(-half_w, half_w, 0.0, TEE_XY_THICKNESS_M, z_bottom, z_top)
    stem_verts = _box_vertices_local(-stem_half_w, stem_half_w, -stem_len, 0.0, z_bottom, z_top)

    offset = np.array(
        [TEE_CENTER_OFFSET_X_M, TEE_CENTER_OFFSET_Y_M, TEE_CENTER_OFFSET_Z_M],
        dtype=np.float64,
    )
    return top_verts + offset, stem_verts + offset


def draw_tee_wireframe_on_image(
    image: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray,
    t_tee_to_cam: np.ndarray,
    color=(0, 255, 255),
    thickness: int = 2,
):
    """Project and draw full 3D tee wireframe (top + stem prisms) on image."""
    R = t_tee_to_cam[:3, :3]
    tvec = t_tee_to_cam[:3, 3].reshape(3, 1)
    rvec, _ = cv2.Rodrigues(R)

    for verts, edges in _tee_wireframe_local():
        pts2d, _ = cv2.projectPoints(verts, rvec, tvec, K, dist)
        uv = np.round(pts2d.reshape(-1, 2)).astype(np.int32)
        for i, j in edges:
            cv2.line(image, tuple(uv[i]), tuple(uv[j]), color, thickness, cv2.LINE_AA)


def draw_tee_outline_on_image(
    image: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray,
    t_tee_to_cam: np.ndarray,
    color=(0, 255, 255),
    thickness: int = 2,
):
    """Backward-compatible alias for tee wireframe overlay."""
    draw_tee_wireframe_on_image(image, K, dist, t_tee_to_cam, color=color, thickness=thickness)