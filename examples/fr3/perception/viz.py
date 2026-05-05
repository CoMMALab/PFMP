from typing import Optional

import cv2
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
# import open3d as o3d

from tee import (
    TEE_CENTER_OFFSET_X_M,
    TEE_CENTER_OFFSET_Y_M,
    TEE_CENTER_OFFSET_Z_M,
    TEE_LENGTH_M,
    TEE_WIDTH_M,
    TEE_XY_THICKNESS_M,
    TEE_Z_THICKNESS_M,
)
from camera import WorldFrame, SERIAL_NUMS_INV

def plot_worldframe(wf: WorldFrame, ps: Optional[np.ndarray] = None, viz_tool: str = "mpl"):
    if viz_tool == "o3d":
        return plot_worldframe_o3d(wf, ps)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlim(-0.5, 0.5)
    ax.set_ylim(-0.5, 0.5)
    ax.set_zlim(-0.5, 0.5)

    def draw_vec(T, label, color, axis_scale=0.05):
        pos = T[:3, 3]
        ax.quiver(0, 0, 0, *pos, color=color, normalize=False, linewidth=2)
        ax.text(*pos, label, fontsize=9, fontweight='bold')
        for i, c in enumerate(["r", "g", "b"]):
            ax.quiver(*pos, *T[:3, i], color=c, length=axis_scale, normalize=False, linewidth=2)

    colors = ["r", "g", "b", "c", "m", "y"]
    ax.scatter([0], [0], [0], c="k", s=40)
    draw_vec(wf.t_aruco, "aruco", "k")
    for (serial, T), color in zip(wf.t_cams.items(), colors):
        draw_vec(T, serial[-4:], color)
    if wf.t_pose is not None:
        draw_vec(wf.t_pose, "pose", "m")

    if ps is not None:
        ax.scatter(ps[:, 0], ps[:, 1], ps[:, 2], c="k", s=10)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.tight_layout()
    plt.show()


def _o3d_frame(T: np.ndarray, size: float = 0.05):
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)
    frame.transform(T)
    return frame


def _o3d_points(points: np.ndarray, color=(0.0, 0.0, 0.0)):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    color_arr = np.asarray(color, dtype=np.float64)
    if color_arr.ndim == 2 and color_arr.shape[0] == len(points) and color_arr.shape[1] == 3:
        colors = color_arr
    else:
        colors = np.tile(color_arr.reshape(1, 3), (len(points), 1))
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


def _o3d_line(a: np.ndarray, b: np.ndarray, color=(0.2, 0.2, 0.2)):
    line = o3d.geometry.LineSet()
    pts = np.array([a, b], dtype=np.float64)
    line.points = o3d.utility.Vector3dVector(pts)
    line.lines = o3d.utility.Vector2iVector(np.array([[0, 1]], dtype=np.int32))
    line.colors = o3d.utility.Vector3dVector(np.array([color], dtype=np.float64))
    return line


def plot_worldframe_o3d(wf: WorldFrame, ps: Optional[np.ndarray] = None):
    geoms = []

    origin = np.zeros(3, dtype=np.float64)
    geoms.append(_o3d_frame(np.eye(4), size=0.08))
    for T in [wf.t_aruco] + list(wf.t_cams.values()):
        geoms.append(_o3d_frame(T, size=0.06))
        geoms.append(_o3d_line(origin, T[:3, 3], color=(0.4, 0.4, 0.4)))
    if wf.t_pose is not None:
        geoms.append(_o3d_frame(wf.t_pose, size=0.06))
        geoms.append(_o3d_line(origin, wf.t_pose[:3, 3], color=(0.8, 0.2, 0.8)))

    if ps is not None and len(ps) > 0:
        geoms.append(_o3d_points(ps, color=(0.0, 0.0, 0.0)))

    o3d.visualization.draw_geometries(geoms, window_name="World Frame")


def print_world_coordinates(wf: WorldFrame):
    """Print key object coordinates in world frame (meters)."""
    objects = [
        ("robot/world", np.array([0.0, 0.0, 0.0], dtype=np.float64)),
        ("aruco", wf.t_aruco[:3, 3]),
        *((serial[-4:], T[:3, 3]) for serial, T in wf.t_cams.items()),
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
    return plot_tee_world_with_backend(
        wf,
        raw_tee_world,
        mean_tee_world,
        viz_tool="mpl",
        world_point_clouds=None,
    )


def plot_tee_world_with_backend(
    wf: WorldFrame,
    raw_tee_world,
    mean_tee_world,
    viz_tool: str = "mpl",
    world_point_clouds=None,
):
    if viz_tool == "o3d":
        return plot_tee_world_o3d(
            wf,
            raw_tee_world,
            mean_tee_world,
            world_point_clouds=world_point_clouds,
        )

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
    cam_entries = [(SERIAL_NUMS_INV.get(s, s), T) for s, T in wf.t_cams.items()]

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


def _box_triangle_faces():
    # Two triangles per box face.
    return np.array(
        [
            [0, 1, 2], [0, 2, 3],
            [4, 5, 6], [4, 6, 7],
            [0, 1, 5], [0, 5, 4],
            [1, 2, 6], [1, 6, 5],
            [2, 3, 7], [2, 7, 6],
            [3, 0, 4], [3, 4, 7],
        ],
        dtype=np.int32,
    )


def _o3d_mesh_from_box_vertices(vertices: np.ndarray, color):
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices.astype(np.float64))
    mesh.triangles = o3d.utility.Vector3iVector(_box_triangle_faces())
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color(np.array(color, dtype=np.float64))
    return mesh


def _o3d_tee_meshes(T: np.ndarray, color):
    top_verts_local, stem_verts_local = _tee_box_vertices_local()
    top_verts = _transform_points(T, top_verts_local)
    stem_verts = _transform_points(T, stem_verts_local)
    return [
        _o3d_mesh_from_box_vertices(top_verts, color),
        _o3d_mesh_from_box_vertices(stem_verts, color),
    ]


def plot_tee_world_o3d(
    wf: WorldFrame,
    raw_tee_world,
    mean_tee_world,
    world_point_clouds=None,
):
    geoms = []
    raw_colors = {
        "top": (0.9, 0.2, 0.2),
        "corner": (0.2, 0.8, 0.2),
        "side": (0.2, 0.4, 0.9),
    }
    cam_entries = [(SERIAL_NUMS_INV.get(s, s), T) for s, T in wf.t_cams.items()]

    geoms.append(_o3d_frame(np.eye(4), size=0.08))
    for cam_name, T in cam_entries:
        geoms.append(_o3d_frame(T, size=0.06))
        geoms.append(_o3d_points(np.array([T[:3, 3]]), color=raw_colors[cam_name]))

    for cam_name, T in raw_tee_world:
        geoms.extend(_o3d_tee_meshes(T, raw_colors.get(cam_name, (0.5, 0.5, 0.5))))

    if mean_tee_world is not None:
        geoms.extend(_o3d_tee_meshes(mean_tee_world, (1.0, 0.55, 0.1)))

    if world_point_clouds:
        for entry in world_point_clouds:
            if len(entry) == 3:
                cam_name, points, color = entry
            else:
                cam_name = "pc"
                points, color = entry
            if points is not None and len(points) > 0:
                geoms.append(_o3d_points(points, color=color))
                center = np.mean(points, axis=0)
                label_frame = np.eye(4, dtype=np.float64)
                label_frame[:3, 3] = center
                geoms.append(_o3d_frame(label_frame, size=0.02))

    o3d.visualization.draw_geometries(geoms, window_name="Tee Poses in World Frame")


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