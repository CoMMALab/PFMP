import time
import numpy as np
import cv2
from camera import *
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def draw_frame(ax, T, label, axis_scale=0.07):
    pos = T[:3, 3]
    ax.text(*pos, label, fontsize=9)
    for i, c in enumerate(["r", "g", "b"]):
        axis_vec = T[:3, i] * axis_scale
        ax.quiver(pos[0], pos[1], pos[2], axis_vec[0], axis_vec[1], axis_vec[2], color=c, normalize=False)

def box_faces_local(xmin, xmax, ymin, ymax, zmin, zmax):
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
        [0, 1, 2, 3],  # bottom
        [4, 5, 6, 7],  # top
        [0, 1, 5, 4],
        [1, 2, 6, 5],
        [2, 3, 7, 6],
        [3, 0, 4, 7],
    ]
    return verts, faces


def transform_points(T, points):
    pts_h = np.hstack((points, np.ones((points.shape[0], 1), dtype=np.float64)))
    return (T @ pts_h.T).T[:, :3]

def draw_t_shape(ax, T, color, alpha, label):
    half_w = TEE_WIDTH_M / 2.0
    half_t = TEE_THICKNESS_M / 2.0
    stem_half_w = TEE_THICKNESS_M / 2.0
    stem_len = TEE_LENGTH_M - TEE_THICKNESS_M  # total length stays 8in with 2in top thickness

    # Tag origin is at the junction where top and stem meet.
    # Top bar extends in +Y; stem extends in -Y.
    top_verts, top_faces = box_faces_local(
        -half_w, half_w, 0.0, TEE_THICKNESS_M, -half_t, half_t
    )
    stem_verts, stem_faces = box_faces_local(
        -stem_half_w, stem_half_w, -stem_len, 0.0, -half_t, half_t
    )

    for verts, faces in [(top_verts, top_faces), (stem_verts, stem_faces)]:
        verts_w = transform_points(T, verts)
        poly = Poly3DCollection(
            [[verts_w[i] for i in face] for face in faces],
            facecolors=color,
            edgecolors=color,
            linewidths=0.8,
            alpha=alpha,
        )
        ax.add_collection3d(poly)

    draw_frame(ax, T, label, axis_scale=0.05)

def plot_tee_world(wf: WorldFrame, raw_tee_world, mean_tee_world: np.array):
    """Plot raw and average Tee poses in world frame with robot/world origin."""
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # World/robot origin: only draw a single 3-axis frame (no point marker).
    draw_frame(ax, np.eye(4, dtype=np.float64), "world")

    # Use consistent colors for each camera and its Tee estimate.
    raw_colors = {"top": "tab:red", "corner": "tab:green", "side": "tab:blue"}
    for cam_name, T in raw_tee_world:
        c = raw_colors.get(cam_name, "gray")
        draw_t_shape(ax, T, c, alpha=0.18, label=f"tee_{cam_name}")

    # Average/fused Tee pose in world framaxe.
    draw_t_shape(ax, mean_tee_world, "orange", alpha=0.35, label="tee_mean")

    # Optional context: camera centers (same color as corresponding Tee estimate).
    all_points = [np.zeros(3)]
    cam_entries = [("top", wf.t_cam1), ("corner", wf.t_cam2), ("side", wf.t_cam3)]
    for cam_name, T in cam_entries:
        color = raw_colors[cam_name]
        label = f"cam_{cam_name}"
        p = T[:3, 3]
        all_points.append(p)
        ax.scatter([p[0]], [p[1]], [p[2]], c=color, s=30)
        ax.text(p[0], p[1], p[2], label, fontsize=8)
    for _, T in raw_tee_world:
        all_points.append(T[:3, 3])
    all_points.append(mean_tee_world[:3, 3])

    pts = np.array(all_points, dtype=np.float64)
    max_abs = float(np.max(np.abs(pts))) if pts.size else 0.8
    lim = max(0.8, max_abs + 0.2)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)

    ax.set_title("Tee Poses in World Frame (Raw + Mean)")
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    plt.tight_layout()
    plt.show()


"""
Localize the tee in the world frame.
"""
def localize_tee(args):
    wf = WorldFrame.load("wf.npz")
    if not wf.is_anchored():
        raise RuntimeError(
            "wf.npz is unanchored (t_aruco = I). Run anchor.py to produce a "
            "robot/world-anchored wf.npz before localizing the tee."
        )
    # Saved world frame stores camera-to-world transforms.
    cam_to_world = {
        "top": wf.t_cam1,
        "corner": wf.t_cam2,
        "side": wf.t_cam3,
    }

    pipelines, intrinsics = init_cameras(args.cams)
    try:
        tee_tag_size_m = args.tee_tag_size_m
        print(f"[TEE] using tag id={TEE_TAG_ID} size={tee_tag_size_m:.4f} m")
        time.sleep(1.0)  # warm up sensors before one-shot capture
        tee_positions_world = []
        tee_transforms_world = []
        raw_tee_world = []

        for cam in args.cams:
            frames = get_rgbd_frames(pipelines[cam], cam, retries=4, timeout_ms=10000)
            if frames is None:
                print(f"[WARN] {cam}: no RGBD frame, skipping Tee localization for this camera")
                continue

            color_image, _ = frames
            K, dist = intrinsics[cam]
            detections, _, debug = detect_apriltags(color_image, K, dist, tee_tag_size_m, staged=True, return_debug=True)
            t_tee_to_cam = get_tag_transform_by_id(detections, TEE_TAG_ID)
            if t_tee_to_cam is None:
                print(f"[WARN] {cam}: Tee tag id={TEE_TAG_ID} not detected")
                continue

            # Tee origin in world frame: p_world = T_cam_to_world @ T_tee_to_cam @ [0,0,0,1].
            t_tee_to_world = cam_to_world[cam] @ t_tee_to_cam
            # Enforce physical constraint: Tee rotates only about +world-Z.
            t_tee_to_world = enforce_world_z_yaw_only(t_tee_to_world)
            p_tee_world_h = t_tee_to_world @ np.array([0.0, 0.0, 0.0, 1.0])
            p_tee_world = p_tee_world_h[:3]
            tee_positions_world.append(p_tee_world)
            tee_transforms_world.append(t_tee_to_world)
            raw_tee_world.append((cam, t_tee_to_world))
            roll_deg, pitch_deg, yaw_deg = rotation_matrix_to_rpy_zyx_deg(t_tee_to_world[:3, :3])
            print(
                f"[TEE][world] {cam:6s} -> x={p_tee_world[0]: .4f}, "
                f"y={p_tee_world[1]: .4f}, z={p_tee_world[2]: .4f}"
            )
            print(
                f"[TEE][world] {cam:6s} -> roll={roll_deg: .1f} deg, "
                f"pitch={pitch_deg: .1f} deg, yaw={yaw_deg: .1f} deg (ZYX)"
            )

        if tee_positions_world:
            mean_pos = np.mean(np.stack(tee_positions_world), axis=0)
            mean_T = fuse_yaw_only_transforms_world(tee_transforms_world)
            print(
                f"[TEE][world] mean  -> x={mean_pos[0]: .4f}, "
                f"y={mean_pos[1]: .4f}, z={mean_pos[2]: .4f}"
            )
            if mean_T is not None:
                m_roll, m_pitch, m_yaw = rotation_matrix_to_rpy_zyx_deg(mean_T[:3, :3])
                print(
                    f"[TEE][world] mean  -> roll={m_roll: .1f} deg, "
                    f"pitch={m_pitch: .1f} deg, yaw={m_yaw: .1f} deg (ZYX, Z-only constrained)"
                )
                print("[TEE][world] plotting raw Tee poses and fused mean pose")
                plot_tee_world(wf, raw_tee_world, mean_T)
        else:
            print("[TEE] No valid Tee detections from selected cameras.")
    finally:
        for pipe in pipelines.values():
            pipe.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cams", nargs="+", default=["top", "corner", "side"])
    parser.add_argument("--tee_tag_size_m", type=float, default=0.0445)
    args = parser.parse_args()
    localize_tee(args)