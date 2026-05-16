"""T-block geometry, SDF, vertices, and Panda forward kinematics.

All T-block coordinates are in the local body frame:
  - origin at stem center
  - +y points toward the cap (bar)
  - units: metres

Dimensions are derived from the CAD model at models/T.stl.
"""

import numpy as np
from pathlib import Path

# ── Load mesh and derive dimensions ───────────────────────────────────────────

def _load_dims():
    import trimesh
    stl_path = Path(__file__).resolve().parent.parent.parent / "models" / "T.stl"
    mesh = trimesh.load(str(stl_path))

    # Unique 2D footprint vertices from the mesh (mm)
    xy_mm = np.unique(np.round(mesh.vertices[:, :2], 1), axis=0)

    # THICKNESS from z-extent (mm -> m)
    thickness_m = float(mesh.extents[2]) * 1e-3

    # Unique x and y values define the T outline.
    x_vals = np.sort(np.unique(xy_mm[:, 0]))
    y_vals = np.sort(np.unique(xy_mm[:, 1]))

    # Stem: inner x bounds, lower y bounds. Cap: outer x bounds, upper y bounds.
    stem_x = x_vals[1:3]           # inner two x values
    cap_x  = np.array([x_vals[0], x_vals[-1]])
    stem_y = y_vals[:2]            # lower two y values
    cap_y  = y_vals[1:]            # upper two y values

    # Convert mm -> m and compute body frame quantities
    stem_x_m = stem_x * 1e-3
    stem_y_m = stem_y * 1e-3
    cap_x_m  = cap_x  * 1e-3
    cap_y_m  = cap_y  * 1e-3

    stem_ctr_m = np.array([(stem_x_m[0] + stem_x_m[1]) / 2,
                            (stem_y_m[0] + stem_y_m[1]) / 2])
    cap_ctr_m  = np.array([(cap_x_m[0]  + cap_x_m[1])  / 2,
                            (cap_y_m[0]  + cap_y_m[1])  / 2])

    stem_half_m = np.array([(stem_x_m[1] - stem_x_m[0]) / 2,
                             (stem_y_m[1] - stem_y_m[0]) / 2])
    cap_half_m  = np.array([(cap_x_m[1]  - cap_x_m[0])  / 2,
                             (cap_y_m[1]  - cap_y_m[0])  / 2])

    # Express cap center in body frame (stem center = origin)
    cap_ctr_body = cap_ctr_m - stem_ctr_m

    return stem_half_m, cap_half_m, cap_ctr_body, thickness_m


STEM_HALF, CAP_HALF, CAP_CTR, THICKNESS = _load_dims()
STEM_CTR = np.array([0.0, 0.0])

# Transform from Optitrack rigid body frame (marker centroid) to tee body frame.
# Rotation encodes R90-CCW + x-reflection; translation measured by visual alignment.
OPTI_RB_TO_TEE = np.array([
    [ 0.,  1.,  0., 0.0307 ],
    [ 1.,  0.,  0., 0.0442 ],
    [ 0.,  0.,  1., 0.03125],  # z: markers above tee top surface
    [ 0.,  0.,  0., 1.     ],
])

# ── Polygon outline, CCW ───────────────────────────────────────────────────────

_sc, _sh = STEM_CTR, STEM_HALF
_cc, _ch = CAP_CTR,  CAP_HALF
VERTS_LOCAL = np.array([
    [_sc[0] - _sh[0], _sc[1] - _sh[1]],  # stem bottom-left
    [_sc[0] + _sh[0], _sc[1] - _sh[1]],  # stem bottom-right
    [_sc[0] + _sh[0], _cc[1] - _ch[1]],  # stem/cap junction, right
    [_cc[0] + _ch[0], _cc[1] - _ch[1]],  # cap bottom-right
    [_cc[0] + _ch[0], _cc[1] + _ch[1]],  # cap top-right
    [_cc[0] - _ch[0], _cc[1] + _ch[1]],  # cap top-left
    [_cc[0] - _ch[0], _cc[1] - _ch[1]],  # cap bottom-left
    [_sc[0] - _sh[0], _cc[1] - _ch[1]],  # stem/cap junction, left
], dtype=np.float32)

# AprilTag: outer border top edge is 4cm below the top of the T cap.
TEE_TAG_SIZE_M = 0.0415
_cap_top_y = float(_cc[1] + _ch[1])
TAG_IN_BODY = np.array([
    [1, 0, 0, 0.0                                    ],
    [0, 1, 0, _cap_top_y - 0.04 - TEE_TAG_SIZE_M / 2 ],
    [0, 0, 1, THICKNESS/2                            ],
    [0, 0, 0, 1.0                                    ],
], dtype=np.float64)


# ── SDF ───────────────────────────────────────────────────────────────────────

def _sdf_box(p: np.ndarray, half: np.ndarray) -> float:
    d = np.abs(p) - half
    outside = np.sqrt(np.sum(np.maximum(d, 0.0) ** 2) + 1e-12)
    inside  = np.minimum(np.max(d), 0.0)
    return outside + inside


def sdf_local(p: np.ndarray) -> float:
    """Signed distance from point p (body frame, shape (2,)) to the T."""
    return np.minimum(
        _sdf_box(p - STEM_CTR, STEM_HALF),
        _sdf_box(p - CAP_CTR,  CAP_HALF),
    )


def sdf_world(p_world: np.ndarray, t_xy: np.ndarray, t_angle: float) -> float:
    """Signed distance from world-frame point p to the T at pose (t_xy, t_angle)."""
    c, s = np.cos(t_angle), np.sin(t_angle)
    R_inv = np.array([[c, s], [-s, c]])
    return sdf_local(R_inv @ (p_world - t_xy))


# ── Vertex helpers ────────────────────────────────────────────────────────────

def verts_world(t_xy: np.ndarray, t_angle: float) -> np.ndarray:
    """Return the 8 outline vertices in world frame (shape (8, 2))."""
    c, s = np.cos(t_angle), np.sin(t_angle)
    R = np.array([[c, -s], [s, c]])
    return (VERTS_LOCAL @ R.T) + t_xy


def plot_wireframe(ax, t_xy: np.ndarray, t_angle: float,
                   z: float = 0.0, color: str = "darkorange", alpha: float = 0.7):
    """Draw the T-block as a 3D wireframe on a Axes3D axes."""
    xy_bot = verts_world(t_xy, t_angle)
    z_bot  = np.full(len(xy_bot), z)
    z_top  = np.full(len(xy_bot), z + THICKNESS)

    loop = np.append(np.arange(len(xy_bot)), 0)
    ax.plot(xy_bot[loop, 0], xy_bot[loop, 1], z_bot[loop], color=color, alpha=alpha)
    ax.plot(xy_bot[loop, 0], xy_bot[loop, 1], z_top[loop], color=color, alpha=alpha)

    for x, y in xy_bot:
        ax.plot([x, x], [y, y], [z, z + THICKNESS], color=color, alpha=alpha * 0.6)

    s = TEE_TAG_SIZE_M / 2.0
    tag_y = TAG_IN_BODY[1, 3]
    tag_corners_body = np.array([[-s, tag_y - s], [s, tag_y - s],
                                  [s, tag_y + s], [-s, tag_y + s], [-s, tag_y - s]])
    c, sv = np.cos(t_angle), np.sin(t_angle)
    R = np.array([[c, -sv], [sv, c]])
    tag_xy = (tag_corners_body @ R.T) + t_xy
    z_tag  = np.full(len(tag_xy), z + THICKNESS)
    ax.plot(tag_xy[:, 0], tag_xy[:, 1], z_tag, color="black", alpha=alpha, linewidth=1.5)


if __name__ == "__main__":
    print(f"STEM_HALF={STEM_HALF}, CAP_HALF={CAP_HALF}, CAP_CTR={CAP_CTR}, THICKNESS={THICKNESS:.4f}m")
    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax  = fig.add_subplot(111, projection="3d")
    t_angle = 0.5
    plot_wireframe(ax, np.array([0.0, 0.0]), t_angle, z=-THICKNESS / 2)

    # Draw body frame axes at origin (rotated by yaw)
    c, s = np.cos(t_angle), np.sin(t_angle)
    Ryaw = np.array([[c,-s,0],[s,c,0],[0,0,1]], dtype=float)
    ax.quiver(0,0,0, *(Ryaw[:,0]*0.05), color='r', label='body x')
    ax.quiver(0,0,0, *(Ryaw[:,1]*0.05), color='g', label='body y')
    ax.quiver(0,0,0, *(Ryaw[:,2]*0.05), color='b', label='body z')

    # Draw tag frame axes rotated by yaw
    tag_pos_w = Ryaw @ TAG_IN_BODY[:3, 3]
    tag_R_w   = Ryaw @ TAG_IN_BODY[:3, :3]
    ax.quiver(*tag_pos_w, *tag_R_w[:, 0]*0.03, color='r', linestyle='dashed')
    ax.quiver(*tag_pos_w, *tag_R_w[:, 1]*0.03, color='g', linestyle='dashed')
    ax.quiver(*tag_pos_w, *tag_R_w[:, 2]*0.03, color='b', linestyle='dashed')
    ax.scatter(*tag_pos_w, color='k', s=20, label='tag origin')

    # Marker triangle at calibrated offset
    AB, BC, AC = 0.16706, 0.09224, 0.19128
    A = np.array([0.0, 0.0])
    B = np.array([AB, 0.0])
    Cx = (AC**2 + AB**2 - BC**2) / (2 * AB)
    Cy = np.sqrt(max(AC**2 - Cx**2, 0.0))
    C = np.array([Cx, Cy])
    tri_local = np.array([A, B, C])
    tri_local -= tri_local.mean(axis=0)
    R90 = np.array([[0, -1], [1, 0]])
    tri_local = (R90 @ tri_local.T).T
    tri_local[:, 0] *= -1
    tri_local += OPTI_RB_TO_TEE[:2, 3]

    R2 = np.array([[c, -s], [s, c]])
    tri_w = (R2 @ tri_local.T).T
    loop = [0, 1, 2, 0]
    ax.plot(tri_w[loop, 0], tri_w[loop, 1],
            [OPTI_RB_TO_TEE[2, 3]] * 4,
            'o--', color='purple', linewidth=1.5, markersize=6, label='markers')

    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    ax.legend(fontsize=7)
    lim = 0.15
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(-lim, lim)
    ax.set_box_aspect([1, 1, 1])
    plt.tight_layout()
    plt.show()
