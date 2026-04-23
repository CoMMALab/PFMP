"""T-block geometry: dimensions, SDF, and object-centric vertices.

All coordinates are in the T-block's local body frame:
  - origin at stem center
  - +y points toward the cap (bar)
  - units: metres
"""

import numpy as np

# ── Dimensions ────────────────────────────────────────────────────────────────
STEM_HALF = np.array([0.0225, 0.0675])   # (hx, hy)
CAP_HALF  = np.array([0.0675, 0.0225])
STEM_CTR  = np.array([0.0,    0.0   ])
CAP_CTR   = np.array([0.0,    0.0675])
THICKNESS = 0.045   # metres, physical block height

# Polygon outline, CCW, derived from dimension params
_sc, _sh = np.array(STEM_CTR), np.array(STEM_HALF)
_cc, _ch = np.array(CAP_CTR),  np.array(CAP_HALF)
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
# Tag center y = 0.09 - 0.04 - TEE_TAG_SIZE_M/2 = 0.02775m in body frame.
# Tag lies flat on the top face (z = THICKNESS). Rotation assumed aligned with body frame.
TEE_TAG_SIZE_M = 0.0445
TAG_IN_BODY = np.array([
    [1, 0, 0, 0.0              ],
    [0, 1, 0, 0.09 - 0.04 - TEE_TAG_SIZE_M / 2],
    [0, 0, 1, THICKNESS        ],
    [0, 0, 0, 1.0              ],
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
    """Draw the T-block as a 3D wireframe on a Axes3D axes.

    t_xy:    (2,) xy position of body origin in world frame
    t_angle: rotation about z in radians
    z:       height of the block's bottom face
    """
    xy_bot = verts_world(t_xy, t_angle)          # (8, 2)
    z_bot  = np.full(len(xy_bot), z)
    z_top  = np.full(len(xy_bot), z + THICKNESS)

    # Close the loop
    loop = np.append(np.arange(len(xy_bot)), 0)

    # Bottom and top outlines
    ax.plot(xy_bot[loop, 0], xy_bot[loop, 1], z_bot[loop],  color=color, alpha=alpha)
    ax.plot(xy_bot[loop, 0], xy_bot[loop, 1], z_top[loop],  color=color, alpha=alpha)

    # Vertical edges at each vertex
    for x, y in xy_bot:
        ax.plot([x, x], [y, y], [z, z + THICKNESS], color=color, alpha=alpha * 0.6)

    # AprilTag border on top face
    s = TEE_TAG_SIZE_M / 2.0
    tag_corners_body = np.array([[-s, TAG_IN_BODY[1, 3] - s],
                                  [ s, TAG_IN_BODY[1, 3] - s],
                                  [ s, TAG_IN_BODY[1, 3] + s],
                                  [-s, TAG_IN_BODY[1, 3] + s],
                                  [-s, TAG_IN_BODY[1, 3] - s]])  # closed
    c, sv = np.cos(t_angle), np.sin(t_angle)
    R = np.array([[c, -sv], [sv, c]])
    tag_xy = (tag_corners_body @ R.T) + t_xy
    z_tag  = np.full(len(tag_xy), z + THICKNESS)
    ax.plot(tag_xy[:, 0], tag_xy[:, 1], z_tag, color="black", alpha=alpha, linewidth=1.5)


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax  = fig.add_subplot(111, projection="3d")
    plot_wireframe(ax, np.array([0.0, 0.0]), 0.0, z=-THICKNESS / 2)
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    lim = 0.15
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(-lim, lim)
    ax.set_box_aspect([1, 1, 1])
    plt.tight_layout()
    plt.show()
