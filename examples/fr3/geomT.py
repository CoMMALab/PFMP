"""T-block geometry: dimensions, SDF, and object-centric vertices.

All coordinates are in the T-block's local body frame:
  - origin at stem center
  - +y points toward the cap (bar)
  - units: metres
"""

import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# ── Dimensions ────────────────────────────────────────────────────────────────
STEM_HALF = jnp.array([0.0225, 0.0675])   # (hx, hy)
CAP_HALF  = jnp.array([0.0675, 0.0225])
STEM_CTR  = jnp.array([0.0,    0.0   ])
CAP_CTR   = jnp.array([0.0,    0.0675])

# Polygon outline, CCW, in body frame
VERTS_LOCAL = np.array([
    [-0.0225, -0.0675],
    [ 0.0225, -0.0675],
    [ 0.0225,  0.0450],
    [ 0.0675,  0.0450],
    [ 0.0675,  0.0900],
    [-0.0675,  0.0900],
    [-0.0675,  0.0450],
    [-0.0225,  0.0450],
], dtype=np.float32)

# AprilTag pose in body frame (4x4).  Set once the physical tag offset is measured.
TAG_IN_BODY = np.eye(4, dtype=np.float64)


# ── SDF (JAX) ─────────────────────────────────────────────────────────────────

def _sdf_box(p: jax.Array, half: jax.Array) -> jax.Array:
    d = jnp.abs(p) - half
    outside = jnp.sqrt(jnp.sum(jnp.maximum(d, 0.0) ** 2) + 1e-12)
    inside  = jnp.minimum(jnp.max(d), 0.0)
    return outside + inside


def sdf_local(p: jax.Array) -> jax.Array:
    """Signed distance from point p (body frame, shape (2,)) to the T."""
    return jnp.minimum(
        _sdf_box(p - STEM_CTR, STEM_HALF),
        _sdf_box(p - CAP_CTR,  CAP_HALF),
    )


def sdf_world(p_world: jax.Array, t_xy: jax.Array, t_angle: jax.Array) -> jax.Array:
    """Signed distance from world-frame point p to the T at pose (t_xy, t_angle)."""
    c, s = jnp.cos(t_angle), jnp.sin(t_angle)
    R_inv = jnp.array([[c, s], [-s, c]])
    return sdf_local(R_inv @ (p_world - t_xy))


# ── Vertex helpers ────────────────────────────────────────────────────────────

THICKNESS = 0.045   # metres, physical block height

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


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("TkAgg")
    fig = plt.figure()
    ax  = fig.add_subplot(111, projection="3d")
    plot_wireframe(ax, np.array([0.0, 0.0]), 0.0, z=-THICKNESS / 2)
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    lim = 0.15
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(-lim, lim)
    ax.set_box_aspect([1, 1, 1])
    plt.tight_layout()
    plt.show()
