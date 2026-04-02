"""Common base for all Push-T task variants (2-D sim, 3-D deployment).

Exports:
  - angle_wrap       — scalar angle normalisation
  - pos_angle_cost   — shared (pos, angle) error cost kernel
  - TERMINAL_SCALE   — multiplier applied to running cost at horizon end
"""

import jax.numpy as jnp
import jax

from geomT import (                        # noqa: F401 — re-export for consumers
    VERTS_LOCAL, STEM_HALF, CAP_HALF,
    STEM_CTR, CAP_CTR, THICKNESS,
    TAG_IN_BODY,
    sdf_local, sdf_world,
    verts_world, plot_wireframe,
)

TERMINAL_SCALE = 10.0


def angle_wrap(a: jax.Array) -> jax.Array:
    return (a + jnp.pi) % (2.0 * jnp.pi) - jnp.pi


def pos_angle_cost(
    t_xy: jax.Array,
    t_angle: jax.Array,
    goal_xy: jax.Array,
    goal_angle: float,
    pos_weight: float,
    angle_weight: float,
) -> jax.Array:
    """Weighted sum of squared position and angle errors."""
    err_pos   = jnp.linalg.norm(t_xy - goal_xy)
    err_angle = jnp.abs(angle_wrap(t_angle - goal_angle))
    # Angle weight ramps up as block nears goal — free to spin when far away
    aw = angle_weight * jnp.clip(1.0 - err_pos / 0.25, 0.0, 1.0)
    return pos_weight * err_pos ** 2 + aw * err_angle ** 2
