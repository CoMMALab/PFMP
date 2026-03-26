"""obstacle_grid.py — shared geometry utilities for 2D obstacle environments."""

import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt


# =============================================================================
# Geometry primitives
# =============================================================================

@jax.jit
def sdf_aabb(pos: jax.Array, bounds: jax.Array) -> float:
    xmin, xmax, ymin, ymax = bounds
    center = jnp.array([(xmin + xmax) / 2, (ymin + ymax) / 2])
    half_extents = jnp.array([(xmax - xmin) / 2, (ymax - ymin) / 2])
    d = jnp.abs(pos - center) - half_extents
    outside_dist = jnp.linalg.norm(jnp.maximum(d, 0.0))
    inside_dist = jnp.minimum(jnp.maximum(d[0], d[1]), 0.0)
    return outside_dist + inside_dist


@jax.jit
def hinge_loss(sdf: float, epsilon: float) -> float:
    return jnp.where(sdf < epsilon, 0.5 * (epsilon - sdf) ** 2, 0.0)


def plot_obstacles(ax, obstacles, color='black'):
    """Plot a list of AABB obstacles as filled rectangles."""
    for obs in obstacles:
        xmin, xmax, ymin, ymax = obs
        rect = plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                              edgecolor=color, facecolor='black', linewidth=1)
        ax.add_patch(rect)


# =============================================================================
# Grid obstacle generation (2d_simple)
# =============================================================================

def generate_obstacle_grid(num_x: int, num_y: int, box_size: float, gap: float):
    """Generate a regular grid of AABB obstacles.

    Args:
        num_x: number of boxes along the x-axis
        num_y: number of boxes along the y-axis
        box_size: side length of each box (square)
        gap: gap between boxes (same in x and y)

    Returns:
        A (num_x * num_y, 4) array of AABB boxes as (xmin, xmax, ymin, ymax)
    """
    obstacles = []
    for i in range(num_x):
        for j in range(num_y):
            xmin = i * (box_size + gap)
            ymin = j * (box_size + gap)
            xmax = xmin + box_size
            ymax = ymin + box_size
            obstacles.append(jnp.array([xmin, xmax, ymin, ymax]))
    return jnp.stack(obstacles)


# =============================================================================
# Random obstacle generation (2d_complex)
# =============================================================================

def generate_random_obstacles(
    rng: np.random.Generator,
    arena_length: float,
    arena_width: float,
    density: float,
    box_size_range: tuple[float, float],
) -> np.ndarray:
    """Scatter random AABB boxes at a given density.

    Returns:
        (N, 4) array of [xmin, xmax, ymin, ymax]
    """
    area = arena_length * arena_width
    num_obstacles = max(2, int(density * area))

    obstacles = []
    for _ in range(num_obstacles):
        w = rng.uniform(*box_size_range)
        h = rng.uniform(*box_size_range)
        cx = rng.uniform(w / 2, arena_length - w / 2)
        cy = rng.uniform(h / 2, arena_width - h / 2)
        obstacles.append([cx - w / 2, cx + w / 2, cy - h / 2, cy + h / 2])

    return np.array(obstacles)


def is_point_clear(point: np.ndarray, obstacles: np.ndarray, margin: float) -> bool:
    """Check that a point is at least `margin` away from all obstacles."""
    for obs in obstacles:
        xmin, xmax, ymin, ymax = obs
        cx = np.clip(point[0], xmin, xmax)
        cy = np.clip(point[1], ymin, ymax)
        dist = np.linalg.norm(point[:2] - np.array([cx, cy]))
        if dist < margin:
            return False
    return True


def generate_map(
    rng: np.random.Generator,
    arena_length: float,
    arena_width: float,
    density: float,
    box_size_range: tuple[float, float],
    margin: float,
    max_attempts: int = 100,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Generate a random map with start/goal guaranteed clear.

    Returns:
        (obstacles, start, goal) or None if infeasible after max_attempts.
    """
    start = np.array([0.5, arena_width / 2, 0.0])   # face right
    goal = np.array([arena_length - 0.5, arena_width / 2])

    for _ in range(max_attempts):
        obstacles = generate_random_obstacles(
            rng, arena_length, arena_width, density, box_size_range
        )
        if is_point_clear(start, obstacles, margin) and \
           is_point_clear(goal, obstacles, margin):
            return obstacles, start, goal

    return None


def generate_map_batch(
    seed: int,
    num_maps: int,
    arena_length: float,
    arena_width: float,
    density: float,
    box_size_range: tuple[float, float],
    margin: float,
) -> tuple[jax.Array, jax.Array, jax.Array, int]:
    """Generate num_maps random maps, padded to uniform obstacle count for batching.

    Returns:
        obstacles_batch: (num_maps, max_obs, 4)
        x0_batch:        (num_maps, 3)
        goals_batch:     (num_maps, 2)
        maps_rejected:   int
    """
    all_obstacles = []
    all_starts = []
    all_goals = []
    maps_rejected = 0
    map_seed = 0

    while len(all_obstacles) < num_maps:
        map_rng = np.random.default_rng(seed + map_seed)
        map_seed += 1

        result = generate_map(
            map_rng, arena_length, arena_width,
            density, box_size_range, margin,
        )
        if result is None:
            maps_rejected += 1
            continue

        obstacles, start, goal = result
        all_obstacles.append(obstacles)
        all_starts.append(start)
        all_goals.append(goal)

    # Pad obstacles to max count with far-away dummy boxes
    max_obs = max(len(o) for o in all_obstacles)
    dummy = np.array([1e6, 1e6 + 0.01, 1e6, 1e6 + 0.01])
    padded = []
    for o in all_obstacles:
        pad_count = max_obs - len(o)
        if pad_count > 0:
            padding = np.tile(dummy, (pad_count, 1))
            o = np.concatenate([o, padding], axis=0)
        padded.append(o)

    return (
        jnp.array(np.stack(padded)),
        jnp.array(np.stack(all_starts)),
        jnp.array(np.stack(all_goals)),
        maps_rejected,
    )
