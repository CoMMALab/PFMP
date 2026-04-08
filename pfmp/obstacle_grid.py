"""obstacle_grid.py — shared geometry utilities for 2D obstacle environments."""

import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.transforms as transforms
from matplotlib.animation import FuncAnimation
from pfmp.interpolate import hermite_spline


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


# =============================================================================
# Shared animation
# =============================================================================

def setup_2d(ax, pfmp, x0, theta, dynamics, cost_fn,
             xlim, ylim, plot_obs=None,
             smooth: bool = False, stride: int = 3, num_top: int = 20,
             color_best: str = 'darkblue', color_rest: str = 'steelblue',
             nominals=None, highlight_best: bool = True):
    """Set up a 2D animation panel on ax.

    Draws the background, runs SIR, precomputes trajectories, and creates
    matplotlib artists. Returns (update_fn, frames, interval) for use with
    FuncAnimation — the caller is responsible for creating the animation.

    Args:
        plot_obs: obstacles to draw (defaults to theta.obstacles; pass filtered
                  array to exclude padding dummies)
    """
    obs_to_plot = plot_obs if plot_obs is not None else theta.obstacles

    # Background
    xs = jnp.linspace(xlim[0], xlim[1], 200)
    ys = jnp.linspace(ylim[0], ylim[1], 150)
    X, Y = jnp.meshgrid(xs, ys)
    points = jnp.stack([X.ravel(), Y.ravel()], axis=1)
    Z = jax.vmap(lambda loc: cost_fn(loc, jnp.zeros(1), theta, 0.0))(points).reshape(X.shape)
    ax.contourf(X, Y, Z, levels=20, cmap='Greys', alpha=0.3)
    plot_obstacles(ax, obs_to_plot)
    ax.plot(float(x0[0]), float(x0[1]), 'o', color='darkgreen', markersize=20,
            markeredgecolor='black', markeredgewidth=1.5, zorder=10)
    ax.plot(float(theta.goal[0]), float(theta.goal[1]), '*', color='gold', markersize=30,
            markeredgecolor='black', markeredgewidth=1.5, zorder=10)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(1.5)

    # Run SIR
    num_knots = pfmp.num_knots
    horizon = pfmp.horizon
    interval_len = horizon / (num_knots - 1)
    steps_per_knot = round(interval_len / pfmp.dt)

    states, ancestors, costs, _ = pfmp._sir(jax.random.key(1), x0, theta)

    total_costs = jnp.sum(costs, axis=-1)
    top_indices = np.array(jnp.argsort(total_costs)[:num_top])
    best_pi = int(np.where(top_indices == int(jnp.argmin(total_costs)))[0][0])

    print("Precomputing trajectories...", end=" ", flush=True)
    frame_xys = []
    for k in range(num_knots - 1):
        current_num_knots = k + 2
        current_states = jax.tree.map(lambda x: x[:current_num_knots], states)
        current_trajs = pfmp.choose_paths(
            current_states, ancestors[:current_num_knots - 1],
            jnp.arange(pfmp.particles))

        current_horizon = (current_num_knots - 1) * interval_len
        tq = jnp.linspace(0, current_horizon, (k + 1) * steps_per_knot)
        tk = jnp.linspace(0, current_horizon, current_num_knots)

        def rollout(knots):
            us = hermite_spline(tq, tk, knots)
            _, xy = jax.lax.scan(lambda x, ut: dynamics(x, ut[0], theta, ut[1]),
                                  x0, (us, tq))
            return xy

        frame_xys.append(np.array(jax.vmap(rollout)(current_trajs[top_indices])))
    print("done")

    T = (num_knots - 1) * steps_per_knot
    alphas = np.linspace(0.4, 0.1, num_top)
    if highlight_best:
        alphas[best_pi] = 1.0

    trail_lines, car_patches, knot_patches = [], [], []
    for pi in range(num_top):
        is_best = pi == best_pi and highlight_best
        color = color_best if is_best else color_rest
        trail, = ax.plot([], [], color=color, alpha=alphas[pi],
                         linewidth=2.0 if is_best else 0.8,
                         zorder=5 if is_best else 2, solid_capstyle='round')
        trail_lines.append(trail)
        car = patches.Rectangle((-theta.L / 2, -theta.L / 4), theta.L, theta.L / 2,
                                 facecolor=color, edgecolor='black',
                                 linewidth=1.5 if is_best else 0.5,
                                 alpha=alphas[pi], zorder=6 if is_best else 3)
        ax.add_patch(car)
        car_patches.append(car)
        kp = []
        for _ in range(num_knots):
            box = patches.Rectangle((-theta.L / 2, -theta.L / 4), theta.L, theta.L / 2,
                                    facecolor=color, edgecolor='black',
                                    linewidth=0.5, alpha=0.0,
                                    zorder=4 if is_best else 1)
            ax.add_patch(box)
            kp.append(box)
        knot_patches.append(kp)

    nominal_artists = []
    if nominals:
        for xy, label, color in nominals:
            line, = ax.plot([], [], color=color, linewidth=2.5, alpha=0.0,
                            zorder=9, solid_capstyle='round', label=label)
            nominal_artists.append((line, np.array(xy)))

    def _reveal_nominals():
        for line, xy in nominal_artists:
            line.set_data(xy[:, 0], xy[:, 1])
            line.set_alpha(1.0)
        if nominal_artists:
            ax.legend(loc='upper left', fontsize=9, framealpha=0.85,
                      edgecolor='black', fancybox=False)

    if smooth:
        def update(t):
            k = min(t // steps_per_knot, num_knots - 2)
            xys = frame_xys[k]
            for pi in range(num_top):
                px, py, th = xys[pi, t]
                car_patches[pi].set_transform(
                    transforms.Affine2D().rotate(th).translate(px, py) + ax.transData)
                trail_lines[pi].set_data(xys[pi, :t + 1, 0], xys[pi, :t + 1, 1])
                for ki in range(num_knots):
                    box = knot_patches[pi][ki]
                    if ki <= k:
                        kx, ky, kth = xys[pi, ki * steps_per_knot]
                        box.set_transform(
                            transforms.Affine2D().rotate(kth).translate(kx, ky) + ax.transData)
                        box.set_alpha(alphas[pi] * 0.5)
                    else:
                        box.set_alpha(0.0)
            if nominal_artists and k == num_knots - 2:
                _reveal_nominals()
        return update, range(0, T, stride), 7
    else:
        def update(frame):
            xys = frame_xys[frame]
            for pi in range(num_top):
                trail_lines[pi].set_data(xys[pi, :, 0], xys[pi, :, 1])
                car_patches[pi].set_alpha(0.0)
                for ki in range(num_knots):
                    box = knot_patches[pi][ki]
                    if ki <= frame + 1:
                        knot_t = min(ki * steps_per_knot, xys.shape[1] - 1)
                        kx, ky, kth = xys[pi, knot_t]
                        box.set_transform(
                            transforms.Affine2D().rotate(kth).translate(kx, ky) + ax.transData)
                        box.set_alpha(alphas[pi] * 0.6)
                    else:
                        box.set_alpha(0.0)
            if nominal_artists and frame == num_knots - 2:
                _reveal_nominals()
        return update, num_knots - 1, 300


def animate_2d(ax, pfmp, x0, theta, dynamics, cost_fn,
               xlim, ylim, plot_obs=None,
               smooth: bool = False, stride: int = 3, num_top: int = 20,
               color_best: str = 'darkblue', color_rest: str = 'steelblue'):
    """Animate PFMP planning on a 2D obstacle environment.

    Sets up the background on `ax`, precomputes trajectories, and returns a
    FuncAnimation. The caller handles figure creation, tight_layout, and
    save/show. For multi-panel use, call setup_2d directly.
    """
    update, frames, interval = setup_2d(
        ax, pfmp, x0, theta, dynamics, cost_fn,
        xlim, ylim, plot_obs, smooth, stride, num_top, color_best, color_rest)
    repeat_delay = 1000 if smooth else 2000
    return FuncAnimation(ax.figure, update, frames=frames,
                         interval=interval, blit=False, repeat=True,
                         repeat_delay=repeat_delay)
