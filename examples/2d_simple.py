"""2d_simple.py — Bicycle-model path planning through a regular obstacle grid."""

import argparse
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
from typing import NamedTuple
from pfmp import PFMP
from pfmp.interpolate import hermite_spline
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.transforms as transforms
from matplotlib.animation import FuncAnimation

from obstacle_grid import (
    generate_obstacle_grid, sdf_aabb, hinge_loss, plot_obstacles
)

OUTPUT_DIR = Path(__file__).parent.parent / "figures"


# =============================================================================
# Model
# =============================================================================

class Params(NamedTuple):
    goal: jax.Array       # desired position
    obstacles: jax.Array  # (N, 4) AABB array
    L: float              # wheelbase (m)
    v: float              # forward speed (m/s)
    du_stdev: float       # tangent noise
    u_stdev: float        # point noise


def cost_fn(x: jax.Array, u: jax.Array, th: Params, t: float) -> float:
    loc = x[:2]
    sdf_val = jnp.min(jax.vmap(lambda bounds: sdf_aabb(loc, bounds))(th.obstacles))
    obstacle_loss = jnp.nan_to_num(100. * hinge_loss(sdf_val, 0.25),
                                   nan=0.0, posinf=jnp.inf, neginf=0.0)
    goal_loss = jnp.linalg.norm(loc - th.goal) / 10.
    return obstacle_loss**2 + goal_loss**2


def terminal_cost(x, th) -> float:
    return 1000 * jnp.linalg.norm(x[:2] - th.goal) ** 2


def dynamics(x: jax.Array, u: jax.Array, theta: Params, t):
    """Bicycle kinematics: x = [px, py, heading], u = steering angle."""
    px, py, th = x
    L, v = theta.L, theta.v
    px_next = px + v * jnp.cos(th) * 0.01
    py_next = py + v * jnp.sin(th) * 0.01
    th_next = jnp.mod(th + (v / L) * jnp.tan(u) * 0.01 + jnp.pi, 2 * jnp.pi) - jnp.pi
    x_next = jnp.array([px_next, py_next, th_next])
    return (x_next, x_next)


def init(key, x, theta):
    return jnp.clip(jax.random.normal(key) * theta.u_stdev, -0.8, 0.8)


def proposal(key, _x, _u, theta: Params, t):
    return jnp.clip(jax.random.normal(key) * theta.u_stdev, -0.8, 0.8)


# =============================================================================
# Environment
# =============================================================================

def make_theta() -> tuple[jax.Array, Params]:
    x0 = jnp.array([0.0, 0.0, jnp.pi / 4])
    theta = Params(
        goal=jnp.array([12.0, 12.0]),
        obstacles=generate_obstacle_grid(5, 5, 1.0, 1.0) + 1.0,
        L=0.3,
        v=2.0,
        du_stdev=0.1,
        u_stdev=0.2,
    )
    return x0, theta


# =============================================================================
# Animation
# =============================================================================

def animate(num_knots: int = 20, particles: int = 32768, save_gif: bool = True,
            smooth: bool = False, stride: int = 3):
    horizon = num_knots * 0.5
    pfmp = PFMP(dynamics, cost_fn, terminal_cost, init, proposal,
                horizon, num_knots, particles, ess_resample_thresh=0.5)

    x0, theta = make_theta()

    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(10, 10))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    # Cost field
    xs = jnp.linspace(0, 12, 200)
    ys = jnp.linspace(0, 12, 200)
    X, Y = jnp.meshgrid(xs, ys)
    points = jnp.stack([X.ravel(), Y.ravel()], axis=1)
    Z = jax.vmap(lambda loc: cost_fn(loc, jnp.zeros(1), theta, 0.0))(points).reshape(X.shape)
    ax.contourf(X, Y, Z, levels=20, cmap='Greys', alpha=0.3)

    plot_obstacles(ax, theta.obstacles)
    ax.plot(x0[0], x0[1], 'o', color='darkgreen', markersize=30,
            markeredgecolor='black', markeredgewidth=1.5, zorder=10)
    ax.plot(theta.goal[0], theta.goal[1], '*', color='gold', markersize=40,
            markeredgecolor='black', markeredgewidth=1.5, zorder=10)

    ax.set_xlim(-0.5, 12.5)
    ax.set_ylim(-0.5, 12.5)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5, color='gray')
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(1.5)

    key = jax.random.key(1)
    interval_len = horizon / (num_knots - 1)
    steps_per_knot = round(interval_len / pfmp.dt)

    states, ancestors, costs, _ = pfmp._sir(key, x0, theta)

    # Use final costs to rank particles consistently across all frames
    NUM_TOP = 20
    total_costs = jnp.sum(costs, axis=0)
    top_indices = np.array(jnp.argsort(total_costs)[:NUM_TOP])
    best_global = int(jnp.argmin(total_costs))
    best_pi = int(np.where(top_indices == best_global)[0][0])

    # Per-knot precompute: choose_paths with partial ancestors reveals the
    # un-diverged cloud early and the full split late.
    # frame_xys[k] has shape (NUM_TOP, T_k, 3), T_k = (k+1)*steps_per_knot
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

    # Opacity by rank: best fully opaque, others fade
    alphas = np.linspace(0.4, 0.1, NUM_TOP)
    alphas[best_pi] = 1.0

    # Pre-create artists — one trail + knot boxes per particle, plus moving car if smooth
    trail_lines = []
    knot_patches = []
    car_patches = []
    for pi in range(NUM_TOP):
        is_best = pi == best_pi
        color = 'darkblue' if is_best else 'steelblue'

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

    if smooth:
        def update(t):
            k = min(t // steps_per_knot, num_knots - 2)
            xys = frame_xys[k]
            for pi in range(NUM_TOP):
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

        anim = FuncAnimation(fig, update,
                             frames=range(0, T, stride), interval=7,
                             blit=False, repeat=True, repeat_delay=1000)
    else:
        def update(frame):
            k = frame
            xys = frame_xys[k]
            for pi in range(NUM_TOP):
                # Trail shows full trajectory up to current horizon
                trail_lines[pi].set_data(xys[pi, :, 0], xys[pi, :, 1])
                # Car patch is invisible — knot boxes do the work
                car_patches[pi].set_alpha(0.0)
                for ki in range(num_knots):
                    box = knot_patches[pi][ki]
                    if ki <= k + 1:
                        knot_t = min(ki * steps_per_knot, xys.shape[1] - 1)
                        kx, ky, kth = xys[pi, knot_t]
                        box.set_transform(
                            transforms.Affine2D().rotate(kth).translate(kx, ky) + ax.transData)
                        box.set_alpha(alphas[pi] * 0.6)
                    else:
                        box.set_alpha(0.0)

        anim = FuncAnimation(fig, update,
                             frames=num_knots - 1, interval=300,
                             blit=False, repeat=True, repeat_delay=2000)
    plt.tight_layout()

    if save_gif:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        anim.save(OUTPUT_DIR / 'pfmp_planning.gif', writer='pillow', fps=5, dpi=150)
    else:
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", action="store_true",
                        help="Save animation as GIF instead of displaying")
    parser.add_argument("--knots", type=int, default=20,
                        help="Number of knots (default: 20)")
    parser.add_argument("--particles", type=int, default=32768,
                        help="Number of particles (default: 32768)")
    parser.add_argument("--smooth", action="store_true",
                        help="Smooth car interpolation between knots")
    args = parser.parse_args()
    animate(num_knots=args.knots, particles=args.particles,
            save_gif=args.save, smooth=args.smooth)
