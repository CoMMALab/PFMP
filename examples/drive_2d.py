"""2d_simple.py — Bicycle-model path planning through a regular obstacle grid."""

import argparse
from pathlib import Path
import jax
import jax.numpy as jnp
from typing import NamedTuple
from pfmp import PFMP, make_perturbation
import matplotlib.pyplot as plt

from pfmp.obstacle_grid import (
    generate_obstacle_grid, sdf_aabb, hinge_loss, animate_2d
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
    mean: jax.Array       # (num_knots,) nominal plan the proposal perturbs
    # Sampling noise. Scalar for the fixed-noise methods; CEM refits it per
    # knot, giving shape (num_knots,). Always read through `sigma_at`.
    stdev: jax.Array
    step_len: float       # horizon / (num_knots - 1)


def cost_fn(x: jax.Array, u: jax.Array, th: Params, t: float) -> float:
    loc = x[:2]
    sdf_val = jnp.min(jax.vmap(lambda bounds: sdf_aabb(loc, bounds))(th.obstacles))
    obstacle_loss = jnp.nan_to_num(100. * hinge_loss(sdf_val, 0.25),
                                   nan=0.0, posinf=1e6, neginf=0.0)
    goal_loss = jnp.linalg.norm(loc - th.goal)
    return obstacle_loss + goal_loss


def terminal_cost(x, th) -> float:
    return 10 * jnp.linalg.norm(x[:2] - th.goal)


def dynamics(x: jax.Array, u: jax.Array, theta: Params, t):
    """Bicycle kinematics: x = [px, py, heading], u = steering angle."""
    px, py, th = x
    L, v = theta.L, theta.v
    px_next = px + v * jnp.cos(th) * 0.01
    py_next = py + v * jnp.sin(th) * 0.01
    th_next = jnp.mod(th + (v / L) * jnp.tan(u) * 0.01 + jnp.pi, 2 * jnp.pi) - jnp.pi
    x_next = jnp.array([px_next, py_next, th_next])
    return (x_next, x_next)


STEER_LIMIT = 0.8      # steering angle clip (rad)
U_STDEV     = 0.2      # perturbation width around theta.mean
WHEELBASE   = 0.3
SPEED       = 2.0
DT          = 0.01     # integration step, matching `dynamics`

# Random-map generation, used by the batched runner.
SEED               = 127
PARTICLES          = 1024
ESS_THRESH         = 0.2
ARENA_WIDTH        = 6.0
OBSTACLE_DENSITY   = 0.3
BOX_SIZE_RANGE     = (0.4, 1.2)
MIN_MARGIN         = 0.3

clip_action = lambda u, th: jnp.clip(u, -STEER_LIMIT, STEER_LIMIT)


# =============================================================================
# Environment
# =============================================================================

def make_theta(num_knots: int, horizon: float) -> tuple[jax.Array, Params]:
    x0 = jnp.array([0.0, 0.0, jnp.pi / 4])
    theta = Params(
        goal=jnp.array([12.0, 12.0]),
        obstacles=generate_obstacle_grid(5, 5, 1.0, 1.0) + 1.0,
        L=WHEELBASE, v=SPEED,
        mean=jnp.zeros(num_knots), stdev=U_STDEV,
        step_len=horizon / (num_knots - 1),
    )
    return x0, theta


# =============================================================================
# Animation
# =============================================================================

def animate(num_knots: int = 20, particles: int = 32768, save_gif: bool = False,
            smooth: bool = False, stride: int = 3):
    horizon = num_knots * 0.5
    init, proposal = make_perturbation(clip_action)
    pfmp = PFMP(dynamics, cost_fn, terminal_cost, init, proposal,
                horizon, num_knots, particles, ess_resample_thresh=0.5)
    x0, theta = make_theta(num_knots, horizon)

    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(10, 10))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5, color='gray')

    anim = animate_2d(ax, pfmp, x0, theta, dynamics, cost_fn,
                      xlim=(-0.5, 12.5), ylim=(-0.5, 12.5),
                      smooth=smooth, stride=stride)
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
