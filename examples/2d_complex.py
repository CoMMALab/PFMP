"""
2d_complex.py — Bicycle-model path planning through random obstacle fields.

Supports animation of a single instance (default) and a full ablation
experiment comparing SMC-MPC against PS/MPPI/CEM baselines.

Usage:
    python 2d_complex.py                        # animate a single instance
    python 2d_complex.py --experiment           # run ablation + plot
    python 2d_complex.py --plot-only            # re-plot from saved data
"""

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.transforms as transforms
from matplotlib.animation import FuncAnimation
import json
import argparse
from pathlib import Path
from functools import partial
from typing import NamedTuple

from pfmp import PFMP
from pfmp.interpolate import hermite_spline
from obstacle_grid import (
    sdf_aabb, hinge_loss, plot_obstacles,
    generate_map_batch,
)


# =============================================================================
# Configuration
# =============================================================================

SEED = 42
NUM_MAPS = 128
PARTICLES = 1024
ESS_THRESH = 0.5
DT = 0.01

ARENA_WIDTH = 6.0
OBSTACLE_DENSITY = 0.3
BOX_SIZE_RANGE = (0.4, 1.2)
MIN_MARGIN = 0.3

KNOT_CONFIGS = [
    {"num_knots": k, "arena_length": round(1.0 * k + 1.0, 1)}
    for k in [5, 8, 10, 14, 18, 22, 28, 34, 40]
]

WHEELBASE = 0.3
SPEED = 2.0
DU_STDEV = 0.1
U_STDEV = 0.2

MPPI_TEMPERATURE = 1.0
CEM_ELITE_FRAC = 0.1

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
    return 10 * jnp.linalg.norm(x[:2] - th.goal) ** 2


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
# Animation
# =============================================================================

def animate(num_knots: int = 20, map_index: int = 0, num_top: int = 20,
            save_gif: bool = False):
    """Animate SIS vs SIS/R side-by-side on a single random problem instance."""
    OUTPUT_DIR.mkdir(exist_ok=True)

    arena_length = round(1.0 * num_knots + 1.0, 1)
    horizon = num_knots * 0.5

    pfmp_sir = PFMP(dynamics, cost_fn, terminal_cost, init, proposal,
                    horizon=horizon, num_knots=num_knots, particles=PARTICLES,
                    ess_resample_thresh=ESS_THRESH, dt=DT)
    pfmp_sis = PFMP(dynamics, cost_fn, terminal_cost, init, proposal,
                    horizon=horizon, num_knots=num_knots, particles=PARTICLES,
                    ess_resample_thresh=0.0, dt=DT)

    obs_batch, x0_batch, goals_batch, _ = generate_map_batch(
        SEED + num_knots * 10000, max(map_index + 1, NUM_MAPS),
        arena_length, ARENA_WIDTH, OBSTACLE_DENSITY, BOX_SIZE_RANGE, MIN_MARGIN,
    )

    obstacles = obs_batch[map_index]
    x0 = x0_batch[map_index]
    goal = goals_batch[map_index]
    theta = Params(goal=goal, obstacles=obstacles,
                   L=WHEELBASE, v=SPEED, du_stdev=DU_STDEV, u_stdev=U_STDEV)

    key = jax.random.key(SEED + map_index)

    print("  running SIS/R ...", end=" ", flush=True)
    sir_states, sir_ancestors, sir_costs, _ = pfmp_sir._sir(key, x0, theta)
    print("done")

    print("  running SIS ...", end=" ", flush=True)
    sis_trajs, sis_costs_raw = pfmp_sis.run_sis(key, x0, theta)
    print("done")

    interval_len = horizon / (num_knots - 1)
    steps_per_knot = round(interval_len / pfmp_sir.dt)

    xs = jnp.linspace(-0.5, arena_length + 0.5, 200)
    ys = jnp.linspace(-0.5, ARENA_WIDTH + 0.5, 150)
    X, Y = jnp.meshgrid(xs, ys)
    points = jnp.stack([X.ravel(), Y.ravel()], axis=1)
    Z = jax.vmap(lambda loc: cost_fn(loc, jnp.zeros(1), theta, 0.0))(points).reshape(X.shape)
    real_obs = obstacles[obstacles[:, 0] < 1e5]

    def render_background(ax):
        ax.set_facecolor("white")
        ax.contourf(X, Y, Z, levels=20, cmap="Greys", alpha=0.2)
        plot_obstacles(ax, real_obs)
        ax.plot(float(x0[0]), float(x0[1]), "o", color="darkgreen", markersize=16,
                markeredgecolor="black", markeredgewidth=1.5, zorder=10)
        ax.plot(float(goal[0]), float(goal[1]), "*", color="gold", markersize=24,
                markeredgecolor="black", markeredgewidth=1.5, zorder=10)
        ax.set_xlim(-0.5, arena_length + 0.5)
        ax.set_ylim(-0.5, ARENA_WIDTH + 0.5)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor("black")
            spine.set_linewidth(1.5)

    def render_sir(ax, current_num_knots):
        render_background(ax)
        current_states = jax.tree.map(lambda x: x[:current_num_knots], sir_states)
        current_ancestors = sir_ancestors[:current_num_knots - 1]
        current_trajs = pfmp_sir.choose_paths(
            current_states, current_ancestors, jnp.arange(pfmp_sir.particles))
        current_costs = sir_costs[:, :current_num_knots - 1]
        total_costs = jnp.sum(current_costs, axis=-1)
        top_indices = jnp.argsort(total_costs)[:num_top]
        best_idx = jnp.argmin(total_costs)

        current_horizon = (current_num_knots - 1) * interval_len
        tq = jnp.linspace(0, current_horizon, int(current_horizon / pfmp_sir.dt))
        tk = jnp.linspace(0, current_horizon, current_num_knots)

        for i in top_indices:
            if i == best_idx:
                continue
            us = hermite_spline(tq, tk, current_trajs[i])
            _, xy = jax.lax.scan(
                lambda x, ut: dynamics(x, ut[0], theta, ut[1]), x0, (us, tq))
            ax.plot(xy[:, 0], xy[:, 1], color="#4A90D9",
                    alpha=0.3, linewidth=1.2, zorder=2)

        us = hermite_spline(tq, tk, current_trajs[best_idx])
        _, best_xy = jax.lax.scan(
            lambda x, ut: dynamics(x, ut[0], theta, ut[1]), x0, (us, tq))
        ax.plot(best_xy[:, 0], best_xy[:, 1], color="white",
                linewidth=5, alpha=0.6, zorder=4, solid_capstyle="round")
        ax.plot(best_xy[:, 0], best_xy[:, 1], color="#1a3a5c",
                linewidth=3, alpha=1.0, zorder=5, solid_capstyle="round")
        for t in range(0, best_xy.shape[0], steps_per_knot):
            ax.plot(best_xy[t, 0], best_xy[t, 1], "o", color="#1a3a5c",
                    markersize=5, markeredgecolor="black", markeredgewidth=0.8, zorder=8)
        ax.set_title("SIS/R (resampling)", fontsize=12)

    def render_sis(ax, current_num_knots):
        render_background(ax)
        current_trajs = sis_trajs[:, :current_num_knots]
        current_costs = sis_costs_raw[:, :current_num_knots - 1]
        total_costs = jnp.sum(current_costs, axis=-1)
        top_indices = jnp.argsort(total_costs)[:num_top]
        best_idx = jnp.argmin(total_costs)

        current_horizon = (current_num_knots - 1) * interval_len
        tq = jnp.linspace(0, current_horizon, int(current_horizon / pfmp_sis.dt))
        tk = jnp.linspace(0, current_horizon, current_num_knots)

        for i in top_indices:
            if i == best_idx:
                continue
            us = hermite_spline(tq, tk, current_trajs[i])
            _, xy = jax.lax.scan(
                lambda x, ut: dynamics(x, ut[0], theta, ut[1]), x0, (us, tq))
            ax.plot(xy[:, 0], xy[:, 1], color="#E74C3C",
                    alpha=0.3, linewidth=1.2, zorder=2)

        us = hermite_spline(tq, tk, current_trajs[best_idx])
        _, best_xy = jax.lax.scan(
            lambda x, ut: dynamics(x, ut[0], theta, ut[1]), x0, (us, tq))
        ax.plot(best_xy[:, 0], best_xy[:, 1], color="white",
                linewidth=5, alpha=0.6, zorder=4, solid_capstyle="round")
        ax.plot(best_xy[:, 0], best_xy[:, 1], color="#8B0000",
                linewidth=3, alpha=1.0, zorder=5, solid_capstyle="round")
        for t in range(0, best_xy.shape[0], steps_per_knot):
            ax.plot(best_xy[t, 0], best_xy[t, 1], "o", color="#8B0000",
                    markersize=5, markeredgecolor="black", markeredgewidth=0.8, zorder=8)
        ax.set_title("SIS (no resampling)", fontsize=12)

    fig_anim, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(16, 4))

    def update(frame):
        ax_left.clear()
        ax_right.clear()
        current_num_knots = min(frame + 2, num_knots)
        render_sis(ax_left, current_num_knots)
        render_sir(ax_right, current_num_knots)

    anim = FuncAnimation(fig_anim, update,
                         frames=num_knots - 1, interval=200,
                         blit=False, repeat=True, repeat_delay=2000)

    if save_gif:
        OUTPUT_DIR.mkdir(exist_ok=True)
        anim.save(OUTPUT_DIR / "fig1_animation.gif", writer="pillow", fps=5, dpi=150)
        print(f"Saved fig1_animation.gif")
    else:
        plt.show()


# =============================================================================
# Experiment
# =============================================================================

def _rollout_cost(pfmp: PFMP, knots, x0, theta):
    tq = jnp.linspace(0, pfmp.horizon, int(pfmp.horizon / pfmp.dt))
    us = hermite_spline(tq, pfmp.tk(), knots)

    def step(x, inp):
        u, t = inp
        x_next, _ = dynamics(x, u, theta, t)
        c = cost_fn(x_next, u, theta, t)
        return x_next, c

    x_final, running_costs = jax.lax.scan(step, x0, (us, tq))
    return jnp.sum(running_costs) + terminal_cost(x_final, theta)


@partial(jax.jit, static_argnames=["pfmp"])
def _rollout_cost_jit(pfmp: PFMP, knots, x0, theta):
    return _rollout_cost(pfmp, knots, x0, theta)


def sample_and_evaluate(key, pfmp, nominal, x0, theta, noise_level):
    K = pfmp.particles
    keys = jax.random.split(key, K)
    noise = jax.vmap(lambda k: jax.random.normal(k, shape=nominal.shape) * noise_level)(keys)
    trajs = jnp.clip(nominal[None, :] + noise, -0.8, 0.8)
    costs = jax.vmap(lambda knots: _rollout_cost_jit(pfmp, knots, x0, theta))(trajs)
    return trajs, costs


def run_experiment():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results = {}

    for cfg in KNOT_CONFIGS:
        num_knots = cfg["num_knots"]
        arena_length = cfg["arena_length"]
        horizon = num_knots * 0.5

        print(f"\n{'='*60}")
        print(f"  num_knots={num_knots}  arena={arena_length:.0f}x{ARENA_WIDTH:.0f}"
              f"  horizon={horizon:.1f}s")
        print(f"{'='*60}")

        pfmp_sis = PFMP(dynamics, cost_fn, terminal_cost, init, proposal,
                        horizon=horizon, num_knots=num_knots, particles=PARTICLES,
                        ess_resample_thresh=0.0, dt=DT)
        pfmp_sir = PFMP(dynamics, cost_fn, terminal_cost, init, proposal,
                        horizon=horizon, num_knots=num_knots, particles=PARTICLES,
                        ess_resample_thresh=ESS_THRESH, dt=DT)

        obs_batch, x0_batch, goals_batch, maps_rejected = generate_map_batch(
            SEED + num_knots * 10000, NUM_MAPS,
            arena_length, ARENA_WIDTH, OBSTACLE_DENSITY, BOX_SIZE_RANGE, MIN_MARGIN,
        )
        print(f"  generated {NUM_MAPS} maps (rejected {maps_rejected})")

        thetas = Params(
            goal=goals_batch,
            obstacles=obs_batch,
            L=jnp.full(NUM_MAPS, WHEELBASE),
            v=jnp.full(NUM_MAPS, SPEED),
            du_stdev=jnp.full(NUM_MAPS, DU_STDEV),
            u_stdev=jnp.full(NUM_MAPS, U_STDEV),
        )
        keys = jax.random.split(jax.random.key(SEED + num_knots * 1000), NUM_MAPS)

        print("  running SMC-MPC ...", end=" ", flush=True)
        sir_trajs, sir_costs = jax.vmap(
            pfmp_sir.run_sir, in_axes=(0, 0, 0))(keys, x0_batch, thetas)
        sir_total = jnp.sum(sir_costs, axis=-1)
        sir_best_idx = jnp.argmin(sir_total, axis=-1)
        sir_nominals = jax.vmap(lambda t, i: t[i])(sir_trajs, sir_best_idx)
        smc_costs_arr = jax.vmap(
            lambda knots, x0, th: _rollout_cost_jit(pfmp_sir, knots, x0, th)
        )(sir_nominals, x0_batch, thetas)
        sir_best_costs = jnp.min(sir_total, axis=-1)
        print("done")

        print("  running SIS ...", end=" ", flush=True)
        keys_sis = jax.random.split(jax.random.fold_in(keys[0], 999), NUM_MAPS)
        sis_trajs, sis_costs_raw = jax.vmap(
            pfmp_sis.run_sis, in_axes=(0, 0, 0))(keys_sis, x0_batch, thetas)
        sis_total = jnp.sum(sis_costs_raw, axis=-1)
        sis_best_costs = jnp.min(sis_total, axis=-1)
        print("done")

        nominal_ps = jnp.zeros((NUM_MAPS, num_knots))
        nominal_mppi = jnp.zeros((NUM_MAPS, num_knots))
        nominal_cem = jnp.zeros((NUM_MAPS, num_knots))
        num_elite = max(1, int(CEM_ELITE_FRAC * PARTICLES))
        noise = U_STDEV

        print(f"  running PS/MPPI/CEM ({num_knots} iters) ...", end=" ", flush=True)
        for i in range(num_knots):
            iter_keys_ps   = jax.vmap(lambda k: jax.random.fold_in(k, i))(keys)
            iter_keys_mppi = jax.vmap(lambda k: jax.random.fold_in(k, num_knots + i))(keys)
            iter_keys_cem  = jax.vmap(lambda k: jax.random.fold_in(k, 2 * num_knots + i))(keys)

            trajs_i, costs_i = jax.vmap(
                sample_and_evaluate, in_axes=(0, None, 0, 0, 0, None)
            )(iter_keys_ps, pfmp_sis, nominal_ps, x0_batch, thetas, noise)
            nominal_ps = jax.vmap(lambda t, idx: t[idx])(trajs_i, jnp.argmin(costs_i, axis=-1))

            trajs_i, costs_i = jax.vmap(
                sample_and_evaluate, in_axes=(0, None, 0, 0, 0, None)
            )(iter_keys_mppi, pfmp_sis, nominal_mppi, x0_batch, thetas, noise)
            weights = jnp.exp(-costs_i / MPPI_TEMPERATURE)
            weights = weights / jnp.sum(weights, axis=-1, keepdims=True)
            if trajs_i.ndim == 4:
                nominal_mppi = jnp.sum(trajs_i * weights[:, :, None, None], axis=1)
            else:
                nominal_mppi = jnp.sum(trajs_i * weights[:, :, None], axis=1)

            trajs_i, costs_i = jax.vmap(
                sample_and_evaluate, in_axes=(0, None, 0, 0, 0, None)
            )(iter_keys_cem, pfmp_sis, nominal_cem, x0_batch, thetas, noise)
            elite_idxs = jnp.argsort(costs_i, axis=-1)[:, :num_elite]
            nominal_cem = jax.vmap(
                lambda t, idx: jnp.mean(t[idx], axis=0))(trajs_i, elite_idxs)

        print("done")

        ps_costs_arr = jax.vmap(
            lambda k, x0, th: _rollout_cost_jit(pfmp_sis, k, x0, th)
        )(nominal_ps, x0_batch, thetas)
        mppi_costs_arr = jax.vmap(
            lambda k, x0, th: _rollout_cost_jit(pfmp_sis, k, x0, th)
        )(nominal_mppi, x0_batch, thetas)
        cem_costs_arr = jax.vmap(
            lambda k, x0, th: _rollout_cost_jit(pfmp_sis, k, x0, th)
        )(nominal_cem, x0_batch, thetas)

        results[str(num_knots)] = {
            "num_knots": num_knots,
            "arena_length": arena_length,
            "horizon": horizon,
            "sis_costs": sis_best_costs.tolist(),
            "sir_costs": sir_best_costs.tolist(),
            "ps_costs": ps_costs_arr.tolist(),
            "mppi_costs": mppi_costs_arr.tolist(),
            "cem_costs": cem_costs_arr.tolist(),
            "smc_costs": smc_costs_arr.tolist(),
            "maps_rejected": maps_rejected,
        }

        print(f"  PS:   med={np.median(ps_costs_arr):.2f}")
        print(f"  MPPI: med={np.median(mppi_costs_arr):.2f}")
        print(f"  CEM:  med={np.median(cem_costs_arr):.2f}")
        print(f"  SMC:  med={np.median(smc_costs_arr):.2f}")
        print(f"  SIS/SIR ratio: "
              f"{np.median(np.array(sis_best_costs)) / np.median(np.array(sir_best_costs)):.3f}x")

    outpath = OUTPUT_DIR / "fig2_data.json"
    with open(outpath, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {outpath}")
    return results


# =============================================================================
# Plotting
# =============================================================================

def make_plots(results: dict):
    knot_counts = sorted([int(k) for k in results.keys()])
    x = np.array(knot_counts)

    methods = {
        "PS":      {"key": "ps_costs",   "color": "#E74C3C", "ls": "--"},
        "MPPI":    {"key": "mppi_costs", "color": "#3498DB", "ls": "--"},
        "CEM":     {"key": "cem_costs",  "color": "#E67E22", "ls": "--"},
        "SMC-MPC": {"key": "smc_costs",  "color": "#27AE60", "ls": "-"},
    }

    stats = {}
    success_rates = {}
    for name, m in methods.items():
        medians, q25, q75, rates = [], [], [], []
        for k in knot_counts:
            data = np.where(np.isfinite(np.array(results[str(k)][m["key"]])),
                            np.array(results[str(k)][m["key"]]), np.nan)
            medians.append(np.nanmedian(data))
            q25.append(np.nanpercentile(data, 25))
            q75.append(np.nanpercentile(data, 75))
            arena_length = results[str(k)]["arena_length"]
            thresh = (arena_length - 1.0) ** 2 * 10 * 3.0
            raw = np.array(results[str(k)][m["key"]])
            rates.append(np.sum(np.isfinite(raw) & (raw < thresh)) / len(raw) * 100)
        stats[name] = {"median": np.array(medians), "q25": np.array(q25), "q75": np.array(q75)}
        success_rates[name] = np.array(rates)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))

    for name, m in methods.items():
        s = stats[name]
        lw = 2.5 if name == "SMC-MPC" else 1.8
        z = 5 if name == "SMC-MPC" else 3
        ax1.plot(x, s["median"], color=m["color"], linestyle=m["ls"],
                 linewidth=lw, marker="o", markersize=5, label=name, zorder=z)
        ax1.fill_between(x, s["q25"], s["q75"], color=m["color"],
                         alpha=0.20 if name == "SMC-MPC" else 0.12, zorder=z - 1)
        ax2.plot(x, success_rates[name], color=m["color"], linestyle=m["ls"],
                 linewidth=lw, marker="o", markersize=5, label=name, zorder=z)

    for ax, title, ylabel in [
        (ax1, "(a) Planning cost", "Executed cost (median)"),
        (ax2, "(b) Success rate", "Success rate (%)"),
    ]:
        ax.set_xlabel("Planning horizon (knots)", fontsize=13)
        ax.set_ylabel(ylabel, fontsize=13)
        ax.tick_params(labelsize=11)
        ax.set_xticks(x)
        ax.legend(fontsize=10, framealpha=0.95, edgecolor="black", fancybox=False)
        for spine in ax.spines.values():
            spine.set_linewidth(1.2)
        ax.set_title(title, fontsize=13, pad=10)

    ax2.set_ylim(-5, 105)
    ax2.axhline(100, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)
    ax2.legend(loc="lower left", fontsize=10, framealpha=0.95, edgecolor="black", fancybox=False)

    fig.suptitle(f"{NUM_MAPS} random maps, {PARTICLES} particles per method",
                 fontsize=12, y=1.02, color="gray")
    fig.tight_layout()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_DIR / "fig2.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / "fig2.png", dpi=300, bbox_inches="tight")
    print("Saved fig2.pdf/png")
    plt.show()


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", action="store_true",
                        help="Run full ablation experiment + plot")
    parser.add_argument("--plot-only", action="store_true",
                        help="Re-plot from saved data")
    parser.add_argument("--save", action="store_true",
                        help="Save animation as GIF instead of displaying")
    parser.add_argument("--animate-knots", type=int, default=20,
                        help="Number of knots for animation (default: 20)")
    parser.add_argument("--animate-map", type=int, default=0,
                        help="Map index for animation (default: 0)")
    args = parser.parse_args()

    if args.plot_only:
        with open(OUTPUT_DIR / "fig2_data.json") as f:
            results = json.load(f)
        make_plots(results)
    elif args.experiment:
        results = run_experiment()
        make_plots(results)
    else:
        animate(num_knots=args.animate_knots, map_index=args.animate_map,
                save_gif=args.save)
