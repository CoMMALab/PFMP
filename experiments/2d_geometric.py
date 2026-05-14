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
from matplotlib.animation import FuncAnimation
import json
import argparse
from pathlib import Path
from functools import partial
from typing import NamedTuple

from pfmp import PFMP
from pfmp.interpolate import hermite_spline
from pfmp.obstacle_grid import (
    sdf_aabb, hinge_loss, plot_obstacles,
    generate_map_batch,
)


# =============================================================================
# Configuration
# =============================================================================

SEED = 123
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
    mean: jax.Array       # (num_knots,) proposal mean
    step_len: float       # horizon / (num_knots - 1)


def cost_fn(x: jax.Array, u: jax.Array, th: Params, t: float) -> float:
    loc = x[:2]
    sdf_val = jnp.min(jax.vmap(lambda bounds: sdf_aabb(loc, bounds))(th.obstacles))
    obstacle_loss = jnp.nan_to_num(100. * hinge_loss(sdf_val, 0.25),
                                   nan=0.0, posinf=jnp.inf, neginf=0.0)
    goal_loss = jnp.linalg.norm(loc - th.goal) / 10.
    return obstacle_loss**2 + goal_loss**2


def terminal_cost(x, th) -> float:
    return 100 * jnp.linalg.norm(x[:2] - th.goal) ** 2


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
    return jnp.clip(jax.random.normal(key) * theta.u_stdev + theta.mean[0], -0.8, 0.8)


def proposal(key, _x, _u, theta: Params, t):
    idx = jnp.clip(jnp.round(t / theta.step_len).astype(int) + 1, 0, theta.mean.shape[0] - 1)
    return jnp.clip(jax.random.normal(key) * theta.u_stdev + theta.mean[idx], -0.8, 0.8)


# =============================================================================
# Animation — shared helpers
# =============================================================================

BASELINE_COLORS = {
    'ps':   ('crimson',    'indianred',      r"PS"),
    'mppi': ('royalblue',  'cornflowerblue', r"MPPI"),
    'cem':  ('darkorange', 'sandybrown',     r"CEM"),
}
NUM_CLOUD = 30   # particle paths shown per baseline per frame
NUM_OUTER = 15   # default outer optimization iterations
SIGMA_MIN_FRAC = 0.1  # final noise = U_STDEV * this


def _sigma_schedule(sigma_max, n):
    return np.geomspace(sigma_max, sigma_max * SIGMA_MIN_FRAC, n)


def _rollout_xy(pfmp, knots, x0, theta):
    tq = jnp.linspace(0, pfmp.horizon, round(pfmp.horizon / pfmp.dt))
    tk = jnp.linspace(0, pfmp.horizon, pfmp.num_knots)
    us = hermite_spline(tq, tk, knots)
    _, xy = jax.lax.scan(lambda x, ut: dynamics(x, ut[0], theta, ut[1]), x0, (us, tq))
    return xy

@partial(jax.jit, static_argnames=['pfmp'])
def _rollout_xy_jit(pfmp, knots, x0, theta):
    return _rollout_xy(pfmp, knots, x0, theta)


def _draw_panel_bg(ax, x0, theta, cost_fn, xlim, ylim, plot_obs):
    obs_to_plot = plot_obs if plot_obs is not None else theta.obstacles
    xs = jnp.linspace(xlim[0], xlim[1], 200)
    ys = jnp.linspace(ylim[0], ylim[1], 150)
    X, Y = jnp.meshgrid(xs, ys)
    points = jnp.stack([X.ravel(), Y.ravel()], axis=1)
    Z = jax.vmap(lambda loc: cost_fn(loc, jnp.zeros(1), theta, 0.0))(
        points).reshape(X.shape)
    ax.contourf(X, Y, Z, levels=20, cmap='Greys', alpha=0.3)
    plot_obstacles(ax, obs_to_plot)
    ax.plot(float(x0[0]), float(x0[1]), 'o', color='darkgreen', markersize=20,
            markeredgecolor='black', markeredgewidth=1.5, zorder=10)
    ax.plot(float(theta.goal[0]), float(theta.goal[1]), '*', color='gold',
            markersize=30, markeredgecolor='black', markeredgewidth=1.5, zorder=10)
    ax.set_xlim(*xlim); ax.set_ylim(*ylim)
    ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor('black'); spine.set_linewidth(1.5)


# =============================================================================
# Animation — baselines
# =============================================================================

def precompute_baselines(names, pfmp, x0, theta, num_outer):
    """N outer iterations of each baseline with annealed noise.

    Returns: dict name -> list of (cloud_xys, nominal_xy) per outer iteration
        cloud_xys:  (NUM_CLOUD, T, 2)
        nominal_xy: (T, 2)
    """
    sigmas = _sigma_schedule(float(theta.u_stdev), num_outer)
    rollout_batch = jax.jit(jax.vmap(partial(_rollout_xy_jit, pfmp, x0=x0, theta=theta)))
    name_seed = {'ps': 0, 'mppi': 1, 'cem': 2}
    results = {name: [] for name in names}

    for name in names:
        nominal = jnp.zeros(pfmp.num_knots)
        print(f"  {name} ({num_outer} iters)...", end=" ", flush=True)

        for i, sigma in enumerate(sigmas):
            key = jax.random.key(i * 3 + name_seed[name])
            keys = jax.random.split(key, pfmp.particles)
            noise = jax.vmap(
                lambda k: jax.random.normal(k, (pfmp.num_knots,)) * sigma)(keys)
            trajs = jnp.clip(nominal[None] + noise, -0.8, 0.8)
            costs = jax.vmap(lambda k: _rollout_cost_jit(pfmp, k, x0, theta))(trajs)

            if name == 'ps':
                nominal = trajs[jnp.argmin(costs)]
            elif name == 'mppi':
                # Temperature anneals with sigma^2 (variance of the proposal)
                temperature = max(float(jnp.std(costs)) * (sigma / sigmas[0]) ** 2, 1e-6)
                weights = jax.nn.softmax(-costs / temperature)
                nominal = jnp.sum(trajs * weights[:, None], axis=0)
            elif name == 'cem':
                num_elite = max(1, int(CEM_ELITE_FRAC * pfmp.particles))
                nominal = jnp.mean(trajs[jnp.argsort(costs)[:num_elite]], axis=0)

            sample_idx = np.random.RandomState(i).choice(
                pfmp.particles, NUM_CLOUD, replace=False)
            cloud_xys = np.array(rollout_batch(trajs[sample_idx]))[:, :, :2]
            nom_xy = np.array(_rollout_xy_jit(pfmp, nominal, x0, theta))[:, :2]
            results[name].append((cloud_xys, nom_xy))

        print("done")

    return results


def setup_baseline_panel(ax, baselines_data, x0, theta, cost_fn, xlim, ylim,
                          plot_obs=None, unroll=False, steps_per_knot=None,
                          num_knots=None):
    """Set up the baselines animation panel. Returns (update_fn, num_frames, interval)."""
    _draw_panel_bg(ax, x0, theta, cost_fn, xlim, ylim, plot_obs)

    cloud_lines = {}
    nominal_lines = {}
    for name in baselines_data:
        color_best, color_rest, label = BASELINE_COLORS[name]
        lines = [ax.plot([], [], color=color_rest, alpha=0.2, linewidth=0.8,
                         zorder=2, solid_capstyle='round')[0]
                 for _ in range(NUM_CLOUD)]
        cloud_lines[name] = lines
        nom_line, = ax.plot([], [], color=color_best, linewidth=2.5, alpha=1.0,
                            zorder=8, solid_capstyle='round', label=label)
        nominal_lines[name] = nom_line

    ax.legend(loc='upper left', fontsize=9, framealpha=0.85,
              edgecolor='black', fancybox=False)

    def _draw(name, cloud_xys, nom_xy):
        for j, line in enumerate(cloud_lines[name]):
            line.set_data(cloud_xys[j, :, 0], cloud_xys[j, :, 1])
        nominal_lines[name].set_data(nom_xy[:, 0], nom_xy[:, 1])

    num_outer = len(next(iter(baselines_data.values())))

    if unroll:
        inner_frames = num_knots - 1
        def update(j):
            outer, inner = divmod(j, inner_frames)
            end = (inner + 1) * steps_per_knot
            for name, iters in baselines_data.items():
                cloud_xys, nom_xy = iters[outer]
                _draw(name, cloud_xys[:, :end, :], nom_xy)
        return update, num_outer * inner_frames, 300
    else:
        def update(frame):
            for name, iters in baselines_data.items():
                cloud_xys, nom_xy = iters[frame]
                _draw(name, cloud_xys, nom_xy)
        return update, num_outer, 300


# =============================================================================
# Animation — SMC
# =============================================================================

def precompute_smc(pfmp_sir, x0, theta, num_outer, num_top=20):
    """N outer SMC runs with annealed noise.

    Returns: list of (xys, best_pi) per outer iteration
        xys:     (num_top, T, 2) full-horizon trajectories
        best_pi: index into xys of the best trajectory
    """
    sigmas = _sigma_schedule(float(theta.u_stdev), num_outer)
    num_knots = pfmp_sir.num_knots
    interval_len = pfmp_sir.horizon / (num_knots - 1)
    steps_per_knot = round(interval_len / pfmp_sir.dt)
    tq = jnp.linspace(0, pfmp_sir.horizon, (num_knots - 1) * steps_per_knot)
    tk = pfmp_sir.tk()

    frames = []
    best_knots = jnp.zeros(pfmp_sir.num_knots)
    print(f"  SMC ({num_outer} iters)...", end=" ", flush=True)
    for i, sigma in enumerate(sigmas):
        theta_i = theta._replace(u_stdev=sigma, mean=best_knots)
        states, ancestors, costs, _ = pfmp_sir._sir(jax.random.key(i), x0, theta_i)
        total_costs = jnp.sum(costs, axis=-1)
        top_indices = np.array(jnp.argsort(total_costs)[:num_top])
        best_pi = int(np.where(top_indices == int(jnp.argmin(total_costs)))[0][0])

        trajs = pfmp_sir.choose_paths(states, ancestors, jnp.array(top_indices))

        def rollout(knots, tq=tq, tk=tk):
            us = hermite_spline(tq, tk, knots)
            _, xy = jax.lax.scan(
                lambda x, ut: dynamics(x, ut[0], theta_i, ut[1]), x0, (us, tq))
            return xy

        xys = np.array(jax.vmap(rollout)(trajs))[:, :, :2]
        best_knots = trajs[best_pi]
        frames.append((xys, best_pi))
    print("done")

    return frames, steps_per_knot


def setup_smc_panel(ax, smc_frames, steps_per_knot, x0, theta, cost_fn, xlim, ylim,
                     plot_obs=None, num_top=20, unroll=False, num_knots=None):
    """Set up the SMC animation panel. Returns (update_fn, num_frames, interval)."""
    _draw_panel_bg(ax, x0, theta, cost_fn, xlim, ylim, plot_obs)

    alphas = np.linspace(0.4, 0.1, num_top)
    trail_lines = []
    for pi in range(num_top):
        line, = ax.plot([], [], color='mediumseagreen', alpha=alphas[pi],
                        linewidth=0.8, zorder=2, solid_capstyle='round')
        trail_lines.append(line)
    best_line, = ax.plot([], [], color='darkgreen', linewidth=2.5, alpha=1.0,
                          zorder=8, solid_capstyle='round')

    def _draw(xys, end, best_pi):
        for pi in range(num_top):
            if pi != best_pi:
                trail_lines[pi].set_data(xys[pi, :end, 0], xys[pi, :end, 1])
            else:
                trail_lines[pi].set_data([], [])
        best_line.set_data(xys[best_pi, :, 0], xys[best_pi, :, 1])

    if unroll:
        inner_frames = num_knots - 1
        def update(j):
            outer, inner = divmod(j, inner_frames)
            xys, best_pi = smc_frames[outer]
            _draw(xys, (inner + 1) * steps_per_knot, best_pi)
        return update, len(smc_frames) * inner_frames, 300
    else:
        def update(frame):
            xys, best_pi = smc_frames[frame]
            _draw(xys, xys.shape[1], best_pi)
        return update, len(smc_frames), 300


# =============================================================================
# Animation
# =============================================================================

def animate(num_knots: int = 20, map_index: int = 0,
            save_gif: bool = False, num_outer: int = NUM_OUTER,
            baseline: str = 'all', unroll: bool = False):
    """Animate MPC baselines vs SMC, synchronized over N outer iterations."""
    arena_length = round(1.0 * num_knots + 1.0, 1)
    horizon = num_knots * 0.5

    pfmp_base = PFMP(dynamics, cost_fn, terminal_cost, init, proposal,
                     horizon=horizon, num_knots=num_knots, particles=PARTICLES,
                     ess_resample_thresh=0.0, dt=DT)
    pfmp_sir = PFMP(dynamics, cost_fn, terminal_cost, init, proposal,
                    horizon=horizon, num_knots=num_knots, particles=PARTICLES,
                    ess_resample_thresh=ESS_THRESH, dt=DT)

    obs_batch, x0_batch, goals_batch, _ = generate_map_batch(
        SEED + num_knots * 10000, max(map_index + 1, 1),
        arena_length, ARENA_WIDTH, OBSTACLE_DENSITY, BOX_SIZE_RANGE, MIN_MARGIN,
    )

    obstacles = obs_batch[map_index]
    x0 = x0_batch[map_index]
    goal = goals_batch[map_index]
    real_obs = obstacles[obstacles[:, 0] < 1e5]
    step_len = horizon / (num_knots - 1)
    theta = Params(goal=goal, obstacles=obstacles,
                   L=WHEELBASE, v=SPEED, du_stdev=DU_STDEV, u_stdev=U_STDEV,
                   mean=jnp.zeros(num_knots), step_len=step_len)

    names = ['ps', 'mppi', 'cem'] if baseline == 'all' else [baseline]

    print("Precomputing baselines:")
    baselines_data = precompute_baselines(names, pfmp_base, x0, theta, num_outer)
    print("Precomputing SMC:")
    smc_frames, steps_per_knot = precompute_smc(pfmp_sir, x0, theta, num_outer)

    plt.style.use('default')
    fig, (ax_right, ax_left) = plt.subplots(2, 1, figsize=(8, 10),
                                              gridspec_kw={'hspace': 0.05})
    fig.patch.set_facecolor('white')
    for ax in (ax_left, ax_right):
        ax.set_facecolor('white')
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5, color='gray')

    xlim = (-0.5, arena_length + 0.5)
    ylim = (-0.5, ARENA_WIDTH + 0.5)

    steps_per_knot = round((horizon / (num_knots - 1)) / DT)

    update_left, frames, interval = setup_baseline_panel(
        ax_left, baselines_data, x0, theta, cost_fn, xlim, ylim, plot_obs=real_obs,
        unroll=unroll, steps_per_knot=steps_per_knot, num_knots=num_knots)
    ax_left.set_title("MPC Baselines", fontsize=12)

    update_right, _, _ = setup_smc_panel(
        ax_right, smc_frames, steps_per_knot, x0, theta, cost_fn, xlim, ylim,
        plot_obs=real_obs, unroll=unroll, num_knots=num_knots)
    ax_right.set_title("SMC", fontsize=12)

    def combined(frame):
        update_left(frame)
        update_right(frame)

    anim = FuncAnimation(fig, combined, frames=frames, interval=interval,
                         blit=False, repeat=True, repeat_delay=2000)

    if save_gif:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        anim.save(OUTPUT_DIR / 'pfmp_complex.gif', writer='pillow', fps=5, dpi=250)
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
            mean=jnp.zeros((NUM_MAPS, num_knots)),
            step_len=jnp.full(NUM_MAPS, horizon / (num_knots - 1)),
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
        "SMC": {"key": "smc_costs",  "color": "#27AE60", "ls": "-"},
    }

    stats = {}
    for name, m in methods.items():
        medians, q25, q75 = [], [], []
        for k in knot_counts:
            data = np.where(np.isfinite(np.array(results[str(k)][m["key"]])),
                            np.array(results[str(k)][m["key"]]), np.nan)
            medians.append(np.nanmedian(data))
            q25.append(np.nanpercentile(data, 25))
            q75.append(np.nanpercentile(data, 75))
        stats[name] = {"median": np.array(medians), "q25": np.array(q25), "q75": np.array(q75)}

    fig, ax = plt.subplots(figsize=(7, 4.5))

    for name, m in methods.items():
        s = stats[name]
        lw = 2.5 if name == "SMC" else 1.8
        z = 5 if name == "SMC" else 3
        ax.plot(x, s["median"], color=m["color"], linestyle=m["ls"],
                linewidth=lw, marker="o", markersize=5, label=name, zorder=z)
        ax.fill_between(x, s["q25"], s["q75"], color=m["color"],
                        alpha=0.20 if name == "SMC" else 0.12, zorder=z - 1)

    ax.set_xlabel("Planning Horizon (knots)", fontsize=13)
    ax.set_ylabel("Executed Cost (median)", fontsize=13)
    ax.tick_params(labelsize=11)
    ax.set_xticks(x)
    ax.set_title("Planning Cost vs. Horizon Length", fontsize=13, pad=10)
    ax.legend(fontsize=10, framealpha=0.95, edgecolor="black", fancybox=False)
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)

    fig.tight_layout()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_DIR / "geometric_cost.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / "geometric_cost.png", dpi=300, bbox_inches="tight")
    print("Saved geometric_cost.pdf/png")
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
    parser.add_argument("--iters", type=int, default=NUM_OUTER,
                        help=f"Outer optimization iterations (default: {NUM_OUTER})")
    parser.add_argument("--baseline", type=str, default="all",
                        choices=["ps", "mppi", "cem", "all"],
                        help="Which baseline(s) to show (default: all)")
    parser.add_argument("--unroll", action="store_true",
                        help="Show inner knot-by-knot unrolls within each outer iteration")
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
                save_gif=args.save, num_outer=args.iters,
                baseline=args.baseline, unroll=args.unroll)
