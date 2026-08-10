"""Qualitative figures for the tasks in `geom_runner.py`.

Counterpart to `geom_analyze.py`: that one draws the sweeps, this one draws the
plans the paper puts beside them.

    drive -> geometric_plan         pushT -> fig_pushT_traces     mbm -> planning1024

Three branches, because they share no rendering path (2D matplotlib with
task-specific geometry; a viser 3D scene for MBM). What they do share is the
task definition, so a figure cannot be drawn from a different cost or proposal
than the numbers next to it. Single-problem and unbatched: these keep whole
particle trajectories, which is what the runner's `measure` discards.

    python experiments/geom_figure.py drive
    python experiments/geom_figure.py experiments/configs/pushT.yaml --seeds 0 1 2 3
    python experiments/geom_figure.py experiments/configs/mbm.yaml --problem 0
"""

import argparse
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from functools import partial

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "examples"))

import yaml

from pfmp import PFMP
from pfmp.interpolate import hermite_spline
from pfmp.obstacle_grid import generate_map_batch, plot_obstacles
# The task itself lives in geom_runner, so the figure and the numbers cannot be
# drawn from different dynamics or costs. Its functions are closed over inside
# the adapter, so they arrive on the GeomTask and are threaded through the
# helpers below rather than imported.
from geom_runner import (DRIVE_ARENA_WIDTH as ARENA_WIDTH,
                         DRIVE_BOX_SIZE_RANGE as BOX_SIZE_RANGE, DRIVE_DT as DT,
                         DRIVE_ESS_THRESH as ESS_THRESH,
                         DRIVE_MIN_MARGIN as MIN_MARGIN,
                         DRIVE_OBSTACLE_DENSITY as OBSTACLE_DENSITY,
                         DRIVE_PARTICLES as PARTICLES, DRIVE_SEED as SEED,
                         DRIVE_U_STDEV as U_STDEV,
                         DRIVE_CEM_ELITE_FRAC as CEM_ELITE_FRAC,
                         DRIVE_MPPI_TEMPERATURE as MPPI_TEMPERATURE,
                         DRIVE_STEER_LIMIT, build_controller, load_task,
                         make_perturbation)



OUTPUT_DIR = Path(__file__).parent.parent / "figures"


def _rollout_cost(pfmp, knots, x0, theta, drive_dynamics, drive_cost, drive_terminal):
    tq = jnp.linspace(0, pfmp.horizon, int(pfmp.horizon / pfmp.dt))
    us = hermite_spline(tq, pfmp.tk(), knots)

    def step(x, inp):
        u, t = inp
        x_next, _ = drive_dynamics(x, u, theta, t)
        c = drive_cost(x_next, u, theta, t)
        return x_next, c

    x_final, running_costs = jax.lax.scan(step, x0, (us, tq))
    return jnp.sum(running_costs) + drive_terminal(x_final, theta)


@partial(jax.jit, static_argnames=["pfmp", "drive_dynamics", "drive_cost", "drive_terminal"])
def _rollout_cost_jit(pfmp, knots, x0, theta, drive_dynamics, drive_cost, drive_terminal):
    return _rollout_cost(pfmp, knots, x0, theta, drive_dynamics, drive_cost, drive_terminal)


def sample_and_evaluate(key, pfmp, nominal, x0, theta, noise_level, fns):
    K = pfmp.particles
    keys = jax.random.split(key, K)
    noise = jax.vmap(lambda k: jax.random.normal(k, shape=nominal.shape) * noise_level)(keys)
    trajs = jnp.clip(nominal[None, :] + noise, -0.8, 0.8)
    costs = jax.vmap(lambda knots: _rollout_cost_jit(pfmp, knots, x0, theta, *fns))(trajs)
    return trajs, costs



def _save(fig, name, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / name
    for ext in ("pdf", "png"):
        fig.savefig(f"{stem}.{ext}", dpi=300, bbox_inches="tight")
    print(f"Saved {stem}.pdf / {stem}.png")



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


def _rollout_xy(pfmp, knots, x0, theta, drive_dynamics):
    tq = jnp.linspace(0, pfmp.horizon, round(pfmp.horizon / pfmp.dt))
    tk = jnp.linspace(0, pfmp.horizon, pfmp.num_knots)
    us = hermite_spline(tq, tk, knots)
    _, xy = jax.lax.scan(lambda x, ut: drive_dynamics(x, ut[0], theta, ut[1]), x0, (us, tq))
    return xy

@partial(jax.jit, static_argnames=['pfmp', 'drive_dynamics'])
def _rollout_xy_jit(pfmp, knots, x0, theta, drive_dynamics):
    return _rollout_xy(pfmp, knots, x0, theta, drive_dynamics)


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

def precompute_baselines(names, pfmp, x0, theta, num_outer, fns):
    """N outer iterations of each baseline with annealed noise.

    Returns: dict name -> list of (cloud_xys, nominal_xy) per outer iteration
        cloud_xys:  (NUM_CLOUD, T, 2)
        nominal_xy: (T, 2)
    """
    sigmas = _sigma_schedule(float(theta.stdev), num_outer)
    rollout_batch = jax.jit(jax.vmap(partial(_rollout_xy_jit, pfmp, x0=x0, theta=theta,
                                            drive_dynamics=fns[0])))
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
            costs = jax.vmap(lambda k: _rollout_cost_jit(pfmp, k, x0, theta, *fns))(trajs)

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
            nom_xy = np.array(_rollout_xy_jit(pfmp, nominal, x0, theta, fns[0]))[:, :2]
            results[name].append((cloud_xys, nom_xy))

        print("done")

    return results


def setup_baseline_panel(ax, baselines_data, x0, theta, cost_fn, fns, xlim, ylim,
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

def precompute_smc(pfmp_sir, x0, theta, num_outer, fns, num_top=20):
    """N outer SMC runs with annealed noise.

    Returns: list of (xys, best_pi) per outer iteration
        xys:     (num_top, T, 2) full-horizon trajectories
        best_pi: index into xys of the best trajectory
    """
    sigmas = _sigma_schedule(float(theta.stdev), num_outer)
    num_knots = pfmp_sir.num_knots
    interval_len = pfmp_sir.horizon / (num_knots - 1)
    steps_per_knot = round(interval_len / pfmp_sir.dt)
    tq = jnp.linspace(0, pfmp_sir.horizon, (num_knots - 1) * steps_per_knot)
    tk = pfmp_sir.tk()

    frames = []
    best_knots = jnp.zeros(pfmp_sir.num_knots)
    print(f"  SMC ({num_outer} iters)...", end=" ", flush=True)
    for i, sigma in enumerate(sigmas):
        theta_i = theta._replace(stdev=sigma, mean=best_knots)
        states, ancestors, costs, _ = pfmp_sir._sir(jax.random.key(i), x0, theta_i)
        total_costs = jnp.sum(costs, axis=-1)
        top_indices = np.array(jnp.argsort(total_costs)[:num_top])
        best_pi = int(np.where(top_indices == int(jnp.argmin(total_costs)))[0][0])

        trajs = pfmp_sir.choose_paths(states, ancestors, jnp.array(top_indices))

        def rollout(knots, tq=tq, tk=tk):
            us = hermite_spline(tq, tk, knots)
            _, xy = jax.lax.scan(
                lambda x, ut: fns[0](x, ut[0], theta_i, ut[1]), x0, (us, tq))
            return xy

        xys = np.array(jax.vmap(rollout)(trajs))[:, :, :2]
        best_knots = trajs[best_pi]
        frames.append((xys, best_pi))
    print("done")

    return frames, steps_per_knot


def setup_smc_panel(ax, smc_frames, steps_per_knot, x0, theta, cost_fn, fns, xlim, ylim,
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

def figure_drive(config, args, output_dir):
    """Two stacked panels, SMC over baselines: particle clouds on the cost
    landscape. A still of `geometric.animate` at its converged frame."""
    knots = int(config.get("num_knots", 20))
    arena, horizon = round(knots + 1.0, 1), knots * 0.5
    particles = int(config.get("num_samples", PARTICLES))
    n_outer = args.num_outer or NUM_OUTER

    cfg = {"task": "geometric", "num": args.problem + 1, "num_knots": knots,
           "num_samples": particles, "seed": config.get("seed", SEED)}
    task, _ = load_task("drive", cfg)
    fns = (task.dynamics, task.running_cost, task.terminal_cost)

    def ctrl(ess):
        return build_controller(task, "ps", particles, ess_thresh=ess)._replace(
            ess_resample_thresh=ess)

    x0, theta = task.sample_problem(args.problem)
    obstacles = theta.obstacles
    real_obs = obstacles[obstacles[:, 0] < 1e5]

    print("Precomputing baselines:")
    baselines = precompute_baselines(["ps", "mppi", "cem"], ctrl(0.0),
                                     x0, theta, n_outer, fns)
    print("Precomputing SMC:")
    smc, _ = precompute_smc(ctrl(config.get("ess_thresh", ESS_THRESH)),
                            x0, theta, n_outer, fns)

    plt.style.use("default")
    # Size from the arena, not a fixed 8x10: the axes hold equal aspect, so a
    # mismatched figure leaves a large dead band between the two panels.
    w = 10.0
    fig, (top, bot) = plt.subplots(
        2, 1, figsize=(w, 2 * w * (ARENA_WIDTH + 1) / (arena + 1) + 0.5),
        gridspec_kw={"hspace": 0.12})
    fig.patch.set_facecolor("white")
    for ax in (top, bot):
        ax.set_facecolor("white")
        ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.5, color="gray")

    lims = dict(xlim=(-0.5, arena + 0.5), ylim=(-0.5, ARENA_WIDTH + 0.5))
    spk = round((horizon / (knots - 1)) / DT)
    panel = dict(x0=x0, theta=theta, cost_fn=task.running_cost, fns=fns, plot_obs=real_obs,
                 unroll=False, num_knots=knots, **lims)

    update_bot, frames, _ = setup_baseline_panel(
        bot, baselines, steps_per_knot=spk, **panel)
    update_top, _, _ = setup_smc_panel(top, smc, spk, **panel)
    bot.set_title("MPC Baselines", fontsize=12)
    top.set_title("SMC", fontsize=12)

    update_bot(frames - 1)          # the still is the converged frame
    update_top(frames - 1)
    _save(fig, args.out or "geometric_plan", output_dir)


PUSHT_DISTANCE = 7.0

def figure_pushT(config, args, output_dir):
    """2x2 grid of scenarios; PS and SMC T-block traces overlaid per panel."""
    import pushT as pt
    from pushT_2d_sweep import PARTICLES, SEED
    fig = pt._make_trace_figure(
        args.seeds or list(range(config.get("seed", SEED),
                                 config.get("seed", SEED) + 4)),
        distance=PUSHT_DISTANCE, n_obs=None, corridor=False,
        particles=PARTICLES)
    _save(fig, args.out or "fig_pushT_traces", output_dir)


def figure_mbm(config, args, output_dir):
    """Viser 3D scene. Interactive by nature, so it serves rather than saves:
    the png comes from the client's own render, via its Save Screenshot button."""
    out = args.out or "planning1024"
    sys.argv = ["mbm_figure",
                "--dataset", args.dataset or config["datasets"][0],
                "--problem", str(args.problem),
                "--seed", str(config.get("seed", 0)), "--out", out]
    print(f"Serving viser; use 'Save Screenshot' to write figures/{out}.png")
    import mbm_figure
    mbm_figure.main()


BRANCHES = {"drive": figure_drive, "pushT": figure_pushT, "mbm": figure_mbm}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("task", help="task name (%s), or a path to a config yaml"
                    % "/".join(BRANCHES))
    ap.add_argument("--problem", type=int, default=0,
                    help="map index (geometric) or problem index (mbm)")
    ap.add_argument("--seed", type=int,
                    help="override the config's seed (all branches)")
    ap.add_argument("--seeds", type=int, nargs="+", help="pushT: scenario seeds")
    ap.add_argument("--dataset", help="mbm: dataset dir")
    ap.add_argument("--num-outer", type=int, help="geometric: outer iterations")
    ap.add_argument("--out", help="output stem (no extension)")
    ap.add_argument("--output-dir")
    args = ap.parse_args()

    # Accept either a task name or a config path; a bare name resolves to that
    # task's config if one exists, since some branches read nothing from it.
    if args.task in BRANCHES:
        path = Path(__file__).parent / "configs" / f"{args.task}.yaml"
        config = yaml.safe_load(open(path)) if path.exists() else {"task": args.task}
    else:
        config = yaml.safe_load(open(args.task))
    task = config["task"]
    if args.seed is not None:      # one place, so every branch honours it
        config["seed"] = args.seed
    if task not in BRANCHES:
        raise SystemExit(f"no figure branch for task {task!r}; "
                         f"have {'/'.join(BRANCHES)}")
    BRANCHES[task](config, args,
                   Path(args.output_dir) if args.output_dir else OUTPUT_DIR)


if __name__ == "__main__":
    main()
