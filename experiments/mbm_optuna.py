"""
Bayesian hyperparameter search for MBM benchmark using Optuna.

Usage:
    python experiments/mbm_tune.py
    python experiments/mbm_tune.py --trials 200 --dataset models/problems/table_pick_panda
"""

import argparse
import sys
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

sys.path.insert(0, str(Path(__file__).parent.parent))

from examples.mbm import (
    load_robot, load_mbm_dataset, Params, Q_MIN, Q_MAX,
    robot_sdf, dynamics, forward_kinematics, get_sphere_positions,
)
from pfmp import PFMP
from pfmp.interpolate import hermite_spline


OUTPUT_DIR = Path(__file__).parent.parent / "figures"
DATASETS   = [
    "models/problems/table_pick_panda",
    "models/problems/table_under_pick_panda",
    "models/problems/bookshelf_small_panda",
    "models/problems/bookshelf_tall_panda",
    "models/problems/bookshelf_thin_panda",
    "models/problems/box_panda",
    "models/problems/cage_panda",
]
URDF       = "models/panda_spherized.urdf"
N_PER_DATASET = 10   # problems per dataset
SEED       = 42


def make_cost_fns(col_weight, goal_weight, margin, terminal_goal_weight):
    def cost_fn(x, u, theta, t):
        sdf = robot_sdf(x, theta.env, theta.robot)
        col  = jnp.where(sdf < 0.0, jnp.inf, col_weight * jnp.maximum(margin - sdf, 0.0) ** 2)
        goal = goal_weight * jnp.linalg.norm(x - theta.goal)
        return col + goal

    def terminal_cost(x, theta):
        sdf = robot_sdf(x, theta.env, theta.robot)
        col  = jnp.where(sdf < 0.0, jnp.inf, col_weight * jnp.maximum(margin - sdf, 0.0) ** 2)
        goal = terminal_goal_weight * jnp.linalg.norm(x - theta.goal) ** 2
        return col + goal

    return cost_fn, terminal_cost


def make_proposal(goal_frac, gauss_frac, gauss_std):
    def init(key, x, theta):
        k1, k2, k3 = jax.random.split(key, 3)
        r = jax.random.uniform(k1)
        branch = jnp.where(r < goal_frac, 0,
                 jnp.where(r < goal_frac + gauss_frac, 1, 2))
        return jax.lax.switch(branch, [
            lambda _: theta.goal,
            lambda _: jnp.clip(x + jax.random.normal(k2, x.shape) * gauss_std,
                               theta.q_min, theta.q_max),
            lambda _: jax.random.uniform(k3, x.shape, minval=theta.q_min, maxval=theta.q_max),
        ], None)

    def proposal(key, x, u, theta, t):
        return init(key, x, theta)

    return init, proposal


def run_trial(problems, robot, num_knots, horizon, particles, ess_thresh,
              col_weight, goal_weight, margin, terminal_goal_weight,
              goal_frac, gauss_frac, gauss_std, step_size=0.05, seed=SEED):
    cost_fn, terminal_cost = make_cost_fns(col_weight, goal_weight, margin, terminal_goal_weight)
    init_fn, proposal_fn   = make_proposal(goal_frac, gauss_frac, gauss_std)

    pfmp = PFMP(dynamics, cost_fn, terminal_cost, init_fn, proposal_fn,
                horizon=horizon, num_knots=num_knots, particles=particles,
                ess_resample_thresh=ess_thresh)

    @jax.jit
    def run_sir(key, q_start, theta):
        traj, scores = pfmp.run_sir(key, q_start, theta)
        return traj[jnp.argmin(jnp.sum(scores, axis=-1))]

    @jax.jit
    def extract_path(knots, q_start, theta):
        tq = jnp.linspace(0, pfmp.horizon, int(pfmp.horizon / pfmp.dt))
        us = hermite_spline(tq, pfmp.tk(), knots)
        _, path = jax.lax.scan(lambda x, u: (dynamics(x, u, theta, 0.0)[0], x), q_start, us)
        return path

    # Warm-up
    p0 = problems[0]
    th0 = Params(goal=p0['q_goal'], env=p0['env'], q_min=Q_MIN, q_max=Q_MAX, robot=robot, step_size=step_size)
    k0 = jax.random.key(0)
    _ = extract_path(run_sir(k0, p0['q_start'], th0), p0['q_start'], th0)

    keys = jax.random.split(jax.random.key(seed), len(problems))
    successes = []
    for i, prob in enumerate(problems):
        theta = Params(goal=prob['q_goal'], env=prob['env'], q_min=Q_MIN, q_max=Q_MAX, robot=robot, step_size=step_size)
        knots = run_sir(keys[i], prob['q_start'], theta)
        path  = extract_path(knots, prob['q_start'], theta)
        col   = bool(jax.vmap(lambda q: robot_sdf(q, theta.env, robot) < 0)(path).any())
        dist  = float(jnp.linalg.norm(path[-1] - prob['q_goal']))
        successes.append((not col) and dist < 0.5)

    return float(np.mean(successes))


def objective(trial, problems, robot):
    col_weight           = trial.suggest_float("col_weight",           100.0, 5000.0, log=True)
    goal_weight          = trial.suggest_float("goal_weight",          1e-4,  1.0,    log=True)
    margin               = trial.suggest_float("margin",               0.01,  0.15)
    terminal_goal_weight = trial.suggest_float("terminal_goal_weight", 10.0,  2000.0, log=True)
    goal_frac            = trial.suggest_float("goal_frac",            0.05,  0.4)
    gauss_frac           = trial.suggest_float("gauss_frac",           0.3,   0.8)
    gauss_std            = trial.suggest_float("gauss_std",            0.05,  0.5)
    num_knots            = trial.suggest_int  ("num_knots",            4,     12)
    horizon              = trial.suggest_float("horizon",              0.5,   3.0)
    ess_thresh           = trial.suggest_float("ess_thresh",           0.1,   0.7)
    step_size            = trial.suggest_float("step_size",            0.01,  0.3)

    return run_trial(problems, robot, num_knots, horizon, particles=512,
                     ess_thresh=ess_thresh, col_weight=col_weight,
                     goal_weight=goal_weight, margin=margin,
                     terminal_goal_weight=terminal_goal_weight,
                     goal_frac=goal_frac, gauss_frac=gauss_frac, gauss_std=gauss_std,
                     step_size=step_size)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials",      type=int, default=100)
    parser.add_argument("--n-per-dataset", type=int, default=N_PER_DATASET)
    parser.add_argument("--out",         type=str, default=None)
    args = parser.parse_args()

    robot = load_robot(URDF)
    problems = []
    for ds in DATASETS:
        problems.extend(load_mbm_dataset(ds, max_problems=args.n_per_dataset))
    print(f"Loaded {len(problems)} problems across {len(DATASETS)} datasets. Running {args.trials} Optuna trials…")

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(lambda t: objective(t, problems, robot),
                   n_trials=args.trials, show_progress_bar=True)

    best = study.best_trial
    print(f"\nBest success rate: {best.value*100:.1f}%")
    print("Best params:")
    for k, v in best.params.items():
        print(f"  {k}: {v}")

    out = args.out or str(OUTPUT_DIR / "mbm_tune.json")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump({"best_value": best.value, "best_params": best.params}, f, indent=2)
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
