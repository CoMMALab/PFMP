"""
Optuna hyperparameter search for the MBM *baselines* (PS, MPPI, CEM, Biased-MPPI,
Biased-CEM), so the SMC-vs-baseline comparison is tuned on both sides.

The task cost and the custom proposal are held fixed at the SMC-tuned values
(imported from fr3.mbm); only each baseline's own knobs are searched:
  PS          : u_stdev
  MPPI        : u_stdev, mppi_temperature
  CEM         : u_stdev, cem_elite_frac
  Biased-MPPI : mppi_temperature          (custom proposal, no u_stdev)
  Biased-CEM  : cem_elite_frac

Saves figures/mbm_baseline_tune.json -> {method: best_params}.

Usage:
    python experiments/mbm_baseline_optuna.py --trials 80
    python experiments/mbm_baseline_optuna.py --trials 80 --methods ps mppi
"""

import argparse
import json
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

sys.path.insert(0, str(Path(__file__).parent.parent))

from fr3.mbm import (
    load_robot, load_mbm_dataset, Params, Q_MIN, Q_MAX, robot_sdf,
    dynamics, cost_fn, terminal_cost, init, proposal,
    perturbation_init, perturbation_proposal, NUM_KNOTS, HORIZON,
)
from pfmp import PFMP
from pfmp.algorithm import ps_update, mppi_update, cem_update
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
URDF          = "models/panda_spherized.urdf"
N_PER_DATASET = 10
PARTICLES     = 512   # tuning budget; knobs transfer to the 1000-particle benchmark
SEED          = 42
PERTURB       = ("ps", "mppi", "cem")


def make_runner(method, robot):
    """One jitted sampler + path extractor per method (reused across trials)."""
    prop = (perturbation_init, perturbation_proposal) if method in PERTURB else (init, proposal)
    ctrl = PFMP(dynamics, cost_fn, terminal_cost, *prop,
                horizon=HORIZON, num_knots=NUM_KNOTS, particles=PARTICLES,
                ess_resample_thresh=0.0)
    sample = jax.jit(ctrl.run_sis)

    @jax.jit
    def extract(knots, q_start, theta):
        tq = jnp.linspace(0, ctrl.horizon, int(ctrl.horizon / ctrl.dt))
        us = hermite_spline(tq, ctrl.tk(), knots)
        _, path = jax.lax.scan(lambda x, u: (dynamics(x, u, theta, 0.0)[0], x), q_start, us)
        return path

    return sample, extract


def make_theta(method, prob, robot, u_stdev):
    if method in PERTURB:
        a = jnp.linspace(0.0, 1.0, NUM_KNOTS)[:, None]
        mean0 = (1 - a) * prob['q_start'] + a * prob['q_goal']
        return Params(goal=prob['q_goal'], env=prob['env'], q_min=Q_MIN, q_max=Q_MAX,
                      robot=robot, mean=mean0, u_stdev=u_stdev)
    return Params(goal=prob['q_goal'], env=prob['env'], q_min=Q_MIN, q_max=Q_MAX, robot=robot)


def eval_method(method, problems, robot, runner, u_stdev, temperature, elite_frac, seed=SEED):
    sample, extract = runner
    keys = jax.random.split(jax.random.key(seed), len(problems))
    succ = []
    for i, prob in enumerate(problems):
        theta = make_theta(method, prob, robot, u_stdev)
        trajs, costs = sample(keys[i], prob['q_start'], theta)
        if method == 'ps':
            th, _ = ps_update(theta, trajs, costs)
        elif method in ('mppi', 'biased_mppi'):
            th, _ = mppi_update(theta, trajs.transpose(1, 0, 2), costs.T, temperature)
        else:  # cem, biased_cem
            th, _ = cem_update(theta, trajs, jnp.sum(costs, axis=1), elite_frac)
        path = extract(th.mean, prob['q_start'], theta)
        col = bool(jax.vmap(lambda q: robot_sdf(q, theta.env, robot) < 0)(path).any())
        dist = float(jnp.linalg.norm(path[-1] - prob['q_goal']))
        succ.append((not col) and dist < 0.5)
    return float(np.mean(succ))


def objective(trial, method, problems, robot, runner):
    u_stdev     = trial.suggest_float("u_stdev", 0.05, 0.6) if method in PERTURB else 0.0
    temperature = trial.suggest_float("mppi_temperature", 0.01, 100.0, log=True) \
                  if method in ("mppi", "biased_mppi") else 1.0
    elite_frac  = trial.suggest_float("cem_elite_frac", 0.005, 0.3, log=True) \
                  if method in ("cem", "biased_cem") else 0.1
    return eval_method(method, problems, robot, runner, u_stdev, temperature, elite_frac)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=80)
    parser.add_argument("--n-per-dataset", type=int, default=N_PER_DATASET)
    parser.add_argument("--methods", type=str, nargs="+",
                        default=["ps", "mppi", "cem", "biased_mppi", "biased_cem"])
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    robot = load_robot(URDF)
    problems = []
    for ds in DATASETS:
        problems.extend(load_mbm_dataset(ds, max_problems=args.n_per_dataset))
    print(f"Loaded {len(problems)} problems across {len(DATASETS)} datasets.")

    out_path = args.out or str(OUTPUT_DIR / "mbm_baseline_tune.json")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_best = {}
    if Path(out_path).exists():
        all_best = json.load(open(out_path))

    for method in args.methods:
        print(f"\n=== Tuning {method} ({args.trials} trials) ===")
        runner = make_runner(method, robot)
        study = optuna.create_study(direction="maximize",
                                    sampler=optuna.samplers.TPESampler(seed=SEED))
        study.optimize(lambda t: objective(t, method, problems, robot, runner),
                       n_trials=args.trials, show_progress_bar=True)
        print(f"  {method}: best success {study.best_value*100:.1f}%  params {study.best_params}")
        all_best[method] = {"best_value": study.best_value, "best_params": study.best_params}
        with open(out_path, "w") as f:
            json.dump(all_best, f, indent=2)

    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
