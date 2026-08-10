"""
Optuna search over MBM hyperparameters, for EVERY method on the same footing.

The task is fixed and shared: cost weights, dynamics step size, horizon, and
knot count all come from `mbm` and are never searched. Tuning them per method
would have each algorithm solving a different problem, and tuning them at all
would fit the benchmark to whoever was tuned first. Only the axes the benchmark
actually compares are searched, per method:

  method       proposal (q)                      update (U)          resample (rho)
  sis          goal_frac/gauss_frac/gauss_std    argmin              0
  sir          goal_frac/gauss_frac/gauss_std    argmin              ess_thresh
  ps           stdev                             argmin              0
  mppi         stdev                             mppi_temperature    0
  cem          stdev                             cem_elite_frac      0
  biased_mppi  goal_frac/gauss_frac/gauss_std    mppi_temperature    0
  biased_cem   goal_frac/gauss_frac/gauss_std    cem_elite_frac      0

Tuning uses the first TUNE_PER_DATASET problems of each dataset and validates on
the remainder, so the reported held-out number is not measured on the problems
the search optimised against.

Saves figures/mbm_tune.json -> {method: {best_params, best_value, validation_value}},
which `mbm.run_benchmark_fast` loads so evaluation cannot drift from tuning.

Usage:
    python experiments/mbm_optuna.py --trials 80
    python experiments/mbm_optuna.py --trials 80 --methods sis mppi
"""

import argparse
import json
import sys
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))   # sibling: mbm.py

from mbm import (
    load_robot, load_mbm_dataset, Params, Q_MIN, Q_MAX, robot_sdf,
    dynamics, cost_fn, terminal_cost, init, proposal,
    perturbation_init, perturbation_proposal, NUM_KNOTS, HORIZON,
    # method-specific defaults, used when a method does not search that knob
    GOAL_FRAC, GAUSS_FRAC, GAUSS_STD, BASE_U_STDEV,
    MPPI_TEMPERATURE, CEM_ELITE_FRAC,
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
# Problems per dataset used for TUNING. Each MBM dataset holds exactly 100
# problems and the benchmark reports on all of them, so tuning and evaluation
# necessarily overlap unless the tuned set is held out. Tune on the first
# TUNE_PER_DATASET and validate on the remainder, mirroring pushT_2d_optuna.
TUNE_PER_DATASET     = 64
PROBLEMS_PER_DATASET = 100
PARTICLES     = 512   # tuning budget; knobs transfer to the 1000-particle benchmark
SEED          = 42
PERTURB       = ("ps", "mppi", "cem")          # Gaussian perturbation proposal
CUSTOM        = ("sis", "sir", "biased_mppi", "biased_cem")   # object-informed
ARGMIN        = ("sis", "sir", "ps")           # MAP update
ALL_METHODS   = ("sis", "sir", "ps", "mppi", "cem", "biased_mppi", "biased_cem")


def make_batched_runner(method, robot, ess_thresh=0.0):
    """jit(vmap(...)) over a batch of problems -> (collided, final_dist).

    Env shapes (obstacle counts) are identical within a dataset but differ across
    them, so batches are per-dataset and need no padding. Batching is worth ~50x
    on GPU: the old Python loop issued one dispatch per problem, far too small to
    fill the device.

    `ess_thresh` is a PFMP field and therefore static, so tuning it (sir only)
    rebuilds the controller and recompiles once per trial. `num_elite` is static
    for the same reason `cem_update` slices a Python int.
    """
    prop = (perturbation_init, perturbation_proposal) if method in PERTURB else (init, proposal)
    ctrl = PFMP(dynamics, cost_fn, terminal_cost, *prop,
                horizon=HORIZON, num_knots=NUM_KNOTS, particles=PARTICLES,
                ess_resample_thresh=ess_thresh)
    sample = ctrl.run_sir if method == "sir" else ctrl.run_sis
    tq = jnp.linspace(0, ctrl.horizon, int(ctrl.horizon / ctrl.dt))
    tk = ctrl.tk()

    def solve_one(key, q_start, theta, temperature, num_elite):
        trajs, costs = sample(key, q_start, theta)
        if method in ARGMIN:                     # sis, sir, ps
            th, _ = ps_update(theta, trajs, costs)
        elif method in ("mppi", "biased_mppi"):
            th, _ = mppi_update(theta, trajs, costs, temperature)
        else:                                    # cem, biased_cem
            th, _ = cem_update(theta, trajs, costs, num_elite / PARTICLES)
        us = hermite_spline(tq, tk, th.mean)
        _, path = jax.lax.scan(
            lambda x, u: (dynamics(x, u, theta, 0.0)[0], x), q_start, us)
        collided = jax.vmap(lambda q: robot_sdf(q, theta.env, theta.robot) < 0)(path).any()
        return collided, jnp.linalg.norm(path[-1] - theta.goal)

    @partial(jax.jit, static_argnums=(4,))
    def run(keys, q_starts, thetas, temperature, num_elite):
        return jax.vmap(solve_one, in_axes=(0, 0, 0, None, None))(
            keys, q_starts, thetas, temperature, num_elite)

    return run


def make_theta(method, prob, robot, hp):
    """hp: the trial's params. Perturbation methods need a warm start + stdev;
    custom-proposal methods need the mixture weights instead."""
    common = dict(goal=prob['q_goal'], env=prob['env'], q_min=Q_MIN, q_max=Q_MAX,
                  robot=robot)
    if method in PERTURB:
        a = jnp.linspace(0.0, 1.0, NUM_KNOTS)[:, None]
        mean0 = (1 - a) * prob['q_start'] + a * prob['q_goal']
        return Params(**common, mean=mean0, stdev=hp.get("stdev", BASE_U_STDEV))
    return Params(**common,
                  goal_frac=hp.get("goal_frac", GOAL_FRAC),
                  gauss_frac=hp.get("gauss_frac", GAUSS_FRAC),
                  gauss_std=hp.get("gauss_std", GAUSS_STD))


def eval_method(method, groups, robot, runner, hp, seed=SEED):
    """Success rate over `groups`, a list of same-dataset problem lists."""
    temperature = hp.get("mppi_temperature", MPPI_TEMPERATURE)
    num_elite = _num_elite(hp.get("cem_elite_frac", CEM_ELITE_FRAC))
    succ = []
    for problems in groups:
        if not problems:
            continue
        thetas = [make_theta(method, p, robot, hp) for p in problems]
        theta_b = jax.tree.map(lambda *xs: jnp.stack(xs), *thetas)
        q0_b = jnp.stack([p['q_start'] for p in problems])
        keys = jax.random.split(jax.random.key(seed), len(problems))
        collided, dist = runner(keys, q0_b, theta_b, temperature, num_elite)
        succ.extend((~np.asarray(collided)) & (np.asarray(dist) < 0.5))
    return float(np.mean(succ))


def _num_elite(elite_frac):
    return max(1, int(elite_frac * PARTICLES))


def suggest_params(trial, method):
    """Search space for `method`: only its own proposal / update / resampling
    knobs. The task (cost, dynamics, horizon, knots) is fixed for everyone."""
    hp = {}
    if method in PERTURB:
        hp["stdev"] = trial.suggest_float("stdev", 0.05, 0.6)
    else:                                        # custom, object-informed proposal
        hp["goal_frac"] = trial.suggest_float("goal_frac", 0.05, 0.4)
        hp["gauss_frac"] = trial.suggest_float("gauss_frac", 0.3, 0.8)
        hp["gauss_std"] = trial.suggest_float("gauss_std", 0.05, 0.5)
    if method in ("mppi", "biased_mppi"):
        hp["mppi_temperature"] = trial.suggest_float("mppi_temperature", 0.01, 100.0, log=True)
    if method in ("cem", "biased_cem"):
        # Lower bound reaches num_elite=1 (argmin, i.e. ps_update); push-T's CEM
        # went straight to k=1 once allowed, so a higher bound may pin it away
        # from its optimum.
        hp["cem_elite_frac"] = trial.suggest_float("cem_elite_frac", 1.0 / PARTICLES, 0.3, log=True)
    if method == "sir":
        hp["ess_thresh"] = trial.suggest_float("ess_thresh", 0.1, 0.7)
    return hp


def objective(trial, method, groups, robot, runner_cache):
    hp = suggest_params(trial, method)
    # sir's ess_thresh is a static PFMP field, so each distinct value needs its
    # own controller; cache so repeated values do not recompile.
    ess = round(float(hp.get("ess_thresh", 0.0)), 4)
    if ess not in runner_cache:
        runner_cache[ess] = make_batched_runner(method, robot, ess_thresh=ess)
    return eval_method(method, groups, robot, runner_cache[ess], hp)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=80)
    parser.add_argument("--tune-per-dataset", type=int, default=TUNE_PER_DATASET,
                        help="problems per dataset used for tuning; the rest of the "
                             "100 are held out to validate the tuned config")
    parser.add_argument("--methods", type=str, nargs="+", default=list(ALL_METHODS))
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    robot = load_robot(URDF)
    # Grouped per dataset, not flattened: obstacle counts differ between datasets,
    # so each group is vmapped separately (see make_batched_runner).
    tune_groups, val_groups = [], []
    for ds in DATASETS:
        all_p = load_mbm_dataset(ds, max_problems=PROBLEMS_PER_DATASET)
        tune_groups.append(all_p[:args.tune_per_dataset])
        val_groups.append(all_p[args.tune_per_dataset:])
    n_tune = sum(len(g) for g in tune_groups)
    n_val = sum(len(g) for g in val_groups)
    print(f"{n_tune} tuning problems, {n_val} held out, across {len(DATASETS)} datasets.")
    if not n_val:
        print("WARNING: no held-out problems; tuned numbers will be in-sample.")

    out_path = args.out or str(OUTPUT_DIR / "mbm_tune.json")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_best = {}
    if Path(out_path).exists():
        all_best = json.load(open(out_path))

    for method in args.methods:
        print(f"\n=== Tuning {method} ({args.trials} trials) ===")
        runner_cache = {}
        study = optuna.create_study(direction="maximize",
                                    sampler=optuna.samplers.TPESampler(seed=SEED))
        study.optimize(lambda t: objective(t, method, tune_groups, robot, runner_cache),
                       n_trials=args.trials, show_progress_bar=True)

        # Held-out score for the winner. The tuning number is in-sample over a
        # fixed problem set, so it is optimistic; this is what generalises.
        bp = study.best_params
        val_value = None
        if n_val:
            ess = round(float(bp.get("ess_thresh", 0.0)), 4)
            best_runner = runner_cache.get(ess) or make_batched_runner(
                method, robot, ess_thresh=ess)
            val_value = eval_method(method, val_groups, robot, best_runner, bp)
        gap = None if val_value is None else study.best_value - val_value
        print(f"  {method}: tuning {study.best_value*100:5.1f}% "
              + ("held-out    n/a" if val_value is None
                 else f"held-out {val_value*100:5.1f}%  (gap {gap*100:+.1f} pts)")
              + f"  params {bp}")
        if gap is not None and gap > 0.15:
            print("    WARNING: large tuning/held-out gap — likely overfit to the "
                  "tuning problems. Report the held-out number.")
        all_best[method] = {"best_value": study.best_value, "best_params": bp,
                            "validation_value": val_value,
                            "n_tune": n_tune, "n_validate": n_val}
        with open(out_path, "w") as f:
            json.dump(all_best, f, indent=2)

    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
