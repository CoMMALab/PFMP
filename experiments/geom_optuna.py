"""Hyperparameter search for the tasks in `geom_runner.py`.

Evaluates through `run_algorithm`, so a tuned parameter means the same thing
here as in the run that reports the number. Tuning and validation draw from
disjoint problems (`problem_offset`) and disjoint rollout keys, so the reported
gap is generalisation. Writes `experiments/checkpoints/<task>_tune.json`.

    python experiments/geom_optuna.py experiments/configs/drive.yaml --methods mppi cem
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from geom_runner import (load_task, run_algorithm, build_simulator,
                         TRACED_HP, ALGORITHMS)
from pfmp import method_decomp

# Tuned params are inputs to a run, not figures: they live beside the
# configs that reference them.
OUTPUT_DIR = Path(__file__).parent / "checkpoints"

# Per-task search setup, inlined rather than read from the config: these are
# properties of the search, not of the evaluation, and putting them in the yaml
# meant tuning inherited the sweep's own settings.
#
#   objective: what the search maximises, and how it reduces across problems.
#   tuning:    config overrides for the search only.
TASK_SETUP = {
    "push_tee": dict(metric="success", minimise=False, agg="mean", ceiling=1.0,
                  overrides={"distance": 4.0, "num": 128, "num_samples": 256}),
    "mbm":   dict(metric="success", minimise=False, agg="mean", ceiling=1.0,
                  overrides={}),
    "drive": dict(metric="cost", minimise=True, agg="median", ceiling=None,
                  overrides={}),
}

# Proposal-mixture knobs, which only MBM puts on theta.
MBM_PROPOSAL = ("goal_frac", "gauss_frac", "gauss_std")


def suggest(trial, method, particles, task):
    """The search space, per method -- only knobs the method actually reads.

    A dead parameter wastes trials and makes a search look converged when it is
    not (DIAL used to search `mppi_temperature`, which it never touches).
    """
    proposal, update, rho = method_decomp(method)
    hp = {}
    # Every method reads `stdev`: the perturbation methods as their noise width,
    # the custom-proposal ones through their own `_sigma_at`. Omitting it for
    # SIS left that method with an empty search space, so every trial tied.
    hp["stdev"] = trial.suggest_float("stdev", 0.01, 0.6, log=True)
    if task == "mbm" and proposal == "informed":
        hp["goal_frac"] = trial.suggest_float("goal_frac", 0.05, 0.4)
        hp["gauss_frac"] = trial.suggest_float("gauss_frac", 0.3, 0.8)
        hp["gauss_std"] = trial.suggest_float("gauss_std", 0.05, 0.5)
    if update == "mean":
        hp["mppi_temperature"] = trial.suggest_float(
            "mppi_temperature", 0.01, 1e4, log=True)
    if update == "topk":
        # Absolute count, not a fraction: `int(frac * N)` is a step function of
        # the particle budget, so a tuned fraction does not transfer across a
        # particle-count sweep (it silently changed k from 1 to 2 between
        # N=256 and N=512 on push-T, taking success from 100% to 12.5%).
        hp["cem_num_elites"] = trial.suggest_int(
            "cem_num_elites", 1, max(2, particles // 4), log=True)
        hp["cem_stdev_floor"] = trial.suggest_float(
            "cem_stdev_floor", 1e-3, 2e-1, log=True)
    # rho is None exactly when the name asks for "the task's ess_thresh"
    # (`sir`, or any `+r` with no number), which is when it is ours to tune.
    # Keying on the literal name `sir` meant `sis+r` and `mppi+r` silently
    # skipped it.
    if rho is None:
        hp["ess_thresh"] = trial.suggest_float("ess_thresh", 0.1, 0.7)
    if update == "sample":
        # Shares MPPI's temperature semantics; searched on the same scale.
        hp["sample_temperature"] = trial.suggest_float(
            "sample_temperature", 0.01, 1e4, log=True)
    if update == "diffuse":
        hp["dial_temp_sample"] = trial.suggest_float(
            "dial_temp_sample", 1e-2, 2.0, log=True)
        # `dial_n_inner` is deliberately NOT searched: it is the number of
        # annealing passes, so each extra level is another full batch of rollouts.
        # Tuning it would let DIAL buy accuracy with compute the other methods
        # are not given, breaking the matched-budget comparison.
    return hp


class Evaluator:
    """Scores a method on a fixed problem slice.

    Holds the built task and one compiled simulator. Searching only traced
    knobs compiles once; a static knob (ess_thresh, elite count) recompiles on
    change, but never keeps more than one program alive.
    """

    def __init__(self, config, num, offset, seed):
        self.cfg = dict(config, num=num, problem_offset=offset)
        self.seed = seed
        self.task, self.group_size = load_task(self.cfg["task"], self.cfg)
        self._sim = self._sim_key = None

    def _simulator(self, method, hp):
        key = (method, tuple(sorted((k, v) for k, v in hp.items()
                                    if k not in TRACED_HP
                                    and isinstance(v, (int, float, bool)))))
        if key != self._sim_key:
            self._sim = build_simulator(self.task, method,
                                        self.cfg["num_samples"], hp)
            self._sim_key = key
        return self._sim

    def __call__(self, method, hp):
        full = dict(self.task.hp_defaults)
        full.update(self.cfg.get("hyperparams", {}))
        full.update(hp)
        return run_algorithm(self.task, method, num=self.cfg["num"],
                             particles=self.cfg["num_samples"], hp=full,
                             seed=self.seed, group_size=self.group_size,
                             sim=self._simulator(method, full))


def score(res, setup):
    """Scalar Optuna maximises: negated when the task minimises."""
    v = np.asarray(res[setup["metric"]], dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return -np.inf
    agg = np.median(v) if setup["agg"] == "median" else np.mean(v)
    return float(-agg if setup["minimise"] else agg)


def tune(config, method, trials, tune_eval, val_eval, setup, out_path,
         early_stop=True):
    import optuna
    from tqdm import tqdm

    particles = config["num_samples"]
    study = optuna.create_study(direction="maximize")

    bar = tqdm(total=trials, desc=method, file=sys.stderr)

    def objective_fn(trial):
        hp = suggest(trial, method, particles, config['task'])
        return score(tune_eval(method, hp), setup)

    def on_done(study, trial):
        bar.update(1)
        if study.best_trial.number != trial.number:
            return
        # Score the held-out set on every new best, not once at the end: a gap
        # that only appears in the final line hides whether the search was
        # generalising throughout or overfitting late.
        val = score(val_eval(method, study.best_params), setup)
        tqdm.write(f"  trial {trial.number:>4}: tuning {study.best_value:>9.4f}"
                   f"   held-out {val:>9.4f}   (gap {study.best_value - val:+.4f})",
                   file=sys.stderr)

        # Stop once the objective is at its ceiling: further trials can only
        # tie, so they cost time and the winner becomes whichever config
        # happened to reach it first. A run that ends this way is reporting a
        # saturated task, not a converged search, and says so.
        ceiling = setup.get("ceiling")
        if early_stop and ceiling is not None and study.best_value >= ceiling:
            tqdm.write(f"  ! {method} hit the ceiling ({ceiling:g}) at trial "
                       f"{trial.number}; stopping. The task cannot rank configs "
                       f"above this, so the result is one of possibly many ties.",
                       file=sys.stderr)
            study.stop()

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(objective_fn, n_trials=trials, callbacks=[on_done])
    bar.close()

    # Re-score the winner on problems and keys it never saw.
    val = score(val_eval(method, study.best_params), setup)
    summarise(study, method, val)

    blob = json.loads(out_path.read_text()) if out_path.exists() else {}
    blob[method] = {"best_params": study.best_params,
                    "best_value": study.best_value,
                    "validation_value": val,
                    "trials": len(study.trials),
                    "requested_trials": trials,
                    "saturated": bool(setup.get("ceiling") is not None
                                      and study.best_value >= setup["ceiling"]),
                    "objective": setup}
    out_path.write_text(json.dumps(blob, indent=2))
    return blob[method]


def summarise(study, method, val):
    """Run statistics. A single best value says nothing about whether the search
    converged, how much of the space was viable, or whether the best trial is an
    outlier the holdout will not reproduce."""
    vals = np.array([t.value for t in study.trials
                     if t.value is not None and np.isfinite(t.value)])
    n_bad = len(study.trials) - len(vals)
    print(f"  {method}: tuning {study.best_value:.4f}  held-out {val:.4f}  "
          f"(gap {study.best_value - val:+.4f})")
    print(f"    {len(study.trials)} trials ({n_bad} non-finite)  "
          f"median {np.median(vals):.4f}  IQR [{np.percentile(vals, 25):.4f}, "
          f"{np.percentile(vals, 75):.4f}]  worst {vals.min():.4f}")
    print(f"    best params: {study.best_params}")
    # A parameter pinned at a search bound means the range is wrong, not that
    # the optimum was found.
    for name, dist in study.best_trial.distributions.items():
        v = study.best_params[name]
        lo, hi = getattr(dist, "low", None), getattr(dist, "high", None)
        if lo is not None and (np.isclose(v, lo) or np.isclose(v, hi)):
            print(f"    ! {name}={v:g} is at a search bound [{lo:g}, {hi:g}]")



def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("config")
    ap.add_argument("--methods", nargs="+", default=None,
                    help=f"default: the config's algorithms. one of {ALGORITHMS}")
    ap.add_argument("--trials", type=int, default=200)
    ap.add_argument("--num", type=int, default=None,
                    help="total problems (tuning + held-out). Overrides the "
                         "task's search setup; lower it to trade precision for "
                         "wall-clock without changing the tuning difficulty.")
    ap.add_argument("--tune-num", type=int, default=None,
                    help="problems used for tuning (default: half of `num`)")
    ap.add_argument("--no-early-stop", action="store_true",
                    help="keep searching after the objective hits its ceiling")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    config = yaml.safe_load(open(args.config))
    algorithms = config.pop("algorithms", [])
    methods = args.methods or algorithms
    config.pop("sweep", None)          # tuning is at one point, not an axis
    config["tuned_params"] = None      # never seed a search with its own output

    setup = TASK_SETUP[config["task"]]
    config.update(setup["overrides"])
    if args.num:
        config["num"] = args.num
    # The task is the authority on how many problems exist; only ask the config
    # when it wants fewer (push-T and drive generate on demand, MBM does not).
    n = config.get("num") or load_task(config["task"], config)[0].num_problems
    n_tune = args.tune_num or n // 2
    n_val = n - n_tune
    if n_val < 1:
        raise SystemExit(f"--tune-num {n_tune} leaves nothing to validate on")

    # Disjoint problems AND disjoint rollout keys: sharing either would let a
    # trial be scored on what it was fit to.
    tune_eval = Evaluator(config, n_tune, 0, config.get("seed", 0))
    val_eval = Evaluator(config, n_val, n_tune, config.get("seed", 0) + 100_000)
    print(f"{n_tune} tuning problems, {n_val} held out; "
          f"{'minimise' if setup['minimise'] else 'maximise'} "
          f"{setup['agg']} {setup['metric']}")

    out = Path(args.out) if args.out else OUTPUT_DIR / f"{config['task']}_tune.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    for method in methods:
        print(f"\n=== Tuning {method} ({args.trials} trials) ===")
        tune(config, method, args.trials, tune_eval, val_eval, setup, out,
             early_stop=not args.no_early_stop)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
