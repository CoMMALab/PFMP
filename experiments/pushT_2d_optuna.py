"""
Hyperparameter sweep for 2D push-T baselines (MPPI, CEM, DIAL-MPC) via Optuna.

Tunes each method independently to give a fair comparison against PFMP.
Best params are saved to figures/pushT_optuna_<method>.json.

Usage:
    python experiments/pushT_2d_optuna.py --method mppi --trials 500
    python experiments/pushT_2d_optuna.py --method cem  --trials 500
    python experiments/pushT_2d_optuna.py --method dial --trials 500
    python experiments/pushT_2d_optuna.py --method all  --trials 200
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

import importlib.util

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from pfmp import PFMP

# pushT.py starts with a digit so we can't import it directly
_spec = importlib.util.spec_from_file_location("pushT", ROOT / "examples/pushT.py")
_pushT = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_pushT)

dynamics           = _pushT.dynamics
running_cost       = _pushT.running_cost
terminal_cost      = _pushT.terminal_cost
init_control       = _pushT.init_control
proposal           = _pushT.proposal
perturbation_init  = _pushT.perturbation_init
perturbation_proposal = _pushT.perturbation_proposal
_build_scenario    = _pushT._build_scenario
_is_success        = _pushT._is_success
PushParams         = _pushT.PushParams
SEED               = _pushT.SEED
dial_step          = _pushT.dial_step
make_batched_simulate = _pushT.make_batched_simulate
STEPS_PER_REPLAN   = _pushT.STEPS_PER_REPLAN
SUCCESS_POS_THRESH = _pushT.SUCCESS_POS_THRESH
SUCCESS_ANGLE_THRESH = _pushT.SUCCESS_ANGLE_THRESH
# The task cost is fixed, not tuned: every method minimises the same J.
POS_WEIGHT      = _pushT.POS_WEIGHT
ANGLE_WEIGHT    = _pushT.ANGLE_WEIGHT
APPROACH_WEIGHT = _pushT.APPROACH_WEIGHT

OUTPUT_DIR = Path(__file__).parent.parent / "figures"
TUNE_DISTANCE = 4.0    # representative difficulty, fast to evaluate
TUNE_BATCH    = 64      # scenarios per trial
TUNE_SEED     = 42
# Held-out set for validating the tuned config. Disjoint from the tuning
# scenarios (which use TUNE_SEED + 0..batch-1) in both scenario seed and
# rollout key, so it measures generalization instead of re-reading the
# objective the search already optimized against.
VALIDATE_SEED  = 1000
VALIDATE_BATCH = 64


def _run_batch(method, scenarios, horizon, num_knots, particles, ess_thresh,
               pos_weight, angle_weight, stdev, approach_weight=0.0,
               mppi_temperature=0.5,
               cem_num_elites=10, cem_stdev_floor=0.01,
               dial_n_inner=2, dial_beta1=1.0, dial_beta2=1.0,
               dial_temp_sample=_pushT.DIAL_TEMP_SAMPLE,
               key_seed=TUNE_SEED):
    """Run one batch of scenarios for a given method and return success rate."""
    dt = 0.01
    n_replans = scenarios[0][4]

    is_baseline = method in ('mppi', 'dial', 'ps', 'cem')
    use_informed = method == 'biased_mppi'
    ctrl = PFMP(
        dynamics, running_cost, terminal_cost,
        perturbation_init     if is_baseline and not use_informed else init_control,
        perturbation_proposal if is_baseline and not use_informed else proposal,
        horizon=horizon,
        num_knots=num_knots,
        particles=particles,
        ess_resample_thresh=0.0 if is_baseline or use_informed else ess_thresh,
        dt=dt,
    )
    # Batch every scenario into one vmapped, jitted call rather than looping in
    # Python. The loop version issued a separate dispatch per (scenario, replan)
    # at only `particles` wide, which leaves the GPU mostly idle; at 32 scenarios
    # x 48 replans that is ~1500 launches per trial. This shares the exact
    # rollout the sweep uses (pushT.make_batched_simulate), so the tuner and
    # the evaluation can no longer diverge.
    n = len(scenarios)
    stdev0 = (jnp.full((num_knots, 2), stdev) if method == 'cem' else stdev)

    def prep(theta_base):
        return theta_base._replace(
            pos_weight=pos_weight,
            angle_weight=angle_weight,
            stdev=stdev0,
            approach_weight=approach_weight,
            step_len=horizon / (num_knots - 1),
        )

    x0_batch    = jnp.stack([s[0] for s in scenarios])
    theta_batch = jax.tree.map(lambda *xs: jnp.stack(xs),
                               *[prep(s[1]) for s in scenarios])
    # Same per-scenario key derivation the Python-loop version used, so results
    # stay comparable across this change.
    keys_batch  = jnp.stack([jax.random.key(key_seed + i) for i in range(n)])

    run = make_batched_simulate(method, ctrl, particles, n_replans,
                                mppi_temp=mppi_temperature,
                                cem_elites=cem_num_elites,
                                cem_stdev_floor=cem_stdev_floor,
                                dial_b1=dial_beta1, dial_b2=dial_beta2,
                                dial_temp_sample=dial_temp_sample,
                                dial_n_inner=dial_n_inner)
    final_states = np.array(run(keys_batch, x0_batch, theta_batch))

    goal_xys    = np.array(theta_batch.goal_xy)
    goal_angles = np.array(theta_batch.goal_angle)
    successes = [_is_success(final_states[i], goal_xys[i], float(goal_angles[i]))
                 for i in range(n)]
    return float(np.mean(successes))


def _build_scenarios(n, distance, seed=TUNE_SEED):
    return [_build_scenario(seed + i, distance, None, False) for i in range(n)]


def _warmup(scenarios, horizon, num_knots, particles):
    """JIT warm-up so trial timing is fair."""
    dt = 0.01
    ctrl = PFMP(
        dynamics, running_cost, terminal_cost,
        perturbation_init, perturbation_proposal,
        horizon=horizon, num_knots=num_knots, particles=particles,
        ess_resample_thresh=0.0, dt=dt,
    )
    x0, theta, *_ = scenarios[0]
    k = jax.random.key(0)
    _ = ctrl.run_sis(k, x0, theta)
    _ = ctrl.run_sis(k, x0, theta)


FIXED_HORIZON   = 2.5
FIXED_NUM_KNOTS = 6
FIXED_PARTICLES = 256


def _kwargs_from_params(method, params):
    """Map a params dict (optuna suggestions, or a study's best_params) onto
    _run_batch kwargs. Shared by the objective and the held-out validation so
    the validated configuration is exactly the tuned one."""
    kwargs = dict(
        horizon=FIXED_HORIZON, num_knots=FIXED_NUM_KNOTS, particles=FIXED_PARTICLES,
        ess_thresh=0.25,
        pos_weight=POS_WEIGHT, angle_weight=ANGLE_WEIGHT,
        approach_weight=APPROACH_WEIGHT,
        stdev=params["stdev"],
    )
    if method in ('mppi', 'biased_mppi'):
        kwargs['mppi_temperature'] = params["mppi_temperature"]
    elif method == 'cem':
        kwargs['cem_num_elites']  = params["cem_num_elites"]
        kwargs['cem_stdev_floor'] = params["cem_stdev_floor"]
    elif method == 'dial':
        # temp_sample divides the std-normalised returns before the softmax,
        # matching dial-mpc's reverse_once. It is DIAL's only concentration
        # knob, the counterpart to MPPI's temperature and CEM's elite count.
        kwargs['dial_n_inner']     = _pushT.DIAL_N_INNER  # static: unrolled loop
        kwargs['dial_temp_sample'] = params["dial_temp_sample"]
        kwargs['dial_beta1']       = params["dial_beta1"]
        kwargs['dial_beta2']       = params["dial_beta2"]
    return kwargs


def make_objective(method, scenarios):
    def objective(trial):
        # Suggest order/ranges are load-bearing: TPE is seeded, so changing them
        # changes the search. Keep them identical if editing.
        # Only algorithm hyperparameters are searched. The cost function is
        # part of the task definition and is identical across methods, so a
        # comparison isolates the proposal/update/resampling axes instead of
        # confounding them with four different objectives.
        params = {
            "stdev": trial.suggest_float("stdev", 0.01, 0.2),
        }

        if method in ('mppi', 'biased_mppi'):
            params['mppi_temperature'] = trial.suggest_float("mppi_temperature", 0.01, 100.0, log=True)
        elif method == 'cem':
            # Lower bound 1, not 5: MPPI's temperature can reach 0.01 (effectively
            # argmax), so clamping CEM's elite count above the degenerate case
            # would let one method express full concentration and not the other.
            # At k=1 the elite mean is the argmin, i.e. exactly ps_update, so if
            # the search selects 1 that is the finding (tuned CEM degenerates to
            # predictive sampling here) rather than a configuration error.
            params['cem_num_elites'] = trial.suggest_int("cem_num_elites", 1, 50)
            # hydrax's sigma_min. Without a floor the refit covariance
            # collapses and the proposal freezes mid-episode.
            params['cem_stdev_floor'] = trial.suggest_float("cem_stdev_floor", 1e-3, 2e-1, log=True)
        elif method == 'dial':
            # dial-mpc's temp_sample (reference default 0.06, configs 0.04-0.1).
            # Range spans well past both ends so the optimum is interior.
            params['dial_temp_sample'] = trial.suggest_float("dial_temp_sample", 1e-2, 2.0, log=True)
            params['dial_beta1']       = trial.suggest_float("dial_beta1",       0.1,  5.0)
            params['dial_beta2']       = trial.suggest_float("dial_beta2",       0.1,  5.0)

        return _run_batch(method, scenarios, **_kwargs_from_params(method, params))

    return objective


def _write_line(msg):
    """Print without corrupting optuna's tqdm progress bar.

    Goes to stderr, not stdout: that is where tqdm draws the bar, so the two
    interleave correctly and neither gets held in a block buffer when stdout is
    redirected to a file or a pipe.
    """
    try:
        from tqdm import tqdm
        tqdm.write(msg, file=sys.stderr)
    except ImportError:
        print(msg, file=sys.stderr, flush=True)


def _make_best_callback(method, val_scenarios):
    """Optuna callback: whenever the search finds a new best, re-score that
    config on the held-out scenarios and print both numbers side by side.

    Only fires on improvement, not every trial, so the extra cost is one
    held-out evaluation per new best rather than per trial.
    """
    state = {"best": float("-inf")}

    def callback(study, trial):
        if study.best_value <= state["best"]:
            return
        state["best"] = study.best_value
        if not val_scenarios:
            _write_line(f"  trial {trial.number:>4}: tuning {study.best_value*100:5.1f}%")
            return
        val = _run_batch(method, val_scenarios, key_seed=VALIDATE_SEED,
                         **_kwargs_from_params(method, study.best_params))
        gap = study.best_value - val
        _write_line(f"  trial {trial.number:>4}: tuning {study.best_value*100:5.1f}%"
                    f"   held-out {val*100:5.1f}%   (gap {gap*100:+.1f} pts)")

    return callback


def tune_method(method, trials, distance=TUNE_DISTANCE, batch=TUNE_BATCH,
                out_dir=None, validate_batch=VALIDATE_BATCH):
    print(f"\n{'='*60}")
    print(f"  Tuning {method.upper()}  ({trials} trials, distance={distance}, batch={batch})")
    print(f"{'='*60}")

    scenarios = _build_scenarios(batch, distance)
    val_scenarios = (_build_scenarios(validate_batch, distance, seed=VALIDATE_SEED)
                     if validate_batch and validate_batch > 0 else [])
    print("JIT warm-up…", end=" ", flush=True)
    _warmup(scenarios, horizon=FIXED_HORIZON, num_knots=FIXED_NUM_KNOTS, particles=FIXED_PARTICLES)
    print("done")
    if val_scenarios:
        print(f"Held-out set: {validate_batch} scenarios, seeds "
              f"{VALIDATE_SEED}..{VALIDATE_SEED + validate_batch - 1} "
              f"(re-scored on each new best)")

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=TUNE_SEED),
    )
    study.optimize(
        make_objective(method, scenarios),
        n_trials=trials,
        show_progress_bar=True,
        callbacks=[_make_best_callback(method, val_scenarios)],
    )

    best = study.best_trial

    # Final held-out score for the winning config. Tuning searches many configs
    # against one fixed set of scenarios AND one fixed set of rollout keys, so
    # the tuning score is an optimistic, in-sample number; this re-scores on
    # disjoint scenario seeds and disjoint rollout keys. Recomputed here rather
    # than reusing the callback's last value, since the final best may have been
    # reached on a trial that tied (and so did not trigger the callback).
    val_value = None
    if val_scenarios:
        print(f"\nFinal held-out score on {validate_batch} scenarios…",
              end=" ", flush=True)
        val_value = _run_batch(method, val_scenarios, key_seed=VALIDATE_SEED,
                               **_kwargs_from_params(method, best.params))
        print("done")

    # Tuning and held-out numbers side by side: the first is what the search
    # optimized (in-sample, optimistic), the second is what it generalizes to.
    if val_value is None:
        print(f"\nBest success rate: {best.value*100:5.1f}%  "
              f"(tuning, {batch} scenarios)   held-out: not run")
    else:
        gap = best.value - val_value
        print(f"\nBest success rate: {best.value*100:5.1f}%  (tuning, {batch} scenarios)"
              f"   held-out: {val_value*100:5.1f}%  ({validate_batch} scenarios,"
              f" gap {gap*100:+.1f} pts)")
        if gap > 0.15:
            print("  WARNING: large tuning/held-out gap — the search likely "
                  "overfit the tuning scenarios. Report the held-out number.")

    print("Best params:")
    for k, v in best.params.items():
        print(f"  {k}: {v}")

    out_dir = OUTPUT_DIR if out_dir is None else Path(out_dir)
    out = out_dir / f"pushT_optuna_{method}.json"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump({"method": method, "best_value": best.value,
                   "best_params": best.params, "distance": distance,
                   "batch": batch,
                   "validation_value": val_value,
                   "validation_batch": validate_batch,
                   "validation_seed": VALIDATE_SEED,
                   "tune_seed": TUNE_SEED}, f, indent=2)
    print(f"Saved to {out}")
    return best.params


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method",   choices=["mppi", "biased_mppi", "cem", "dial", "ps", "all"], default="all")
    parser.add_argument("--trials",   type=int,   default=200)
    parser.add_argument("--distance", type=float, default=TUNE_DISTANCE)
    parser.add_argument("--batch",    type=int,   default=TUNE_BATCH)
    parser.add_argument("--out-dir",  default=None,
                        help="where to write pushT_optuna_<method>.json "
                             "(default figures/). Point this at a scratch dir for "
                             "smoke tests, or a short run will overwrite real results.")
    parser.add_argument("--validate-batch", type=int, default=VALIDATE_BATCH,
                        help=f"held-out scenarios to re-score the tuned config on "
                             f"(default {VALIDATE_BATCH}; 0 disables). Uses disjoint "
                             f"scenario seeds and rollout keys from seed {VALIDATE_SEED}.")
    args = parser.parse_args()

    # biased_* deliberately inherit their base method's tuned params at eval time
    # (see the _base map in examples/pushT.py), so the sweep never reads a
    # pushT_optuna_biased_*.json. Tuning them here would just burn compute.
    methods = ["mppi", "cem", "dial", "ps"] if args.method == "all" else [args.method]
    for m in methods:
        tune_method(m, args.trials, args.distance, args.batch, out_dir=args.out_dir,
                    validate_batch=args.validate_batch)
