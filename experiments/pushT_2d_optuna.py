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
from pfmp.algorithm import ps_update, mppi_update, cem_update
from pfmp.interpolate import hermite_spline

# 2d_pushT.py starts with a digit so we can't import it directly
_spec = importlib.util.spec_from_file_location("pushT", ROOT / "examples/2d_pushT.py")
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
dial_sigma         = _pushT.dial_sigma
STEPS_PER_REPLAN   = _pushT.STEPS_PER_REPLAN
SUCCESS_POS_THRESH = _pushT.SUCCESS_POS_THRESH
SUCCESS_ANGLE_THRESH = _pushT.SUCCESS_ANGLE_THRESH

OUTPUT_DIR = Path(__file__).parent.parent / "figures"
TUNE_DISTANCE = 4.0    # 1 obstacle — representative difficulty, fast to evaluate
TUNE_BATCH    = 32     # scenarios per trial
TUNE_SEED     = 42


def _run_batch(method, scenarios, horizon, num_knots, particles, ess_thresh,
               pos_weight, angle_weight, u_stdev, approach_weight=0.0,
               mppi_temperature=0.5,
               cem_num_elites=10,
               dial_n_inner=5, dial_beta1=1.0, dial_beta2=1.0):
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
    tk = ctrl.tk()

    def mean_from_state(x):
        r_xy   = x[6:8]
        t_xy   = x[0:2]
        alphas = jnp.linspace(0.0, 1.0, num_knots)
        return (1.0 - alphas[:, None]) * r_xy + alphas[:, None] * t_xy

    successes = []
    for seed_i, (x0, theta_base, ws_min, ws_max, _) in enumerate(scenarios):
        theta = theta_base._replace(
            pos_weight=pos_weight,
            angle_weight=angle_weight,
            u_stdev=u_stdev,
            approach_weight=approach_weight,
            step_len=horizon / (num_knots - 1),
            du_stdev=jnp.full((num_knots, 2), u_stdev),
        )
        if is_baseline or use_informed:
            theta = theta._replace(mean=mean_from_state(x0))

        key = jax.random.key(TUNE_SEED + seed_i)
        x = x0

        for _ in range(n_replans):
            key, subkey = jax.random.split(key)

            if method == 'dial':
                for i_inner in range(dial_n_inner, 0, -1):
                    sigmas = jnp.array([
                        dial_sigma(i_inner, h, u_stdev, dial_n_inner, num_knots,
                                   dial_beta1, dial_beta2)
                        for h in range(num_knots)
                    ])
                    key, subkey_i = jax.random.split(key)
                    theta_i = theta._replace(u_stdev=float(sigmas.mean()))
                    trajs_i, costs_i = ctrl.run_sis(subkey_i, x, theta_i)
                    total_i = jnp.sum(costs_i, axis=1)
                    temp = float(jnp.std(jnp.where(jnp.isfinite(total_i), total_i, 0.0))) * mppi_temperature
                    theta, _ = mppi_update(theta, trajs_i.transpose(1, 0, 2), costs_i.T, max(temp, 1e-6))
            else:
                trajs, costs = ctrl.run_sis(subkey, x, theta)
                if method in ('mppi', 'biased_mppi'):
                    temp  = mppi_temperature  # fixed inverse-temperature (canonical MPPI), not adaptively scaled by cost std
                    theta, _ = mppi_update(theta, trajs.transpose(1, 0, 2), costs.T, max(temp, 1e-6))
                elif method == 'cem':
                    theta, _ = cem_update(theta, trajs, jnp.sum(costs, axis=1),
                                          elite_frac=cem_num_elites / particles)
                else:  # ps / pfmp
                    theta, _ = ps_update(theta, trajs, costs)

            tq = jnp.arange(STEPS_PER_REPLAN) * dt
            us = hermite_spline(tq, tk, theta.mean)

            def phys_step(x, args):
                u, t = args
                xn, _ = dynamics(x, u, theta, t)
                return xn, xn

            x, _ = jax.lax.scan(phys_step, x, (us, tq))
            new_tk = tk + STEPS_PER_REPLAN * dt
            theta  = theta._replace(mean=hermite_spline(new_tk, tk, theta.mean))

        successes.append(_is_success(np.array(x), np.array(theta.goal_xy), float(theta.goal_angle)))

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
FIXED_PARTICLES = 2048


def make_objective(method, scenarios):
    def objective(trial):
        pos_weight      = trial.suggest_float("pos_weight",      1.0,  50.0)
        angle_weight    = trial.suggest_float("angle_weight",    0.1,  10.0)
        u_stdev         = trial.suggest_float("u_stdev",         0.01, 0.2)
        approach_weight = trial.suggest_float("approach_weight", 0.0,  5.0)

        kwargs = dict(
            horizon=FIXED_HORIZON, num_knots=FIXED_NUM_KNOTS, particles=FIXED_PARTICLES,
            ess_thresh=0.25,
            pos_weight=pos_weight, angle_weight=angle_weight, u_stdev=u_stdev,
            approach_weight=approach_weight,
        )

        if method in ('mppi', 'biased_mppi'):
            kwargs['mppi_temperature'] = trial.suggest_float("mppi_temperature", 0.01, 100.0, log=True)
        elif method == 'cem':
            kwargs['cem_num_elites'] = trial.suggest_int("cem_num_elites", 5, 50)
        elif method == 'dial':
            kwargs['mppi_temperature'] = trial.suggest_float("mppi_temperature", 0.05, 5.0, log=True)
            kwargs['dial_n_inner']     = 5   # fixed to avoid recompilation
            kwargs['dial_beta1']       = trial.suggest_float("dial_beta1",       0.1,  5.0)
            kwargs['dial_beta2']       = trial.suggest_float("dial_beta2",       0.1,  5.0)

        return _run_batch(method, scenarios, **kwargs)

    return objective


def tune_method(method, trials, distance=TUNE_DISTANCE, batch=TUNE_BATCH):
    print(f"\n{'='*60}")
    print(f"  Tuning {method.upper()}  ({trials} trials, distance={distance}, batch={batch})")
    print(f"{'='*60}")

    scenarios = _build_scenarios(batch, distance)
    print("JIT warm-up…", end=" ", flush=True)
    _warmup(scenarios, horizon=FIXED_HORIZON, num_knots=FIXED_NUM_KNOTS, particles=FIXED_PARTICLES)
    print("done")

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=TUNE_SEED),
    )
    study.optimize(
        make_objective(method, scenarios),
        n_trials=trials,
        show_progress_bar=True,
    )

    best = study.best_trial
    print(f"\nBest success rate: {best.value*100:.1f}%")
    print("Best params:")
    for k, v in best.params.items():
        print(f"  {k}: {v}")

    out = OUTPUT_DIR / f"pushT_optuna_{method}.json"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump({"method": method, "best_value": best.value,
                   "best_params": best.params, "distance": distance,
                   "batch": batch}, f, indent=2)
    print(f"Saved to {out}")
    return best.params


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method",   choices=["mppi", "biased_mppi", "cem", "dial", "ps", "all"], default="all")
    parser.add_argument("--trials",   type=int,   default=200)
    parser.add_argument("--distance", type=float, default=TUNE_DISTANCE)
    parser.add_argument("--batch",    type=int,   default=TUNE_BATCH)
    args = parser.parse_args()

    methods = ["mppi", "biased_mppi", "cem", "dial", "ps"] if args.method == "all" else [args.method]
    for m in methods:
        tune_method(m, args.trials, args.distance, args.batch)
