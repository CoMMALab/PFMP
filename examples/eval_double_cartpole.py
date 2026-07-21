"""
Ablation: particle count vs performance for Double CartPole swingup.
Compares learned proposal (mjlab checkpoint) vs linear Gaussian baseline.

Usage:
    python examples/eval_double_cartpole.py --checkpoint logs/rsl_rl/double_cartpole/<run>/model_*.pt
"""

import argparse
import numpy as np
import jax.numpy as jnp
import mujoco
import mujoco.mjx as mjx

import pfmp
from pfmp.run_batch import run_batch
from hydrax.tasks.double_cart_pole import DoubleCartPole

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from mjlab_double_cartpole import load_actor_params, make_proposal

HORIZON   = 1.0
NUM_KNOTS = 4
NOISE     = 0.06
FREQUENCY = 50.0
STEPS     = 500
N_SEEDS   = 50
METHOD    = "sir"

task      = DoubleCartPole()
mj_model  = task.mj_model
mjx_model = mjx.put_model(mj_model)
running_cost, terminal_cost = pfmp.mjx_hydrax_cost(task)


def sample_problem(seed: int):
    rng = np.random.default_rng(seed)
    d = mujoco.MjData(mj_model)
    d.qpos[0] = rng.uniform(-1.0, 1.0)
    d.qpos[1] = rng.uniform(-np.pi, np.pi)
    d.qpos[2] = rng.uniform(-np.pi, np.pi)
    d.qvel[:] = rng.uniform(-0.1, 0.1, size=3)
    mujoco.mj_forward(mj_model, d)
    theta = pfmp.GenericMJXTaskParams(
        mjx_model=mjx_model,
        mean=jnp.zeros((NUM_KNOTS, mj_model.nu)),
        noise_level=NOISE,
        actor_params=None,
    )
    return d, theta


def evaluate(proposal_type: str, particles: int, actor_params=None, method: str = "sir", n_seeds: int = N_SEEDS) -> tuple[float, float]:
    if proposal_type == "learned":
        init_fn, proposal_fn = make_proposal(deterministic=False, blend=0.0)
    elif proposal_type == "blend":
        init_fn, proposal_fn = make_proposal(deterministic=False, blend=0.5)
    else:
        init_fn, proposal_fn = pfmp.mjx_linear_gaussian_proposal(task, HORIZON, NUM_KNOTS)

    ctrl = pfmp.PFMP(
        pfmp.mjx_physics, running_cost, terminal_cost,
        init_fn, proposal_fn,
        horizon=HORIZON, num_knots=NUM_KNOTS, particles=particles,
        dt=mj_model.opt.timestep, ess_resample_thresh=0.3,
    )
    theta = pfmp.GenericMJXTaskParams(
        mjx_model=mjx_model,
        mean=jnp.zeros((NUM_KNOTS, mj_model.nu)),
        noise_level=NOISE,
        actor_params=actor_params,
    )

    def _sample(seed):
        d, th = sample_problem(seed)
        th = th._replace(actor_params=actor_params)
        return d, th

    traj = run_batch(
        ctrl, theta, mj_model, mujoco.MjData(mj_model),
        frequency=FREQUENCY, algorithm=method,
        max_step=STEPS, num=n_seeds,
        sample_problem=_sample,
    )

    half = traj.cost.shape[1] // 2
    costs = np.nan_to_num(traj.cost[:, half:], nan=1.0)
    mean_cost = float(np.mean(costs))
    std_cost  = float(np.std(np.mean(costs, axis=1)))
    return mean_cost, std_cost


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--particles", type=int, nargs="+", default=[1, 16, 128])
    parser.add_argument("--seeds", type=int, default=N_SEEDS)
    parser.add_argument("--method", type=str, default="sir", choices=["sir", "sis"])
    args = parser.parse_args()
    METHOD = args.method
    N_SEEDS = args.seeds
    actor_params = load_actor_params(args.checkpoint)

    results = []
    for proposal_type in ["linear", "learned", "blend"]:
        ap = actor_params if proposal_type in ("learned", "blend") else None
        for particles in args.particles:
            print(f"\n=== {proposal_type} N={particles} ===")
            mean, std = evaluate(proposal_type, particles, ap, method=args.method, n_seeds=args.seeds)
            results.append((proposal_type, particles, mean, std))
            print(f"mean_cost={mean:.4f}  std={std:.4f}")

    print(f"\n{'Proposal':10s} {'Particles':>10s} {'Mean Cost':>10s} {'Std':>8s}")
    print("-" * 44)
    for proposal_type, particles, mean, std in results:
        print(f"{proposal_type:10s} {particles:>10d} {mean:>10.4f} {std:>8.4f}")
