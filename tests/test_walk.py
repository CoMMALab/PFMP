import argparse

import mujoco
import jax
import jax.numpy as jnp
import pfmp

from mujoco import mjx
from hydrax.task_base import Task
from hydrax.tasks.walker import Walker
from hydrax.algs import PredictiveSampling, MPPI
from hydrax.simulation.deterministic import run_interactive
from typing import NamedTuple


task = Walker()

# Define the model used for simulation
mj_model = task.mj_model
mj_data = mujoco.MjData(mj_model)

# Parse command-line arguments
parser = argparse.ArgumentParser(
    description="Run an interactive simulation of the inverted cart-pole task."
)
subparsers = parser.add_subparsers(
    dest="algorithm", help="Sampling algorithm (choose one)", required=True
)
subparsers.add_parser("ps", help="Predictive Sampling")
subparsers.add_parser("mppi", help="Model Predictive Path Integral")
subparsers.add_parser("sis", help="Particle Filter Motion Planning (SIS)")
subparsers.add_parser("sir", help="Particle Filter Motion Planning (SIR)")
args = parser.parse_args()

if args.algorithm == "ps":
    ctrl = PredictiveSampling(
        task,
        num_samples=128,
        noise_level=0.5,
        plan_horizon=0.6,
        spline_type="zero",
        num_knots=5,
    )
    theta = None
if args.algorithm:
    horizon = 0.6
    num_knots = 5

    def dynamics(x, j, th, _t):
        x = mjx.step(th.mjx_model, x.replace(ctrl=j))
        return x, j

    # Cost from Hydrax
    running_cost = lambda x, j, th, _t: task.running_cost(x, j) * 0.01
    terminal_cost = lambda x, th: task.terminal_cost(x)

    def init(key, x, th):
        u = jax.random.normal(key, th.mjx_model.nu) * th.noise_level + th.mean[0]
        u = jnp.clip(u, task.u_min, task.u_max)
        return u

    # Linear Proposal
    def proposal(key, x: mjx.Data, u: jax.Array, th, t: float) -> jax.Array:
        step_len = horizon / (num_knots - 1)
        idx = (t / step_len).astype(int) + 1
        u = jax.random.normal(key, th.mjx_model.nu) * th.noise_level + th.mean[idx]
        u = jnp.clip(u, task.u_min, task.u_max)
        return u

    class DroneParams(NamedTuple):
        mjx_model: mjx.Model
        mean: jax.Array
        noise_level: float

    theta = DroneParams(
        mjx_model = mjx.put_model(mj_model),
        mean = jnp.zeros((num_knots,mj_model.nu)),
        noise_level = 0.5
    )

    ctrl = pfmp.PFMP(
        dynamics, running_cost, terminal_cost, init, proposal,
        horizon=horizon, num_knots=num_knots, particles=128, dt=mj_model.opt.timestep,
        ess_resample_thresh=0.5
        )

pfmp.run_interactive(
    ctrl,
    theta,
    mj_model,
    mj_data,
    frequency=50,
    method=args.algorithm,
    fixed_camera_id=None,
    record_video=False,
    record_traj=False,
    max_step=1500
)