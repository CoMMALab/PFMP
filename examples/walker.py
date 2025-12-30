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
if args.algorithm == "sir" or args.algorithm == "sis":
    horizon = 0.6
    num_knots = 5

    running_cost, terminal_cost = pfmp.mjx_hydrax_cost(task)
    init, proposal = pfmp.mjx_linear_gaussian_proposal(task, horizon, num_knots)
    ctrl = pfmp.PFMP(
        pfmp.mjx_physics,
        running_cost, terminal_cost,
        init, proposal,
        horizon=horizon,
        num_knots=num_knots,
        particles=128,
        dt=mj_model.opt.timestep,
        ess_resample_thresh=0.3
        )
    theta = pfmp.GenericMJXTaskParams(
        mjx_model = mjx.put_model(mj_model),
        mean = jnp.zeros((num_knots,mj_model.nu)),
        noise_level = 0.3
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