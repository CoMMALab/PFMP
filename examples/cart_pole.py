import argparse

import jax
import jax.numpy as jnp
import mujoco

from hydrax.algs import PredictiveSampling, MPPI
from hydrax.tasks.cart_pole import CartPole
from pfmp.interpolate import Jet2
from typing import NamedTuple
import pfmp
import mujoco.mjx as mjx


task = CartPole()

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
subparsers.add_parser("sir", help="Particle Filter Motion Planning")
subparsers.add_parser("sis", help="Particle Filter Motion Planning")
args = parser.parse_args()

if args.algorithm == "ps":
    ctrl = PredictiveSampling(
        task,
        num_samples = 128,
        noise_level = 0.3,
        spline_type = "cubic",
        plan_horizon = 1.0,
        num_knots = 4
    )
    theta = None
elif args.algorithm == "mppi":
    ctrl = MPPI(
        task,
        num_samples=128,
        noise_level=0.3,
        temperature=0.1,
        spline_type="cubic",
        plan_horizon=1.0,
        num_knots=4,
    )
    theta = None
elif args.algorithm == "sir" or args.algorithm == "sis":
    horizon = 1.0
    num_knots = 4

    class CartPoleParams(NamedTuple):
        mjx_model: mjx.Model
        mean: jax.Array
        coarse_control: float

    u_min = jnp.where(
        mj_model.actuator_ctrllimited,
        mj_model.actuator_ctrlrange[:, 0],
        -jnp.inf,
    )
    u_max = jnp.where(
        mj_model.actuator_ctrllimited,
        mj_model.actuator_ctrlrange[:, 1],
        jnp.inf,
    )

    # Mujoco XLA dynamics
    def dynamics(x, j, th, _t):
        x = mjx.step(th.mjx_model, x.replace(ctrl=jnp.array([j])))
        return x, None

    # Cost from Hydrax
    running_cost = lambda x, u, th, _t: task.running_cost(x, u) * th.mjx_model.opt.timestep
    terminal_cost = lambda x, th: task.terminal_cost(x)

    def init(key, x, th):
        u = jax.random.normal(key) * th.coarse_control + th.mean[0]
        u = jnp.clip(u, u_min, u_max)
        return u

    # Linear Proposal
    def proposal(key, x: mjx.Data, u: jax.Array, th: CartPoleParams, t: float) -> jax.Array:
        step_len = horizon / (num_knots - 1)
        idx = (jnp.round(t / step_len)).astype(int) + 1
        u = jax.random.normal(key) * th.coarse_control + th.mean[idx]
        u = jnp.clip(u, u_min, u_max)
        return u

    theta = CartPoleParams(
        mjx_model = mjx.put_model(mj_model),
        mean = jnp.zeros((num_knots,mj_model.nu)),
        coarse_control = 0.3
    )

    ctrl = pfmp.PFMP(
        dynamics, running_cost, terminal_cost, init, proposal,
        horizon=horizon, num_knots=num_knots, particles=128, dt=mj_model.opt.timestep,
        ess_resample_thresh=0.3
        )

# Run the interactive simulation
pfmp.run_interactive(
    ctrl,
    theta,
    mj_model,
    mj_data,
    frequency=50,
    fixed_camera_id=0,
    method=args.algorithm,
    record_video=True,
    record_traj=False,
    max_step=1000
)