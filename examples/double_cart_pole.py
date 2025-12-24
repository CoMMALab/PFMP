import argparse

import jax
import jax.numpy as jnp
import mujoco

from hydrax.algs import PredictiveSampling, MPPI
from hydrax.simulation.deterministic import run_interactive
from hydrax.tasks.double_cart_pole import DoubleCartPole
from typing import NamedTuple

import pfmp
import mujoco.mjx as mjx


task = DoubleCartPole()

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
        num_samples=1024,
        noise_level=0.3,
        plan_horizon=1.0,
        spline_type="linear",
        num_knots=4,
    )
    theta = None
elif args.algorithm == "mppi":
    ctrl = MPPI(
        task,
        num_samples=1024,
        noise_level=0.3,
        temperature=0.1,
        spline_type="linear",
        plan_horizon=1.0,
        num_knots=4,
    )
    theta = None
elif args.algorithm == "sis" or args.algorithm == "sir":
    horizon = 1.0
    num_knots = 4

    class CartPoleParams(NamedTuple):
        mjx_model: mjx.Model
        mean: jax.Array
        coarse_control: float

    # Coarse-and-fine proposal
    # def proposal(key, x: mjx.Data, u: jax.Array, th: CartPoleParams, t: int) -> jax.Array:
    #     # theta = x.qpos[1]
    #     # near_upright = 1 + jnp.cos(theta) < th.fine_angle_bounds
    #     # scale = jnp.where(near_upright, th.fine_control, th.coarse_control)
    #     key1, key2 = jax.random.split(key)
    #     # print(th.mean.shape)
    #     print(type(th.mean.u))
    #     print(th.mean.u.shape)
    #     u = jax.random.normal(key1) * th.coarse_control # + th.mean.u[(t+1).astype(int),0]
    #     du = jax.random.normal(key2) * th.fine_control
    #     u = jnp.clip(u, u_min, u_max)  # () -> (1,)
    #     return Jet2(u, jnp.array([du]))
    #     # return jnp.array([u, du])

    # theta = CartPoleParams(
    #     mjx_model = mjx.put_model(mj_model),
    #     mean = Jet2(jnp.zeros((num_knots-1,mj_model.nu)), jnp.zeros((num_knots-1,mj_model.nu))),,
    #     coarse_control = 0.3,
    #     fine_control = 0.2,
    #     fine_angle_bounds = 0.1
    # )

    # Mujoco XLA dynamics
    def dynamics(x, j, th, _t):
        x = mjx.step(th.mjx_model, x.replace(ctrl=jnp.array([j])))
        return x, j

    # Cost from Hydrax
    running_cost = lambda x, j, th, _t: task.running_cost(x, j) * 0.01
    terminal_cost = lambda x, th: task.terminal_cost(x)

    def init(key, x, th):
        u = jax.random.normal(key) * th.coarse_control + th.mean[0]
        u = jnp.clip(u, task.u_min, task.u_max)
        return u

    # Linear Proposal
    def proposal(key, x: mjx.Data, u: jax.Array, th: CartPoleParams, t: float) -> jax.Array:
        step_len = horizon / (num_knots - 1)
        idx = (jnp.round(t / step_len)).astype(int) + 1
        u = jax.random.normal(key) * th.coarse_control + th.mean[idx]
        u = jnp.clip(u, task.u_min, task.u_max)
        return u

    theta = CartPoleParams(
        mjx_model = mjx.put_model(mj_model),
        mean = jnp.zeros((num_knots,mj_model.nu)),
        coarse_control = 0.3
    )

    ctrl = pfmp.PFMP(
        dynamics, running_cost, terminal_cost, init, proposal,
        horizon=horizon, num_knots=num_knots, particles=1024, dt=mj_model.opt.timestep,
        ess_resample_thresh=0.5
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
    record_video=False,
    record_traj=False,
    max_step=1000
)