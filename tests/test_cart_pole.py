import argparse

import jax
import jax.numpy as jnp
import mujoco

from hydrax.algs import PredictiveSampling
from hydrax.simulation.deterministic import run_interactive
from hydrax.tasks.cart_pole import CartPole
from pfmp.interpolate import Jet2
from typing import NamedTuple

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
subparsers.add_parser("pfmp", help="Particle Filter Motion Planning")
args = parser.parse_args()

if args.algorithm == "ps":
    ctrl = PredictiveSampling(
        task,
        num_samples = 128,
        noise_level = 0.3,
        spline_type = "linear",
        plan_horizon = 1.0,
        num_knots = 4
    )
elif args.algorithm == "pfmp":
    import pfmp
    import mujoco.mjx as mjx

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
    running_cost = lambda x, j, th, _t: task.running_cost(x, j) * th.mjx_model.opt.timestep

    # For some reason, adding this breaks PFMP
    terminal_cost = lambda x, th: 0.0 # task.terminal_cost(x)

    # Linear Proposal
    def proposal(key, x: mjx.Data, u: jax.Array, th: CartPoleParams, t: float) -> jax.Array:
        step_len = horizon / (num_knots - 1)
        idx = (jnp.round(t / step_len)).astype(int)
        u = jax.random.normal(key) * th.coarse_control + th.mean[idx+1]
        u = jnp.clip(u, u_min, u_max)
        return u

    theta = CartPoleParams(
        mjx_model = mjx.put_model(mj_model),
        mean = jnp.zeros((num_knots,mj_model.nu)),
        coarse_control = 0.3
    )

    ctrl = pfmp.PFMP(
        dynamics, running_cost, terminal_cost, proposal,
        horizon=horizon, num_knots=num_knots, particles=128, dt=mj_model.opt.timestep,
        ess_resample_thresh=0.5
        )

    u0 = jnp.ones(mj_model.nu)
    # u0 = jnp.zeros(mj_model.nu)
    # u0 = Jet2(jnp.zeros(mj_model.nu,), jnp.zeros(mj_model.nu,))

    pfmp.run_interactive(
        ctrl,
        u0,
        theta,
        mj_model,
        mj_data,
        frequency=50,
        fixed_camera_id=0,
        method="sir",
        record_video=True
    )
    exit()

# Run the interactive simulation
run_interactive(
    ctrl,
    mj_model,
    mj_data,
    frequency=50,
    fixed_camera_id=0,
    show_traces=False,
    max_traces=1,
    record_video=True
)