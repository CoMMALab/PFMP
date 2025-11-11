import argparse

import jax
import jax.numpy as jnp
import mujoco

from hydrax.algs import PredictiveSampling
from hydrax.simulation.deterministic import run_interactive
from hydrax.tasks.cart_pole import CartPole
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
        spline_type = "cubic",
        plan_horizon = 1.0,
        num_knots = 4
    )
elif args.algorithm == "pfmp":
    # list of TODO
    #
    # remove PPL dependency
    #   - allows ==Jax v0.5.2 -> >=Jax v0.7.0
    # make API more user-friendly:
    #   - H -> horizon
    #   - K -> particles
    #   - l -> num_knots (and move to run_sir)
    #   - rho -> ess_resample_thresh
    #   - dynamics/proposals/cost should not be time-dependent (standard MPC)
    # update splines
    #   - run_sir: add spline_type
    #   - proposals returns knot vector (instead of ndarray)
    #   - pfmp/interpolate -> depend on interpax
    #   - run_sir: compute time-of-knots (tk)
    # later
    #   - bundle CartPoleModel(task, dynamics, proposal) into MPC class.
    #   - bundle PFMP -> InferSMC(model)
    import pfmp
    import mujoco.mjx as mjx

    num_knots = 4

    # Mujoco XLA dynamics
    def dynamics(x, u, th, _t):
        x = mjx.step(th.mjx_model, x.replace(ctrl=jnp.array([u])))
        return x, (x.ctrl, x.qpos, x.qvel)

    cost = lambda x, u, _th, _t: task.running_cost(x, u)

    class CartPoleParams(NamedTuple):
        mjx_model: mjx.Model
        mean: jax.Array
        coarse_control: float
        fine_control: float
        fine_angle_bounds: float

    theta = CartPoleParams(
        mjx_model = mjx.put_model(mj_model),
        mean = jnp.zeros((num_knots,mj_model.nu,)),
        coarse_control = 5.0,
        fine_control = 0.01,
        fine_angle_bounds = 0.1
    )

    def proposal(key, x: mjx.Data, u: jax.Array, th: CartPoleParams, t: int) -> jax.Array:
        theta = x.qpos[1]
        near_upright = 1 + jnp.cos(theta) < th.fine_angle_bounds
        scale = jnp.where(near_upright, th.fine_control, th.coarse_control)
        u = jax.random.normal(key, (th.mjx_model.nu,)) * jnp.sqrt(scale)
        return u

    # simple, clean. No PPL, just proposals and dynamics
    ctrl = pfmp.PFMP(
        dynamics, cost, proposal,
        horizon=1.0, num_knots=4, particles=128
        )

    # must be new because of the sample-resample kernel
    pfmp.run_interactive(
        ctrl,
        theta,  # model-specific parameters 
        mj_model,
        mj_data,
        frequency=50,
        fixed_camera_id=0,
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
)