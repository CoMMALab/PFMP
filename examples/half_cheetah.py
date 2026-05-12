import argparse
import os

import jax
import jax.numpy as jnp
import mujoco
import pfmp

from hydrax.algs import MPPI, PredictiveSampling
from hydrax.task_base import Task
from mujoco import mjx

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "half_cheetah.xml")


class HalfCheetah(Task):
    """A planar half-cheetah tasked with running forward."""

    def __init__(self) -> None:
        mj_model = mujoco.MjModel.from_xml_path(MODEL_PATH)
        super().__init__(mj_model)

        self.target_velocity = 2.0
        self.target_rootz = 0.0
        self.initial_qpos = jnp.array([
            0.0, 0.0, 0.0,
            0.0, 0.0, 0.0,
            0.0, 0.0, 0.0,
        ])

    def running_cost(self, state: mjx.Data, control: jax.Array) -> jax.Array:
        x_velocity = state.qvel[0]
        root_z = state.qpos[1]
        torso_pitch = state.qpos[2]

        velocity_cost = jnp.square(x_velocity - self.target_velocity)
        height_cost = jnp.square(root_z - self.target_rootz)
        pitch_cost = jnp.square(torso_pitch)
        control_cost = jnp.sum(jnp.square(control))

        return velocity_cost + 5.0 * height_cost + 0.5 * pitch_cost + 0.01 * control_cost

    def terminal_cost(self, state: mjx.Data) -> jax.Array:
        x_velocity = state.qvel[0]
        root_z = state.qpos[1]
        torso_pitch = state.qpos[2]
        return (
            5.0 * jnp.square(x_velocity - self.target_velocity)
            + 20.0 * jnp.square(root_z - self.target_rootz)
            + 2.0 * jnp.square(torso_pitch)
        )


def main() -> None:
    task = HalfCheetah()
    mj_model = task.mj_model
    mj_data = mujoco.MjData(mj_model)
    mj_data.qpos[:] = task.initial_qpos
    mujoco.mj_forward(mj_model, mj_data)

    parser = argparse.ArgumentParser(
        description="Run an interactive simulation of the half-cheetah task."
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
    elif args.algorithm == "mppi":
        ctrl = MPPI(
            task,
            num_samples=128,
            noise_level=0.5,
            temperature=0.1,
            plan_horizon=0.6,
            spline_type="zero",
            num_knots=5,
        )
        theta = None
    elif args.algorithm == "sis" or args.algorithm == "sir":
        horizon = 0.6
        num_knots = 5

        running_cost, terminal_cost = pfmp.mjx_hydrax_cost(task)
        init, proposal = pfmp.mjx_linear_gaussian_proposal(task, horizon, num_knots)
        ctrl = pfmp.PFMP(
            pfmp.mjx_physics,
            running_cost,
            terminal_cost,
            init,
            proposal,
            horizon=horizon,
            num_knots=num_knots,
            particles=128,
            dt=mj_model.opt.timestep,
            ess_resample_thresh=0.3,
        )
        theta = pfmp.GenericMJXTaskParams(
            mjx_model=mjx.put_model(mj_model),
            mean=jnp.zeros((num_knots, mj_model.nu)),
            noise_level=0.3,
        )

    pfmp.run_interactive(
        ctrl,
        theta,
        mj_model,
        mj_data,
        frequency=50,
        method=args.algorithm,
        fixed_camera_id=0,
        record_video=False,
        record_traj=False,
        max_step=1500,
    )


if __name__ == "__main__":
    main()
