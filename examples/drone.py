# drone.py

import argparse

import mujoco
import jax
import jax.numpy as jnp
import pfmp

from mujoco import mjx
from hydrax.task_base import Task
from hydrax.algs import PredictiveSampling
from typing import NamedTuple


class DroneRace(Task):
    def __init__(self) -> None:
        mj_model = mujoco.MjModel.from_xml_path("./models/x2_scene.xml")

        self.goal = jnp.array([5.0, 0.0, 1.0])
        
        # Cost weights for tuning
        self.w_dist = 5.
        self.w_ang_vel = 0.5     # Angular velocity penalty
        self.w_vel = 0.05          # Velocity penalty weight
        self.w_orient = 0.01       # Orientation error weight
        self.w_control = 0.01     # Control effort penalty

        # For visual marker
        self.target_bid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "target_body")
        mj_model.body_pos[self.target_bid] = self.goal
        
        super().__init__(mj_model)
    
    def running_cost(self, state: mjx.Data, control: jax.Array) -> jax.Array:
        pos = state.qpos[:3]
        quat = state.qpos[3:7]  # Assuming quaternion at indices 3-6
        vel = state.qvel[:3]
        ang_vel = state.qvel[3:6]
        
        # Velocity penalty (discourage excessive speed for stability)
        vel_cost = self.w_vel * jnp.dot(vel, vel)
        
        # Orientation cost (penalize deviation from upright)
        # For upright: quaternion should be close to [1, 0, 0, 0] or [w, 0, 0, 0]
        # We want the z-axis to point up, which means quat[0] (w) close to 1
        orient_cost = self.w_orient * ((1 - quat[0])**2 + quat[1]**2 + quat[2]**2 + quat[3]**2)
        
        # Angular velocity penalty (discourage spinning)
        ang_vel_cost = self.w_ang_vel * jnp.dot(ang_vel, ang_vel)
        
        # Control effort penalty (energy efficiency)
        control_cost = self.w_control * jnp.dot(control, control)
        
        # Distance to goal
        goal_cost = self.w_dist * jnp.linalg.norm(pos - self.goal)
        
        total_cost = (goal_cost + vel_cost + orient_cost + 
                     control_cost + ang_vel_cost)
        
        return total_cost
    
    def terminal_cost(self, state: mjx.Data) -> jax.Array:
        pos = state.qpos[:3]
        vel = state.qvel[:3]
        quat = state.qpos[3:7]
        ang_vel = state.qvel[3:6]
        
        # Position error at terminal state
        pos_err = pos - self.goal
        pos_cost = 50 * jnp.dot(pos_err, pos_err)

        ang_vel_cost = 5 * jnp.dot(ang_vel, ang_vel)
        
        # Terminal velocity penalty (want to stop at goal)
        vel_cost = 5 * jnp.dot(vel, vel)
        
        # Terminal orientation cost (want to be upright at goal)
        orient_cost = self.w_orient * ((1 - quat[0])**2 + quat[1]**2 + quat[2]**2 + quat[3]**2)
        
        return pos_cost + vel_cost + orient_cost + ang_vel_cost

task = DroneRace()

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
subparsers.add_parser("sis", help="Particle Filter Motion Planning (SIR)")
subparsers.add_parser("sir", help="Particle Filter Motion Planning (SIR)")
args = parser.parse_args()

if args.algorithm == "ps":
    ctrl = PredictiveSampling(
        task,
        num_samples=1024,
        noise_level=5.0,
        plan_horizon=1.0,
        spline_type="linear",
        num_knots=4
    )
    theta = None
elif args.algorithm == "sis" or "sir":
    horizon = 1.0
    num_knots = 4

    running_cost, terminal_cost = pfmp.mjx_hydrax_cost(task)
    init, proposal = pfmp.mjx_linear_gaussian_proposal(task, horizon, num_knots)
    ctrl = pfmp.PFMP(
        pfmp.mjx_physics,
        running_cost, terminal_cost,
        init, proposal,
        horizon=horizon,
        num_knots=num_knots,
        particles=1024,
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