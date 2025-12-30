import os
import jax
import jax.numpy as jnp
import yaml
import mujoco
import pfmp

from fire import Fire
from hydrax.task_base import Task
from hydrax.tasks.cart_pole import CartPole
from hydrax.tasks.double_cart_pole import DoubleCartPole
from hydrax.tasks.walker import Walker
from hydrax.tasks.cube import CubeRotation
from hydrax.algs import PredictiveSampling, MPPI, CEM
from mujoco import mjx
from typing import Literal, NamedTuple


class MujocoParams(NamedTuple):
    mjx_model: mjx.Model
    mean: jax.Array
    noise_level: float

AlgorithmType = Literal["sis", "sir", "ps", "mppi", "cem"]

class Experiment(NamedTuple):
    task: str
    algorithm: AlgorithmType
    max_step: int
    num: int
    plan_horizon: float
    num_knots: int
    num_samples: int
    noise_level: float
    spline_type: str
    mppi_temperature: float
    cem_num_elites: int
    cem_sigma_start: float
    cem_sigma_min: float
    frequency: int = 50

def main(yamlfile, outdir):
    os.makedirs(outdir, exist_ok=True)

    with open(yamlfile, "r") as f:
        config = yaml.safe_load(f)

    task = eval(config["task"])()
    mj_model = task.mj_model
    mj_data = mujoco.MjData(mj_model)

    for algorithm in config.pop("algorithms"):
        cfg = Experiment(
            **config,
            algorithm=algorithm
        )
        theta = None
        if cfg.algorithm == "sis" or cfg.algorithm == "sir":
            running_cost, terminal_cost = pfmp.mjx_hydrax_cost(task)
            init, proposal = pfmp.mjx_linear_gaussian_proposal(task, cfg.horizon, cfg.num_knots)
            ctrl = pfmp.PFMP(
                pfmp.mjx_physics,
                running_cost, terminal_cost,
                init, proposal,
                horizon=cfg.horizon,
                num_knots=cfg.num_knots,
                particles=cfg.num_samples,
                dt=mj_model.opt.timestep,
                ess_resample_thresh=cfg.ess_resample_thresh
                )
            theta = pfmp.GenericMJXTaskParams(
                mjx_model = mjx.put_model(mj_model),
                mean = jnp.zeros((cfg.num_knots,mj_model.nu)),
                noise_level = cfg.noise_level
                )
        elif cfg.algorithm == "ps":
            ctrl = PredictiveSampling(
                task,
                num_samples = cfg.num_samples,
                noise_level = cfg.noise_level,
                spline_type = cfg.spline_type,
                plan_horizon = cfg.plan_horizon,
                num_knots = cfg.num_knots
            )
        elif cfg.algorithm == "mppi":
            ctrl = MPPI(
                task,
                num_samples=cfg.num_samples,
                noise_level=cfg.noise_level,
                temperature=cfg.mppi_temperature,
                spline_type=cfg.spline_type,
                plan_horizon=cfg.plan_horizon,
                num_knots=cfg.num_knots,
            )
        elif cfg.algorithm == "cem":
            ctrl = CEM(
                task,
                num_samples=cfg.num_samples,
                num_elites=cfg.cem_num_elites,
                sigma_start=cfg.cem_sigma_start,
                sigma_min=cfg.cem_sigma_min,
                spline_type=cfg.spline_type,
                plan_horizon=cfg.plan_horizon,
                num_knots=cfg.num_knots,
            )
        else:
            raise ValueError(f"Unsupported MPC algorithm: {cfg.algorithm}")

        traj = pfmp.run_passive(
            ctrl,
            theta,
            mj_model,
            mj_data,
            frequency=cfg.frequency,
            algorithm=cfg.algorithm,
            max_step=cfg.max_step,
            num=cfg.num
        )
        traj.save(os.path.join(outdir, cfg.algorithm))

if __name__ == "__main__":
    Fire(main)