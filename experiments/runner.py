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
            def dynamics(x, j, th, _t):
                x = mjx.step(th.mjx_model, x.replace(ctrl=j.reshape(mj_model.nu)))
                return x, None

            running_cost = lambda x, j, th, _t: task.running_cost(x, j) * th.mjx_model.opt.timestep
            terminal_cost = lambda x, th: task.terminal_cost(x)

            def init(key, x, th):
                u = jax.random.normal(key) * th.noise_level + th.mean[0]
                u = jnp.clip(u, task.u_min, task.u_max)
                return u

            def proposal(key, x: mjx.Data, u: jax.Array, th: MujocoParams, t: float) -> jax.Array:
                step_len = cfg.plan_horizon / (cfg.num_knots - 1)
                idx = (jnp.round(t / step_len)).astype(int) + 1
                u = jax.random.normal(key) * th.noise_level + th.mean[idx]
                u = jnp.clip(u, task.u_min, task.u_max)
                return u

            ctrl = pfmp.PFMP(
                dynamics, running_cost, terminal_cost, init, proposal,
                horizon=cfg.plan_horizon, num_knots=cfg.num_knots, particles=cfg.num_samples, dt=mj_model.opt.timestep,
                ess_resample_thresh=0.3
                )

            theta = MujocoParams(
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