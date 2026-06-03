import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import jax
import jax.numpy as jnp
import yaml
import mujoco
import pfmp
import importlib.util

from fire import Fire
from hydrax.task_base import Task
from hydrax.tasks.cart_pole import CartPole
from hydrax.tasks.double_cart_pole import DoubleCartPole
from hydrax.tasks.walker import Walker
from hydrax.tasks.cube import CubeRotation
from hydrax.algs import PredictiveSampling, MPPI, CEM, DIAL
from mujoco import mjx
from typing import Literal, NamedTuple


class MujocoParams(NamedTuple):
    mjx_model: mjx.Model
    mean: jax.Array
    noise_level: float

AlgorithmType = Literal["sis", "sir", "ps", "mppi", "cem", "dial"]

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
    dial_beta_opt_iter: float = 1.0
    dial_beta_horizon: float = 1.0
    dial_iterations: int = 5

def _load_task(name):
    """Instantiate a task by name. PFMP-native tasks are imported lazily."""
    if name == "PushT":
        from pfmp.tasks.pushT import PushT
        return PushT()
    if name == "HalfCheetah":
        path = os.path.join(os.path.dirname(__file__), "..", "examples", "half_cheetah.py")
        spec = importlib.util.spec_from_file_location("half_cheetah", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.HalfCheetah()
    return eval(name)()

def main(yamlfile, outdir):
    os.makedirs(outdir, exist_ok=True)

    with open(yamlfile, "r") as f:
        config = yaml.safe_load(f)

    task = _load_task(config["task"])
    mj_model = task.mj_model
    mj_data = mujoco.MjData(mj_model)
    if hasattr(task, "initial_qpos"):
        mj_data.qpos[:] = task.initial_qpos
        mujoco.mj_forward(mj_model, mj_data)

    # Check if this is a PFMP-native task (provides its own controller)
    pfmp_native = hasattr(task, 'make_controller')

    for algorithm in config.pop("algorithms"):
        cfg = Experiment(
            **config,
            algorithm=algorithm
        )
        theta = None
        ik_interp = None

        sample_problem = None

        if pfmp_native:
            # PFMP-native task (e.g. PushT) — provides its own dynamics,
            # cost functions, proposals, and optionally ik_interp.
            if cfg.algorithm not in ("sis", "sir"):
                raise ValueError(
                    f"Task {cfg.task} only supports PFMP algorithms (sis/sir), "
                    f"got '{cfg.algorithm}'"
                )
            ctrl, theta = task.make_controller(cfg)
            task.init_data(mj_data, theta)
            ik_interp = getattr(task, 'ik_interp', None)
            sample_problem = getattr(task, 'sample_problem', None)
        elif cfg.algorithm == "sis" or cfg.algorithm == "sir":
            running_cost, terminal_cost = pfmp.mjx_hydrax_cost(task)
            init, proposal = pfmp.mjx_linear_gaussian_proposal(task, cfg.plan_horizon, cfg.num_knots)
            ctrl = pfmp.PFMP(
                pfmp.mjx_physics,
                running_cost, terminal_cost,
                init, proposal,
                horizon=cfg.plan_horizon,
                num_knots=cfg.num_knots,
                particles=cfg.num_samples,
                dt=mj_model.opt.timestep,
                ess_resample_thresh=0.5,
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
        elif cfg.algorithm == "dial":
            ctrl = DIAL(
                task,
                num_samples=cfg.num_samples,
                noise_level=cfg.noise_level,
                beta_opt_iter=cfg.dial_beta_opt_iter,
                beta_horizon=cfg.dial_beta_horizon,
                temperature=cfg.mppi_temperature,
                spline_type=cfg.spline_type,
                plan_horizon=cfg.plan_horizon,
                num_knots=cfg.num_knots,
                iterations=cfg.dial_iterations,
            )
        else:
            raise ValueError(f"Unsupported MPC algorithm: {cfg.algorithm}")

        traj = pfmp.run_batch(
            ctrl,
            theta,
            mj_model,
            mj_data,
            frequency=cfg.frequency,
            algorithm=cfg.algorithm,
            max_step=cfg.max_step,
            num=cfg.num,
            ik_interp=ik_interp,
            sample_problem=sample_problem,
        )
        traj.save(os.path.join(outdir, cfg.algorithm))

if __name__ == "__main__":
    Fire(main)
