import jax
import jax.numpy as jnp
import mujoco
import matplotlib.pyplot as plt
import time
import os, math, pickle, itertools, time, traceback, sys
import numpy as np

from genjax import gen, mv_normal
from mujoco import mjx
from typing import NamedTuple, Type
from mujoco.viewer import launch_passive
from pathlib import Path
from typing import Sequence, Tuple, Any, Iterable
from collections import OrderedDict
from matplotlib.lines import Line2D
from pfmp.algorithm_spline import make_transition_model, PFMP
from pfmp.interpolate import hermite_spline

from runner.arg_utils import build_parser_and_parse_args
from runner.file_utils import _to_jsonable, _atomic_write_json, save_trajs_npz_named

from tests.models.problem import ProblemDesc

N_SAMPLES: int = 10


class Config(NamedTuple):
    K: int
    H: int
    L: int
    sigma: float
    deg: int
    rho: float        
        

class Metrics(NamedTuple):
    runtime: str
    average_cost: str



class Results(NamedTuple):
    name: str
    config: Config
    comments: str
    metrics: Metrics
    sampled_trajs: jax.tree.map



class ExperimentRunner:
    def __init__(self, problem: ProblemDesc, config: Config, results_dir):

        self.problem = problem
        self.config = config
        self.name = problem.name
        self.comments = ""
        self.results_dir = results_dir


        self.reset_experiment()

    def reset_experiment(self):
        """
        Resets set values to None.
        The config is left untouched

        """

        self.mjx_model = None
        self.x0 = None
        self.theta = None
        self.mpc_model = None

        # Setting to 42 for the answer to everything
        self.key = jax.random.key(42)





    def init_mjx_mpc(self):
        """
        Build an MPC transition model with a proposal matched to `deg` and `l`.

        deg: 1 (linear), 3 (cubic), or 5 (quintic)
        l:   segment length
        """

        proposal_map = {
            1: self.problem.linear_proposal_fn,
            3: self.problem.cubic_proposal_fn,
            5: self.problem.quintic_proposal_fn,
        }

        try:
            proposal_fn = proposal_map[self.config.deg]
        except KeyError:
            valid = ", ".join(map(str, sorted(proposal_map)))
            raise ValueError(
                f"Unsupported degree: deg={self.config.deg}. Valid options are {{{valid}}} "
                "(1=linear, 3=cubic, 5=quintic)."
            )

        # basic sanity
        for name, fn in [("proposal_fn", proposal_fn), ("cost_fn", self.problem.cost_fn), ("dynamics_fn", self.problem.dynamics_fn)]:
            if not callable(fn):
                raise TypeError(f"{name} must be callable; got {type(fn).__name__}")

        return make_transition_model(proposal_fn, self.problem.cost_fn, self.problem.dynamics_fn, l=self.config.L)


    def run_model_and_benchmark(
            self,
            x0,
            u0,
            theta
        ):
        """
        This is just a placeholder. Will be replaced later
        """

        t0 = time.time()

        trajs = self.mpc_model(
            key = self.key, 
            x0 = x0, 
            u0 = u0,
            theta = theta,
            H=self.config.H, 
            K=self.config.K, 
            rho=self.config.rho,
            n_samples = N_SAMPLES
        )
        t1 = time.time()


        # # run and profile algorithm
        # # time grid as float
        # t_seq = jnp.arange(self.config.H * self.config.L, dtype=jnp.float32) / jnp.float32(self.config.L)
        # # sum over T for a single trajectory (leaves shaped (T, ...))
        # traj_cost_fn = lambda tau: jnp.sum(
        #     jax.vmap(self.problem.cost_fn, in_axes=(0, 0, None, 0))(tau, tau.ctrl, theta, t_seq)
        # )

        # # mean over K of per-trajectory totals (leaves shaped (K, T, ...))
        # average_cost_fn = lambda trajs_flat: jnp.mean(
        #     jax.vmap(traj_cost_fn, in_axes=0)(trajs_flat)
        # )

        # # (K,H,L,...) -> (K,H*L,...)
        # trajs_flat = jax.tree.map(
        #     lambda x: x.reshape((x.shape[0], self.config.H * self.config.L) + x.shape[3:]),
        #     trajs,
        # )

        return Metrics(
            runtime = t1 - t0,
            # average_cost = average_cost_fn(trajs_flat),
            average_cost = 0.,
        ), jax.tree.map(lambda x: x[:N_SAMPLES], trajs),


    def visualize(self, traj):

        self.problem.visualize()

        # with launch_passive(self.mj_model, data=self.mj_data, show_left_ui=False, show_right_ui=False) as viewer:
        #     for t in range(traj.qpos.shape[0]):
        #         for s in range(self.config.L):
        #             self.mj_data.qpos[:] = jax.device_get(traj.qpos[t,s])
        #             self.mj_data.qvel[:] = jax.device_get(traj.qvel[t,s])
        #             mujoco.mj_forward(self.mj_model, self.mj_data)
        #             time.sleep(0.01)
        #             viewer.sync()



    def evaluate(self, visualize: bool = False, save_results: bool = True):

        # mj_model, mj_data = load_mj(self.name)
        # self.problem.init_mjx(mj_model, mj_data)

        self.mpc_model = PFMP(self.init_mjx_mpc())
        x0, u0, theta = self.problem.initialize_problem(self.config)

        metrics, sampled_trajs = self.run_model_and_benchmark(x0, u0, theta)

        results = Results(
            name = self.name,
            comments = self.comments,
            config = self.config,
            metrics = metrics,
            sampled_trajs = sampled_trajs
        )

        if visualize:
            self.visualize(results.sampled_trajs[0])

        if save_results:
            self.save_results(results)

    def get_file_name_from_config(self):
        _ffmt = lambda x: f"{float(x):g}".replace(".", "p")

        params_str = f"_k{self.config.K}_d{self.config.deg}_h{self.config.H}_l{self.config.L}_s{_ffmt(self.config.sigma)}"
        json_name  = f"{self.name}{params_str}.json"
        json_path  = Path(self.results_dir, json_name)

        return json_name, json_path


    def save_results(self, results: Results):

        # Get file name
        json_name, json_path = self.get_file_name_from_config()

        sample_trajs = results.sampled_trajs
        n_samples = sample_trajs[0].shape[0]

        trajs_fname = f"{self.name}_trajs_n{n_samples}{json_path.stem}.npz"
        trajs_path  = os.path.join(self.results_dir, trajs_fname)

        results = results._replace(sampled_trajs = None)

        payload = {
            "meta": {
                "name": self.name,
                "k": self.config.K,
                "deg": self.config.deg,
                "h": self.config.H,
                "l": self.config.L,
                "sigma": self.config.sigma,
                "n_samples": n_samples,
                "timestamp": time.time(),
                "elapsed_sec": None,  # filled below
                "jax_version": getattr(jax, "__version__", "unknown"),
                "mujoco_version": getattr(mujoco, "__version__", "unknown"),
            },
            # convert remaining results to NumPy → JSON-safe
            "results": _to_jsonable(results),
        }
        payload["meta"]["elapsed_sec"] = results.metrics.runtime


        if sample_trajs is not None:
            save_trajs_npz_named(sample_trajs, trajs_path)
            payload["meta"]["trajs_file"] = trajs_fname
            # Add a tiny schema preview for convenience (optional)
            with np.load(trajs_path) as z:
                payload["meta"]["trajs_fields"] = [
                    {"key": k, "shape": list(z[k].shape), "dtype": str(z[k].dtype)}
                    for k in sorted(z.files)
                ]

        _atomic_write_json(payload, json_path, indent=2)
        print(f"  → saved {json_name} in {payload['meta']['elapsed_sec']:.2f}s")
        print(f"    cost = {results.metrics.average_cost}")
