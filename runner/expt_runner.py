import jax
import jax.numpy as jnp
import mujoco
import matplotlib.pyplot as plt
import time
import os, math, pickle, itertools, time, traceback, sys
import numpy as np

from genjax import gen, mv_normal
from typing import NamedTuple, Type
from pathlib import Path
from typing import Sequence, Tuple, Any, Iterable
from collections import OrderedDict
from matplotlib.lines import Line2D
from pfmp.algorithm_spline import make_transition_model, PFMP
from pfmp.interpolate import hermite_spline

from runner.arg_utils import build_parser_and_parse_args
from runner.file_utils import _to_jsonable, _atomic_write_json, save_trajs_npz_named, save_traj_costs_npz_named, load_trajs_npz_named

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
    traj_costs: jnp.array



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

        self.mpc_model = None

        # Setting to 42 for the answer to everything
        self.key = jax.random.key(42)

        self.problem = None
        self.config = None
        self.name = None
        self.comments = None
        self.results_dir = None


    def init_transition_model_mpc(self):
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

        costs = self.mpc_model.project_costs(self.problem.cost_fn, trajs, theta)

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
        ), trajs, costs


    def visualize(self, traj):

        x0, u0, theta = self.problem.initialize_problem(self.config)
        self.problem.visualize(theta = theta, config = self.config, traj=traj)


    def evaluate(self, visualize: bool = False, save_results: bool = True):


        self.mpc_model = PFMP(self.init_transition_model_mpc())
        x0, u0, theta = self.problem.initialize_problem(self.config)

        metrics, trajs, traj_costs = self.run_model_and_benchmark(x0, u0, theta)
        # (K, H, L, D)

        results = Results(
            name = self.name,
            comments = self.comments,
            config = self.config,
            metrics = metrics,
            sampled_trajs = trajs,
            traj_costs = traj_costs,
        )

        total_costs = traj_costs.reshape(traj_costs.shape[0], -1).sum(axis=1)
        map_idx = jnp.argmin(total_costs)
        map_traj = jax.tree.map(lambda x: x[map_idx].reshape(self.config.H*self.config.L,-1), trajs)


        if visualize:
            self.problem.visualize(theta = theta, config = self.config, traj=map_traj)

        if save_results:
            self.save_results(results)

    def get_file_name_from_config(self):
        _ffmt = lambda x: f"{float(x):g}".replace(".", "p")

        params_str = f"_k{self.config.K}_d{self.config.deg}_h{self.config.H}_l{self.config.L}_s{_ffmt(self.config.sigma)}_rho{_ffmt(self.config.rho)}"
        json_name  = f"{self.name}{params_str}.json"
        json_path  = Path(self.results_dir, json_name)

        return json_name, json_path


    def save_results(self, results: Results):

        # Get file name
        json_name, json_path = self.get_file_name_from_config()

        sample_trajs = results.sampled_trajs
        traj_costs = results.traj_costs
        n_samples = sample_trajs[0].shape[0]

        trajs_fname = f"{self.name}_trajs_n{n_samples}{json_path.stem}.npz"
        trajs_path  = os.path.join(self.results_dir, trajs_fname)

        traj_costs_fname = f"{self.name}_traj_costs_n{n_samples}{json_path.stem}.npz"
        traj_costs_path  = os.path.join(self.results_dir, traj_costs_fname)

        results = results._replace(sampled_trajs = None)
        results = results._replace(traj_costs = None)

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
        if traj_costs is not None:
            save_traj_costs_npz_named(traj_costs, traj_costs_path)
            payload["meta"]["traj_costs_fname"] = traj_costs_fname
            # Add a tiny schema preview for convenience (optional)
            with np.load(traj_costs_path) as z:
                payload["meta"]["traj_costs_fields"] = [
                    {"key": k, "shape": list(z[k].shape), "dtype": str(z[k].dtype)}
                    for k in sorted(z.files)
                ]


        _atomic_write_json(payload, json_path, indent=2)
        print(f"  → saved {json_name} in {payload['meta']['elapsed_sec']:.2f}s")
        print(f"    cost = {results.metrics.average_cost}")
