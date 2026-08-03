import jax
import jax.numpy as jnp

from functools import partial
from jaxtyping import PyTree, PRNGKeyArray
from typing import NamedTuple, Callable
from dataclasses import is_dataclass, fields
from hydrax.task_base import Task
from mujoco import mjx
from .interpolate import hermite_interp1d, Jet1, Jet2, Jet3
# Re-exported so existing `from pfmp.algorithm import ps_update` (etc.) call
# sites keep working — the update rules (U) live in update.py. Four of them
# (ps/sample/mppi/cem) are pure readouts over a completed rollout batch and
# never call back into the PFMP sampler defined below; dial_step is the
# exception (it drives its own re-sampling loop, taking the sampler as an
# argument), included here too for a uniform import surface.
from .update import (
    ess, dial_sigma, _check_update_shapes,
    ps_update, sample_update, mppi_update, cem_update, dial_step,
)


def get_field_info(x):
    """
    Return:
      - field_names: tuple of keys
      - accessor(obj, key) -> leaf array (indexable by [tau, idx] for arrays with (horizon,particles,...))
      - builder(vals) -> reconstructs the same container type as `x`
    """
    # dataclass
    if is_dataclass(x):
        names = tuple(f.name for f in fields(x))
        accessor = lambda obj, n: getattr(obj, n)
        builder  = lambda vals: type(x)(**{n: v for n, v in zip(names, vals)})
        return names, accessor, builder

    # namedtuple
    if hasattr(x, "_fields") and isinstance(x._fields, tuple):
        names = x._fields
        accessor = lambda obj, n: getattr(obj, n)
        # namedtuple accepts kwargs with field names
        builder  = lambda vals: type(x)(**{n: v for n, v in zip(names, vals)})
        return names, accessor, builder

    # jax.Array -> treat as ONE field: the whole array (horizon, particles, ...)
    if jax.Array and isinstance(x, jax.Array):
        names = ("__array__",)
        accessor = lambda obj, _k: obj               # shape (horizon, particles, ...)
        builder  = lambda vals: vals[0]              # just return the array leaf
        return names, accessor, builder

    # plain tuple
    if isinstance(x, tuple):
        names = tuple(range(len(x)))
        accessor = lambda obj, i: obj[i]
        builder  = lambda vals: tuple(vals)
        return names, accessor, builder

    raise TypeError(f"Unsupported state type: {type(x)}")


# Mujoco XLA helper functions
def mjx_hydrax_cost(task: Task):
    running_cost = lambda x, j, th, _t: task.running_cost(x, j) * 0.01
    terminal_cost = lambda x, th: task.terminal_cost(x)
    return running_cost, terminal_cost

def mjx_linear_gaussian_proposal(task: Task, horizon: float, num_knots: int):
    def init(key, x, th):
        u = jax.random.normal(key, th.mjx_model.nu) * th.noise_level + th.mean[0]
        u = jnp.clip(u, task.u_min, task.u_max)
        return u

    # Linear Proposal
    def proposal(key, x: mjx.Data, u: jax.Array, th, t: float) -> jax.Array:
        step_len = horizon / (num_knots - 1)
        idx = (t / step_len).astype(int) + 1
        u = jax.random.normal(key, th.mjx_model.nu) * th.noise_level + th.mean[idx]
        u = jnp.clip(u, task.u_min, task.u_max)
        return u

    return init, proposal

def mjx_physics(x, j, th, _t):
    if j.shape == ():
        x = mjx.step(th.mjx_model, x.replace(ctrl=jnp.array([j])))
    else:
        x = mjx.step(th.mjx_model, x.replace(ctrl=j))
    return x, None

class GenericMJXTaskParams(NamedTuple):
    mjx_model: mjx.Model
    mean: jax.Array
    noise_level: float
    actor_params: any = None

class PFMP(NamedTuple):
    """ Particle Filtering Motion Planning (PFMP) """

    mpc_dynamics: Callable[[PyTree, jax.Array, PyTree, float], tuple[PyTree, jax.Array]]
    mpc_running_cost: Callable[[PyTree, jax.Array, PyTree, float], float]
    mpc_terminal_cost: Callable[[PyTree, PyTree], float]    # should be deterministic
    mpc_init: Callable[[PRNGKeyArray, PyTree, PyTree], jax.Array]
    mpc_proposal: Callable[[PRNGKeyArray, PyTree, jax.Array, PyTree, float], jax.Array]

    horizon: float
    num_knots: int
    particles: int
    ess_resample_thresh: float = 0.5
    dt: float = 0.01
    # How `u` varies within a knot interval. "hermite" splines u0 -> u1; "zoh"
    # holds u1 for the whole interval. Use "zoh" when the executor tracks
    # waypoints itself (e.g. franky moving to each knot), so the rollout scores
    # the trajectory that is actually run rather than an interpolant that isn't.
    interp: str = "hermite"

    def tk(self, endpoint: bool = True):
        if endpoint:
            return jnp.linspace(0., self.horizon, self.num_knots)
        else:
            return jnp.linspace(0., self.horizon, self.num_knots)[:-1]

    def _interpolate(self, x0: PyTree, u0: jax.Array, u1: jax.Array,
                     t0: float, theta: PyTree, return_traj: bool = False):
        """Integrate the dynamics over one knot interval [t0, t0+step_len], with
        the control spline-interpolated from u0 to u1. Returns (x_end, cost), or
        (x_end, cost, xs) with the per-substep states when `return_traj`. Shared
        by `transition_model` (one proposed interval, endpoint only) and `predict`
        (a fixed sequence, dense). `return_traj` is off in the rollout so the
        planner's scans never carry the extra states."""
        step_len = self.horizon / (self.num_knots - 1)
        steps_per_knot = int(step_len / self.dt)
        t1 = t0 + step_len

        if isinstance(u0, (Jet1, jax.Array)):
            nout = 1
        elif isinstance(u0, Jet2):
            nout = 2
        elif isinstance(u0, Jet3):
            nout = 3

        if self.interp == "zoh":
            tq = jnp.linspace(t0, t1, steps_per_knot + 1)
            us = jax.tree.map(
                lambda v: jnp.broadcast_to(v, (steps_per_knot + 1,) + jnp.shape(v)), u1)
        else:
            tq, us = hermite_interp1d(t0, t1, u0, u1, num=steps_per_knot + 1, nout=nout)

        def interpolate_step(carry, inps):
            x, c = carry
            u, s = inps
            (x_next, _y) = self.mpc_dynamics(x, u, theta, s)
            c_next = self.mpc_running_cost(x_next, u, theta, s)
            return (x_next, c + c_next), (x_next if return_traj else None)

        (x1, cost), xs = jax.lax.scan(
            interpolate_step, (x0, 0.0), (jax.tree.map(lambda u: u[:-1], us), tq[:-1]))
        return (x1, cost, xs) if return_traj else (x1, cost)

    @partial(jax.jit, static_argnames=['self'])
    def transition_model(self, key, x0: PyTree, u0: jax.Array, theta: PyTree, t0: float) -> PyTree:
        # Propose a new spline knot, then integrate the interval that reaches it.
        u1 = self.mpc_proposal(key, x0, u0, theta, t0)
        x1, cost = self._interpolate(x0, u0, u1, t0, theta)
        return (x1, u1), (u1, cost)

    @partial(jax.jit, static_argnames=['self'])
    def predict(self, x0: PyTree, us: jax.Array, theta: PyTree) -> PyTree:
        """Deterministically roll a fixed knot sequence through the dynamics.

        Unlike `transition_model`/`unroll`, this proposes nothing and samples
        nothing: it is the forward model of an already-chosen plan (e.g.
        `theta.mean`). Returns the *dense* trajectory. Every integration substep,
        starting at x0. The leaves have shape (num_intervals*steps_per_knot + 1, ...).
        Use for scoring, analysis, or visualising what a plan is predicted to do.

        us: (num_knots, action_dim) control knots.
        """
        def interval_step(x, inp):
            u0, u1, t0 = inp
            x_end, _cost, xs = self._interpolate(x, u0, u1, t0, theta, return_traj=True)
            return x_end, xs      # xs: (steps_per_knot, ...) states within the interval

        _, seg = jax.lax.scan(
            interval_step, x0, (us[:-1], us[1:], self.tk(endpoint=False)))
        # (num_intervals, steps_per_knot, ...) -> (num_intervals*steps_per_knot, ...).
        # Merge the two leading axes explicitly: an inferred -1 divides by zero on
        # mjx.Data leaves that have a zero-sized dim (e.g. empty contact buffers).
        dense = jax.tree.map(lambda a: a.reshape((a.shape[0] * a.shape[1],) + a.shape[2:]), seg)
        return jax.tree.map(lambda a, b: jnp.concatenate([a[None], b], axis=0), x0, dense)

    @partial(jax.jit, static_argnames=['self'])
    def unroll(
        self,
        key: PRNGKeyArray,
        x0: PyTree,
        theta: PyTree,
    ) -> tuple[jax.Array, jax.Array, PyTree]:
        """Training primitive. Returns (us, costs, xs) per particle per knot.

        us:    (particles, num_knots, action_dim)
        costs: (particles, num_knots)
        xs:    per-knot states, shape (particles, num_knots-1, ...) — state at end of each interval
        """
        key1, key2 = jax.random.split(key)
        keys1 = jax.random.split(key1, (self.particles,))
        u0s = jax.vmap(self.mpc_init, in_axes=(0, None, None))(keys1, x0, theta)
        keys2 = jax.random.split(key2, (self.particles, self.num_knots - 1))

        def scan_one(particle_keys, u0):
            def step(carry, inp):
                (x1, u1), (u, cost) = self.transition_model(inp[0], carry[0], carry[1], theta, inp[1])
                return (x1, u1), (u, cost, x1)
            (final_x, _), (us, costs, xs) = jax.lax.scan(
                step, (x0, u0), (particle_keys, self.tk(endpoint=False))
            )
            return final_x, us, costs, xs

        final_xs, us, costs, xs = jax.vmap(scan_one)(keys2, u0s)
        final_costs = jax.vmap(self.mpc_terminal_cost, in_axes=(0, None))(final_xs, theta)
        us    = jax.tree.map(lambda head, tail: jnp.concat([head[:, None], tail], axis=1), u0s, us)
        costs = jnp.concatenate((costs, final_costs[:, None]), axis=-1)
        # xs: (particles, num_knots-1, ...) — state at end of each knot interval
        return us, costs, xs

    @partial(jax.jit, static_argnames=['self'])
    def run_sis(
        self,
        key,
        x0: PyTree,
        theta: PyTree
    ) -> tuple[jax.Array, jax.Array]:
        # SIS is `unroll` without the per-knot physical states.
        trajs, costs, _xs = self.unroll(key, x0, theta)
        return trajs, costs

    @partial(jax.jit, static_argnames=["self"])
    def _sir(
        self,
        key: PRNGKeyArray,
        x0: PyTree, theta: PyTree
    ) -> tuple[PyTree[jax.Array], jax.Array]:
        key1, key2 = jax.random.split(key)
        keys = jax.random.split(key1, (self.particles,))
        num_intervals = self.num_knots - 1
        step_keys = jax.random.split(key2, num_intervals * self.particles).reshape(num_intervals, self.particles)
        resample_keys = jax.random.split(key2, num_intervals)
        x0s = jax.tree.map(lambda x: jnp.repeat(x[None], self.particles, axis=0), x0)
        u0s = jax.vmap(self.mpc_init, in_axes=(0, None, None))(keys, x0, theta)

        def step(carry, inp):
            log_ws, xs, us = carry
            step_keys, resample_key, t = inp
            (new_xs, new_us), (_, costs) = jax.vmap(
                lambda k, x, u: self.transition_model(k, x, u, theta, t))(step_keys, xs, us)

            # Reweight, and resample only when the effective sample size collapses.
            log_ws = log_ws - costs
            w = jnp.exp(log_ws - jnp.max(log_ws))
            ps = w / jnp.sum(w)
            should_resample = 1 / jnp.sum(ps ** 2) < self.ess_resample_thresh * self.particles

            def do_resample(_):
                parents = jax.random.choice(resample_key, self.particles, shape=(self.particles,), p=ps, replace=True)
                return parents, jnp.zeros(self.particles)

            def dont_resample(_):
                return jnp.arange(self.particles), log_ws

            parents, log_ws = jax.lax.cond(should_resample, do_resample, dont_resample, operand=None)

            new_xs = jax.tree.map(lambda a: a[parents], new_xs)
            new_us = jax.tree.map(lambda a: a[parents], new_us)
            return (log_ws, new_xs, new_us), (new_us, parents, costs)

        (_, final_xs, _), (us, ancestors, costs) = jax.lax.scan(
            step, (jnp.zeros(self.particles), x0s, u0s),
            (step_keys, resample_keys, self.tk(endpoint=False)))
        states = jax.tree.map(lambda u0, u: jnp.concat([u0[None], u], axis=0), u0s, us)
        final_costs = jax.vmap(self.mpc_terminal_cost, in_axes=(0, None))(final_xs, theta)
        costs = jnp.concatenate((costs.T, final_costs[:, None]), axis=-1)
        return states, ancestors, costs, final_xs

    @partial(jax.jit, static_argnames=['self'])
    def run_sir(
        self,
        key: PRNGKeyArray,
        x0: PyTree, theta: PyTree
    ) -> tuple[PyTree[jax.Array], jax.Array]:
        states, ancestors, costs, _ = self._sir(key, x0, theta)
        trajs = self.choose_paths(states, ancestors, jnp.arange(self.particles))
        return trajs, costs

    @staticmethod
    @jax.jit
    def choose_path(states, ancestors, index):
        final_idx = ancestors.shape[0]

        # names + accessor + builder from the *instance*
        field_names, accessor, builder = get_field_info(states)

        def body(idx, tau):
            vals = [accessor(states, name)[tau, idx] for name in field_names]
            state = builder(vals)  # rebuild in the original container type
            return ancestors[tau-1, idx], state

        back_indices = jnp.arange(final_idx, -1, -1)
        _, traj_back = jax.lax.scan(body, index, back_indices)
        return jax.tree.map(lambda x: x[::-1], traj_back)

    @staticmethod
    @jax.jit
    def choose_paths(
        states: PyTree[jax.Array],  # (num_knots,particles,...)
        ancestors: jax.Array,       # (num_knots,particles)
        indices: jax.Array          # (N,) where self.particles >= N
    ) -> PyTree[jax.Array]:         # (N,horizon,...)
        """
        Trace back the trajectories for N self.particles among self.particles, for finite horizon horizon.

        Args:
            states: PyTree with leaves of shape (horizon, self.particles, ...) representing the full simulation state at each time step.
            ancestors: (horizon, self.particles) int32 array specifying indices for `states` that were resampled.
            indices: jax.Array, array of indices for the final N particles at time horizon.

        Returns:
            NamedTuple: containing fields with shape (N, horizon, ...), representing N reconstructed trajectories.
        """
        return jax.vmap(PFMP.choose_path, in_axes=(None, None, 0))(states, ancestors, indices)
