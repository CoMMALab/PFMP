import jax
import jax.numpy as jnp

from functools import partial
from jaxtyping import PyTree, PRNGKeyArray
from typing import NamedTuple, Callable
from dataclasses import is_dataclass, fields
from hydrax.task_base import Task
from mujoco import mjx
from .interpolate import hermite_interp1d, Jet1, Jet2, Jet3


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

# Predictive Sampling
def ps_update(theta: PyTree, trajs: jax.Array, costs: jax.Array):
    best_idx = jnp.argmin(jnp.sum(costs, axis=1))
    new_mean = jax.tree.map(lambda u: u[best_idx], trajs)
    theta = theta._replace(mean = new_mean)
    return theta, costs[:,best_idx]

# Model Predictive Path Integral
def mppi_update(theta: PyTree, trajs: jax.Array, costs: jax.Array, temperature: float):
    weights = jnp.exp(-costs / temperature)
    weights = weights / jnp.sum(weights, axis=1, keepdims=True)
    new_mean = jax.tree.map(lambda u: jnp.sum(u * weights[:,:,None], axis=1), trajs)
    theta = theta._replace(mean = new_mean)
    return theta, jnp.sum(costs * weights, axis=1)

# Cross-Entropy Method
def cem_update(theta: PyTree, trajs: jax.Array, costs: jax.Array, elite_frac: float):
    num_elite = int(elite_frac * trajs.shape[0])
    elite_idxs = jnp.argsort(costs, axis=0)[:num_elite]
    new_mean = jax.tree.map(lambda u: jnp.mean(u[elite_idxs], axis=0), trajs)
    new_std  = jax.tree.map(lambda u: jnp.std(u[elite_idxs], axis=0), trajs)
    theta = theta._replace(mean=new_mean, du_stdev=new_std)
    return theta, jnp.mean(costs[elite_idxs], axis=0)

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

    def tk(self, endpoint: bool = True):
        if endpoint:
            return jnp.linspace(0., self.horizon, self.num_knots)
        else:
            return jnp.linspace(0., self.horizon, self.num_knots)[:-1]

    @partial(jax.jit, static_argnames=['self'])
    def transition_model(self, key, x0: PyTree, u0: jax.Array, theta: PyTree, t0: float) -> PyTree:
        step_len = self.horizon / (self.num_knots - 1)
        steps_per_knot = int(step_len / self.dt)
        t1 = t0 + step_len

        # Propose new spline knot
        u1 = self.mpc_proposal(key, x0, u0, theta, t0)

        # Interpolate `u` over the spline
        if isinstance(u0, (Jet1, jax.Array)):
            nout = 1
        elif isinstance(u0, Jet2):
            nout = 2
        elif isinstance(u0, Jet3):
            nout = 3

        tq, us = hermite_interp1d(
            t0, t1,
            u0, u1,
            num=steps_per_knot+1,
            nout=nout
        )

        def interpolate_step(
            carry: tuple[jax.Array, float, float],
            inps: tuple[jax.Array, float]
        ) -> tuple[tuple[jax.Array, float, float], None]:
            x, c = carry
            u, s = inps
            (x_next, y) = self.mpc_dynamics(x, u, theta, s)
            c_next = self.mpc_running_cost(x_next, u, theta, s)
            return (x_next, c + c_next), y

        # Scan over the spline and numerically integrate cost
        (x1, cost), _ys = jax.lax.scan(interpolate_step, (x0, 0.0), (jax.tree.map(lambda u: u[:-1], us), tq[:-1]))
        return (x1, u1), (u1, cost)

    @partial(jax.jit, static_argnames=['self'])
    def run_sis(
        self,
        key,
        x0: PyTree,
        theta: PyTree
    ) -> tuple[jax.Array, jax.Array]:
        key1, key2 = jax.random.split(key)
        keys1 = jax.random.split(key1, (self.particles,))
        u0s = jax.vmap(self.mpc_init, in_axes=(0,None,None))(keys1, x0, theta)
        keys2 = jax.random.split(key2, (self.particles, self.num_knots - 1))
        (final_xs, _), (us, costs) = jax.vmap(
            lambda particle_keys, u0: jax.lax.scan(
                lambda carry, inp: self.transition_model(inp[0], carry[0], carry[1], theta, inp[1]),
                (x0, u0),
                (particle_keys, self.tk(endpoint=False))
            )
        )(keys2, u0s)
        final_costs = jax.vmap(self.mpc_terminal_cost, in_axes=(0,None))(final_xs, theta)
        trajs = jax.tree.map(lambda head, tail: jnp.concat([head[:,None], tail], axis=1), u0s, us)
        costs = jnp.concatenate((costs, final_costs[:,None]), axis=-1)
        return trajs, costs

    def sample_resample_step(
        self,
        carry: tuple[jax.Array, PyTree[jax.Array], PyTree],
        inp: tuple[PRNGKeyArray, PRNGKeyArray, int, float]
    ) -> tuple[jax.Array, PyTree[jax.Array], PyTree]:
        (log_ws, xs, us, theta) = carry
        (step_keys, resample_key, t) = inp

        (new_xs, new_us), (_, costs) = jax.vmap(
            lambda key, xs, us: self.transition_model(key, xs, us, theta, t)
        )(step_keys, xs, us)

        new_log_ws = log_ws - costs
        ws = jnp.exp(new_log_ws - jnp.max(new_log_ws))
        ps = ws / jnp.sum(ws)
        ess = 1 / jnp.sum(ps ** 2)
        should_resample = ess < self.ess_resample_thresh * self.particles

        def do_resample(_):
            parents = jax.random.choice(resample_key, self.particles, shape=(self.particles,), p=ps, replace=True)
            return parents, jnp.zeros(self.particles)

        def dont_resample(_):
            return jnp.arange(self.particles), new_log_ws

        parents, resampled_log_ws = jax.lax.cond(
            should_resample, do_resample, dont_resample, operand=None
        )

        resampled_xs = jax.tree.map(lambda x: x[parents], new_xs)
        resampled_us = jax.tree.map(lambda u: u[parents], new_us)
        # resampled_ys = jax.tree.map(lambda x: x[parents], interp_ys)

        return (resampled_log_ws, resampled_xs, resampled_us, theta), (resampled_us, parents, costs)

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
        u0s = jax.vmap(self.mpc_init, in_axes=(0,None,None))(keys, x0, theta)
        (_, final_xs, _, _), (us, ancestors, costs) = jax.lax.scan(
            lambda carry, inp: self.sample_resample_step(carry, inp),
            (jnp.zeros(self.particles), x0s, u0s, theta),
            (step_keys, resample_keys, self.tk(endpoint=False))
        )
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
