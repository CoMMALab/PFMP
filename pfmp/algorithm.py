import jax
import jax.numpy as jnp

from functools import partial
from jaxtyping import PyTree, PRNGKeyArray
from typing import NamedTuple, Callable
from dataclasses import is_dataclass, fields
from .interpolate import hermite_interp1d


DEFAULT_MUJOCO_DT = 0.002

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

class PFMP(NamedTuple):
    """ Particle Filtering Motion Planning (PFMP) """

    mpc_dynamics: Callable
    mpc_cost: Callable
    mpc_proposal: Callable

    horizon: int
    num_knots: int
    particles: int
    ess_resample_thresh: float = 0.8
    dt: float = DEFAULT_MUJOCO_DT

    def tk(self):
        return jnp.linspace(0., self.horizon, self.num_knots)

    @partial(jax.jit, static_argnames=['self'])
    def transition_model(self, key, x0: PyTree, u0: jax.Array, theta: PyTree, t: int) -> PyTree:
        interval_len = self.horizon / (self.num_knots - 1)
        steps_per_knot = round(interval_len / self.dt)
        t0 = t.astype(float); t1 = t0 + interval_len

        # Propose new spline knot
        u1 = self.mpc_proposal(key, x0, u0, theta, t)

        num_derivs = int(u0.shape[0])
        D = int(u0.size // num_derivs)
        J0 = jnp.reshape(u0, (num_derivs, D))
        J1 = jnp.reshape(u1, (num_derivs, D))

        # Interpolate `u` over the spline
        t_grid, us = hermite_interp1d(
            t0, t1,
            J0, J1,
            ord=2*num_derivs-1,
            num=steps_per_knot,
            nout=num_derivs
        )

        def interpolate_step(
            carry: tuple[jax.Array, float, float],
            inps: tuple[jax.Array, float]
        ) -> tuple[tuple[jax.Array, float, float], None]:
            x, c = carry
            u, s = inps
            (x_next, y) = self.mpc_dynamics(x, u, theta, s)
            c_next = self.mpc_cost(x_next, u, theta, s)
            return (x_next, c + c_next), y

        # Scan over the spline and numerically integrate cost
        (x1, cost), ys = jax.lax.scan(interpolate_step, (x0, 0.0), (us.T, t_grid))
        return (x1, u1), (ys, cost)

    @partial(jax.jit, static_argnames=['self'])
    def run_sis(
        self,
        key,
        x0: PyTree,
        u0: PyTree,
        theta: PyTree
    ) -> tuple[jax.Array, jax.Array]:
        keys = jax.random.split(key, (self.particles, self.num_knots))
        (states, costs) = jax.vmap(
            lambda particle_keys: jax.lax.scan(
                lambda carry, inp: self.transition_model(inp[0], carry[0], carry[1], theta, inp[1]),
                (x0, u0),
                (particle_keys, jnp.linspace(0, self.horizon, self.num_knots))
            )[1]
        )(keys)
        return states, -costs

    def sample_resample_step(
        self,
        carry: tuple[jax.Array, PyTree[jax.Array], PyTree],
        inp: tuple[PRNGKeyArray, PRNGKeyArray, int, float]
    ) -> tuple[jax.Array, PyTree[jax.Array], PyTree]:
        (log_ws, xs, us, theta) = carry
        (step_keys, resample_key, t) = inp

        (new_xs, new_us), (interp_ys, costs) = jax.vmap(
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
        resampled_ys = jax.tree.map(lambda x: x[parents], interp_ys)

        return (resampled_log_ws, resampled_xs, resampled_us, theta), (resampled_ys, parents, costs)

    @partial(jax.jit, static_argnames=['self'])
    def run_sir(
        self,
        key: PRNGKeyArray,
        x0: PyTree, u0: jax.Array, theta: PyTree
    ) -> tuple[PyTree[jax.Array], jax.Array]:
        step_keys = jax.random.split(key, self.num_knots * self.particles).reshape(self.num_knots, self.particles)
        resample_keys = jax.random.split(key, self.num_knots)
        x0s = jax.tree.map(lambda x: jnp.repeat(x[None], self.particles, axis=0), x0)
        u0s = jax.tree.map(lambda u: jnp.repeat(u[None], self.particles, axis=0), u0)
        (states, ancestors, costs) = jax.lax.scan(
            lambda carry, inp: self.sample_resample_step(carry, inp),
            (jnp.zeros(self.particles), x0s, u0s, theta),
            (step_keys, resample_keys, jnp.linspace(0, self.horizon, self.num_knots))
        )[1]
        trajs = self.choose_paths(states, ancestors, jnp.arange(self.particles))
        return trajs, -costs

    @staticmethod
    @jax.jit
    def choose_path(states, ancestors, index):
        num_knots = ancestors.shape[0]

        # names + accessor + builder from the *instance*
        field_names, accessor, builder = get_field_info(states)

        def body(idx, tau):
            # pull out each leaf at (tau, idx)
            vals = [accessor(states, name)[tau, idx] for name in field_names]
            state = builder(vals)  # rebuild in the original container type
            return ancestors[tau, idx], state

        times = jnp.arange(num_knots - 1, -1, -1)
        _, traj_back = jax.lax.scan(body, index, times)
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
