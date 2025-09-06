import jax
import jax.numpy as jnp
import genjax

from genjax import gen, ChoiceMapBuilder as C
from jaxtyping import PyTree, PRNGKeyArray
from typing import NamedTuple, Callable
from dataclasses import is_dataclass, fields


def get_field_info(x):
    """
    Given a states‐object x, return
      • field_names: tuple of keys (str names or integer positions)
      • accessor:  fn(obj, key) -> getattr or obj[key]
    """
    # dataclass?
    if is_dataclass(x):
        names = tuple(f.name for f in fields(x))
        return names, lambda obj, n: getattr(obj, n)
    # namedtuple?
    if hasattr(x, "_fields") and isinstance(x._fields, tuple):
        names = x._fields
        return names, lambda obj, n: getattr(obj, n)
    # plain tuple?
    if isinstance(x, tuple):
        names = tuple(range(len(x)))
        return names, lambda obj, i: obj[i]
    raise TypeError(f"Unsupported state type: {type(x)}")


class _PFMP_V1(NamedTuple):
    """
    Particle Flow Motion Planning (PFMP): a modular GPU-accelerated JIT-compiled planning-as-
    inference kernel that generalizes MPPI, compatible with both MJX and tuple/dataclass types.

    Given constant shape (H, K) ∈ ℕ^2, we denote the following relations

    1) PyTree            => 1-Particle
    2) PyTree[jax.Array] => K-Particles | (H,K)-Particles

    Where `Particle` is the type of the return value of `stoch_transition`.

    Args:
        key: JAX PRNGKey.
        (PAR) stoch_transition : (Particle, PyTree, t) -> Particle. Stochastic transition model for tuple or dataclass state.
        (MJX) stoch_transition : (mjx.Data, PyTree, t) -> (mjx.Data, Particle). Stochastic transition model for Mujoco XLA.
        (PAR) cost_fn : (Particle, PyTree, t) -> float. Cost function for a Particle.
        (MJX) cost_fn : (mjx.Data, PyTree, t) -> float. Cost function for Mujoco XLA Data.
        carry_mjx: bool. Indicates if the type of the carry in `self.kernel` is `mjx.Data` or `Particle`.
    """

    stoch_transition: Callable[[PyTree,PyTree,int],PyTree]
    cost_fn: Callable[[PyTree,PyTree,int],float]
    carry_mjx: bool = False

    def kernel(
        self,
        carry: tuple[jax.Array, PyTree[jax.Array], PyTree],
        inp: tuple[PRNGKeyArray, PRNGKeyArray, bool, int]
    ) -> tuple[jax.Array, PyTree[jax.Array], PyTree]:
        (log_ws, xs, u) = carry
        (generate_key, resample_key, should_resample, t) = inp

        k = log_ws.shape[0]

        @gen
        def model_transition(x: PyTree, u: PyTree, t: int) -> PyTree:
            new_x = self.stoch_transition(x, u, t) @ "MPC"
            if self.carry_mjx:
                genjax.bernoulli(-self.cost_fn(new_x[0], u, t + 1)) @ "O"
            else:
                genjax.bernoulli(-self.cost_fn(new_x, u, t + 1)) @ "O"
            return new_x

        O1 = C.n().at["O"].set(True)
        def generate_cond_O1(key: PRNGKeyArray, x: PyTree, u: PyTree, t: int) -> tuple[PyTree, float]:
            trace, score = model_transition.generate(key, O1, (x, u, t))
            return trace.retval, score
        v_generate_cond_O1 = jax.vmap(generate_cond_O1, in_axes=[0, 0, None, None])

        if self.carry_mjx:
            (new_xs, new_ys), log_ws_inc = v_generate_cond_O1(generate_key, xs, u, t)
        else:
            new_xs, log_ws_inc = v_generate_cond_O1(generate_key, xs, u, t)

        new_log_ws = log_ws + log_ws_inc

        if self.carry_mjx:
            def do_resample(_):
                k = new_log_ws.shape[0]
                ws = jnp.exp(new_log_ws - jnp.max(new_log_ws))
                ps = ws / jnp.sum(ws)
                parents = jax.random.choice(resample_key, k, shape=(k,), p=ps, replace=True)
                resampled_xs = jax.tree.map(lambda x: x[parents], new_xs)
                resampled_ys = jax.tree.map(lambda y: y[parents], new_ys)
                return resampled_xs, resampled_ys, parents, jnp.zeros(k)

            def dont_resample(_):
                return new_xs, new_ys, jnp.arange(k), new_log_ws

            # adaptive resampling
            ws = jnp.exp(new_log_ws - jnp.max(new_log_ws))
            ps = ws / jnp.sum(ws)
            should_resample = 1 / jnp.sum(ps ** 2) < 0.8*k

            next_xs, next_ys, parents, next_log_ws = jax.lax.cond(
                should_resample, do_resample, dont_resample, operand=None
            )

            return (next_log_ws, next_xs, u), (next_ys, parents)
        else:
            def do_resample(_):
                k = new_log_ws.shape[0]
                ws = jnp.exp(new_log_ws - jnp.max(new_log_ws))
                ps = ws / jnp.sum(ws)
                parents = jax.random.choice(resample_key, k, shape=(k,), p=ps, replace=True)
                resampled_xs = jax.tree.map(lambda x: x[parents], new_xs)
                return resampled_xs, parents, jnp.zeros(k)

            def dont_resample(_):
                return new_xs, jnp.arange(k), new_log_ws

            next_xs, parents, next_log_ws = jax.lax.cond(
                should_resample, do_resample, dont_resample, operand=None
            )

            return (next_log_ws, next_xs, u), (next_xs, parents)

    def sample_forward(
        self,
        key: PRNGKeyArray,
        block_shape: tuple[int, int],
        x0: PyTree, u: PyTree,
        resampler_schedule: jax.Array
    ) -> tuple[PyTree[jax.Array], jax.Array]:
        (h, k) = block_shape

        generate_keys = jax.random.split(key, h * k).reshape(h, k)
        resample_keys = jax.random.split(key, h)

        x0s = jax.tree.map(lambda x: jnp.repeat(x[None], k, axis=0), x0)

        return jax.lax.scan(
            self.kernel,
            (jnp.zeros(k), x0s, u),
            (generate_keys, resample_keys, resampler_schedule, jnp.arange(h))
        )[1]

    def sample_forward_weighted(
        self,
        key: PRNGKeyArray,
        block_shape: tuple[int, int],
        x0: PyTree, u: PyTree,
    ) -> tuple[PyTree[jax.Array], jax.Array]:
        (h, k) = block_shape

        generate_keys = jax.random.split(key, h * k).reshape(h, k)
        resample_keys = jax.random.split(key, h)

        xs0 = jax.tree.map(lambda x: jnp.broadcast_to(x, (k,) + jnp.shape(x)), x0)

        out = jax.lax.scan(
            self.kernel,
            (jnp.zeros(k), xs0, u),
            (generate_keys, resample_keys, jnp.full(h, False), jnp.arange(h))
        )
        return (jax.tree.map(lambda x: jnp.swapaxes(x, 0, 1), out[1][0]), out[0][0])

    @staticmethod
    @jax.jit
    def choose_path(
        states,                  # dataclass/namedtuple instance or plain tuple of arrays
        ancestors: jnp.ndarray,  # (H, K) int32
        index: int               # final particle index at time H
    ):
        """
        Trace back the trajectory for one particle among K, for finite horizon H.

        Args:
            states: PyTree with leaves of shape (H, K, ...) representing the full simulation state at each time step.
            ancestors: (H, K) int32 array specifying indices for `states` that were resampled.
            index: int, index of the final particle at time H.

        Returns:
            NamedTuple: containing fields with shape (H, ...), representing the reconstructed trajectory.
        """
        h = ancestors.shape[0]
        data_type = type(states)

        # introspect on the *instance* to get names + an accessor
        field_names, accessor = get_field_info(states)

        def body(idx, tau):
            # pull out each leaf at (tau, idx)
            vals = [accessor(states, name)[tau, idx] for name in field_names]

            # rebuild exactly the same container type
            if isinstance(field_names[0], str):
                # dataclass or namedtuple
                state = data_type(**{n: v for n, v in zip(field_names, vals)})
            else:
                # plain tuple
                state = tuple(vals)

            return ancestors[tau, idx], state

        # scan from h-1 → 0, then reverse
        times = jnp.arange(h-1, -1, -1)
        _, traj_back = jax.lax.scan(body, index, times)
        return jax.tree_map(lambda x: x[::-1], traj_back)

    @staticmethod
    @jax.jit
    def choose_paths(
        states: PyTree[jax.Array],  # (H,K,...)
        ancestors: jax.Array,       # (H,K)
        indices: jax.Array          # (L,) where K >= L
    ) -> PyTree[jax.Array]:         # (L,H,...)
        """
        Trace back the trajectories for L particles among K, for finite horizon H.

        Args:
            states: PyTree with leaves of shape (H, K, ...) representing the full simulation state at each time step.
            ancestors: (H, K) int32 array specifying indices for `states` that were resampled.
            indices: jax.Array, array of indices for the final L particles at time H.

        Returns:
            NamedTuple: containing fields with shape (L, H, ...), representing L reconstructed trajectories.
        """
        return jax.vmap(PFMP.choose_path, in_axes=(None, None, 0))(states, ancestors, indices)