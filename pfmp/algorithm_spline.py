import jax
import jax.numpy as jnp
import genjax

from genjax import gen, ChoiceMapBuilder as C
from jaxtyping import PyTree, PRNGKeyArray
from typing import NamedTuple, Callable
from dataclasses import is_dataclass, fields
from .interpolate import hermite_spline


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


def make_zoh_model(mpc_transition, mpc_cost):

    @gen
    def transition_model(x: PyTree, theta: PyTree, t: int) -> PyTree:
        new_x = mpc_transition(x, theta, t) @ "MPC transition"
        genjax.bernoulli(-mpc_cost(new_x, theta, t + 1)) @ "O"
        return new_x

    return transition_model

def make_transition_model(mpc_transition, mpc_cost, l: int):

    @gen
    def spline_transition_model(x: PyTree, theta: PyTree, t: int) -> PyTree:
        # Sample next-step u proposal
        ctrl_proposed = (mpc_transition(x, theta, t) @ "MPC transition").ctrl

        # Interpolate u over spline
        t0, t1 = t.astype(float), (t + 1).astype(float)
        u_spline = jnp.permute_dims(
            hermite_spline(
                jnp.reshape(x.ctrl, (x.ctrl.shape[0], -1)),
                jnp.reshape(ctrl_proposed, (ctrl_proposed.shape[0], -1)),
                t0, t1, l + 1
            ),
            (1, 0, 2)
        )

        dt = 1 / l

        def interpolate_ctrl(carry: tuple[jax.Array, float], ui: tuple[jax.Array, float]) -> tuple[tuple[jax.Array, float], None]:
            xi, cost, t = carry
            x = jnp.squeeze(ui[0])
            v = jnp.squeeze(ui[1])
            a = jnp.squeeze(ui[2])
            cmap = C.n().at["u"].set(x) \
                        .at["du"].set(v) \
                        .at["ddu"].set(a)
            xi_next = mpc_transition.generate(jax.random.key(0), cmap, (xi, theta, t))[0].retval
            ci = mpc_cost(xi_next, theta, t)
            return (xi_next, cost + ci, t + dt), xi_next

        (_, total_cost, _), interpolated = jax.lax.scan(interpolate_ctrl, (x, 0.0, t0), u_spline[1:])

        # Probability of optimality is proportional
        # to exponential negative cost
        genjax.bernoulli(probs=jnp.exp(-total_cost)) @ 'O'

        return interpolated

    return spline_transition_model


class PFMP(NamedTuple):
    """
    Particle Flow Motion Planning (PFMP): a modular JIT-compiled and GPU-accelerated planning-as-
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
        carry_mjx: bool. Indicates if the type of the carry in `kernel` is `mjx.Data` or `Particle`.
    """

    transition_model: Callable[[PyTree,PyTree,int,int],PyTree]
    cond_O1 = C.n().at['O'].set(True)  # Optimality Condition

    @jax.jit
    def generate_cond_O1(self, theta, carry, inp) -> tuple[PyTree, float]:
        (x, score) = carry
        (key, t) = inp
        trace, dscore = self.transition_model.generate(key, self.cond_O1, (x, theta, t))
        return (jax.tree.map(lambda x: x[-1], trace.retval), score + dscore), trace.retval

    @jax.jit
    def unfold_cond_O1(self, keys, x0, theta):
        h = jnp.shape(keys)[0]
        (carry_out, trace_retvals) = jax.lax.scan(
            lambda carry, inp: self.generate_cond_O1(theta, carry, inp),
            (x0, 0.0),
            (keys, jnp.arange(h))
        )
        return trace_retvals, carry_out[1]

    def run_sis(
        self,
        key,
        x0: PyTree,
        theta: PyTree,
        h: int,
        k: int
    ) -> tuple[jax.Array, jax.Array]:
        generate_keys = jax.random.split(key, h * k).reshape(k, h)
        traj_retvals, scores = jax.vmap(
            lambda key: self.unfold_cond_O1(key, x0, theta)
        )(generate_keys)
        return traj_retvals, scores

    @jax.jit
    def sample_resample_step(
        self,
        carry: tuple[jax.Array, PyTree[jax.Array], PyTree],
        inp: tuple[PRNGKeyArray, PRNGKeyArray, int, float],
        rho: float
    ) -> tuple[jax.Array, PyTree[jax.Array], PyTree]:
        (log_ws, xs, theta) = carry
        (generate_keys, resample_key, t) = inp
        k = log_ws.shape[0]

        (new_xs, new_log_ws), new_interp = jax.vmap(
            lambda xs, log_ws, key: self.generate_cond_O1(theta, (xs, log_ws), (key, t))
        )(xs, log_ws, generate_keys)

        ws = jnp.exp(new_log_ws - jnp.max(new_log_ws))
        ps = ws / jnp.sum(ws)
        # ess = 1 / jnp.sum(ps ** 2)
        # should_resample = ess / k < rho
        should_resample = 1 / jnp.sum(ps ** 2) < k * rho

        def do_resample(_):
            parents = jax.random.choice(resample_key, k, shape=(k,), p=ps, replace=True)
            return parents, jnp.zeros(k)

        def dont_resample(_):
            return jnp.arange(k), new_log_ws

        parents, resampled_log_ws = jax.lax.cond(
            should_resample, do_resample, dont_resample, operand=None
        )

        resampled_xs = jax.tree.map(lambda x: x[parents], new_xs)
        resampled_interp = jax.tree.map(lambda x: x[parents], new_interp)

        return (resampled_log_ws, resampled_xs, theta), (resampled_interp, parents)

    def run_sir(
        self,
        key: PRNGKeyArray,
        x0: PyTree, theta: PyTree,
        h: int, k: int, rho: float
    ) -> tuple[PyTree[jax.Array], jax.Array]:
        generate_keys = jax.random.split(key, h * k).reshape(h, k)
        resample_keys = jax.random.split(key, h)
        x0s = jax.tree.map(lambda x: jnp.repeat(x[None], k, axis=0), x0)
        return jax.lax.scan(
            lambda carry, inp: self.sample_resample_step(carry, inp, rho),
            (jnp.zeros(k), x0s, theta),
            (generate_keys, resample_keys, jnp.arange(h))
        )[1]

    @staticmethod
    @jax.jit
    def choose_path(
        states: PyTree,          # dataclass/namedtuple instance or plain tuple of arrays
        ancestors: jnp.ndarray,  # (H, K) int32
        index: int               # final particle index at time H
    ) -> PyTree:
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