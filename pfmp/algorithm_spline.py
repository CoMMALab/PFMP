import jax
import jax.numpy as jnp
import genjax
import numpy as np

from functools import partial
from genjax import gen, ChoiceMapBuilder as C
from jaxtyping import PyTree, PRNGKeyArray
from typing import NamedTuple, Callable
from dataclasses import is_dataclass, fields
from .interpolate import hermite_spline
# from .interpolateh import hermite_interp1d

import jax.numpy as jnp
from jax.tree_util import tree_leaves, tree_map


def get_field_info(x):
    """
    Return:
      - field_names: tuple of keys
      - accessor(obj, key) -> leaf array (indexable by [tau, idx] for arrays with (H,K,...))
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

    # jax.Array -> treat as ONE field: the whole array (H, K, ...)
    if jax.Array and isinstance(x, jax.Array):
        names = ("__array__",)
        accessor = lambda obj, _k: obj               # shape (H, K, ...)
        builder  = lambda vals: vals[0]              # just return the array leaf
        return names, accessor, builder

    # plain tuple
    if isinstance(x, tuple):
        names = tuple(range(len(x)))
        accessor = lambda obj, i: obj[i]
        builder  = lambda vals: tuple(vals)
        return names, accessor, builder

    raise TypeError(f"Unsupported state type: {type(x)}")


def make_transition_model(
        mpc_proposal: Callable,
        mpc_cost: Callable,
        mpc_dynamics: Callable,
        l: int = 1
    ):

    @gen
    def mpc_hermite_spline(x0: PyTree, u0: jax.Array, theta: PyTree, t: int) -> PyTree:
        # Propose new Hermite spline model predictive control point `u` from a generative function
        u1 = mpc_proposal(x0, u0, theta, t) @ 'u'

        t0, t1 = t.astype(float), (t + 1).astype(float)
        nderivs = int(u0.shape[0])  # number of constraints (1,2,3)
        D = int(u0.size // nderivs)
        J0 = jnp.reshape(u0, (nderivs, D))
        J1 = jnp.reshape(u1, (nderivs, D))

        # Interpolate `u` over the spline

        # TODO: deprecate old spline
        Js = hermite_spline(
            J0, J1,
            t0, t1,
            num_points = l + 1
        )
        t_grid = jnp.linspace(t0, t1, l + 1, dtype=J0.dtype)

        # TODO: use new spline
        # t_grid, Js = hermite_interp1d(
        #     t0, t1,
        #     J0, J1,
        #     ord=2*nderivs-1,
        #     num=l+1,
        #     nout=nderivs
        # )

        # (nderivs, l+1, D) -> (l+1, nderivs, D)
        us = jnp.permute_dims(Js, (1, 0, 2)).squeeze()

        def interpolate_step(
            carry: tuple[jax.Array, float, float],
            inps: tuple[jax.Array, float]
        ) -> tuple[tuple[jax.Array, float, float], None]:
            x_i, c = carry
            u_i, t_i = inps
            (x_i_next, y_i_next) = mpc_dynamics(x_i, u_i, theta, t_i)
            c_i = mpc_cost(y_i_next, theta, t_i)
            return (x_i_next, c + c_i), y_i_next

        # Scan over the spline and numerically integrate cost
        (x1, cost), interp_ys = jax.lax.scan(interpolate_step, (x0, 0.0), (us[1:], t_grid[1:]))

        # `u` are optimal with transient probability
        # equal to exponential negative cost
        genjax.bernoulli(probs=jnp.exp(-cost)) @ 'O'

        return (x1, u1), interp_ys

    return mpc_hermite_spline


class PFMP(NamedTuple):
    """
    Particle Filtering for Model Predictive Control (PFMPC)
    """

    transition_model: Callable[[PyTree,PyTree,PyTree,int],PyTree]
    cond_O1 = C.n().at['O'].set(True)  # Optimality Condition

    @jax.jit
    def generate_cond_O1(self, theta, carry, inp) -> tuple[PyTree, float]:
        (x0, u0, score) = carry
        (key, t) = inp
        (trace, dscore) = self.transition_model.generate(key, self.cond_O1, (x0, u0, theta, t))
        (x1, u1) = trace.retval[0]
        interp_ys = trace.retval[1]
        return (x1, u1, score + dscore), interp_ys

    @jax.jit
    def unfold_cond_O1(self, keys, x0, u0, theta):
        h = jnp.shape(keys)[0]
        (carry_out, trajs) = jax.lax.scan(
            lambda carry, inp: self.generate_cond_O1(theta, carry, inp),
            (x0, u0, 0.0),
            (keys, jnp.arange(h))
        )
        return trajs, carry_out[2]

    @partial(jax.jit, static_argnames=['h', 'k'])
    def run_sis(
        self,
        key,
        x0: PyTree,
        u0: PyTree,
        theta: PyTree,
        h: int,
        k: int
    ) -> tuple[jax.Array, jax.Array]:
        generate_keys = jax.random.split(key, h * k).reshape(k, h)
        k_trajs, k_scores = jax.vmap(
            lambda key: self.unfold_cond_O1(key, x0, u0, theta)
        )(generate_keys)
        return k_trajs, k_scores

    def sample_resample_step(
        self,
        carry: tuple[jax.Array, PyTree[jax.Array], PyTree[jax.Array], PyTree],
        inp: tuple[PRNGKeyArray, PRNGKeyArray, float],
        rho: float
    ) -> tuple[jax.Array, PyTree[jax.Array], PyTree]:
        (log_ws, xs, us, theta) = carry
        (generate_keys, resample_key, t) = inp
        k = log_ws.shape[0]

        (sampled_xs, sampled_us, sampled_log_ws), sampled_interp_ys = jax.vmap(
            lambda xs, us, log_ws, key: self.generate_cond_O1(theta, (xs, us, log_ws), (key, t))
        )(xs, us, log_ws, generate_keys)
        # (sampled_xs, sampled_log_ws), sampled_us = jax.vmap(self.generate_cond_O1, in_axes=[0, (0,)])(xs, us, log_ws, generate_keys)

        ws = jnp.exp(sampled_log_ws - jnp.max(sampled_log_ws))
        ps = ws / jnp.sum(ws)
        ess = 1 / jnp.sum(ps ** 2)
        should_resample = ess < rho * k

        def do_resample(_):
            parents = jax.random.choice(resample_key, k, shape=(k,), p=ps, replace=True)
            return parents, jnp.zeros(k)

        def dont_resample(_):
            return jnp.arange(k), sampled_log_ws

        parents, resampled_log_ws = jax.lax.cond(
            should_resample, do_resample, dont_resample, operand=None
        )

        resampled_xs = jax.tree.map(lambda x: x[parents], sampled_xs)
        resampled_us = jax.tree.map(lambda u: u[parents], sampled_us)
        resampled_interp_ys = jax.tree.map(lambda x: x[parents], sampled_interp_ys)

        return (resampled_log_ws, resampled_xs, resampled_us, theta), (resampled_interp_ys, parents)

    @partial(jax.jit, static_argnames=['h', 'k', 'rho'])
    def run_sir(
        self,
        key: PRNGKeyArray,
        x0: PyTree, u0: jax.Array, theta: PyTree,
        h: int, k: int, rho: float
    ) -> tuple[PyTree[jax.Array], jax.Array]:
        generate_keys = jax.random.split(key, h * k).reshape(h, k)
        resample_keys = jax.random.split(key, h)
        x0s = jax.tree.map(lambda x: jnp.repeat(x[None], k, axis=0), x0)
        u0s = jax.tree.map(lambda u: jnp.repeat(u[None], k, axis=0), u0)
        (states, ancestors) = jax.lax.scan(
            lambda carry, inp: self.sample_resample_step(carry, inp, rho),
            (jnp.zeros(k), x0s, u0s, theta),
            (generate_keys, resample_keys, jnp.arange(h))
        )[1]
        return (states, ancestors)

    @staticmethod
    @jax.jit
    def choose_path(states, ancestors, index):
        h = ancestors.shape[0]

        # names + accessor + builder from the *instance*
        field_names, accessor, builder = get_field_info(states)

        def body(idx, tau):
            # pull out each leaf at (tau, idx)
            vals = [accessor(states, name)[tau, idx] for name in field_names]
            state = builder(vals)  # rebuild in the original container type
            return ancestors[tau, idx], state

        times = jnp.arange(h - 1, -1, -1)
        _, traj_back = jax.lax.scan(body, index, times)
        return jax.tree_map(lambda x: x[::-1], traj_back)

    @staticmethod
    @jax.jit
    def choose_paths(
        states: PyTree[jax.Array],  # (H,K,...)
        ancestors: jax.Array,       # (H,K)
        indices: jax.Array          # (N,) where N <= K
    ) -> PyTree[jax.Array]:         # (N,H,...)
        """
        Trace back the trajectories for N particles among K, for finite horizon H.

        Args:
            states: PyTree with leaves of shape (H, K, ...) representing the tracked simulation state at each time step.
            ancestors: (H, K) int32 array specifying indices for `states` that were resampled.
            indices: jax.Array, array of indices for the final L particles at time H.

        Returns:
            NamedTuple: containing fields with shape (L, H, ...), representing N reconstructed trajectories.
        """
        return jax.vmap(PFMP.choose_path, in_axes=(None, None, 0))(states, ancestors, indices)

    @staticmethod
    @partial(jax.jit, static_argnames=("cost_fn",))
    def project_cost(cost_fn, traj, th):
        """
        Project a cost function over a single trajectory.

        Args:
            cost_fn: callable (traj_slice, th, t_scalar) -> array_like
                     where traj_slice corresponds to the PyTree leaves at a single (h, l).
            traj:    PyTree whose leaves have shape (H, L, ...).
            th:      static params (PyTree), not batched.

        Returns:
            array_like with shape (H, L, ...cost_fn output shape...).
        """
        leaf0 = tree_leaves(traj)[0]
        assert leaf0.ndim >= 2, "Each leaf must start with (H, L, ...)"
        H, L = leaf0.shape[:2]

        # (Optional) sanity-check that all leaves share (H, L)
        def _check(x):
            assert x.shape[:2] == (H, L), "All leaves must share leading (H, L)."
            return x
        _ = tree_map(_check, traj)

        # Build time grid t[h, l] = h + l/L  (shape: (H, L))
        t_dtype = leaf0.dtype if jnp.issubdtype(leaf0.dtype, jnp.floating) else jnp.float32
        t_HL = (jnp.arange(H, dtype=t_dtype)[:, None] +
                jnp.arange(L, dtype=t_dtype)[None, :] / L)

        def f_one(a_slice, t_scalar):
            return cost_fn(a_slice, th, t_scalar)

        # vmap over H, then L (or vice versa); traj leaves are (H, L, ...)
        vmapped = jax.vmap(                    # over H axis
                    jax.vmap(f_one,            # over L axis
                             in_axes=(0, 0)),
                    in_axes=(0, 0)
                  )

        return vmapped(traj, t_HL)

    @staticmethod
    @partial(jax.jit, static_argnames=("cost_fn",))
    def project_costs(cost_fn, trajs, th):
        """
        Batched version: vmap project_cost over leading K.

        Args:
            cost_fn: callable (traj_slice, th, t_scalar) -> array_like
            trajs:   PyTree whose leaves have shape (K, H, L, ...).
            th:      static params (PyTree), not batched.

        Returns:
            array_like with leading shape (K, H, L, ...cost_fn output shape...).
        """
        # Quick shape probe for (K, H, L)
        leaf0 = tree_leaves(trajs)[0]
        assert leaf0.ndim >= 3, "Each leaf must start with (K, H, L, ...)"
        K = leaf0.shape[0]

        # (Optional) sanity-check all leaves share (K, H, L)
        def _check(x):
            assert x.shape[:3] == leaf0.shape[:3], "All leaves must share leading (K, H, L)."
            return x
        _ = tree_map(_check, trajs)

        # vmap the single-trajectory projector across K
        return jax.vmap(PFMP.project_cost, in_axes=(None, 0, None))(cost_fn, trajs, th)


    def __call__(
        self,
        key: PRNGKeyArray,
        x0: PyTree,
        u0: jax.Array,
        theta: PyTree,
        H: int,
        K: int,
        rho: float,
        n_samples: int,
        index: int = 0,
        return_best = False
    ) -> PyTree[jax.Array]:
        """
        Run the model to return trajectories

        Args:
            All args needed to run the model
        
        Returns:
            NamedTuple: containing fields with shape (L, H, ...), representing N reconstructed trajectories.
       
        """

        (states, ancestors) = self.run_sir(
            key=key, 
            x0=x0, 
            u0=u0,
            theta=theta, 
            h=H, 
            k=K, 
            rho=rho
        )
        if return_best:
            return self.choose_path(states, ancestors, index)
        return self.choose_paths(states, ancestors, jnp.arange(K))