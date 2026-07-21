import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
from typing import Callable, NamedTuple
from jaxtyping import PRNGKeyArray, PyTree

from .algorithm import PFMP


ObsFn = Callable[[PyTree, jax.Array, PyTree, float], jax.Array]
# (x, u_prev, theta, t) -> flat observation array


class ProposalLearner(NamedTuple):
    actor:        nn.Module   # (obs) -> (mean, log_std); user-defined architecture
    actor_params: PyTree
    actor_opt:    optax.OptState
    obs_fn:       ObsFn
    action_dim:   int

    @staticmethod
    def create(actor: nn.Module, obs_fn: ObsFn,
               obs_dim: int, action_dim: int, key: PRNGKeyArray,
               lr: float = 3e-4) -> "ProposalLearner":
        params = actor.init(key, jnp.zeros(obs_dim))
        tx = optax.adam(lr)
        return ProposalLearner(
            actor=actor,
            actor_params=params,
            actor_opt=tx.init(params),
            obs_fn=obs_fn,
            action_dim=action_dim,
        )

    # ── PFMP proposal interface ───────────────────────────────────────────────

    def as_init(self) -> Callable:
        """mpc_init: (key, x, theta) -> u"""
        def init(key: PRNGKeyArray, x: PyTree, theta: PyTree) -> jax.Array:
            mean, log_std = self.actor.apply(self.actor_params, self.obs_fn(x, jnp.zeros(self.action_dim), theta, 0.0))
            std = jnp.exp(jnp.clip(log_std, -5, 2))
            return jnp.tanh(mean + std * jax.random.normal(key, mean.shape))
        return init

    def as_proposal(self) -> Callable:
        """mpc_proposal: (key, x, u, theta, t) -> u_next"""
        def proposal(key: PRNGKeyArray, x: PyTree, u: jax.Array, theta: PyTree, t: float) -> jax.Array:
            mean, log_std = self.actor.apply(self.actor_params, self.obs_fn(x, u, theta, t))
            std = jnp.exp(jnp.clip(log_std, -5, 2))
            return jnp.tanh(mean + std * jax.random.normal(key, mean.shape))
        return proposal
