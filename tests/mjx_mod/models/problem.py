from typing import NamedTuple, Any, Tuple
import jax
from mujoco import mjx
from genjax import gen

class Problem(NamedTuple):
    name: str
    linear_proposal_fn: callable
    cubic_proposal_fn: callable
    quintic_proposal_fn: callable
    cost_fn: callable
    dynamics_fn: callable


class ProblemDesc:
    def __init__(self):
        pass

    def initialize_problem(self):
        pass

    def get_problem_description(self):

        return Problem(
            name = self.name,
            linear_proposal_fn = self.linear_proposal_fn,
            cubic_proposal_fn = self.cubic_proposal_fn,
            quintic_proposal_fn = self.quintic_proposal_fn,
            cost_fn = self.cost_fn,
            dynamics_fn = self.dynamics_fn
        )

    @gen
    def linear_proposal_fn(self, x: mjx.Data, u: jax.Array, th: Any, t: int) -> jax.Array:
        pass

    @gen
    def cubic_proposal_fn(self, x: mjx.Data, u: jax.Array, th: Any, t: int) -> jax.Array:
        pass

    @gen
    def quintic_proposal_fn(self, x: mjx.Data, u: jax.Array, th: Any, t: int) -> jax.Array:
        pass

    @jax.jit
    def dynamics_fn(self, x: mjx.Data, u: jax.Array, th: Any, t: int) -> Tuple[jax.Array,jax.Array]:
        pass

    @jax.jit
    def cost_fn(self, x: mjx.Data, _u, th: Any, _t) -> float:
        pass

    def visualize(self):
        pass
