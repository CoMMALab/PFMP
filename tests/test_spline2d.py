from typing import NamedTuple
from pfmp import hermite_spline
from pfmp.algorithm_spline import PFMP, make_transition_model
from genjax import gen, mv_normal_diag, mv_normal
import jax
import jax.numpy as jnp
import matplotlib
import matplotlib.pyplot as plt
import time
matplotlib.use('Qt5Agg')


class ParParams(NamedTuple):
    # goal: jax.Array
    # obstacles: jax.Array
    # collision_multiplier: float
    u_stdev: float
    du_stdev: float
    ddu_stdev: float


def init():
    x0 = jnp.array([0., 0.])

    theta = ParParams(
        u_stdev = 0.5,
        du_stdev = 0.1,
        ddu_stdev = 0.1
    )

    # u0 = jnp.array([0., 0.])                     # linear
    # u0 = jnp.array([[0., 0.], [0., 0.]])           # cubic
    u0 = jnp.array([[0., 0.], [0., 0.], [0., 0]])  # quintic

    return (x0, u0, theta)

@gen
def proposal_fn(x: jax.Array, u: jax.Array, theta: ParParams, t: int) -> jax.Array:
    nu = u.shape[1]
    pred_x = mv_normal(u[0] + u[1], jnp.eye(nu) * theta.u_stdev) @ "u"
    pred_v = mv_normal(u[1], jnp.eye(nu) * theta.du_stdev) @ "du"
    pred_a = mv_normal(jnp.zeros(nu), jnp.eye(nu) * theta.ddu_stdev) @ "ddu"
    # return jnp.array([pred_x])                # linear
    # return jnp.array([pred_x, pred_v])        # cubic
    return jnp.array([pred_x, pred_v, pred_a])  # quintic

def dynamics_fn(x: jax.Array, u: jax.Array, theta, t):
    return u[0], u

@jax.jit
def cost_fn(x: jax.Array, u: jax.Array, theta: ParParams, t: float) -> float:
    # Circle parameters
    angle = 2 * jnp.pi * t / (H - 1)
    r = 1.0

    # Unit radial and tangential directions at time t
    radial = jnp.array([jnp.cos(angle), jnp.sin(angle)])      # pointing outward
    tangent = jnp.array([-jnp.sin(angle), jnp.cos(angle)])    # CCW tangent

    # Extract control components
    x = u[0]      # position or control signal
    v = u[1]      # assumed shape (2,)
    a = u[2]  # assumed shape (2,)

    # 1. Position cost: minimize distance from point on circle
    x_target = r * radial
    pos_cost = jnp.sum((x - x_target) ** 2)

    # 2. Velocity cost: penalize non-tangential components
    vel_proj = jnp.dot(v, tangent)     # projection onto tangent
    tangential_velocity = vel_proj * tangent  # reconstructed tangential component
    vel_cost = jnp.sum((v - tangential_velocity) ** 2)

    # 3. Acceleration cost: encourage pointing toward center
    acc_proj = jnp.dot(a, -radial)  # projection toward center
    centripetal_acc = acc_proj * -radial
    acc_cost = jnp.sum((a - centripetal_acc) ** 2)

    # Weighted total cost (weights can be tuned)
    return pos_cost + 0.4 * vel_cost + 0.1 * acc_cost

K, H, L = 1000, 10, 128

def test_navigate():
    N = 1000
    model_t = make_transition_model(proposal_fn, cost_fn, dynamics_fn, l = L)
    pfmp = PFMP(model_t)
    (x0, u0, theta) = init()

    key = jax.random.key(42)
    (states, ancestors) = pfmp.run_sir(key, x0, u0, theta, h=H, k=K, rho=0.8)
    trajs = pfmp.choose_paths(states, ancestors, jnp.arange(N))
    fig, ax = plt.subplots(figsize=(8, 8))
    print(trajs.shape)

    for k in range(N):
        u_knots = jnp.concatenate((u0[None,0], trajs[k,:,0,0]), axis=0)
        ax.scatter(u_knots[:,0], u_knots[:,1], color='magenta', zorder=10)

        us = trajs[k,:,:,0,:].reshape(-1,2)
        ax.plot(us[:,0], us[:,1], color='magenta', alpha=0.5, ls='dotted')

    ax.set_title("Hermite-Interpolated Control (u₁ vs u₂)")
    ax.set_xlabel("u₁")
    ax.set_ylabel("u₂")
    ax.grid(False)
    ax.axis('equal')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    test_navigate()