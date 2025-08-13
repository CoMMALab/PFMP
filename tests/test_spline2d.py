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


class ParState(NamedTuple):
    u: jax.Array

class ParParams(NamedTuple):
    # goal: jax.Array
    # obstacles: jax.Array
    # collision_multiplier: float
    u_stdev: float
    du_stdev: float
    ddu_stdev: float


def init():
    x0 = ParState(
        # u = jnp.array([[0., 0.]]),             # linear
        # u = jnp.array([[0., 0.], [0., 0.]]),       # cubic
        u = jnp.array([[0., 0.], [0., 0.], [0., 0]]), # quintic
    )

    theta = ParParams(
        u_stdev = 0.5,
        du_stdev = 0.1,
        ddu_stdev = 0.1
    )

    return (x0, theta)

@gen
def proposal_fn(state: ParState, theta: ParParams, t: int) -> ParState:
    nu = state.u.shape[1]
    pred_u = mv_normal(state.u[0] + state.u[1], jnp.eye(nu) * theta.u_stdev) @ "u"
    pred_du = mv_normal(state.u[1], jnp.eye(nu) * theta.du_stdev) @ "du"
    pred_ddu = mv_normal(jnp.zeros(nu), jnp.eye(nu) * theta.ddu_stdev) @ "ddu"
    # pred_du = mv_normal(jnp.zeros(nu), jnp.eye(nu) * theta.du_stdev) @ "du"
    # pred_ddu = mv_normal(jnp.zeros(nu), jnp.eye(nu) * theta.ddu_stdev) @ "ddu"
    return jnp.array([pred_u, pred_du, pred_ddu])
    # return state._replace(
    #     # u = jnp.array([pred_u]),                  # linear
    #     # u = jnp.array([pred_u, pred_du]),         # cubic
    #     u = jnp.array([pred_u, pred_du, pred_ddu])  # quintic
    # )

def dynamics_fn(x, u, theta, t):
    return x._replace(u=u)

@jax.jit
def cost_fn(state: ParState, theta: ParParams, t: float) -> float:
    # Circle parameters
    angle = 2 * jnp.pi * t / (H - 1)
    r = 1.0

    # Unit radial and tangential directions at time t
    radial = jnp.array([jnp.cos(angle), jnp.sin(angle)])      # pointing outward
    tangent = jnp.array([-jnp.sin(angle), jnp.cos(angle)])    # CCW tangent

    # Extract control components
    u_actual = state.u[0]      # position or control signal
    velocity = state.u[1]      # assumed shape (2,)
    acceleration = state.u[2]  # assumed shape (2,)

    # 1. Position cost: minimize distance from point on circle
    u_target = r * radial
    pos_cost = jnp.sum((u_actual - u_target) ** 2)

    # 2. Velocity cost: penalize non-tangential components
    vel_proj = jnp.dot(velocity, tangent)     # projection onto tangent
    tangential_velocity = vel_proj * tangent  # reconstructed tangential component
    vel_cost = jnp.sum((velocity - tangential_velocity) ** 2)

    # 3. Acceleration cost: encourage pointing toward center
    acc_proj = jnp.dot(acceleration, -radial)  # projection toward center
    centripetal_acc = acc_proj * -radial
    acc_cost = jnp.sum((acceleration - centripetal_acc) ** 2)

    # Weighted total cost (weights can be tuned)
    return pos_cost + 0.4 * vel_cost + 0.1 * acc_cost

# @jax.jit
# def cost_fn(state: ParState, theta: ParParams, t: float) -> float:
#     # Compute target control on a circle of radius r
#     angle = 2 * jnp.pi * t / 10
#     r = 1.0  # radius of the target circle

#     u_target = jnp.array([r * jnp.cos(angle), r * jnp.sin(angle)])
#     u_actual = state.u[0]  # shape (2,) for 2D control

#     return jnp.sum((u_actual - u_target) ** 2)



K, H, L = 100000, 8, 100

def test_navigate():
    N = 10
    model_t = make_transition_model(proposal_fn, cost_fn, dynamics_fn, l = L)
    pfmp = PFMP(model_t)
    (x0, theta) = init()

    key = jax.random.key(42)
    (states, ancestors) = pfmp.run_sir(key, x0, theta, h=H, k=K, rho=0.8)
    trajs = pfmp.choose_paths(states, ancestors, jnp.arange(N)).u
    fig, ax = plt.subplots(figsize=(8, 8))

    for k in range(N):
        u1_full = []
        u2_full = []
        knot_points = []

        for h in range(H):
            u1 = trajs[k, h, :, 0, 0]
            u2 = trajs[k, h, :, 0, 1]

            if h < H - 1:
                u1_full.append(u1[:-1])
                u2_full.append(u2[:-1])
            else:
                u1_full.append(u1)
                u2_full.append(u2)

            # Add the first point of this segment as a knot
            knot_points.append((u1[0], u2[0]))

        u1_full = jnp.concatenate(u1_full)
        u2_full = jnp.concatenate(u2_full)

        ax.plot(u1_full, u2_full, color='magenta', alpha=0.5, ls='dotted', lw=2)

        # Plot knot points
        knot_points = jnp.array(knot_points)
        ax.scatter(knot_points[:, 0], knot_points[:, 1], color='magenta', s=50, zorder=10)


    ax.set_title("Hermite-Interpolated Control (u₁ vs u₂)")
    ax.set_xlabel("u₁")
    ax.set_ylabel("u₂")
    ax.grid(True)
    ax.axis('equal')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    test_navigate()