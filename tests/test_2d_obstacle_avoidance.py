from typing import NamedTuple
from pfmp import PFMP, make_transition_model
from genjax import gen, normal
import jax
import jax.numpy as jnp
import matplotlib
import matplotlib.pyplot as plt
import time
matplotlib.use('Qt5Agg')


class ParState(NamedTuple):
    delta: jax.Array
    location: jax.Array

class ParUniform(NamedTuple):
    goal: jax.Array
    obstacles: jax.Array
    heading_nominal: jax.Array
    heading_stdev: float
    speed: float
    collision_buffer: float
    collision_multiplier: float

def generate_obstacle_grid(num_x: int, num_y: int, box_size: float, gap: float):
    """ Generate a regular grid of AABB obstacles.

    Args:
        num_x: number of boxes along the x-axis
        num_y: number of boxes along the y-axis
        box_size: side length of each box (square)
        gap: gap between boxes (same in x and y)

    Returns:
        A (num_x * num_y, 4) array of AABB boxes as (xmin, xmax, ymin, ymax)
    """
    obstacles = []
    for i in range(num_x):
        for j in range(num_y):
            xmin = i * (box_size + gap)
            ymin = j * (box_size + gap)
            xmax = xmin + box_size
            ymax = ymin + box_size
            obstacles.append(jnp.array([xmin, xmax, ymin, ymax]))
    return jnp.stack(obstacles)
    
@jax.jit
def sdf_aabb(pos: jax.Array, bounds: jax.Array) -> float:
    xmin, xmax, ymin, ymax = bounds
    center = jnp.array([(xmin + xmax) / 2, (ymin + ymax) / 2])
    half_extents = jnp.array([(xmax - xmin) / 2, (ymax - ymin) / 2])
    
    d = jnp.abs(pos - center) - half_extents
    outside_dist = jnp.linalg.norm(jnp.maximum(d, 0.0))
    inside_dist = jnp.minimum(jnp.maximum(d[0], d[1]), 0.0)
    return outside_dist + inside_dist

@jax.jit
def hinge_loss(sdf: float, epsilon: float) -> float:
    return jnp.where(sdf < epsilon, 0.5 * (epsilon - sdf) ** 2, 0.0)

# @gen
# def transition_fn(state: ParState, u: ParUniform, t: int) -> ParState:
#     delta = normal(0., u.heading_stdev) @ "delta"
#     heading = u.heading_nominal[t] + delta
#     velocity = u.speed * jnp.array([jnp.cos(heading), jnp.sin(heading)])
#     location = state.location + velocity

@gen
def proposal_fn(state: ParState, u: ParUniform, th, t: int) -> ParState:
    heading = normal(th.heading_nominal[t], th.heading_stdev) @ "heading"
    return jnp.array([heading])

def dynamics_fn(state, u, th, t: float):
    heading = u[0]
    velocity = th.speed * jnp.array([jnp.cos(heading), jnp.sin(heading)])
    location = state.location + velocity
    return ParState(
        heading - th.heading_nominal[t.astype(int)],
        location
    ), location

@jax.jit
def cost_fn(xy: jax.Array, th, t: int) -> float:
    sdf_vals = jax.vmap(lambda bounds: sdf_aabb(xy, bounds))(th.obstacles)
    sdf_val = jnp.min(sdf_vals)
    obstacle_loss = th.collision_multiplier * hinge_loss(sdf_val, th.collision_buffer)
    # this is in-case `u.collision_multiplier == jnp.inf`
    obstacle_loss = jnp.nan_to_num(obstacle_loss, nan=0.0, posinf=jnp.inf, neginf=0.0)
    goal_loss = t * jnp.linalg.norm(xy - th.goal)
    return obstacle_loss**2 + goal_loss**2

def plot_goal(goal):
    plt.plot(goal[0], goal[1], marker='*', markersize=30, c='green')

def plot_obstacles(obstacles, color='black'):
    """Plot a list of AABB obstacles."""
    ax = plt.gca()
    for obs in obstacles:
        xmin, xmax, ymin, ymax = obs
        width = xmax - xmin
        height = ymax - ymin
        rect = plt.Rectangle((xmin, ymin), width, height,
                             edgecolor=color, facecolor='black', linewidth=1)
        ax.add_patch(rect)

def plot_locations(locs):
    K = locs.shape[0]
    for i in range(K):
        plt.plot(locs[i,:,0], locs[i,:,1], linestyle='--', color='black', alpha=0.3)
        plt.scatter(locs[i,:,0], locs[i,:,1], color='black', s=10)

def init(H: int):
    x0 = ParState(
        delta = jnp.array(0.),
        location = jnp.array([0., 0.])
    )

    th = ParUniform(
        goal = jnp.array([1.25, 1.25]),
        obstacles = generate_obstacle_grid(5, 5, 0.1, 0.1) + 0.2,
        heading_nominal = jnp.full(H, 0.),
        heading_stdev = 1.0,
        speed = 0.05,
        # collision_buffer = 0.03,
        collision_buffer = 0.1,
        collision_multiplier = 2000.
    )

    return (x0, th)

# K = num particles, H = horizon, L = grid sample resolution
K, H, L = 1000, 40, 1
RHO = 0.8

def test_pfmp():
    model = make_transition_model(proposal_fn, cost_fn, dynamics_fn, l=L)
    pfmp = PFMP(model)
    (x0, th) = init(H)
    u0 = jnp.array([0.])

    key = jax.random.key(42)
    for i in range(5):
        t0 = time.time()
        (key, subkey) = jax.random.split(key)
        (states, ancestors) = pfmp.run_sir(subkey, x0, u0, th, H, K, RHO)
        t1 = time.time()
        if i == 0:
            print(f"sample_forward (first run): {t1 - t0}")
        else:
            print(f"sample_forward (after jit): {t1 - t0}")

    # assert states.delta.shape == (H,K)
    # assert states.location.shape == (H,K,2)
    # assert ancestors.shape == (H,K)

    (key, subkey) = jax.random.split(key)
    # traj = pfmp.choose_path(states, ancestors, 0)
    # traj = jax.tree.map(lambda head, tail: jnp.concat((head[None], tail)), x0, traj)

    # print(traj.delta.shape)
    # assert traj.delta.shape == (H+1,)
    # assert traj.location.shape == (H+1,2)

    n_samples = 100
    trajs = pfmp.choose_paths(states, ancestors, jnp.arange(n_samples))
    print(trajs.shape)
    g = jax.vmap(
            jax.vmap(
                jax.vmap(cost_fn, in_axes=(0, None, None)),
                in_axes=(0, None, None)
            ),
            in_axes=(0, None, None)
        )
    mat_costs = PFMP.project_cost(cost_fn, trajs, th)
    total_costs = mat_costs.reshape(mat_costs.shape[0], -1).sum(axis=1)
    besti = jnp.argmin(total_costs)
    print(besti)
    print(total_costs[besti])
    print(trajs[besti])
    trajs = trajs[jnp.array([besti])]
    # trajs = states[:n_samples]

    # l = 10
    # for i in range(5):
    #     t0 = time.time()
    #     trajs = jax.block_until_ready(
    #         pfmp.choose_paths(states, ancestors, jnp.arange(l))
    #     )
    #     trajs = jax.tree.map(lambda head, tail: jnp.concat((jnp.broadcast_to(head, (l,1) + jnp.shape(head)), tail), axis=1), x0, trajs)
    #     t1 = time.time()
    #     if i == 0:
    #         print(f"choose_paths (first run): {t1 - t0}")
    #     else:
    #         print(f"choose_paths (after jit): {t1 - t0}")

    # plt.axis('equal')
    # plt.xlabel('x')
    # plt.ylabel('y')
    # plt.title('Obstacle Avoidance (PFMP)')
    # plt.grid(True)

    plot_goal(th.goal)
    plot_obstacles(th.obstacles)
    locs = trajs.reshape(1,-1,2)
    plot_locations(locs)

    plt.show()

def test_mppi():
    mppi = PFMP(transition_fn, cost_fn)
    (x0, u) = init(H)

    # MPPI does better with a higher multiplier
    u = u._replace(collision_multiplier = 5000.)
    u = u._replace(heading_stdev = 0.05)

    key = jax.random.key(42)
    j_sample_forward = jax.jit(mppi.sample_forward_weighted, static_argnames=['block_shape'])
    t0 = time.time()
    for i in range(30):
        (key, subkey) = jax.random.split(key)
        if i == 2:
            t0 = time.time()
        (trajs, log_ws) = jax.block_until_ready(
            j_sample_forward(subkey, (H, K), x0, u)
        )
        t1 = time.time()
        if i == 0:
            print(f"sample_forward (first run): {t1 - t0}")
        else:
            print(f"sample_forward (after jit): {t1 - t0}")
        ws = jnp.exp(log_ws - jnp.max(log_ws))
        ps = ws / jnp.sum(ws)
        expected = jax.tree.map(lambda field: jnp.tensordot(ps, field, axes=([0], [0])), trajs)
        u = u._replace(
            heading_nominal = u.heading_nominal + expected.delta,
            # heading_stdev = u.heading_stdev - 0.49/30  # annealing
        )

    print()

    assert trajs.delta.shape == (K,H)
    assert trajs.location.shape == (K,H,2)

    traj = jax.tree.map(lambda x: x[0], trajs)

    assert traj.delta.shape == (H,)
    assert traj.location.shape == (H,2)

    l = 10
    trajs = jax.tree.map(lambda x: x[:l], trajs)
    trajs = jax.tree.map(lambda head, tail: jnp.concat((jnp.broadcast_to(head, (l,1) + jnp.shape(head)), tail), axis=1), x0, trajs)

    plt.axis('equal')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Obstacle Avoidance (MPPI)')
    plt.grid(True)

    plot_goal(th.goal)
    plot_obstacles(th.obstacles)
    plot_locations(trajs.location)

    plt.show()

def test_cem():
    cem = PFMP(transition_fn, cost_fn)
    (x0, u) = init(H)

    # CEM does better with a higher multiplier
    u = u._replace(collision_multiplier = 5000.)

    key = jax.random.key(42)
    j_sample_forward = jax.jit(cem.sample_forward_weighted, static_argnames=['block_shape'])
    t0 = time.time()
    for i in range(30):
        (key, subkey) = jax.random.split(key)
        if i == 2:
            t0 = time.time()
        (trajs, log_ws) = jax.block_until_ready(
            j_sample_forward(subkey, (H, K), x0, u)
        )
        t1 = time.time()
        if i == 0:
            print(f"sample_forward (first run): {t1 - t0}")
        else:
            print(f"sample_forward (after jit): {t1 - t0}")

        l = 10
        elite_indices = jnp.argsort(-log_ws)[:l]
        elite_choices = jax.random.randint(key, shape=(K,), minval=0, maxval=l)
        resampled_indices = elite_indices[elite_choices]  # shape (K,)
        resampled_trajs = jax.tree.map(lambda x: x[resampled_indices], trajs)
        expected = jax.tree.map(lambda field: jnp.mean(field, axis=0), resampled_trajs)

        # Update control
        u = u._replace(
            heading_nominal = u.heading_nominal + expected.delta,
            heading_stdev = u.heading_stdev - 0.49 / 30  # annealing
        )

    print()

    assert trajs.delta.shape == (K,H)
    assert trajs.location.shape == (K,H,2)

    traj = jax.tree.map(lambda x: x[0], trajs)

    assert traj.delta.shape == (H,)
    assert traj.location.shape == (H,2)

    l = 10
    trajs = jax.tree.map(lambda x: x[:l], trajs)
    trajs = jax.tree.map(lambda head, tail: jnp.concat((jnp.broadcast_to(head, (l,1) + jnp.shape(head)), tail), axis=1), x0, trajs)

    plt.axis('equal')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Obstacle Avoidance (CEM)')
    plt.grid(True)

    plot_goal(th.goal)
    plot_obstacles(th.obstacles)
    plot_locations(trajs.location)

    plt.show()

def test_ps():
    pass

def test_dial_mpc():
    pass

if __name__ == "__main__":
    test_pfmp()
    print("done")
    # test_mppi()
    # test_cem()