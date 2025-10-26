from typing import NamedTuple
from pfmp import PFMP, make_transition_model
from genjax import gen, normal
import jax
import jax.numpy as jnp
import matplotlib
import matplotlib.pyplot as plt
import time
matplotlib.use('Qt5Agg')


class StateTracker(NamedTuple):
    delta: jax.Array
    location: jax.Array

class ObstacleGridParams(NamedTuple):
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
# def proposal_fn(state: jax.Array, u: jax.Array, th: ObstacleGridParams, t: int) -> jax.Array:
#     delta = normal(0., th.heading_stdev) @ "delta"
#     return jnp.array([delta])

# def dynamics_fn(x: jax.Array, u: jax.Array, th: ObstacleGridParams, t: float) -> tuple[jax.Array, StateTracker]:
#     delta = u[0]
#     heading = th.heading_nominal[t.astype(int)] + delta
#     direction = jnp.array([jnp.cos(heading), jnp.sin(heading)])
#     x += th.speed * direction
#     y = StateTracker(
#         delta = heading - th.heading_nominal[t.astype(int)],
#         location = x
#     )
#     return x, y
@gen
def proposal_fn(state: jax.Array, u: jax.Array, th: ObstacleGridParams, t: int) -> jax.Array:
    # Sample a heading offset
    delta = normal(0., th.heading_stdev) @ "delta"
    theta = th.heading_nominal[t] + delta

    # Direction unit vector
    direction = jnp.array([jnp.cos(theta), jnp.sin(theta)])  # (2,)

    # Position = surface point
    pos = state + th.speed * direction  # (2,)
    # Velocity = constant radial speed
    vel = th.speed * direction  # (2,)

    # Package as 2×2 array
    return jnp.stack([pos, vel], axis=0)  # shape (2,2)


def dynamics_fn(_: jax.Array, u: jax.Array, th: ObstacleGridParams, t: float) -> tuple[jax.Array, StateTracker]:
    # Just copy the proposal’s position
    pos = u[0]   # (2,)
    vel = u[1]   # (2,)

    y = StateTracker(
        delta = jnp.arctan2(vel[1], vel[0]) - th.heading_nominal[t.astype(int)],
        location = pos
    )
    # Return 2×2 array [[x,y],[vx,vy]]
    return pos, y

@jax.jit
def cost_fn(y: StateTracker, u, th, t: int) -> float:
    loc = y.location
    sdf_vals = jax.vmap(lambda bounds: sdf_aabb(loc, bounds))(th.obstacles)
    sdf_val = jnp.min(sdf_vals)
    obstacle_loss = th.collision_multiplier * hinge_loss(sdf_val, th.collision_buffer)
    # this is in-case `u.collision_multiplier == jnp.inf`
    obstacle_loss = jnp.nan_to_num(obstacle_loss, nan=0.0, posinf=jnp.inf, neginf=0.0)
    goal_loss = t * jnp.linalg.norm(loc - th.goal)
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
    x0 = jnp.array([0., 0.])

    th = ObstacleGridParams(
        goal = jnp.array([1.25, 1.25]),
        obstacles = generate_obstacle_grid(5, 5, 0.1, 0.1) + 0.2,
        heading_nominal = jnp.full(H, jnp.pi/4),
        heading_stdev = 1.0,
        speed = 0.05 * L,
        # collision_buffer = 0.03,
        collision_buffer = 0.075,
        collision_multiplier = 2000.
    )

    return (x0, th)

# K = num particles, H = horizon, L = grid sample resolution
K, H, L = 100000, 8, 5
RHO = 0.8

def test_pfmp():
    model = make_transition_model(proposal_fn, cost_fn, dynamics_fn, l=L)
    pfmp = PFMP(model)
    (x0, th) = init(H)
    u0 = jnp.array([[0.,0.], [0.,0.]])

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

    # print(states.delta.shape)
    # print(states.location.shape)

    assert states.delta.shape == (H,K,L)
    assert states.location.shape == (H,K,L,2)
    assert ancestors.shape == (H,K)

    (key, subkey) = jax.random.split(key)
    n_samples = 100

    trajs = pfmp.choose_paths(states, ancestors, jnp.arange(n_samples))
    # mat_costs = PFMP.project_costs(cost_fn, trajs, th)

    # total_costs = mat_costs.reshape(n_samples, -1).sum(axis=1)
    # map_idx = jnp.argmin(total_costs)
    # map_traj = jax.tree.map(lambda x: x[map_idx], trajs)
    # map_cost = total_costs[map_idx]
    # print(f"MAP Trajectory: {map_traj}")
    # print(f"MAP cost: {map_cost}")

    # assert map_cost == jnp.sum(PFMP.project_cost(cost_fn, map_traj, th))
    # trajs = jax.tree.map(lambda x: x[jnp.array([map_idx])], trajs)

    plot_goal(th.goal)
    plot_obstacles(th.obstacles)
    # print(trajs.location.shape)
    locs = trajs.location.reshape(n_samples,-1,2)
    print(locs.shape)
    plot_locations(locs)

    plt.show()

def test_mppi():
    model = make_transition_model(proposal_fn, cost_fn, dynamics_fn, l=L)
    mppi = PFMP(model)
    (x0, th) = init(H)
    u0 = jnp.array([[0.,0.], [0.,0.]])

    n_iters = 30
    init_stdev = 0.05

    # MPPI does better with a higher multiplier
    th = th._replace(collision_multiplier = 5000.)
    th = th._replace(heading_stdev = init_stdev)

    key = jax.random.key(42)
    t0 = time.time()
    for i in range(n_iters):
        (key, subkey) = jax.random.split(key)
        if i == 2:
            t0 = time.time()
        (trajs, log_ws) = mppi.run_sis(
            subkey, x0, u0, th, H, K
        )
        trajs = jax.tree.map(lambda x: x[:,:,0], trajs)
        t1 = time.time()
        if i == 0:
            print(f"sample_forward (first run): {t1 - t0}")
        else:
            print(f"sample_forward (after jit): {t1 - t0}")
        ws = jnp.exp(log_ws - jnp.max(log_ws))
        ps = ws / jnp.sum(ws)
        expected = jax.tree.map(lambda field: jnp.tensordot(ps, field, axes=([0], [0])).squeeze(), trajs)
        th = th._replace(
            heading_nominal = th.heading_nominal + expected.delta,
            heading_stdev = th.heading_stdev - (init_stdev - 1e-3)/n_iters  # annealing
        )
        print(th.heading_stdev)

    print()

    print(trajs.delta.shape)
    assert trajs.delta.shape == (K,H)
    assert trajs.location.shape == (K,H,2)

    traj = jax.tree.map(lambda x: x[0], trajs)

    assert traj.delta.shape == (H,)
    assert traj.location.shape == (H,2)

    l = 10
    trajs = jax.tree.map(lambda x: x[:1], trajs)
    # trajs = jax.tree.map(lambda head, tail: jnp.concat((jnp.broadcast_to(head, (l,1) + jnp.shape(head)), tail), axis=1), x0, trajs)

    plt.axis('equal')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Obstacle Avoidance (MPPI)')
    plt.grid(True)

    plot_goal(th.goal)
    plot_obstacles(th.obstacles)
    locs = trajs.location.reshape(1,-1,2)
    plot_locations(locs)

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
    # print("done")
    # test_mppi()
    # test_cem()