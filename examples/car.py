# obstacle_avoidance.py

import jax
import jax.numpy as jnp
from typing import NamedTuple
from pfmp import PFMP
from pfmp.interpolate import Jet2, hermite_spline
import matplotlib.pyplot as plt


class Params(NamedTuple):
    goal: jax.Array           # Desired position
    obstacles: jax.Array      # ObstacleGrid
    L: float                  # wheelbase (m)
    v: float                  # forward speed (m/s)
    du_stdev: float           # tangent noise
    u_stdev: float            # point noise (local jitter)

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

def cost_fn(x: jax.Array, u: jax.Array, th: Params, t: float) -> float:
    loc = x[:2]
    sdf_vals = jax.vmap(lambda bounds: sdf_aabb(loc, bounds))(th.obstacles)
    sdf_val = jnp.min(sdf_vals)
    obstacle_loss = 100. * hinge_loss(sdf_val, 0.25)
    # this is in-case `u.collision_multiplier == jnp.inf`
    obstacle_loss = jnp.nan_to_num(obstacle_loss, nan=0.0, posinf=jnp.inf, neginf=0.0)
    goal_loss = jnp.linalg.norm(loc - th.goal) / 10.
    return obstacle_loss**2 + goal_loss**2

def terminal_cost(x, th) -> float:
    loc = x[:2]
    return 10 * jnp.linalg.norm(loc - th.goal) ** 2

def dynamics(x: jax.Array, u: jax.Array, theta: Params, t):
    """
    x: [px, py, theta]
    j: Jet2(u, du) where u is steering angle δ
    """
    px, py, th = x
    L, v = theta.L, theta.v
    
    px_next = px + v * jnp.cos(th) * 0.01
    py_next = py + v * jnp.sin(th) * 0.01
    th_next = th + (v / L) * jnp.tan(u) * 0.01

    th_next = jnp.mod(th_next + jnp.pi, 2 * jnp.pi) - jnp.pi

    x_next = jnp.array([px_next, py_next, th_next])
    return (x_next, x_next)

def init(key, x, theta):
    u_next = jax.random.normal(key) * theta.u_stdev
    u_bound = 0.8
    return jnp.clip(u_next, -u_bound, u_bound)

def proposal(key, _x, _u: float, theta: Params, t):
    """
    key: PRNGKey
    x: state (ignored)
    u = current steering angle and rate
    theta: Params
    """

    u_next = jax.random.normal(key) * theta.u_stdev
    u_bound = 0.8
    u_next = jnp.clip(u_next, -u_bound, u_bound)
    return u_next

def plot_obstacles(ax, obstacles, color='black'):
    """Plot a list of AABB obstacles."""
    for obs in obstacles:
        xmin, xmax, ymin, ymax = obs
        width = xmax - xmin
        height = ymax - ymin
        rect = plt.Rectangle((xmin, ymin), width, height,
                             edgecolor=color, facecolor='black', linewidth=1)
        ax.add_patch(rect)

if __name__ == "__main__":
    horizon = 10
    num_knots = int(2*horizon)
    particles = 128
    pfmp = PFMP(dynamics, cost_fn, terminal_cost, init, proposal,
                horizon, num_knots, particles,
                ess_resample_thresh=0.5)
    x0 = jnp.array([0,0,jnp.pi/4])
    # u0 = Jet2(0., 0.)
    u0 = 0.0
    theta = Params(
        goal = jnp.array([12.0, 12.0]),
        obstacles = generate_obstacle_grid(5, 5, 1.0, 1.0) + 1.0,
        L=0.3,          # typical wheelbase for normalized bicycle model
        v=2.0,          # moderate forward speed (m/s)
        du_stdev=0.1,  # tangent noise — gives gentle arc variation
        u_stdev=0.2    # tiny point jitter — stabilizes proposal
    )

    key = jax.random.key(1)
    traj, scores = pfmp.run_sir(key, x0, theta)
    tq = jnp.arange(0, horizon, pfmp.dt)

    import matplotlib.patches as patches
    import matplotlib.transforms as transforms

    car_length = theta.L
    car_width  = theta.L/2

    ax = plt.gca()

    xs = jnp.linspace(0, 12, 200)
    ys = jnp.linspace(0, 12, 200)
    X, Y = jnp.meshgrid(xs, ys)

    # Flatten for vectorization
    points = jnp.stack([X.ravel(), Y.ravel()], axis=1)

    # Wrap cost_fn so it only depends on position
    def cost_at_loc(loc):
        # Build a dummy x and u consistent with your system
        x = loc  # fill in based on your state dimension
        u = jnp.zeros(1)                          # fill in based on your control dimension
        return cost_fn(x, u, theta, t=0.0)

    cost_vmap = jax.vmap(cost_at_loc)
    Z = cost_vmap(points).reshape(X.shape)

    ax.contourf(X, Y, Z, levels=50)

    plot_obstacles(ax, theta.obstacles)

    interval_len = horizon / (num_knots - 1)
    steps_per_knot = round(interval_len / pfmp.dt)
    best_idx = jnp.argmin(jnp.sum(scores, axis=-1))
    tq = jnp.linspace(0,pfmp.horizon,int(pfmp.horizon/pfmp.dt))
    for i in range(20):
        us = hermite_spline(tq, pfmp.tk(), traj[i])
        _, xy = jax.lax.scan(lambda x, ut: dynamics(x, ut[0], theta, ut[1]), x0, (us, tq))
        plt.plot(xy[:,0], xy[:,1], c="magenta", alpha=0.5)

        for t in range(0, xy.shape[0], steps_per_knot):  # every 5 steps
            x, y, th = xy[t]

            # Create a rectangle centered at (0,0)
            rect = patches.Rectangle(
                (-car_length/2, -car_width/2),  # center the box
                car_length,
                car_width,
                facecolor="cyan",
                edgecolor="black",
                alpha=1.0,
                zorder=5
            )

            # Build the transform: rotate, then translate
            trans = transforms.Affine2D()
            trans.rotate(th)
            trans.translate(x, y)

            rect.set_transform(trans + ax.transData)
            ax.add_patch(rect)
    us = hermite_spline(tq, pfmp.tk(), traj[best_idx])
    _, best_xy = jax.lax.scan(lambda x, ut: dynamics(x, ut[0], theta, ut[1]), x0, (us, tq))
    plt.plot(best_xy[:,0], best_xy[:,1], c="gold", lw=3, alpha=1)

    plt.axis("equal")
    plt.show()


