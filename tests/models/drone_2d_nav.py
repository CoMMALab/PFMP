import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
from typing import NamedTuple, Tuple
import numpy as np
import time
from genjax import gen, mv_normal

# for urdf
from .problem import ProblemDesc

from functools import partial

# for viz
# import PyQt6
import matplotlib
# matplotlib.use('qtagg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# ---------- Constants ----------
DT = 0.01   # s, integration timestep
G  = 9.81   # m/s^2, gravity


# ---------- Helpers ----------
@jax.jit
def _wrap_angle(theta: jax.Array) -> jax.Array:
    # Wrap to (-pi, pi]
    return (theta + jnp.pi) % (2.0 * jnp.pi) - jnp.pi

@jax.jit
def _rot(theta: jax.Array) -> jax.Array:
    c, s = jnp.cos(theta), jnp.sin(theta)
    return jnp.array([[c, -s],
                      [s,  c]])



class Gains(NamedTuple):
    kp: float
    kd: float

class DParams(NamedTuple):
    mass: float
    half_length: float
    moi: float
    max_thrust: float
    sigma: float
    target: jax.Array
    gains: Gains
    rho: float

    # with obstacles
    obstacles: jax.Array
    collision_buffer: float
    collision_multiplier: float

class DState(NamedTuple):
    qpos: jax.Array
    qvel: jax.Array


@gen
def quintic_proposal_fn(x: DState, u: jax.Array, th: DParams, t: float) -> jax.Array:
    rho = th.rho   # correlation coefficient, -1 <= rho <= 1
    Sigma = th.sigma**2 * jnp.array([
        [1.0, rho],
        [rho, 1.0]
    ])
    thrust = mv_normal(G/2, Sigma) @ "thrust"
    dthrust = mv_normal(G/2, Sigma/2) @ "dthrust"
    ddthrust = mv_normal(G/2, Sigma/4) @ "ddthrust"
    return jnp.array([thrust, dthrust, ddthrust])

@gen
def cubic_proposal_fn(x: DState, u: jax.Array, th: DParams, t: float) -> jax.Array:
    rho = th.rho   # correlation coefficient, -1 <= rho <= 1
    Sigma = th.sigma**2 * jnp.array([
        [1.0, rho],
        [rho, 1.0]
    ])
    thrust = mv_normal(G/2, Sigma) @ "thrust"
    dthrust = mv_normal(G/2, Sigma/2) @ "dthrust"
    return jnp.array([thrust, dthrust])

@gen
def linear_proposal_fn(x: DState, u: jax.Array, th: DParams, t: float) -> jax.Array:
    rho = th.rho   # correlation coefficient, -1 <= rho <= 1
    Sigma = th.sigma**2 * jnp.array([
        [1.0, rho],
        [rho, 1.0]
    ])
    thrust = mv_normal(G/2, Sigma) @ "thrust"
    return jnp.array([thrust])



class Drone2DNav(ProblemDesc):
    def __init__(self):
        # ===============================
        # Robot setup
        # ===============================

        self.name = "drone_2d_nav"
        # self.xml_filename = "mjx_t_push"


        self.linear_proposal_fn = linear_proposal_fn
        self.quintic_proposal_fn = quintic_proposal_fn
        self.cubic_proposal_fn = cubic_proposal_fn

        super().__init__()


    def initialize_problem(self, config):
        """
        Set up an initial state and initialize the mjx

        """
        mass = 1.0
        half_length = 0.5
        moi = (1.0/3.0) * mass * half_length**2  # ≈ 0.0833333

        u0 = jnp.array([
            [0.0, 0.0],  # ( x0,  y0)
            [0.0, 0.0],  # (vx0, vy0)
            [0.0, 0.0]   # (ax0, ay0)
        ])
        if config.deg == 1:
            u0 = u0[:1]
        elif config.deg == 3:
            u0 = u0[:2]
        elif config.deg == 5:
            pass
        else:
            raise ValueError(f"Polynomial order not supported: {config.deg}")


        theta = DParams(
            mass = mass,
            half_length = half_length,
            moi = moi,
            max_thrust = 100.0,
            sigma = config.sigma,
            target = jnp.array([4.5, 0.]),
            # target = jnp.array([2., -2.]),
            gains = Gains(0.2, 0.05),
            obstacles = self.generate_obstacle_grid(5, 5, 0.1, 1),
            collision_buffer = 0.25,
            collision_multiplier = 1e3,
            rho = 0.8

        )
        theta = theta._replace(rho = config.rho)


        # x0 = DState(qpos=jnp.array([-4.0,3.0,0.8]), qvel=jnp.array([0.,0.,0.]))

        x0 = DState(qpos=jnp.array([-4.5,-3.0,0.8]), qvel=jnp.array([0.,0.,0.]))
        # return th, x0

        return x0, u0, theta




    @partial(jax.jit, static_argnames=['self'])
    def dynamics_fn(self, x: DState, u: jax.Array, th: DParams, t: float) -> Tuple[DState,DState]:

        # Saturate thrusters
        u = jnp.clip(u, 0.0, th.max_thrust)
        uL, uR = u[0,0], u[0,1]

        # Unpack state
        px, py, tht = x.qpos
        vx, vy, omg  = x.qvel

        # Net body-frame force (along +y_body)
        u_sum = uL + uR
        f_body = jnp.array([0.0, u_sum])               # (Fx_body, Fy_body)

        # Rotate to world frame
        Fw = _rot(tht) @ f_body                         # (Fx_world, Fy_world)

        # Add gravity
        Fw = Fw + jnp.array([0.0, -th.mass * G])

        # Torque about COM from opposing thrusters
        tau = (uR - uL) * th.half_length

        # Accelerations
        ax, ay = Fw / th.mass
        alpha = tau / th.moi

        # Semi-implicit Euler (symplectic)
        vx_new = vx + DT * ax
        vy_new = vy + DT * ay
        omg_new = omg + DT * alpha

        px_new = px + DT * vx_new
        py_new = py + DT * vy_new
        tht_new = _wrap_angle(tht + DT * omg_new)

        x_next = DState(qpos=jnp.array([px_new, py_new, tht_new]),
                        qvel=jnp.array([vx_new, vy_new, omg_new]))

        obs = x_next
        return x_next, obs


    @partial(jax.jit, static_argnames=['self'])
    def cost_fn(self, x: DState, theta: DParams, t: float) -> float:

        p = x.qpos[:2] - theta.target
        perr = jnp.dot(p,p) + x.qpos[2]**2

        d = x.qvel[:2]
        derr = jnp.dot(d,d) #+ (x.qvel[2] * 0.001 )**2

        sdf_vals = jax.vmap(lambda bounds: self.sdf_aabb(x.qpos, theta.half_length, bounds))(theta.obstacles)
        sdf_val = jnp.min(sdf_vals)
        obstacle_loss = theta.collision_multiplier * self.hinge_loss(sdf_val, theta.collision_buffer)
        # this is in-case `u.collision_multiplier == jnp.inf`
        obstacle_loss = jnp.nan_to_num(obstacle_loss, nan=0.0, posinf=jnp.inf, neginf=0.0)

        # return (obstacle_loss) ** 2
        return (obstacle_loss) ** 2 + (theta.gains.kp*perr * ((t + 1.0) / 160.0)) + theta.gains.kd*derr * ((t + 1.0) / 160.0)
        # return theta.gains.kp*perr + theta.gains.kd*derr


    # Adding obstacles for cost function
    def generate_obstacle_grid(self, num_x: int, num_y: int, box_size: float, gap: float):
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

        # obstacles.append(jnp.array([-3.5, -3.3, -1, 3]))
        # obstacles.append(jnp.array([-5.75, -1.5, -2.3, -2.1]))
        # obstacles.append(jnp.array([-0.1, 4, -2.3, -2.1]))
        # obstacles.append(jnp.array([-5.0, -4.95, -5.0, 5.0]))
        # obstacles.append(jnp.array([3.9, 4.0, -6.0, 6.0]))
        obstacles.append(jnp.array([-5.0, 5.0, -5.0, -4.9]))
        # obstacles.append(jnp.array([-6.0, 6.0, 3.9, 5.0]))

        obstacles.append(jnp.array([-3.5, -3.3, -6, -1]))
        obstacles.append(jnp.array([-1.5, -1.3, -0.5, 6]))
        obstacles.append(jnp.array([0.5, 0.7, -6, 0]))
        obstacles.append(jnp.array([2.5, 2.6, -0.5, 6]))
        # obstacles.append(jnp.array([-5.75, -1.5, -2.3, -2.1]))
        # obstacles.append(jnp.array([-0.1, 4, -2.3, -2.1]))
        # obstacles.append(jnp.array([-6.0, -5.8, -6.0, 6.0]))
        # obstacles.append(jnp.array([3.9, 4.0, -6.0, 6.0]))
        # obstacles.append(jnp.array([-6.0, 6.0, -6.0, -5.9]))
        # obstacles.append(jnp.array([-6.0, 6.0, 3.9, 5.0]))

        # for i in range(num_x):
        #     for j in range(num_y):
        #         xmin = i * (box_size + gap)
        #         ymin = j * (box_size + gap)
        #         xmax = xmin + box_size
        #         ymax = ymin + box_size
        #         obstacles.append(jnp.array([xmin, xmax, ymin, ymax]))
        return jnp.stack(obstacles)

    @partial(jax.jit, static_argnames=['self'])
    def segment_endpoints(self, xpos, half_length):
        x, y, theta = xpos
        # theta *= 0.0
        dx = jnp.cos(theta) * half_length
        dy = jnp.sin(theta) * half_length
        p1 = jnp.array([x - dx, y - dy])
        p2 = jnp.array([x + dx, y + dy])
        return p1, p2


    @partial(jax.jit, static_argnames=['self'])
    def sdf_aabb(self, pos: jax.Array, half_length: jax.Array, bounds: jax.Array) -> float:
        xmin, xmax, ymin, ymax = bounds
        center = jnp.array([(xmin + xmax) / 2, (ymin + ymax) / 2])
        half_extents = jnp.array([(xmax - xmin) / 2, (ymax - ymin) / 2])

        p1, p2 = self.segment_endpoints(pos, half_length * 0.8)

        d1 = jnp.abs(p1 - center) - half_extents
        d2 = jnp.abs(p2 - center) - half_extents
        outside_dist1 = jnp.linalg.norm(jnp.maximum(d1, 0.0))
        inside_dist1 = jnp.minimum(jnp.maximum(d1[0], d1[1]), 0.0)

        outside_dist2 = jnp.linalg.norm(jnp.maximum(d2, 0.0))
        inside_dist2 = jnp.minimum(jnp.maximum(d2[0], d2[1]), 0.0)

        return jnp.minimum(outside_dist1 + inside_dist1, outside_dist2 + inside_dist2)

    @partial(jax.jit, static_argnames=['self'])
    def hinge_loss(self, sdf: float, epsilon: float) -> float:
        return jnp.where(sdf < epsilon, 0.5 * (epsilon - sdf) ** 2, 0.0)




    def draw_drone(self, ax, x: DState, th: DParams = None):

        px, py, theta = x.qpos
        hl = th.half_length if th is not None else 0.5
        
        # Rotation
        c, s = jnp.cos(theta), jnp.sin(theta)
        R = jnp.array([[c, -s],
                    [s,  c]])
        
        # Endpoints of the body (local coords: left = [-hl, 0], right = [+hl, 0])
        pL_local = jnp.array([-hl, 0.0])
        pR_local = jnp.array([ hl, 0.0])
        
        pL_world = jnp.array([px, py]) + R @ pL_local
        pR_world = jnp.array([px, py]) + R @ pR_local
        
        # Draw body
        ax.plot([pL_world[0], pR_world[0]], [pL_world[1], pR_world[1]], 'k-', lw=3)
        
        # Draw thrusters
        ax.add_patch(plt.Circle(pL_world, 0.05, color='red', alpha=0.6))
        ax.add_patch(plt.Circle(pR_world, 0.05, color='blue', alpha=0.6))
        
        # Draw COM
        ax.plot(px, py, 'ko', ms=6)

        # Set equal aspect
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlabel("x")
        ax.set_ylabel("y")


    def plot_obstacles(self, obstacles, color='black'):
        """Plot a list of AABB obstacles."""
        ax = plt.gca()
        for obs in obstacles:
            xmin, xmax, ymin, ymax = obs
            width = xmax - xmin
            height = ymax - ymin
            rect = plt.Rectangle((xmin, ymin), width, height,
                                edgecolor=color, facecolor='black', linewidth=1)
            ax.add_patch(rect)


    def visualize(self, theta, config, traj):

        interval=1
        save_path = None
        # save_path = "experiments/drone_nav_co/path.gif"
        fig, ax = plt.subplots()
        ax.set_xlim(-5, 5)
        ax.set_ylim(-5, 5)
        ax.set_aspect("equal", adjustable="box")

        # Container for artists (we’ll re-draw the drone each frame)
        artists = []



        def init():
            ax.clear()
            ax.set_xlim(-5, 5)
            ax.set_ylim(-5, 5)
            ax.set_aspect("equal", adjustable="box")
            return []

        def update(frame_idx):
            ax.clear()
            ax.set_xlim(-5, 5)
            ax.set_ylim(-5, 5)
            ax.set_aspect("equal", adjustable="box")

            plt.plot(theta.target[0], theta.target[1], marker='*', linestyle='-', color='blue', markersize=10)
            self.plot_obstacles(theta.obstacles)
            # pick frame (you had [t,s], so flatten here if needed)
            # t, s = divmod(frame_idx, traj.qpos.shape[1])
            state = jax.tree.map(lambda x: x[frame_idx], traj)
            self.draw_drone(ax, state, theta)
            return []

        n_frames = traj.qpos.shape[0]# * traj.qpos.shape[1]
        ani = animation.FuncAnimation(
            fig, update, frames=n_frames, init_func=init,
            blit=False, interval=interval, repeat=True
        )

        if save_path is not None:
            ani.save(save_path, writer="ffmpeg", fps=1000/interval)
        else:
            plt.show()
