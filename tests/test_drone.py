import matplotlib.pyplot as plt
import matplotlib.animation as animation
import jax
import jax.numpy as jnp
from typing import NamedTuple, Tuple
from genjax import gen, mv_normal
from pfmp import PFMP, make_transition_model


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

class DState(NamedTuple):
    qpos: jax.Array
    qvel: jax.Array

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

# ---------- Init ----------
def init():
    mass = 1.0
    half_length = 0.5
    moi = (1.0/3.0) * mass * half_length**2  # ≈ 0.0833333

    th = DParams(
        mass = mass,
        half_length = half_length,
        moi = moi,
        max_thrust = 100.0,
        sigma = 10.0,
        target = jnp.array([2., -2.]),
        gains = Gains(10, 0.1)
    )
    x0 = DState(qpos=jnp.array([0.,0.,0.8]), qvel=jnp.array([0.,0.,0.]))
    return th, x0

### Thrust-based Controller

@gen
def thrust_proposal_fn(x: DState, u: jax.Array, th: DParams, t: float) -> jax.Array:
    rho = 0.8   # correlation coefficient, -1 <= rho <= 1
    Sigma = th.sigma**2 * jnp.array([
        [1.0, rho],
        [rho, 1.0]
    ])
    thrust = mv_normal(G/2, Sigma) @ "thrust"
    dthrust = mv_normal(G/2, Sigma/2) @ "dthrust"
    return jnp.array([thrust, dthrust])

@jax.jit
def newtonian_dynamics_fn(x: DState, u: jax.Array, th: DParams, t: float) -> Tuple[DState, DState]:
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

@jax.jit
def error_fn(x: DState, u: jax.Array, th: DParams, t: float) -> float:
    p = x.qpos[:2] - th.target
    perr = jnp.dot(p,p) + x.qpos[2]**2

    d = x.qvel[:2]
    derr = jnp.dot(d,d) + x.qvel[2]**2

    return th.gains.kp*perr + th.gains.kd*derr

## PD-controller 
# @gen
# def traj_proposal_fn(x: DState, u: jax.Array, th: DParams, t: float) -> jax.Array:
#     qpos = mv_normal(jnp.zeros(2), jnp.eye(2) * th.sigma) @ "qpos"
#     qvel = mv_normal(jnp.zeros(2), jnp.eye(2) * th.sigma/2) @ "qvel"
#     return jnp.array([qpos, qvel])

# @jax.jit
# def pd_dynamics_fn(x: DState, u: jax.Array, th: DParams, t: float) -> Tuple[DState, DState]:
#     return (x, x)

# @jax.jit
# def track_error_fn(x: DState, u: jax.Array, th: DParams, t: float) -> float:
#     return 0.

def draw_drone(ax, x: DState, th: DParams = None):
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

def animate_traj(traj, th, interval=50, save_path=None):
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

        # pick frame (you had [t,s], so flatten here if needed)
        t, s = divmod(frame_idx, traj.qpos.shape[1])
        state = jax.tree.map(lambda x: x[t, s], traj)
        draw_drone(ax, state, th)
        return []

    n_frames = traj.qpos.shape[0] * traj.qpos.shape[1]
    ani = animation.FuncAnimation(
        fig, update, frames=n_frames, init_func=init,
        blit=False, interval=interval, repeat=True
    )

    if save_path is not None:
        ani.save(save_path, writer="ffmpeg", fps=1000/interval)
    else:
        plt.show()

def test_model():
    th, x0 = init()
    u0 = jnp.array([[0., 0.], [0., 0.]])
    pfmp = PFMP(make_transition_model(thrust_proposal_fn, error_fn, newtonian_dynamics_fn, l=4))
    key = jax.random.key(42)
    (states, ancestors) = pfmp.run_sir(key, x0, u0, th, 80, 100000, 0.8)
    print(states.qpos.shape)
    traj = pfmp.choose_path(states, ancestors, 0)
    print(traj.qpos.shape)

    animate_traj(traj, th, interval=50)


if __name__ == "__main__":
    test_model()