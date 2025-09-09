import jax
import jax.numpy as jnp
from mujoco import mjx
from .inverted_pendulum import transition_fn, ControlParams, DataTracer, u, mjx_step
from genjax import gen, mv_normal
from typing import NamedTuple

class PParams(NamedTuple):
    mjx_model: mjx.Model
    nominal_u: jax.Array
    sigma: float
    fine_angle_bounds: float
    fine_angle_sigma: float

class StateTracker(NamedTuple):
    qpos: jax.Array
    qvel: jax.Array
    ctrl: jax.Array

@jax.jit
def cost_fn(data: mjx.Data, _u: jax.Array, _th: PParams, _t: float) -> float:
    p, theta1, theta2 = data.qpos
    dp, dtheta1, dtheta2 = data.qvel
    
    # Angular deviations from upright (more direct formulation)
    angle_cost1 = 1 + jnp.cos(theta1)
    angle_cost2 = 1 - jnp.cos(theta2)
    
    # Position and velocity costs with different weights
    pos_cost = 0.1 * p**2    # light penalty on cart position
    vel_cost = 0.02 * dp**2  # light penalty on cart velocity
    
    # Angular velocity costs (prevent excessive spinning)
    ang_vel_cost1 = 0.01 * dtheta1**2
    ang_vel_cost2 = 0.01 * dtheta2**2
    
    total_cost = (
        angle_cost1 + angle_cost2 +    # Primary objective: keep upright
        pos_cost + vel_cost +          # Secondary: control cart
        ang_vel_cost1 + ang_vel_cost2  # + Prevent excessive motion
    )
    
    return total_cost

@gen
def linear_proposal_fn(x: mjx.Data, u: jax.Array, th: PParams, _t: float) -> jax.Array:
    angle = x.qpos[1]
    near_upright = 1 + jnp.cos(angle) < th.fine_angle_bounds
    scale = jnp.where(near_upright, th.fine_angle_sigma, th.sigma)
    # ctrl = mv_normal(u[None,0], jnp.eye(th.mjx_model.nu) * scale) @ 'ctrl'
    ctrl = mv_normal(jnp.zeros(th.mjx_model.nu), jnp.eye(th.mjx_model.nu) * scale) @ 'ctrl'
    u = jnp.array([ctrl[0]])
    return u

@gen
def cubic_proposal_fn(x: mjx.Data, u: jax.Array, th: PParams, _t: float) -> jax.Array:
    angle = x.qpos[1]
    near_upright = 1 + jnp.cos(angle) < th.fine_angle_bounds
    scale = jnp.where(near_upright, th.fine_angle_sigma, th.sigma)
    # ctrl = mv_normal(u[None,0], jnp.eye(th.mjx_model.nu) * scale) @ 'ctrl'
    ctrl = mv_normal(jnp.zeros(th.mjx_model.nu), jnp.eye(th.mjx_model.nu) * scale) @ 'ctrl'
    dctrl = mv_normal(jnp.zeros(th.mjx_model.nu), jnp.eye(th.mjx_model.nu) * scale/2) @ 'dctrl'
    u = jnp.array([ctrl[0], dctrl[0]])
    return u

@gen
def quintic_proposal_fn(x: mjx.Data, u: jax.Array, th: PParams, _t: float) -> jax.Array:
    angle = x.qpos[1]
    near_upright = 1 + jnp.cos(angle) < th.fine_angle_bounds
    scale = jnp.where(near_upright, th.fine_angle_sigma, th.sigma)
    # ctrl = mv_normal(u[None,0], jnp.eye(th.mjx_model.nu) * scale) @ 'ctrl'
    ctrl = mv_normal(jnp.zeros(th.mjx_model.nu), jnp.eye(th.mjx_model.nu) * scale) @ 'ctrl'
    dctrl = mv_normal(jnp.zeros(th.mjx_model.nu), jnp.eye(th.mjx_model.nu) * scale/2) @ 'dctrl'
    ddctrl = mv_normal(jnp.zeros(th.mjx_model.nu), jnp.eye(th.mjx_model.nu) * scale/2) @ 'ddctrl'
    u = jnp.array([ctrl[0], dctrl[0], ddctrl[0]])
    return u

@jax.jit
def dynamics_fn(x: mjx.Data, u: jax.Array, theta: PParams, t: float) -> mjx.Data:
    x = x.replace(ctrl=u[None,0])
    return jax.jit(mjx.step)(theta.mjx_model, x), StateTracker(x.qpos, x.qvel, u)
