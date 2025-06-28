import jax
import jax.numpy as jnp
from genjax import gen, mv_normal
from typing import NamedTuple
from mujoco import mjx


class DataTracer(NamedTuple):
    ctrl: jax.Array
    qpos: jax.Array
    qvel: jax.Array

class ControlParams(NamedTuple):
    mjx_model: mjx.Data
    coarse_control: float
    fine_control: float
    fine_angle_bounds: float

@gen
def transition_fn(data: mjx.Data, u: ControlParams, t: int) -> tuple[mjx.Data, DataTracer]:
    theta = data.qpos[1]
    near_upright = 1 + jnp.cos(theta) < u.fine_angle_bounds
    scale = jnp.where(near_upright, u.fine_control, u.coarse_control)
    ctrl = mv_normal(jnp.zeros(u.mjx_model.nu), jnp.eye(u.mjx_model.nu) * scale) @ "control"
    data = data.replace(ctrl=ctrl)
    data = jax.jit(mjx.step)(u.mjx_model, data)
    return data, DataTracer(
        ctrl = data.ctrl,
        qpos = data.qpos,
        qvel = data.qvel
    )

@jax.jit
def cost_fn(data: mjx.Data, _: ControlParams, t: int) -> float:
    (p, theta), (dp, dtheta) = data.qpos, data.qvel
    pos_cost = 0.1 * p**2
    vel_cost = 0.02 * dp**2
    angle_cost = 2 * (1 + jnp.cos(theta))
    ang_vel_cost = 0.01 * dtheta**2
    return (
        pos_cost +
        vel_cost +
        angle_cost +
        ang_vel_cost
    )

u = ControlParams(
    mjx_model = None,
    coarse_control = 5.0,
    fine_control = 0.01,
    fine_angle_bounds = 0.1
)