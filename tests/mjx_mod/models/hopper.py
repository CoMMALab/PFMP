import jax
import jax.numpy as jnp
from genjax import gen, mv_normal
from typing import NamedTuple
from mujoco import mjx


class DataTracer(NamedTuple):
    ctrl_delta: jax.Array
    ctrl: jax.Array
    qpos: jax.Array
    qvel: jax.Array

class ControlParams(NamedTuple):
    mjx_model: mjx.Data
    ctrl_nominal: jax.Array
    ctrl_stdev: float

mjx_step = jax.jit(mjx.step)

@gen
def transition_fn(data: mjx.Data, u: ControlParams, t: int) -> tuple[mjx.Data, DataTracer]:
    ctrl_delta = mv_normal(jnp.zeros(u.mjx_model.nu), jnp.eye(u.mjx_model.nu) * u.ctrl_stdev) @ "control"
    data = data.replace(ctrl=u.ctrl_nominal[t] + ctrl_delta)
    data = mjx_step(u.mjx_model, data)
    return data, DataTracer(
        ctrl_delta = ctrl_delta,
        ctrl = data.ctrl,
        qpos = data.qpos,
        qvel = data.qvel
    )

@jax.jit
def cost_fn(data: mjx.Data, _: ControlParams, t: int) -> float:
    x = data.qpos[0]
    return -10 * x

u = ControlParams(
    mjx_model = None,
    ctrl_nominal = None,
    ctrl_stdev = 5.0
)