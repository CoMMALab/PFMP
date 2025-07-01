import jax
import jax.numpy as jnp
import mujoco as mj
from genjax import gen, mv_normal
from typing import NamedTuple
from mujoco import mjx, mj_name2id


class DataTracer(NamedTuple):
    ctrl_delta: jax.Array
    ctrl: jax.Array
    qpos: jax.Array
    qvel: jax.Array

class ControlParams(NamedTuple):
    mjx_model: mjx.Data
    ctrl_nominal: jax.Array
    ctrl_stdev: float
    goal: jax.Array

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

def get_body_id(model, body_name):
    body_names = []
    for i in range(model.nbody):
        start = model.name_bodyadr[i]
        # Handle the last body name differently
        if i + 1 < len(model.name_bodyadr):
            end = model.name_bodyadr[i + 1]
        else:
            end = len(model.names)
        
        name = model.names[start:end].decode('utf-8').rstrip('\x00')
        body_names.append(name)
    
    try:
        return body_names.index(body_name)
    except ValueError:
        return -1  # Body not found

@jax.jit
def fk_fn(model: mjx.Model, data: mjx.Data) -> jnp.ndarray:
    # Get body ID for the end-effector (only needs to be done once — cache if needed)
    body_id = get_body_id(model, "hand")

    # Extract rotation matrix and position vector from data
    rot = data.xmat[body_id].reshape(3, 3)  # MJX stores xmat as flat 9
    pos = data.xpos[body_id]                # shape (3,)

    # Construct 4x4 homogeneous transform
    ee_pose = jnp.eye(4)
    ee_pose = ee_pose.at[:3, :3].set(rot)
    ee_pose = ee_pose.at[:3, 3].set(pos)
    return ee_pose

@jax.jit
def cost_fn(data: mjx.Data, u: ControlParams, t: int) -> float:
    ee_pose = fk_fn(u.mjx_model, data)
    return 100 * jnp.linalg.norm(ee_pose - u.goal, ord='fro')
    # return 100 * jnp.linalg.norm(ee_pose[:3,3] - u.goal[:3,3])

u = ControlParams(
    mjx_model = None,
    ctrl_nominal = None,
    ctrl_stdev = 5000.0,
    goal = jnp.eye(4).at[:3,3].set([0.3, 0.4, 0.5])
)