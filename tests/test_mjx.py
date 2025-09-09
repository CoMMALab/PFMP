
import fire
import time
import jax
import jax.numpy as jnp
import mujoco
import matplotlib.pyplot as plt
import importlib

from mujoco.viewer import launch_passive
from mujoco import mjx
from pfmp import PFMP


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

def get_site_id(model, site_name):
    for i in range(model.nsite):
        start = model.name_siteadr[i]
        end = model.name_siteadr[i + 1] if i + 1 < model.nsite else len(model.names)
        name = model.names[start:end].decode('utf-8').rstrip('\x00')
        if name == site_name:
            return i
    return -1  # Site not found

def print_all_site_names(model):
    for i in range(model.nsite):
        start = model.name_siteadr[i]
        end = model.name_siteadr[i + 1] if i + 1 < model.nsite else len(model.names)
        name = model.names[start:end].decode('utf-8').rstrip('\x00')
        print(f"Site {i}: '{name}'")

def load_model(name: str):
    with open(f"assets/{name}.xml", 'r') as f:
        xml = f.read()

    mj_model = mujoco.MjModel.from_xml_string(xml)
    mj_data = mujoco.MjData(mj_model)
    mjx_model = mjx.put_model(mj_model)
    x0 = mjx.put_data(mj_model, mj_data)

    parent = importlib.import_module(f"mjx_mod")
    module = getattr(parent, name)
    data = {k: getattr(module, k) for k in dir(module) if not k.startswith("_")}

    pfmp = PFMP(
        data['transition_fn'],
        data['cost_fn'],
        carry_mjx=True
    )
    return pfmp, x0, data['u']._replace(mjx_model=mjx_model), mj_model, mj_data, data['mjx_step']

def decelerating_schedule(H: int, start_gap: int = 3, max_gap: int = 20):
    gaps = jnp.linspace(start_gap, max_gap, H // 2)
    indices = jnp.cumsum(gaps).astype(int)
    indices = indices[indices < H]
    return jnp.zeros(H, dtype=bool).at[indices].set(True)

# def run_pfmp(pfmp, block_shape, x0, u):

def run_mppi(pfmp, block_shape, x0, u, iters=10):
    key = jax.random.key(0)
    j_sample_forward = jax.jit(pfmp.sample_forward_weighted, static_argnames=['block_shape'])
    for i in range(iters):
        print(i)
        (trajs, log_ws) = jax.block_until_ready(
            j_sample_forward(key, block_shape, x0, u)
        )
        ws = jnp.exp(log_ws - jnp.max(log_ws))
        ps = ws / jnp.sum(ws)
        expected = jax.tree.map(lambda field: jnp.tensordot(ps, field, axes=([0], [0])), trajs)
        u = u._replace(
            ctrl_nominal = u.ctrl_nominal + expected.ctrl_delta,
            coarse_control = u.coarse_control - 0.49/30  # annealing
        )
    traj = jax.tree.map(lambda x: x[0], trajs)
    return traj

def run_cem(pfmp, block_shape, x0, u):
    j_sample_forward = jax.jit(pfmp.sample_forward, static_argnames=['block_shape'])
    pass

def test_mjx(
    name: str,
    h: int,
    k: int,
    rc: int = 20,
    rt: float = 0.66,
    meth: str = "pfmp"
):
    pfmp, x0, u, mj_model, mj_data, mjx_step = load_model(name)
    ctrl_buffer = jnp.zeros((h, u.mjx_model.nu))  # or replan_horizon x nu
    u = u._replace(ctrl_nominal = ctrl_buffer)

    block_shape = (h,k)
    t = 0; t_local = 0
    key = jax.random.key(0)
    replan_cadence = rc
    with launch_passive(mj_model, data=mj_data, show_left_ui=False, show_right_ui=False) as viewer:
        if meth == "pfmp":
            j_sample_forward = jax.jit(pfmp.sample_forward, static_argnames=['block_shape'])
            while viewer.is_running():
                if t % replan_cadence == 0:
                    t0_prof = time.time()

                    (key, subkey) = jax.random.split(key)
                    (states, ancestors) = j_sample_forward(subkey, block_shape, x0, u, jnp.full(block_shape[0], True))
                    # trajs = pfmp.choose_paths(states, ancestors, jnp.arange(k))
                    # print(trajs.ctrl.shape)
                    # costs = jax.vmap(lambda trajs: jnp.sum(jax.vmap(pfmp.cost_fn, in_axes=[1,None,0])(trajs,u,jnp.arange(h))), in_axes=[0])(trajs)
                    # best_idx = jnp.argmin(costs)
                    # traj = jax.tree.map(lambda x: x[:,best_idx], traj)
                    traj = pfmp.choose_path(states, ancestors, 0)

                    ctrl_nominal = u.ctrl_nominal.at[:h-replan_cadence].set(traj.ctrl[replan_cadence:])
                    u = u._replace(ctrl_nominal = ctrl_nominal)
                    t_local = 0

                    t1_prof = time.time()
                    print(f"STEP RUNTIME: {t1_prof - t0_prof}")

                x0 = x0.replace(ctrl = traj.ctrl[t_local])
                x0 = mjx_step(u.mjx_model, x0)
                print(f"ctrl: {x0.ctrl}")
                print(f"cost: {pfmp.cost_fn(x0, u, t_local)}")

                mj_data.qpos[:] = jax.device_get(x0.qpos)
                mj_data.qvel[:] = jax.device_get(x0.qvel)
                mj_data.ctrl[:] = jax.device_get(x0.ctrl)
                mujoco.mj_forward(mj_model, mj_data)
                viewer.sync()

                t += 1; t_local += 1
        elif meth == "mppi":
            while viewer.is_running():
                if t % replan_cadence == 0:
                    t0_prof = time.time()
                    j_sample_forward = jax.jit(pfmp.sample_forward_weighted, static_argnames=['block_shape'])

                    if t_local != 0:
                        u = u._replace(
                            ctrl_nominal = jnp.concat((u.ctrl_nominal[replan_cadence:], jnp.zeros((replan_cadence, 1)))),
                            coarse_control = 5.0
                        )

                    for iter in range(5):
                        print(iter)
                        (key, subkey) = jax.random.split(key)
                        (trajs, log_ws) = j_sample_forward(subkey, block_shape, x0, u)

                        ws = jnp.exp(log_ws - jnp.max(log_ws))
                        ps = ws / jnp.sum(ws)
                        expected = jax.tree.map(lambda field: jnp.tensordot(ps, field, axes=([0], [0])), trajs)

                        u = u._replace(
                            ctrl_nominal = u.ctrl_nominal + expected.ctrl,
                            coarse_control = u.coarse_control * 0.8
                        )
                    t_local = 0

                    t1_prof = time.time()
                    print(f"STEP RUNTIME: {t1_prof - t0_prof}")

                x0 = x0.replace(ctrl = u.ctrl_nominal[t_local])
                x0 = mjx_step(u.mjx_model, x0)
                print(f"ctrl: {x0.ctrl}")
                print(f"cost: {pfmp.cost_fn(x0, u, t_local)}")

                mj_data.qpos[:] = jax.device_get(x0.qpos)
                mj_data.qvel[:] = jax.device_get(x0.qvel)
                mj_data.ctrl[:] = jax.device_get(x0.ctrl)
                mujoco.mj_forward(mj_model, mj_data)
                viewer.sync()

                t += 1; t_local += 1
        elif meth == "cem":
            traj = run_cem(pfmp, (h, k), x0, u)
        else:
            raise TypeError(f"Unsupported MPC Method: {meth}")

    # loop_viewer(mj_model, mj_data, traj, u, pfmp.cost_fn)

if __name__ == "__main__":
    fire.Fire(test_mjx)