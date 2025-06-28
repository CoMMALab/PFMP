
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


def load_model(name: str):
    with open(f"tests/mjx_mod/assets/{name}.xml", 'r') as f:
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
    return pfmp, x0, data['u']._replace(mjx_model=mjx_model), mj_model, mj_data

def sync_to_mujoco(mj_data, traj, t):
    mj_data.qpos[:] = jax.device_get(traj.qpos[t])
    mj_data.qvel[:] = jax.device_get(traj.qvel[t])
    mj_data.ctrl[:] = jax.device_get(traj.ctrl[t])

def loop_viewer(mj_model, mj_data, traj, u, cost_fn):
    dt = mj_model.opt.timestep
    with launch_passive(mj_model, data=mj_data, show_left_ui=False, show_right_ui=False) as viewer:
        t = 0
        # --- before the viewer loop ---
        plt.ion()
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        
        # Cost plot
        cost_line, = ax1.plot([], [], 'b-', lw=2, label='Cost')
        ax1.set_xlim(0, 500)
        ax1.set_ylim(0, 1000)
        ax1.set_ylabel("Cost")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Control plot
        ctrl_line, = ax2.plot([], [], 'r-', lw=2, label='Control')
        ax2.set_xlim(0, 500)
        ax2.set_ylim(-100, 100)  # adjust based on expected control range
        ax2.set_xlabel("Time step")
        ax2.set_ylabel("Control Input")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        costs = []
        ctrls = []
        times = []
        
        while viewer.is_running():
            print(t)
            sync_to_mujoco(mj_data, traj, t)
            mujoco.mj_forward(mj_model, mj_data)
            x = jax.tree.map(lambda field: field[t], traj)
            print(f"qpos: {x.qpos}")
            
            cost = float(cost_fn(x, u, t))
            ctrl = float(x.ctrl[0]) if len(x.ctrl) > 0 else 0.0  # assuming single control input
            
            # --- update data ---
            costs.append(cost)
            ctrls.append(ctrl)
            times.append(t)
            
            # --- update live plots ---
            # Cost plot
            cost_line.set_data(times, costs)
            ax1.set_xlim(0, max(500, t + 10))
            ax1.set_ylim(0, max(costs) * 1.1 if costs else 1000)
            
            # Control plot
            ctrl_line.set_data(times, ctrls)
            ax2.set_xlim(0, max(500, t + 10))
            if ctrls:
                ctrl_min, ctrl_max = min(ctrls), max(ctrls)
                ctrl_range = max(abs(ctrl_min), abs(ctrl_max))
                ax2.set_ylim(-ctrl_range * 1.1, ctrl_range * 1.1)
            
            fig.canvas.draw()
            fig.canvas.flush_events()
            viewer.sync()
            # time.sleep(dt)
            t += 1

def decelerating_schedule(H: int, start_gap: int = 3, max_gap: int = 20):
    gaps = jnp.linspace(start_gap, max_gap, H // 2)
    indices = jnp.cumsum(gaps).astype(int)
    indices = indices[indices < H]
    return jnp.zeros(H, dtype=bool).at[indices].set(True)

def test_mjx(
    name: str,
    h: int,
    k: int
):
    key = jax.random.key(2)
    pfmp, x0, u, mj_model, mj_data = load_model(name)

    t0 = time.time()
    j_sample_forward = jax.jit(pfmp.sample_forward, static_argnames=['block_shape'])
    # sched = decelerating_schedule(h, 3, 3)
    sched = jnp.full(h, True)
    (states, ancestors) = jax.block_until_ready(
        j_sample_forward(key, (h, k), x0, u, sched)
    )
    t1 = time.time()
    print(f"total time: {t1 - t0}")

    traj = pfmp.choose_path(states, ancestors, 0)
    loop_viewer(mj_model, mj_data, traj, u, pfmp.cost_fn)

if __name__ == "__main__":
    fire.Fire(test_mjx)