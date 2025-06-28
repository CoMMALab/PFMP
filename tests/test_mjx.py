import time
import jax
import jax.numpy as jnp
import mujoco
import matplotlib.pyplot as plt

from typing import NamedTuple
from genjax import gen, mv_normal
from mujoco import mjx
from mujoco.viewer import launch_passive
from pfmp import PFMP


class DataTracer(NamedTuple):
    ctrl: jax.Array
    qpos: jax.Array
    qvel: jax.Array

class ControlParams(NamedTuple):
    coarse_control: float
    fine_control: float
    fine_angle_bounds: float

# === Load inverted_pendulum.xml ===
with open("tests/mjx_mod/assets/inverted_double_pendulum.xml", "r") as f:
    xml = f.read()

# with open("tests/mujoco/assets/inverted_double_pendulum.xml", "r") as f:
#     xml = f.read()

mj_model = mujoco.MjModel.from_xml_string(xml)
mj_data = mujoco.MjData(mj_model)
mjx_model = mjx.put_model(mj_model)
# mjx_data = mjx.put_data(mj_model, mj_data)
n_act = mjx_model.nu  # == 1 for inverted (single|double) pendulum

mjx_step = jax.jit(mjx.step)

@gen
def transition_fn(data: mjx.Data, params: ControlParams, t: int) -> tuple[mjx.Data, DataTracer]:
    theta = data.qpos[1]
    near_upright = jnp.abs(theta - jnp.pi) < params.fine_angle_bounds
    scale = jnp.where(near_upright, params.fine_control, params.coarse_control)
    ctrl = mv_normal(jnp.zeros(n_act), jnp.eye(n_act) * scale) @ "control"
    data = data.replace(ctrl=ctrl)
    data = mjx_step(mjx_model, data)
    return data, DataTracer(
        ctrl = data.ctrl,
        qpos = data.qpos,
        qvel = data.qvel
    )

# @jax.jit
# def cost_fn(data: mjx.Data, _: ControlParams, t: int) -> float:
#     (p, theta), (dp, dtheta) = data.qpos, data.qvel
#     pos_cost = 0.1 * p**2
#     vel_cost = 0.02 * dp**2
#     angle_cost = 1 + jnp.cos(theta)
#     ang_vel_cost = 0.01 * dtheta**2
#     return (
#         pos_cost +
#         vel_cost +
#         angle_cost +
#         ang_vel_cost
#     )

@jax.jit
def cost_fn(data: mjx.Data, _: ControlParams, t: int) -> float:
    p, theta1, theta2 = data.qpos
    dp, dtheta1, dtheta2 = data.qvel
    
    # Angular deviations from upright (more direct formulation)
    angle_cost1 = (1 + jnp.cos(theta1)) / 2  # 0 when upright, 20 when downward
    angle_cost2 = (1 - jnp.cos(theta2)) / 2  # 0 when upright, 20 when downward
    
    # Position and velocity costs with different weights
    pos_cost = 0.1 * p**2     # light penalty on cart position
    vel_cost = 0.02 * dp**2   # light penalty on cart velocity
    
    # Angular velocity costs (prevent excessive spinning)
    ang_vel_cost1 = 0.01 * dtheta1**2
    ang_vel_cost2 = 0.01 * dtheta2**2
    
    # Control effort penalty (energy efficiency)
    control_cost = 0.001 * data.ctrl[0]**2
    
    total_cost = (
        angle_cost1 + angle_cost2 +      # Primary objective: keep upright
        pos_cost + vel_cost +            # Secondary: control cart
        ang_vel_cost1 + ang_vel_cost2 +  # Prevent excessive motion
        control_cost                     # Penalize large control efforts
    )
    
    return total_cost

def decelerating_schedule(H: int, start_gap: int = 3, max_gap: int = 20):
    gaps = jnp.linspace(start_gap, max_gap, H // 2)
    indices = jnp.cumsum(gaps).astype(int)
    indices = indices[indices < H]
    return jnp.zeros(H, dtype=bool).at[indices].set(True)

# === Copy MJX data into MjData for visualization ===
def sync_to_mujoco(mj_data, traj, t):
    mj_data.qpos[:] = jax.device_get(traj.qpos[t])
    mj_data.qvel[:] = jax.device_get(traj.qvel[t])
    mj_data.ctrl[:] = jax.device_get(traj.ctrl[t])

# def loop_viewer(traj, u):
#     dt = mj_model.opt.timestep
#     with launch_passive(mj_model, data=mj_data, show_left_ui=False, show_right_ui=False) as viewer:
#         t = 0

#         # --- before the viewer loop ---
#         plt.ion()
#         fig, ax = plt.subplots()
#         line, = ax.plot([], [], lw=2)
#         ax.set_xlim(0, 500)           # adjust based on expected t
#         ax.set_ylim(0, 1000)          # adjust based on expected cost
#         ax.set_xlabel("Time step")
#         ax.set_ylabel("Cost")

#         costs = []
#         times = []

#         while viewer.is_running():
#             print(t)
#             sync_to_mujoco(mj_data, traj, t)
#             mujoco.mj_forward(mj_model, mj_data)

#             x = jax.tree.map(lambda field: field[t], traj)
#             print(f"qpos: {x.qpos}")
#             cost = float(cost_fn(x, u, t))  # convert to Python float

#             # --- update live plot ---
#             costs.append(cost)
#             times.append(t)
#             line.set_data(times, costs)
#             ax.set_xlim(0, max(500, t + 10))
#             ax.set_ylim(0, max(costs) * 1.1)
#             fig.canvas.draw()
#             fig.canvas.flush_events()

#             viewer.sync()
#             time.sleep(dt)
#             t += 1

def loop_viewer(traj, u):
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
            sync_to_mujoco(mj_data, traj, t)
            mujoco.mj_forward(mj_model, mj_data)
            x = jax.tree.map(lambda field: field[t], traj)
            # print(f"qpos[{t}] = {x.qpos}")
            
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
            time.sleep(dt)
            t += 1

def test_mjx():
    H, K = 800, 100000
    pfmp = PFMP(transition_fn, cost_fn, carry_mjx=True)
    key = jax.random.key(0)

    x0 = mjx.put_data(mj_model, mj_data)

    u = ControlParams(
        coarse_control = 5.0,
        fine_control = 0.01,
        fine_angle_bounds = 0.1
    )

    j_sample_forward = jax.jit(pfmp.sample_forward, static_argnames=['block_shape'])
    for i in range(1):
        t0 = time.time()
        (key, subkey) = jax.random.split(key)
        sched = decelerating_schedule(H, 3, 3)
        # sched = jnp.full(H, True)
        (states, ancestors) = jax.block_until_ready(
            j_sample_forward(subkey, (H, K), x0, u, sched)
        )
        t1 = time.time()
        if i == 0:
            print(f"sample_forward (first run): {t1 - t0}")
        else:
            print(f"sample_forward (after jit): {t1 - t0}")

    traj = pfmp.choose_path(states, ancestors, 0)
    loop_viewer(traj, u)

if __name__ == "__main__":
    test_mjx()