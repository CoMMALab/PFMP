"""
push_t_2d.py — T-block pushing with PFMP in 2D (no MuJoCo), fr3 variant.

Imports geometry and cost utilities from push_t_task / geom_t.

Usage:
    python examples/fr3/push_t_2d.py
    python examples/fr3/push_t_2d.py --save-gif
"""

import argparse
import itertools
from typing import NamedTuple

import jax
import jax.numpy as jnp
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle, Polygon

from pfmp import PFMP
from pfmp.algorithm import ps_update
from pfmp.interpolate import hermite_spline

from pushT_task import angle_wrap, pos_angle_cost, TERMINAL_SCALE
from geomT import VERTS_LOCAL, sdf_world, verts_world


# ============================================================
# Configuration
# ============================================================

SEED             = 42
DT               = 0.01
HORIZON          = 2.5
NUM_KNOTS        = 6
PARTICLES        = 2048
ESS_THRESH       = 0.5
N_REPLANS        = 12
STEPS_PER_REPLAN = 100

WORKSPACE_MIN = np.array([-0.5, -0.6])
WORKSPACE_MAX = np.array([ 0.7,  0.6])
T_BLOCK_MIN   = np.array([-0.3, -0.4])
T_BLOCK_MAX   = np.array([ 0.65, 0.5])

# Physics
ROBOT_RADIUS   = 0.015
R_MASS         = 0.05
K_P            = 80.0
K_D            = 8.0
K_CONTACT      = 1200.0
K_CONTACT_DAMP = 80.0
T_MASS         = 0.30
T_INERTIA      = 0.003
T_LIN_DAMP     = 6.0
T_ANG_DAMP     = 2.5

# Planner
U_STDEV          = 0.05
POS_WEIGHT       = 15.0
ANGLE_WEIGHT     = 1.5
MPPI_TEMPERATURE = 0.5

# DIAL-MPC
DIAL_N_INNER = 5
DIAL_BETA1   = 1.0
DIAL_BETA2   = 1.0


# ============================================================
# Params
# ============================================================

class PushParams(NamedTuple):
    mean:         jax.Array   # (num_knots, 2)
    goal_xy:      jax.Array   # (2,)
    goal_angle:   float
    pos_weight:   float
    angle_weight: float
    u_stdev:      float
    step_len:     float
    obstacles:    jax.Array   # (N, 4) [cx, cy, hx, hy]


# ============================================================
# SDF helpers
# ============================================================

def _sdf_box(p, half):
    d = jnp.abs(p) - half
    outside = jnp.sqrt(jnp.sum(jnp.maximum(d, 0.0) ** 2) + 1e-12)
    inside  = jnp.minimum(jnp.max(d), 0.0)
    return outside + inside


# ============================================================
# Dynamics
#   x = [t_x, t_y, t_angle, t_vx, t_vy, t_vangle, r_x, r_y, r_vx, r_vy]
# ============================================================

def dynamics(x, u, theta, t):
    t_xy    = x[0:2]
    t_angle = x[2]
    t_vel   = x[3:5]
    t_vang  = x[5]
    r_xy    = x[6:8]
    r_vel   = x[8:10]

    r_acc = K_P * (u - r_xy) - K_D * r_vel

    sdf_val     = sdf_world(r_xy, t_xy, t_angle)
    penetration = jnp.maximum(0.0, ROBOT_RADIUS - sdf_val)
    sdf_grad    = jax.grad(lambda p: sdf_world(p, t_xy, t_angle))(r_xy)
    n           = sdf_grad / (jnp.linalg.norm(sdf_grad) + 1e-7)

    f_contact = K_CONTACT * penetration
    f_on_t =  -f_contact * n
    f_on_r =   f_contact * n

    contact_pt = r_xy - ROBOT_RADIUS * n
    moment_arm = contact_pt - t_xy
    torque = moment_arm[0] * f_on_t[1] - moment_arm[1] * f_on_t[0]

    c, s = jnp.cos(t_angle), jnp.sin(t_angle)
    R = jnp.array([[c, -s], [s, c]])

    _verts = [np.array(v) for v in VERTS_LOCAL]
    _pts = list(itertools.chain.from_iterable(
        [_verts[i] + (_verts[(i+1) % len(_verts)] - _verts[i]) * k / 4
         for k in range(5)]
        for i in range(len(_verts))
    ))
    t_surface_local = jnp.array(np.unique(np.round(_pts, 6), axis=0))
    t_surface_world = jax.vmap(lambda p: R @ p + t_xy)(t_surface_local)

    def wall_contacts(obs):
        obs_ctr, obs_half = obs[:2], obs[2:]
        r_sdf  = _sdf_box(r_xy - obs_ctr, obs_half)
        r_pen  = jnp.maximum(0.0, ROBOT_RADIUS - r_sdf)
        r_grad = jax.grad(lambda p: _sdf_box(p - obs_ctr, obs_half))(r_xy)
        r_n    = r_grad / (jnp.linalg.norm(r_grad) + 1e-7)
        r_vel_n = jnp.dot(r_vel, r_n)
        df_r   = r_pen * (K_CONTACT * r_n - K_CONTACT_DAMP * r_vel_n * r_n)
        t_pens  = jax.vmap(lambda p: jnp.maximum(0.0, -_sdf_box(p - obs_ctr, obs_half)))(t_surface_world)
        t_grads = jax.vmap(jax.grad(lambda p: _sdf_box(p - obs_ctr, obs_half)))(t_surface_world)
        t_norms = t_grads / (jnp.linalg.norm(t_grads, axis=-1, keepdims=True) + 1e-7)
        t_vel_ns = jnp.sum(t_vel[None] * t_norms, axis=-1)
        t_forces = t_pens[:, None] * (K_CONTACT * t_norms - K_CONTACT_DAMP * t_vel_ns[:, None] * t_norms)
        df_t    = jnp.sum(t_forces, axis=0)
        arms    = t_surface_world - t_xy
        dtorque = jnp.sum(arms[:, 0] * t_forces[:, 1] - arms[:, 1] * t_forces[:, 0])
        return df_r, df_t, dtorque

    df_rs, df_ts, dtorques = jax.vmap(wall_contacts)(theta.obstacles)
    f_on_r = f_on_r + jnp.sum(df_rs, axis=0)
    f_on_t = f_on_t + jnp.sum(df_ts, axis=0)
    torque  = torque + jnp.sum(dtorques)

    t_vel_new  = t_vel + (f_on_t / T_MASS) * DT
    t_vel_new  = t_vel_new * (1.0 - T_LIN_DAMP * DT)
    t_vang_new = t_vang + (torque / T_INERTIA) * DT
    t_vang_new = t_vang_new * (1.0 - T_ANG_DAMP * DT)
    t_xy_new   = t_xy + t_vel_new * DT
    t_ang_new  = t_angle + t_vang_new * DT

    r_vel_new = r_vel + (r_acc + f_on_r / R_MASS) * DT
    r_xy_new  = r_xy + r_vel_new * DT

    r_xy_new = jnp.clip(r_xy_new, jnp.array(WORKSPACE_MIN), jnp.array(WORKSPACE_MAX))
    t_xy_new = jnp.clip(t_xy_new, jnp.array(T_BLOCK_MIN),   jnp.array(T_BLOCK_MAX))

    x_next = jnp.concatenate([
        t_xy_new, jnp.array([t_ang_new]),
        t_vel_new, jnp.array([t_vang_new]),
        r_xy_new, r_vel_new,
    ])
    return x_next, x_next


def running_cost(x, u, theta, t):
    return pos_angle_cost(x[0:2], x[2], theta.goal_xy, theta.goal_angle,
                          theta.pos_weight, theta.angle_weight)


def terminal_cost(x, theta):
    return TERMINAL_SCALE * pos_angle_cost(x[0:2], x[2], theta.goal_xy, theta.goal_angle,
                                           theta.pos_weight, theta.angle_weight)


# ============================================================
# Proposals
# ============================================================

PROPOSAL_MIX = 0.5
STEP_LEN_VEL = 0.05


def init_control(key, x, theta):
    key_mix, key_a, key_b = jax.random.split(key, 3)
    r_xy, t_xy = x[6:8], x[0:2]
    direction  = t_xy - r_xy
    dist       = jnp.linalg.norm(direction) + 1e-7
    directed   = r_xy + (direction / dist) * jnp.minimum(dist, STEP_LEN_VEL)
    directed   = directed + jax.random.normal(key_a, (2,)) * theta.u_stdev
    random_loc = r_xy + jax.random.normal(key_b, (2,)) * theta.u_stdev * 2.0
    return jnp.where(jax.random.uniform(key_mix) < PROPOSAL_MIX, directed, random_loc)


def proposal(key, x, _u, theta, t):
    key_mix, key_a, key_b = jax.random.split(key, 3)
    r_xy, t_xy = x[6:8], x[0:2]
    direction  = t_xy - r_xy
    dist       = jnp.linalg.norm(direction) + 1e-7
    directed   = r_xy + (direction / dist) * jnp.minimum(dist, STEP_LEN_VEL)
    directed   = directed + jax.random.normal(key_a, (2,)) * theta.u_stdev
    random_loc = r_xy + jax.random.normal(key_b, (2,)) * theta.u_stdev * 2.0
    return jnp.where(jax.random.uniform(key_mix) < PROPOSAL_MIX, directed, random_loc)


def perturbation_init(key, _x, theta):
    return theta.mean[0] + jax.random.normal(key, (2,)) * theta.u_stdev


def perturbation_proposal(key, _x, _u, theta, t):
    idx = jnp.clip(jnp.round(t / theta.step_len).astype(jnp.int32) + 1,
                   0, theta.mean.shape[0] - 1)
    return theta.mean[idx] + jax.random.normal(key, (2,)) * theta.u_stdev


# ============================================================
# Simulation
# ============================================================

def dial_sigma(i, h, sigma_max, N, H, beta1, beta2):
    return sigma_max * float(np.exp(-(N - i) / (beta1 * N) - (H - h) / (beta2 * H)))


def mppi_update(theta, trajs, costs, temperature):
    total_costs = jnp.sum(costs, axis=1)
    weights     = jax.nn.softmax(-total_costs / temperature)
    new_mean    = jnp.sum(trajs * weights[:, None, None], axis=0)
    return theta._replace(mean=new_mean)


def simulate(method, x0, theta, key, particles):
    is_baseline = method in ('mppi', 'dial')
    ctrl = PFMP(
        dynamics, running_cost, terminal_cost,
        perturbation_init     if is_baseline else init_control,
        perturbation_proposal if is_baseline else proposal,
        horizon=HORIZON, num_knots=NUM_KNOTS, particles=particles,
        ess_resample_thresh=0.0 if is_baseline else ESS_THRESH, dt=DT,
    )

    def mean_from_state(x):
        r_xy, t_xy = x[6:8], x[0:2]
        alphas = jnp.linspace(0.0, 1.0, NUM_KNOTS)
        return (1.0 - alphas[:, None]) * r_xy + alphas[:, None] * t_xy

    if is_baseline:
        theta = theta._replace(mean=mean_from_state(x0))

    print("JIT warm-up…", end=" ", flush=True)
    key, subkey = jax.random.split(key)
    _ = ctrl.run_sis(subkey, x0, theta)
    _ = ctrl.run_sis(subkey, x0, theta)
    print("done")

    print(f"Simulating {N_REPLANS} replan steps × {STEPS_PER_REPLAN} sim steps…")
    dynamics_jit = jax.jit(dynamics)
    goal_xy      = np.array(theta.goal_xy)
    goal_angle   = float(theta.goal_angle)

    states, plans = [np.array(x0)], []
    x  = x0
    tk = ctrl.tk()

    for step_i in range(N_REPLANS):
        key, subkey = jax.random.split(key)
        trajs, costs = ctrl.run_sis(subkey, x, theta)

        if method == 'mppi':
            total_costs = jnp.sum(costs, axis=1)
            temperature = float(jnp.std(total_costs)) * MPPI_TEMPERATURE
            theta = mppi_update(theta, trajs, costs, max(temperature, 1e-6))
        elif method == 'dial':
            for i_inner in range(DIAL_N_INNER, 0, -1):
                sigmas = jnp.array([
                    dial_sigma(i_inner, h, U_STDEV, DIAL_N_INNER, NUM_KNOTS,
                               DIAL_BETA1, DIAL_BETA2)
                    for h in range(NUM_KNOTS)
                ])
                key, subkey_i = jax.random.split(key)
                theta_i = theta._replace(u_stdev=float(sigmas.mean()))
                trajs_i, costs_i = ctrl.run_sis(subkey_i, x, theta_i)
                total_costs = jnp.sum(costs_i, axis=1)
                temperature = float(jnp.std(total_costs)) * MPPI_TEMPERATURE
                theta = mppi_update(theta, trajs_i, costs_i, max(temperature, 1e-6))
        else:
            theta, _ = ps_update(theta, trajs, costs)

        plans.append(np.array(theta.mean))
        tq = jnp.arange(STEPS_PER_REPLAN) * DT
        us = hermite_spline(tq, tk, theta.mean)
        for i in range(STEPS_PER_REPLAN):
            x, _ = dynamics_jit(x, us[i], theta, float(i * DT))
            states.append(np.array(x))

        new_tk = tk + STEPS_PER_REPLAN * DT
        theta  = theta._replace(mean=hermite_spline(new_tk, tk, theta.mean))
        if (step_i + 1) % 4 == 0:
            s = states[-1]
            pos_err   = float(np.linalg.norm(s[0:2] - goal_xy))
            angle_err = float(abs(((s[2] - goal_angle + np.pi) % (2 * np.pi)) - np.pi))
            print(f"  step {step_i+1:3d} | pos_err={pos_err:.3f}m  angle_err={angle_err:.3f}rad")

    print(f"Simulation complete — {len(states)} states recorded.")
    return states, plans


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save-gif', action='store_true')
    parser.add_argument('--seed', type=int, default=SEED)
    parser.add_argument('--particles', type=int, default=PARTICLES)
    parser.add_argument('--method', choices=['pfmp', 'mppi', 'dial'], default='pfmp')
    args = parser.parse_args()

    wall_hw  = 0.02
    wall_len = 0.20
    gap_y    = 0.073
    x_left   = 0.03
    x_right  = 0.27

    rng = np.random.default_rng(args.seed)
    if args.seed != SEED:
        t_xy_0    = rng.uniform([WORKSPACE_MIN[0] + 0.05, -gap_y + 0.05],
                                 [x_left - 0.10,           gap_y - 0.05])
        t_angle_0 = rng.uniform(-np.pi, np.pi)
        goal_xy   = rng.uniform([x_right + 0.15, -gap_y + 0.05],
                                 [WORKSPACE_MAX[0] - 0.05, gap_y - 0.05])
        goal_angle = rng.uniform(-np.pi, np.pi)
    else:
        t_xy_0    = np.array([-0.15, 0.15])
        t_angle_0 = 0.0
        goal_xy   = np.array([ 0.40, 0.0])
        goal_angle = 0.0

    r_xy_0 = np.clip(t_xy_0 - np.array([0.12, 0.0]),
                     WORKSPACE_MIN + 0.05, WORKSPACE_MAX - 0.05)

    x0 = jnp.array([
        t_xy_0[0], t_xy_0[1], t_angle_0,
        0.0, 0.0, 0.0,
        r_xy_0[0], r_xy_0[1], 0.0, 0.0,
    ])

    obstacles = np.array([
        [x_left,   gap_y + wall_len, wall_hw, wall_len],
        [x_left,  -gap_y - wall_len, wall_hw, wall_len],
        [x_right,  gap_y + wall_len, wall_hw, wall_len],
        [x_right, -gap_y - wall_len, wall_hw, wall_len],
    ])

    theta = PushParams(
        mean=jnp.zeros((NUM_KNOTS, 2)),
        goal_xy=jnp.array(goal_xy),
        goal_angle=float(goal_angle),
        pos_weight=POS_WEIGHT,
        angle_weight=ANGLE_WEIGHT,
        u_stdev=U_STDEV,
        step_len=HORIZON / (NUM_KNOTS - 1),
        obstacles=jnp.array(obstacles),
    )

    key = jax.random.key(args.seed)
    states, plans = simulate(args.method, x0, theta, key, args.particles)

    # ── Animation ────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 7))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('#f5f5f5')
    ax.set_xlim(WORKSPACE_MIN[0] - 0.05, WORKSPACE_MAX[0] + 0.05)
    ax.set_ylim(WORKSPACE_MIN[1] - 0.05, WORKSPACE_MAX[1] + 0.05)
    ax.set_aspect('equal')
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor('black'); spine.set_linewidth(1.5)
    ax.set_title(f"{args.method.upper()}: T-block Pushing", fontsize=13)

    ws_rect = mpatches.FancyBboxPatch(
        WORKSPACE_MIN, WORKSPACE_MAX[0] - WORKSPACE_MIN[0],
        WORKSPACE_MAX[1] - WORKSPACE_MIN[1],
        boxstyle="square,pad=0", fill=False,
        edgecolor='#aaaaaa', linewidth=1.0, linestyle='--', zorder=0,
    )
    ax.add_patch(ws_rect)

    for obs in obstacles:
        cx, cy, hx, hy = obs
        ax.add_patch(mpatches.Rectangle(
            (cx - hx, cy - hy), 2 * hx, 2 * hy,
            facecolor='dimgray', edgecolor='black', linewidth=1.5, zorder=3,
        ))

    goal_patch = Polygon(verts_world(goal_xy, goal_angle), closed=True,
                         facecolor='darkorange', alpha=0.25,
                         edgecolor='darkorange', linewidth=1.5,
                         linestyle='--', zorder=1)
    ax.add_patch(goal_patch)
    ax.plot(*goal_xy, marker='*', color='darkorange', markersize=12,
            markeredgecolor='black', markeredgewidth=0.8, zorder=2)

    t_patch = Polygon(verts_world(t_xy_0, t_angle_0), closed=True,
                      facecolor='darkorange', edgecolor='saddlebrown',
                      linewidth=1.5, zorder=5)
    ax.add_patch(t_patch)

    robot_circle = Circle(r_xy_0, ROBOT_RADIUS,
                          facecolor='limegreen', edgecolor='darkgreen',
                          linewidth=1.5, zorder=8)
    ax.add_patch(robot_circle)

    trail_line, = ax.plot([], [], color='limegreen', alpha=0.4, linewidth=1.2, zorder=4)
    plan_dots,  = ax.plot([], [], 'o', color='royalblue', markersize=4, alpha=0.8, zorder=7)
    plan_line,  = ax.plot([], [], color='royalblue', alpha=0.4, linewidth=1.0, zorder=6)
    info_text   = ax.text(0.02, 0.98, '', transform=ax.transAxes, fontsize=9,
                          verticalalignment='top',
                          bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

    legend_handles = [
        mpatches.Patch(facecolor='darkorange', edgecolor='saddlebrown', label='T-block'),
        mpatches.Patch(facecolor='darkorange', alpha=0.25, linestyle='--',
                       edgecolor='darkorange', label='Goal pose'),
        mpatches.Patch(facecolor='limegreen', edgecolor='darkgreen', label='Robot'),
        mpatches.Patch(facecolor='royalblue', alpha=0.6, label='Planned waypoints'),
    ]
    ax.legend(handles=legend_handles, loc='upper right', fontsize=8,
              framealpha=0.85, edgecolor='black', fancybox=False)

    trail_x, trail_y = [], []
    TRAIL_LEN = 60

    def update(frame):
        s       = states[frame]
        t_xy    = s[0:2]
        t_angle = s[2]
        r_xy    = s[6:8]

        t_patch.set_xy(verts_world(t_xy, t_angle))
        robot_circle.set_center(r_xy)

        trail_x.append(r_xy[0]); trail_y.append(r_xy[1])
        if len(trail_x) > TRAIL_LEN:
            del trail_x[0]; del trail_y[0]
        trail_line.set_data(trail_x, trail_y)

        plan_idx = min(frame // STEPS_PER_REPLAN, len(plans) - 1)
        wpts = plans[plan_idx]
        plan_dots.set_data(wpts[:, 0], wpts[:, 1])
        plan_line.set_data(wpts[:, 0], wpts[:, 1])

        pos_err   = float(np.linalg.norm(t_xy - goal_xy))
        angle_err = float(abs(((t_angle - goal_angle + np.pi) % (2 * np.pi)) - np.pi))
        info_text.set_text(f"t = {frame * DT:.2f}s\n"
                           f"pos err = {pos_err:.3f} m\n"
                           f"angle err = {angle_err:.3f} rad")
        return t_patch, robot_circle, trail_line, plan_dots, plan_line, info_text

    anim = FuncAnimation(fig, update, frames=len(states),
                         interval=20, blit=True, repeat=True, repeat_delay=1500)

    plt.tight_layout()

    if args.save_gif:
        from pathlib import Path
        out = Path(__file__).parent.parent.parent / "figures" / f"push_t_2d_{args.method}.gif"
        out.parent.mkdir(parents=True, exist_ok=True)
        print(f"Saving gif to {out}…")
        anim.save(str(out), writer='pillow', fps=30, dpi=120)
        print("Saved.")
    else:
        plt.show()


if __name__ == "__main__":
    main()
