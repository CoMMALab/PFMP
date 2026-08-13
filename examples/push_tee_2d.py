"""
pushT.py — T-block pushing with PFMP in 2D (no MuJoCo).

A circular robot (green) pushes an orange T-shaped block toward a goal
pose (x, y, angle). Robot motion is governed by impedance-style PD control
toward waypoints produced by the PFMP planner.

Usage:
    python examples/pushT.py
    python examples/pushT.py --save-gif
"""

import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')

import argparse
from typing import NamedTuple

import jax
import jax.numpy as jnp
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle, FancyArrowPatch, Polygon

from pfmp import (PFMP, method_decomp, make_perturbation,
                  sigma_at as _sigma_at, knot_index as _knot_index)
from pfmp.algorithm import ps_update, mppi_update, cem_update
from pfmp.interpolate import hermite_spline


# ============================================================
# Configuration
# ============================================================

SEED = 42
DT = 0.01            # simulation timestep (s)
HORIZON = 2.5        # planner horizon (s)
NUM_KNOTS = 6         # spline knots
PARTICLES = 2048      # PFMP particles
ESS_THRESH = 0.25

N_REPLANS = 12      # closed-loop replan steps to precompute
STEPS_PER_REPLAN = 100  # sim steps between replans

# 2D workspace — same as pushT_bench
WORKSPACE_MIN = np.array([-0.5, -0.6])
WORKSPACE_MAX = np.array([ 0.7,  0.6])
T_BLOCK_MIN   = np.array([-0.3, -0.4])
T_BLOCK_MAX   = np.array([ 0.65,  0.5])

# T-block geometry — matches mjx_t_push.xml (MuJoCo half-extents)
#   stem: size="0.0225 0.0675", centered at body origin
#   bar:  size="0.0675 0.0225", pos="0 0.0675"
T_STEM_HALF = jnp.array([0.0225, 0.0675])
T_CAP_HALF  = jnp.array([0.0675, 0.0225])
T_STEM_CTR  = jnp.array([0.0,    0.0   ])
T_CAP_CTR   = jnp.array([0.0,    0.0675])

# Polygon outline of the T (bar bottom = 0.0675 - 0.0225 = 0.045)
_T_VERTS_LOCAL = np.array([
    [-0.0225, -0.0675],
    [ 0.0225, -0.0675],
    [ 0.0225,  0.0450],
    [ 0.0675,  0.0450],
    [ 0.0675,  0.0900],
    [-0.0675,  0.0900],
    [-0.0675,  0.0450],
    [-0.0225,  0.0450],
], dtype=np.float32)

# Physics
ROBOT_RADIUS  = 0.015
R_MASS        = 0.05
K_P           = 80.0    # robot position spring gain
K_D           = 8.0     # robot velocity damping gain
K_CONTACT     = 1200.0  # contact stiffness (N/m)
K_CONTACT_DAMP = 120.0  # contact damping
T_MASS        = 0.15
T_INERTIA     = 0.003   # moment of inertia (kg·m²)
T_LIN_DAMP    = 3.0     # T-block linear damping coefficient
T_ANG_DAMP    = 1.5     # T-block angular damping coefficient

# Planner
U_STDEV          = 0.05  # proposal position noise (m)
# Task cost weights. Fixed across every method: the cost defines the task, so
# tuning it per method would have each algorithm solving a different problem and
# would confound the proposal/update/resampling axes with four different
# objectives. Values are round numbers near the median of what independent
# per-method Optuna searches selected (pos 33.1, angle 2.64, approach 3.06),
# so they are reasonable for all methods rather than favouring any one.
POS_WEIGHT       = 30.0
ANGLE_WEIGHT     = 2.5
# Auxiliary robot->block shaping. Default was 0, but all four baseline searches
# independently moved it to 1.4-4.6: perturbation proposals have no
# object-directed sampling, so they lean on the cost for that guidance. Shared
# by every method (PFMP included), which keeps the proposal-axis comparison
# clean while giving the baselines the term they demonstrably want.
APPROACH_WEIGHT  = 3.0
MPPI_TEMPERATURE = 0.5

# DIAL-MPC
# Canonical dial-mpc's steady-state Ndiffuse (their config default; FR3 uses
# the same). Was 5, which matched neither the reference nor our own hardware
# path and cost 2.5x the rollouts per replan.
DIAL_N_INNER     = 2
DIAL_BETA1       = 1.0     # trajectory-level annealing temperature
DIAL_BETA2       = 1.0     # action-level annealing temperature
# dial-mpc's `temp_sample`: divides the std-normalised returns before the
# softmax. Their default is 0.06 (configs use 0.04-0.1); pfmp.dial_step
# defaults to 1.0 to preserve already-collected FR3 trials, so Push-T passes
# it explicitly. Tuned by pushT_2d_optuna.
DIAL_TEMP_SAMPLE = 0.06

# CEM
CEM_NUM_ELITES   = 10
# Lower bound on CEM's refitted per-knot covariance, i.e. hydrax's `sigma_min`,
# which the reference implementation takes as a required argument. Without it
# the elite std collapses (measured: 0.152 -> 0.0006 over 18 replans at 5
# elites), the proposal freezes, and CEM stops exploring for most of the
# episode. Tuned per-method by pushT_2d_optuna.
CEM_STDEV_FLOOR  = 0.01

# Success thresholds, read by geom_runner's push-T `measure`
SUCCESS_POS_THRESH   = 0.05   # metres
SUCCESS_ANGLE_THRESH = 0.2    # radians


# ============================================================
# Params
# ============================================================

class PushParams(NamedTuple):
    mean:        jax.Array   # (num_knots, 2) — robot target waypoints
    goal_xy:     jax.Array   # (2,)
    goal_angle:  float
    pos_weight:  float
    angle_weight: float
    # Sampling noise. Scalar for the fixed-noise methods (PS/MPPI/SIS/SIR); CEM
    # refits it per knot, giving shape (num_knots, 2), and DIAL overwrites it per
    # annealing level. Always read through `_sigma_at`, never directly.
    stdev:       jax.Array
    step_len:    float       # horizon / (num_knots - 1)
    obstacles:   jax.Array   # (N, 4) each row: [cx, cy, hx, hy]
    ws_min:      jax.Array   # (2,) robot outer clip min
    ws_max:      jax.Array   # (2,) robot outer clip max
    t_ws_min:    jax.Array   # (2,) T-block inner clip min
    t_ws_max:    jax.Array   # (2,) T-block inner clip max
    # Robot-to-T shaping cost (0 disables it). Callers should pass
    # APPROACH_WEIGHT explicitly; this default exists only so old constructions
    # keep working, and does NOT reflect the configured task cost.
    approach_weight: float    = 0.0


# ============================================================
# T-block SDF (JAX)
# ============================================================

def _sdf_box(p: jax.Array, half: jax.Array) -> jax.Array:
    d = jnp.abs(p) - half
    # Use sqrt with eps to avoid 0/0 gradient at exterior corners where d=(0,0)
    outside = jnp.sqrt(jnp.sum(jnp.maximum(d, 0.0) ** 2) + 1e-12)
    inside  = jnp.minimum(jnp.max(d), 0.0)
    return outside + inside


def t_sdf_local(p: jax.Array) -> jax.Array:
    return jnp.minimum(
        _sdf_box(p - T_STEM_CTR, T_STEM_HALF),
        _sdf_box(p - T_CAP_CTR,  T_CAP_HALF),
    )


def t_sdf_world(p_world: jax.Array, t_xy: jax.Array, t_angle: jax.Array) -> jax.Array:
    c, s = jnp.cos(t_angle), jnp.sin(t_angle)
    R_inv = jnp.array([[c, s], [-s, c]])   # world → local
    return t_sdf_local(R_inv @ (p_world - t_xy))


# ============================================================
# State layout
#   x = [t_x, t_y, t_angle, t_vx, t_vy, t_vangle, r_x, r_y, r_vx, r_vy]
# ============================================================

def dynamics(x: jax.Array, u: jax.Array, theta: PushParams, t: float):
    """
    Impedance-controlled robot + rigid-body T-block with contact physics.

    u: (2,) target waypoint for the robot end-effector.
    """
    t_xy    = x[0:2]
    t_angle = x[2]
    t_vel   = x[3:5]
    t_vang  = x[5]
    r_xy    = x[6:8]
    r_vel   = x[8:10]

    # Robot PD control toward waypoint
    r_acc = K_P * (u - r_xy) - K_D * r_vel

    # Contact: robot circle vs T-block
    sdf_val    = t_sdf_world(r_xy, t_xy, t_angle)
    penetration = jnp.maximum(0.0, ROBOT_RADIUS - sdf_val)

    # Outward normal of T at robot position (gradient of SDF points away from T)
    sdf_grad = jax.grad(lambda p: t_sdf_world(p, t_xy, t_angle))(r_xy)
    n = sdf_grad / (jnp.linalg.norm(sdf_grad) + 1e-7)

    # Contact force on T: in -n direction (robot pushes T away from itself).
    # Force on robot: in +n direction (T pushes robot back).
    f_contact = K_CONTACT * penetration
    f_on_t =  -f_contact * n
    f_on_r =   f_contact * n

    # T-block torque from robot contact
    contact_pt = r_xy - ROBOT_RADIUS * n   # point on robot surface facing T
    moment_arm = contact_pt - t_xy
    torque = moment_arm[0] * f_on_t[1] - moment_arm[1] * f_on_t[0]

    # Obstacle contacts — loop over all walls
    c, s = jnp.cos(t_angle), jnp.sin(t_angle)
    R = jnp.array([[c, -s], [s, c]])
    # Dense perimeter samples — 5 points per edge segment catches any wall thickness
    def _edge(a, b, n=5):
        return [a + (b - a) * i / (n - 1) for i in range(n)]
    import itertools
    _verts = [
        np.array([-0.0225, -0.0675]),
        np.array([ 0.0225, -0.0675]),
        np.array([ 0.0225,  0.0450]),
        np.array([ 0.0675,  0.0450]),
        np.array([ 0.0675,  0.0900]),
        np.array([-0.0675,  0.0900]),
        np.array([-0.0675,  0.0450]),
        np.array([-0.0225,  0.0450]),
    ]
    _pts = list(itertools.chain.from_iterable(
        _edge(_verts[i], _verts[(i + 1) % len(_verts)]) for i in range(len(_verts))
    ))
    t_surface_local = jnp.array(np.unique(np.round(_pts, 6), axis=0))
    t_surface_world = jax.vmap(lambda p: R @ p + t_xy)(t_surface_local)

    def wall_contacts(obs):
        obs_ctr, obs_half = obs[:2], obs[2:]
        # Robot vs wall — spring-damper
        r_sdf  = _sdf_box(r_xy - obs_ctr, obs_half)
        r_pen  = jnp.maximum(0.0, ROBOT_RADIUS - r_sdf)
        r_grad = jax.grad(lambda p: _sdf_box(p - obs_ctr, obs_half))(r_xy)
        r_n    = r_grad / (jnp.linalg.norm(r_grad) + 1e-7)
        r_vel_n = jnp.dot(r_vel, r_n)   # velocity along normal
        df_r   = r_pen * (K_CONTACT * r_n - K_CONTACT_DAMP * r_vel_n * r_n)
        # T-block vs wall — spring-damper per surface point
        t_pens  = jax.vmap(lambda p: jnp.maximum(0.0, -_sdf_box(p - obs_ctr, obs_half)))(t_surface_world)
        t_grads = jax.vmap(jax.grad(lambda p: _sdf_box(p - obs_ctr, obs_half)))(t_surface_world)
        t_norms = t_grads / (jnp.linalg.norm(t_grads, axis=-1, keepdims=True) + 1e-7)
        t_vel_ns = jnp.sum(t_vel[None] * t_norms, axis=-1)   # (N,) normal velocities
        t_forces = t_pens[:, None] * (K_CONTACT * t_norms - K_CONTACT_DAMP * t_vel_ns[:, None] * t_norms)
        df_t    = jnp.sum(t_forces, axis=0)
        arms    = t_surface_world - t_xy
        dtorque = jnp.sum(arms[:, 0] * t_forces[:, 1] - arms[:, 1] * t_forces[:, 0])
        return df_r, df_t, dtorque

    df_rs, df_ts, dtorques = jax.vmap(wall_contacts)(theta.obstacles)
    f_on_r = f_on_r + jnp.sum(df_rs, axis=0)
    f_on_t = f_on_t + jnp.sum(df_ts, axis=0)
    torque  = torque + jnp.sum(dtorques)

    # Integrate T-block (Euler + linear damping) — after all forces accumulated
    t_vel_new  = t_vel + (f_on_t / T_MASS) * DT
    t_vel_new  = t_vel_new * (1.0 - T_LIN_DAMP * DT)
    t_vang_new = t_vang + (torque / T_INERTIA) * DT
    t_vang_new = t_vang_new * (1.0 - T_ANG_DAMP * DT)
    t_xy_new   = t_xy + t_vel_new * DT
    t_ang_new  = t_angle + t_vang_new * DT

    # Integrate robot
    r_vel_new  = r_vel + (r_acc + f_on_r / R_MASS) * DT
    r_xy_new   = r_xy + r_vel_new * DT

    # Clip to workspace — robot has outer bounds, T clips tightly per rotation angle
    r_xy_new = jnp.clip(r_xy_new, theta.ws_min, theta.ws_max)
    _V = jnp.array(_T_VERTS_LOCAL)   # (8, 2)
    _c, _s = jnp.cos(t_ang_new), jnp.sin(t_ang_new)
    # World x of vertex v: t_xy_x + v_x*c - v_y*s
    right_ext = jnp.max( _V[:, 0] * _c - _V[:, 1] * _s)
    left_ext  = jnp.max(-_V[:, 0] * _c + _V[:, 1] * _s)
    top_ext   = jnp.max( _V[:, 0] * _s + _V[:, 1] * _c)
    bot_ext   = jnp.max(-_V[:, 0] * _s - _V[:, 1] * _c)
    t_clip_min = jnp.array([theta.t_ws_min[0] + left_ext,  theta.t_ws_min[1] + bot_ext])
    t_clip_max = jnp.array([theta.t_ws_max[0] - right_ext, theta.t_ws_max[1] - top_ext])
    t_xy_new = jnp.clip(t_xy_new, t_clip_min, t_clip_max)

    x_next = jnp.concatenate([
        t_xy_new, jnp.array([t_ang_new]),
        t_vel_new, jnp.array([t_vang_new]),
        r_xy_new, r_vel_new,
    ])
    return x_next, x_next


def _angle_wrap(a: jax.Array) -> jax.Array:
    return (a + jnp.pi) % (2.0 * jnp.pi) - jnp.pi


def running_cost(x: jax.Array, u: jax.Array, theta: PushParams, t: float) -> jax.Array:
    err_pos   = jnp.linalg.norm(x[0:2] - theta.goal_xy)
    err_angle = jnp.abs(_angle_wrap(x[2] - theta.goal_angle))
    aw = theta.angle_weight * jnp.clip(1.0 - err_pos / 0.25, 0.0, 1.0)
    # Approach: pull robot toward T, gated by T-to-goal distance (off when near goal)
    approach  = theta.approach_weight * jnp.linalg.norm(x[6:8] - x[0:2]) * jnp.clip(err_pos / 0.25, 0.0, 1.0)
    return theta.pos_weight * err_pos**2 + aw * err_angle**2 + approach


def terminal_cost(x: jax.Array, theta: PushParams) -> jax.Array:
    err_pos   = jnp.linalg.norm(x[0:2] - theta.goal_xy)
    err_angle = jnp.abs(_angle_wrap(x[2] - theta.goal_angle))
    aw = theta.angle_weight * jnp.clip(1.0 - err_pos / 0.25, 0.0, 1.0)
    return 10.0 * (theta.pos_weight * err_pos**2 + aw * err_angle**2)


PROPOSAL_MIX  = 0.5   # probability of sampling directed (vs random local)
STEP_LEN_VEL  = 0.05  # step size toward T per knot (m)
PUSH_OVERSHOOT = 1.0  # step to T centroid (no overshoot)


def init_control(key: jax.Array, x: jax.Array, theta: PushParams) -> jax.Array:
    key_mix, key_a, key_b = jax.random.split(key, 3)
    sigma = _sigma_at(theta, 0)
    r_xy = x[6:8]
    t_xy = x[0:2]
    direction = t_xy - r_xy
    dist      = jnp.linalg.norm(direction) + 1e-7
    directed  = r_xy + (direction / dist) * (jnp.minimum(dist, STEP_LEN_VEL) * PUSH_OVERSHOOT)
    directed  = directed + jax.random.normal(key_a, (2,)) * sigma
    random_loc = r_xy + jax.random.normal(key_b, (2,)) * sigma * 2.0
    return jnp.where(jax.random.uniform(key_mix) < PROPOSAL_MIX, directed, random_loc)


def proposal(key: jax.Array, x: jax.Array, _u: jax.Array,
             theta: PushParams, t: float) -> jax.Array:
    """Mixture: push through T centroid (50%) or random walk (50%)."""
    key_mix, key_a, key_b = jax.random.split(key, 3)
    sigma = _sigma_at(theta, _knot_index(theta, t))
    r_xy = x[6:8]
    t_xy = x[0:2]
    direction = t_xy - r_xy
    dist      = jnp.linalg.norm(direction) + 1e-7
    directed  = r_xy + (direction / dist) * (jnp.minimum(dist, STEP_LEN_VEL) * PUSH_OVERSHOOT)
    directed  = directed + jax.random.normal(key_a, (2,)) * sigma
    random_loc = r_xy + jax.random.normal(key_b, (2,)) * sigma * 2.0
    return jnp.where(jax.random.uniform(key_mix) < PROPOSAL_MIX, directed, random_loc)


# ============================================================
# Visualisation helpers
# ============================================================

def _t_verts_world(t_xy: np.ndarray, t_angle: float) -> np.ndarray:
    """Return T-polygon vertices in world frame."""
    c, s = np.cos(t_angle), np.sin(t_angle)
    R = np.array([[c, -s], [s, c]])
    return (_T_VERTS_LOCAL @ R.T) + t_xy


# ============================================================
# Main
# ============================================================

from pfmp import dial_step


def simulate(method: str, x0, theta, key, particles: int, n_replans: int = N_REPLANS):
    """Run closed-loop simulation, return (states, plans)."""
    # One decoder for the three axes, shared with geom_runner (pfmp.method_decomp).
    proposal_kind, update_kind, rho = method_decomp(method)
    informed = proposal_kind == "informed"
    perturb_init, perturb_proposal = make_perturbation()   # push-T clips in the dynamics

    ctrl = PFMP(
        dynamics, running_cost, terminal_cost,
        init_control if informed else perturb_init,
        proposal     if informed else perturb_proposal,
        horizon=HORIZON,
        num_knots=NUM_KNOTS,
        particles=particles,
        ess_resample_thresh=(ESS_THRESH if rho is None else rho),
        dt=DT,
    )

    def mean_from_state(x):
        """Linear interp from current robot pos to T-block pos."""
        r_xy   = x[6:8]
        t_xy   = x[0:2]
        alphas = jnp.linspace(0.0, 1.0, NUM_KNOTS)
        return (1.0 - alphas[:, None]) * r_xy + alphas[:, None] * t_xy

    # Only the non-resampling methods need a nominal plan to perturb around.
    if rho == 0.0:
        theta = theta._replace(mean=mean_from_state(x0))

    # CEM refits a per-knot covariance, so widen the scalar stdev to (K, 2)
    # first: cem_update only adapts when stdev is already per-knot, which is
    # what keeps it from clobbering the scalar the fixed-noise methods use.
    if update_kind == "topk" and jnp.ndim(theta.stdev) == 0:
        theta = theta._replace(stdev=jnp.full((NUM_KNOTS, 2), theta.stdev))

    print("JIT warm-up…", end=" ", flush=True)
    key, subkey = jax.random.split(key)
    _ = ctrl.run_sis(subkey, x0, theta)
    _ = ctrl.run_sis(subkey, x0, theta)
    print("done")

    print(f"Simulating {n_replans} replan steps × {STEPS_PER_REPLAN} sim steps…")
    goal_xy    = np.array(theta.goal_xy)
    goal_angle = float(theta.goal_angle)

    states = [np.array(x0)]
    plans  = []
    x  = x0
    tk = ctrl.tk()

    for step_i in range(n_replans):
        key, subkey = jax.random.split(key)

        if update_kind == "diffuse":
            # DIAL owns its re-sampling: one fresh batch per annealing level,
            # via the shared pfmp.dial_step (see pfmp/update.py).
            theta = dial_step(subkey, ctrl, x, theta, n=DIAL_N_INNER,
                              beta1=DIAL_BETA1, beta2=DIAL_BETA2,
                              temp_sample=DIAL_TEMP_SAMPLE)
        else:
            trajs, costs = ctrl.run_sis(subkey, x, theta)
            if update_kind == "mean":
                temperature = MPPI_TEMPERATURE  # fixed inverse-temperature (canonical MPPI), not adaptively scaled by cost std
                theta, _ = mppi_update(theta, trajs, costs, max(temperature, 1e-6))
            elif update_kind == "topk":
                # cem_update takes (P, K) per-knot costs and sums internally
                theta, _ = cem_update(theta, trajs, costs,
                                      elite_frac=CEM_NUM_ELITES / particles,
                                      stdev_floor=CEM_STDEV_FLOOR)
            else:
                theta, _ = ps_update(theta, trajs, costs)

        plans.append(np.array(theta.mean))

        tq = jnp.arange(STEPS_PER_REPLAN) * DT
        us = hermite_spline(tq, tk, theta.mean)
        def phys_step(x, args):
            u, t = args
            xn, _ = dynamics(x, u, theta, t)
            return xn, xn
        x, xs = jax.lax.scan(phys_step, x, (us, tq))
        states.extend(np.asarray(xs))

        new_tk = tk + STEPS_PER_REPLAN * DT
        theta  = theta._replace(mean=hermite_spline(new_tk, tk, theta.mean))

        if (step_i + 1) % 20 == 0:
            s = states[-1]
            pos_err   = float(np.linalg.norm(s[0:2] - goal_xy))
            angle_err = float(abs(((s[2] - goal_angle + np.pi) % (2 * np.pi)) - np.pi))
            print(f"  step {step_i+1:4d} | pos_err={pos_err:.3f}m  angle_err={angle_err:.3f}rad")

    print(f"Simulation complete — {len(states)} states recorded.")
    return states, plans


def _obs_sdf_np(pt, obs):
    """2D point-to-AABB signed distance (numpy). obs = [cx, cy, hx, hy]."""
    cx, cy, hx, hy = obs
    dx = max(abs(pt[0] - cx) - hx, 0.0)
    dy = max(abs(pt[1] - cy) - hy, 0.0)
    return np.sqrt(dx**2 + dy**2)


def _is_clear(pt, obstacles, margin):
    return all(_obs_sdf_np(pt, obs) >= margin for obs in obstacles)


T_WIDTH = float(2 * 0.0675)   # 0.135 m -- full width of the T cap
T_MARGIN = 0.12               # clearance for the T-block centre


def _random_obstacles_arena(rng, n_obs, half, size_lo, size_hi,
                            clear_pts=(), margin=T_MARGIN, max_tries=200):
    """Generate exactly n_obs obstacles via per-obstacle rejection sampling."""
    pad = size_hi + 0.01
    obs = []
    for _ in range(n_obs):
        for _ in range(max_tries):
            hx = rng.uniform(size_lo, size_hi)
            hy = rng.uniform(size_lo, size_hi)
            cx = rng.uniform(-half + pad, half - pad)
            cy = rng.uniform(-half + pad, half - pad)
            candidate = [cx, cy, hx, hy]
            if all(_is_clear(pt, [candidate], margin) for pt in clear_pts):
                obs.append(candidate)
                break
        else:
            raise RuntimeError(f"Could not place obstacle {len(obs)+1}/{n_obs} — try smaller n_obs")
    return np.array(obs)


def _build_scenario(seed: int, distance: float, n_obs_override: int | None = None):
    """Return (x0, theta, ws_min, ws_max, n_replans) for a given seed/distance.

    `distance` sizes the square arena in T-widths and sets the obstacle count
    (`int(distance) - 3`) and the replan budget. The corridor and fixed-
    workspace layouts this used to support were retired with the legacy sweep.
    """
    rng = np.random.default_rng(seed)

    half       = distance * T_WIDTH / 2.0
    t_xy_0     = np.array([-half + T_MARGIN, -half + T_MARGIN])
    t_angle_0  = rng.uniform(-np.pi, np.pi)
    goal_xy    = np.array([ half - T_MARGIN,  half - T_MARGIN])
    goal_angle = rng.uniform(-np.pi, np.pi)
    n_obs      = max(0, int(distance) - 3) if n_obs_override is None else n_obs_override
    if n_obs > 0:
        obstacles = _random_obstacles_arena(rng, n_obs, half,
                                            size_lo=T_WIDTH * 0.3, size_hi=T_WIDTH * 0.8,
                                            clear_pts=(t_xy_0, goal_xy))
    else:
        obstacles = np.array([[1e6, 1e6, 0.001, 0.001]])  # inactive sentinel
    ws_min   = np.array([-half - 0.05, -half - 0.05])  # robot outer
    ws_max   = np.array([ half + 0.05,  half + 0.05])
    t_ws_min = np.array([-half, -half])                # T inner (actual wall)
    t_ws_max = np.array([ half,  half])

    r_xy_0 = np.clip(t_xy_0 - np.array([0.12, 0.0]),
                     ws_min + 0.05, ws_max - 0.05)
    x0 = jnp.array([
        t_xy_0[0], t_xy_0[1], t_angle_0,
        0.0, 0.0, 0.0,
        r_xy_0[0], r_xy_0[1],
        0.0, 0.0,
    ])
    theta = PushParams(
        mean=jnp.zeros((NUM_KNOTS, 2)),
        goal_xy=jnp.array(goal_xy),
        goal_angle=float(goal_angle),
        pos_weight=POS_WEIGHT,
        angle_weight=ANGLE_WEIGHT,
        approach_weight=APPROACH_WEIGHT,
        stdev=U_STDEV,
        step_len=HORIZON / (NUM_KNOTS - 1),
        obstacles=jnp.array(obstacles),
        ws_min=jnp.array(ws_min),
        ws_max=jnp.array(ws_max),
        t_ws_min=jnp.array(t_ws_min),
        t_ws_max=jnp.array(t_ws_max),
    )
    n_replans = int(N_REPLANS * max(1.0, distance))
    return x0, theta, ws_min, ws_max, n_replans


def _is_success(final_state: np.ndarray, goal_xy, goal_angle: float) -> bool:
    pos_err   = float(np.linalg.norm(final_state[0:2] - goal_xy))
    angle_err = float(abs(((final_state[2] - goal_angle + np.pi) % (2 * np.pi)) - np.pi))
    return pos_err < SUCCESS_POS_THRESH and angle_err < SUCCESS_ANGLE_THRESH


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save-gif', action='store_true')
    parser.add_argument('--seed', type=int, default=SEED)
    parser.add_argument('--particles', type=int, default=PARTICLES)
    parser.add_argument('--method', choices=['pfmp', 'mppi', 'biased_mppi', 'biased_cem', 'dial', 'ps', 'cem'], default='pfmp')
    parser.add_argument('--n-obs', type=int, default=None,
                        help='Obstacle count (default: int(distance) - 3)')
    parser.add_argument('--distance', type=float, default=4.0,
                        help='Arena size as a multiple of T-width; T starts bottom-left, goal top-right.')
    args = parser.parse_args()


    # ── Interactive viewer ───────────────────────────────────────
    x0, theta, ws_min, ws_max, n_replans = _build_scenario(
        args.seed, args.distance, args.n_obs)

    key = jax.random.key(args.seed)
    states, plans = simulate(args.method, x0, theta, key, args.particles, n_replans)
    goal_xy    = np.array(theta.goal_xy)
    goal_angle = float(theta.goal_angle)
    obstacles  = np.array(theta.obstacles)
    t_xy_0     = np.array(x0[0:2])
    t_angle_0  = float(x0[2])
    r_xy_0     = np.array(x0[6:8])

    # ── Animation ────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 7))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('#f5f5f5')
    ax.set_xlim(ws_min[0], ws_max[0])
    ax.set_ylim(ws_min[1], ws_max[1])
    ax.set_aspect('equal')
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor('black'); spine.set_linewidth(1.5)
    ax.set_title(f"{args.method.upper()}: T-block Pushing "
                 f"({args.distance:.1f}x T-width arena)", fontsize=13)

    # Outer robot margin (solid, subtle)
    ax.add_patch(mpatches.FancyBboxPatch(
        ws_min, ws_max[0] - ws_min[0], ws_max[1] - ws_min[1],
        boxstyle="square,pad=0", fill=True,
        facecolor='#e8e8e8', edgecolor='#bbbbbb', linewidth=1.0, zorder=0,
    ))
    # Inner arena wall (dotted) — ws_min/max with the 0.05 margin removed
    inner_min = ws_min + 0.05
    inner_max = ws_max - 0.05
    ax.add_patch(mpatches.FancyBboxPatch(
        inner_min, inner_max[0] - inner_min[0], inner_max[1] - inner_min[1],
        boxstyle="square,pad=0", fill=True,
        facecolor='#f5f5f5', edgecolor='#888888', linewidth=1.2, linestyle=':', zorder=0,
    ))

    # Obstacles
    for obs in obstacles:
        cx, cy, hx, hy = obs
        ax.add_patch(mpatches.Rectangle(
            (cx - hx, cy - hy), 2 * hx, 2 * hy,
            facecolor='dimgray', edgecolor='black', linewidth=1.5, zorder=3,
        ))

    # Goal T (dashed, semi-transparent)
    goal_verts = _t_verts_world(goal_xy, goal_angle)
    goal_patch = Polygon(goal_verts, closed=True,
                         facecolor='darkorange', alpha=0.25,
                         edgecolor='darkorange', linewidth=1.5,
                         linestyle='--', zorder=1)
    ax.add_patch(goal_patch)
    ax.plot(*goal_xy, marker='*', color='darkorange', markersize=12,
            markeredgecolor='black', markeredgewidth=0.8, zorder=2)

    # T-block (orange)
    t_patch = Polygon(_t_verts_world(t_xy_0, t_angle_0), closed=True,
                      facecolor='darkorange', edgecolor='saddlebrown',
                      linewidth=1.5, zorder=5)
    ax.add_patch(t_patch)

    # Robot (green circle)
    robot_circle = Circle(r_xy_0, ROBOT_RADIUS,
                          facecolor='limegreen', edgecolor='darkgreen',
                          linewidth=1.5, zorder=8)
    ax.add_patch(robot_circle)

    # Robot trail
    trail_line, = ax.plot([], [], color='limegreen', alpha=0.4,
                          linewidth=1.2, zorder=4)

    # Planned waypoints (robot target path)
    plan_dots, = ax.plot([], [], 'o', color='royalblue', markersize=4,
                         alpha=0.8, zorder=7)
    plan_line, = ax.plot([], [], color='royalblue', alpha=0.4,
                         linewidth=1.0, zorder=6)

    # Info text
    info_text = ax.text(
        0.02, 0.98, '', transform=ax.transAxes,
        fontsize=9, verticalalignment='top',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7),
    )

    # Legend
    legend_handles = [
        mpatches.Patch(facecolor='darkorange', edgecolor='saddlebrown',
                       label='T-block'),
        mpatches.Patch(facecolor='darkorange', alpha=0.25, linestyle='--',
                       edgecolor='darkorange', label='Goal pose'),
        mpatches.Patch(facecolor='limegreen', edgecolor='darkgreen',
                       label='Robot'),
        mpatches.Patch(facecolor='royalblue', alpha=0.6,
                       label='Planned waypoints'),
    ]
    ax.legend(handles=legend_handles, loc='upper right', fontsize=8,
              framealpha=0.85, edgecolor='black', fancybox=False)

    # Trail buffer
    trail_x, trail_y = [], []
    TRAIL_LEN = 60  # steps

    def update(frame):
        # Each animation frame = one sim step
        # Plan refreshes once per STEPS_PER_REPLAN steps
        s = states[frame]
        t_xy    = s[0:2]
        t_angle = s[2]
        r_xy    = s[6:8]

        # Update T-block polygon
        t_patch.set_xy(_t_verts_world(t_xy, t_angle))

        # Update robot circle
        robot_circle.set_center(r_xy)

        # Update robot trail
        trail_x.append(r_xy[0])
        trail_y.append(r_xy[1])
        if len(trail_x) > TRAIL_LEN:
            del trail_x[0]; del trail_y[0]
        trail_line.set_data(trail_x, trail_y)

        # Update planned waypoints (use latest available plan)
        plan_idx = min(frame // STEPS_PER_REPLAN, len(plans) - 1)
        wpts = plans[plan_idx]
        plan_dots.set_data(wpts[:, 0], wpts[:, 1])
        plan_line.set_data(wpts[:, 0], wpts[:, 1])

        # Info text
        pos_err   = float(np.linalg.norm(t_xy - goal_xy))
        angle_err = float(abs(((t_angle - goal_angle + np.pi) % (2 * np.pi)) - np.pi))
        t_sim = frame * DT
        info_text.set_text(
            f"t = {t_sim:.2f}s\n"
            f"pos err = {pos_err:.3f} m\n"
            f"angle err = {angle_err:.3f} rad"
        )

        return t_patch, robot_circle, trail_line, plan_dots, plan_line, info_text

    n_frames = len(states)
    anim = FuncAnimation(fig, update, frames=n_frames,
                         interval=20, blit=True, repeat=True, repeat_delay=1500)

    plt.tight_layout()

    if args.save_gif:
        from pathlib import Path
        suffix = f"dist{args.distance:.0f}_s{args.seed}"
        out = Path(__file__).parent.parent / "figures" / f"pushT_{args.method}_{suffix}.gif"
        out.parent.mkdir(parents=True, exist_ok=True)
        print(f"Saving gif to {out}…")
        anim.save(str(out), writer='pillow', fps=30, dpi=120)
        print("Saved.")
    else:
        plt.show()


if __name__ == "__main__":
    main()
