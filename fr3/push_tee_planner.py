"""
T-block pushing task using PFMP with Panda robot.

Real world workspace is 53' 1/8" x 23' 1/4"
"""

import argparse
import datetime
import json
import os
import time
from typing import NamedTuple, Tuple

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from mujoco import mjx

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from geom_tee import STEM_HALF, CAP_HALF, CAP_CTR

import pfmp

from ik_cache import (
    IKCache, ik_from_cache, get_or_build_cache,
    WORKSPACE_XY_MIN, WORKSPACE_XY_MAX, IK_CACHE_RESOLUTION, EE_HEIGHT,
)

# ===============================
# Configuration
# ===============================

SCENE_XML_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "mjx_t_push.xml")

# T-block qpos offset in the MuJoCo state
T_QPOS_OFFSET = 7


# ===============================
# Task Parameters
# ===============================


class PushParams(NamedTuple):
    """Parameters for the T-block pushing task."""
    mean: jax.Array
    mjx_model: mjx.Model
    ik_cache: IKCache
    goal_xy: jax.Array
    goal_angle: jax.Array
    pos_weight: float
    angle_weight: float
    u_stdev: float
    gripper_site_id: int
    approach_weight: float = 0.0


# ===============================
# Helper Functions
# ===============================


@jax.jit
def angle_wrap(a: jax.Array) -> jax.Array:
    return (a + jnp.pi) % (2.0 * jnp.pi) - jnp.pi


@jax.jit
def compute_error(th: PushParams, t_qpos: jax.Array) -> jax.Array:
    """Compute [x_err, y_err, yaw_err] for T-block. t_qpos = [x, y, yaw]."""
    return jnp.array([
        t_qpos[0] - th.goal_xy[0],
        t_qpos[1] - th.goal_xy[1],
        angle_wrap(t_qpos[2] - th.goal_angle),
    ])


# ===============================
# PFMP Task Functions
# ===============================


HORIZON   = 8.0
NUM_KNOTS = 5
PARTICLES = 2048

# 2s knots at dt=0.02 (mjx_scene.xml) = 100 substeps per knot. Reach saturates
# around 1s — the arm covers ~12cm in 0.5s and ~28cm by 1.0s, against a proposal
# whose p90 step is 23cm — so the extra second is not for travel but to let the
# arm arrive and settle before the knot ends, and to give contact enough substeps
# to resolve.
KNOT_INTERVAL = HORIZON / (NUM_KNOTS - 1)
# Replan once per knot, matching run_realtime, which sends theta.mean[1] and then
# replans. Derived rather than written out so it cannot drift from the horizon: a
# longer replan period executes the tail of a plan open-loop, which the robot
# never does, and lets model error compound uncorrected.
RUN_INTERACTIVE_FREQUENCY = 1.0 / KNOT_INTERVAL

VIZ_INTERP_STEPS = 6        # sub-frames to glide the sim arm between replan poses
VIZ_INTERP_DT    = 0.01     # seconds per sub-frame (~60ms glide at defaults)

SUCCESS_POS_RADIUS = 0.05   # metres
SUCCESS_ANGLE_TOL  = 0.35   # radians (~20°)
TRIAL_TIMEOUT_S    = 300.0  # abandon a trial after 5 min, logged as timeout

# OptiTrack loses the tee when the arm occludes it, and there are dead zones the
# cameras never cover. Planning continues on the last known pose, so warn when
# that pose goes stale — a silently stale tee looks identical to a tracked one.
#
# Counted in consecutive missed detections, not seconds: the replan loop runs at
# ~1Hz (2000 rollouts x multiple knots), so a wall-clock threshold below the loop
# period would fire on every single miss and every duration would be inflated by
# a full replan cycle. One miss is one lost look at the tee, whatever it cost.
TRACK_STALE_MISSES = 2    # consecutive missed detections before warning

PROPOSAL_MIX    = 0.4   # directed toward tee
ORBITAL_MIX     = 0.2   # orbit around tee at push radius
STEP_LEN_VEL    = 0.05
PUSH_OVERSHOOT  = 1.0
PUSH_RADIUS     = 0.16  # just outside tee cap half-extent
ORBITAL_Y_OFFSET = 0.04  # metres up the tee stem for the orbital centre


# Franky drives the arm with Ruckig, which ramps the *joint* reference under
# velocity/acceleration/jerk limits rather than jumping to the target. Commanding
# the IK solution directly instead leaves an overdamped PD chasing a target tens
# of cm away: a 7cm move was still 16mm short after 400 steps, so every rollout
# scored pushes from an arm that never arrived. Ramping the reference at the same
# rate the robot uses keeps the tracking error small, as it is on hardware.
#
# Velocity-limited ramp only — Ruckig also shapes acceleration and jerk, which
# matters for the first few steps after a direction change but not for where the
# arm ends up.
FR3_DQ_MAX     = jnp.array([2.62, 2.62, 2.62, 2.62, 5.26, 4.18, 5.26])   # rad/s
VELOCITY_SCALE = 0.2   # franky relative_dynamics_factor, see push_tee_controller


def dynamics(x: mjx.Data, u: jax.Array, th: PushParams, t: float) -> Tuple[mjx.Data, jax.Array]:
    """Step MuJoCo, ramping the joint reference toward the waypoint's IK solution."""
    dt      = th.mjx_model.opt.timestep
    dq_max  = FR3_DQ_MAX * VELOCITY_SCALE * dt

    target_qpos = ik_from_cache(u, th.ik_cache)
    delta = target_qpos - x.ctrl[:7]
    # Scale the whole delta by one factor rather than clipping per joint, so the
    # joints stay synchronised and the reference keeps its direction. Clipping
    # each joint separately lets the short-travel ones arrive early while others
    # saturate, which bends the path and shows up as twitching mid-motion.
    scale = jnp.minimum(1.0, jnp.min(dq_max / (jnp.abs(delta) + 1e-9)))
    step  = delta * scale

    # No acceleration limit: the profile is rectangular (full speed on the first
    # substep, full stop on arrival) where Ruckig would ramp. Limiting it needs the
    # *reference* velocity as state, which dynamics does not carry — using the
    # arm's own qvel as a stand-in throttles the reference against the arm it is
    # supposed to lead, and the ramp never reaches v_max.
    ctrl = x.ctrl.at[:7].set(x.ctrl[:7] + step)
    x_next = mjx.step(th.mjx_model, x.replace(ctrl=ctrl))
    return x_next, u


TEE_ANNULUS_CENTER = (0.25 + 0.78) / 2   # ideal radial distance from base
TEE_ANNULUS_WEIGHT = 0.3

def running_cost(x: mjx.Data, u: jax.Array, th: PushParams, t: float) -> jax.Array:
    t_qpos    = x.qpos[T_QPOS_OFFSET:T_QPOS_OFFSET + 3]
    t_xy      = t_qpos[:2]
    error     = compute_error(th, t_qpos)
    err_pos   = jnp.linalg.norm(error[:2])
    err_angle = jnp.abs(error[2])
    aw        = th.angle_weight   # ungated: orientation penalty always on
    radial    = jnp.linalg.norm(t_xy) - TEE_ANNULUS_CENTER
    approach  = th.approach_weight * jnp.linalg.norm(u - t_xy) * jnp.clip(err_pos / 0.25, 0.0, 1.0)
    return th.pos_weight * err_pos ** 2 + aw * err_angle ** 2 + TEE_ANNULUS_WEIGHT * radial ** 2 + approach


def terminal_cost(x: mjx.Data, th: PushParams) -> jax.Array:
    t_qpos    = x.qpos[T_QPOS_OFFSET:T_QPOS_OFFSET + 3]
    error     = compute_error(th, t_qpos)
    err_pos   = jnp.linalg.norm(error[:2])
    err_angle = jnp.abs(error[2])
    aw = th.angle_weight   # ungated: orientation penalty always on
    return 10.0 * (th.pos_weight * err_pos + aw * err_angle)


def init_control(key: jax.Array, x: mjx.Data, th: PushParams) -> jax.Array:
    """Same mixture as `proposal`, seeded from the warm-started mean.

    The orbital branch is anchored to the tee rather than the mean, so particles
    stay feasible even if the mean has drifted outside the reachable set.
    """
    return _project_reachable(_sample_mixture(key, x, th.mean[0], th), th)


def mppi_proposal(key: jax.Array, _x: mjx.Data, _u: jax.Array, th: PushParams, t: float) -> jax.Array:
    """Gaussian proposal centred on the mean knot at time t (for PS/MPPI baselines).

    Snapped to the reachable mask, same as `proposal`. This is a feasibility
    constraint, not a task-informed bias, so every method gets it.
    """
    step_len = HORIZON / (NUM_KNOTS - 1)
    idx = jnp.clip(jnp.floor(t / step_len).astype(jnp.int32) + 1, 0, NUM_KNOTS - 1)
    return _project_reachable(
        th.mean[idx] + jax.random.normal(key, (2,)) * th.u_stdev, th)


def _project_reachable(u: jax.Array, th: PushParams) -> jax.Array:
    """Snap an EE waypoint to the nearest reachable grid cell.

    Projecting into a sector *box* is not enough: the true reachable set is not a
    box (the mask has holes and a curved boundary), so a box-projected control can
    still land on an unreachable cell, and ik_from_cache then returns NaN. Snapping
    to the nearest cell that is actually in the mask guarantees a reachable corner
    for the bilinear lookup, so no control can produce NaN regardless of where the
    sample fell.
    """
    cache = th.ik_cache
    Nx, Ny = cache.reachable.shape
    xs = jnp.linspace(cache.xy_min[0], cache.xy_max[0], Nx)
    ys = jnp.linspace(cache.xy_min[1], cache.xy_max[1], Ny)
    X, Y = jnp.meshgrid(xs, ys, indexing="ij")            # (Nx, Ny)
    d2 = (X - u[0]) ** 2 + (Y - u[1]) ** 2
    d2 = jnp.where(cache.reachable > 0.5, d2, jnp.inf)     # only reachable cells
    flat = jnp.argmin(d2)
    return jnp.array([X.reshape(-1)[flat], Y.reshape(-1)[flat]])


def _sample_mixture(key: jax.Array, x: mjx.Data, u: jax.Array, th: PushParams) -> jax.Array:
    """One draw from the three-way mixture: directed / orbital / random walk."""
    key_mix, key_a, key_b, key_ang = jax.random.split(key, 4)
    t_xy  = x.qpos[T_QPOS_OFFSET:T_QPOS_OFFSET + 2]
    t_yaw = x.qpos[T_QPOS_OFFSET + 2]
    r     = jax.random.uniform(key_mix)

    # Branch 1: step toward tee centroid
    direction = t_xy - u
    dist      = jnp.linalg.norm(direction) + 1e-7
    directed  = u + (direction / dist) * (jnp.minimum(dist, STEP_LEN_VEL) * PUSH_OVERSHOOT)
    directed  = directed + jax.random.normal(key_a, (2,)) * th.u_stdev

    # Branch 2: orbit around a point shifted up the tee stem (body-frame +y).
    stem_up     = jnp.array([-jnp.sin(t_yaw), jnp.cos(t_yaw)])
    orbital_ctr = t_xy + ORBITAL_Y_OFFSET * stem_up
    angle       = jax.random.uniform(key_ang, minval=0.0, maxval=2.0 * jnp.pi)
    orbital     = orbital_ctr + PUSH_RADIUS * jnp.array([jnp.cos(angle), jnp.sin(angle)])

    # Branch 3: random walk
    random_loc = u + jax.random.normal(key_b, (2,)) * th.u_stdev * 2.0

    return jnp.where(r < PROPOSAL_MIX, directed,
           jnp.where(r < PROPOSAL_MIX + ORBITAL_MIX, orbital, random_loc))


def proposal(key: jax.Array, x: mjx.Data, u: jax.Array, th: PushParams, t: float) -> jax.Array:
    """Three-way mixture, resampled until the draw lands in the reachable mask, so
    the IK-resolved joint motion tracks the intended EE path instead of arcing
    across a wrist-flip / unreachable cell."""
    return _project_reachable(_sample_mixture(key, x, u, th), th)


# ===============================
# Visualization Helpers
# ===============================


def add_goal_T(
    user_scn,
    goal_xy: np.ndarray,
    yaw_rad: float,
    rgba=(0., 0., 0., 0.25),
    z_shadow=1e-3,
    thickness=5e-4
):
    """Draw a translucent T-shaped goal marker."""
    if user_scn.ngeom + 2 > user_scn.maxgeom:
        return

    stem_half = np.array([STEM_HALF[0], STEM_HALF[1], thickness], dtype=float)
    bar_half  = np.array([CAP_HALF[0],  CAP_HALF[1],  thickness], dtype=float)
    bar_off_y = float(CAP_CTR[1])

    x, y = np.asarray(goal_xy, dtype=float)
    pos_world = np.array([x, y, float(z_shadow)], dtype=float)

    c, s = float(np.cos(yaw_rad)), float(np.sin(yaw_rad))
    R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=float)

    geom = user_scn.geoms[user_scn.ngeom]
    mujoco.mjv_initGeom(
        geom, mujoco.mjtGeom.mjGEOM_BOX,
        size=stem_half, pos=pos_world, mat=R.flatten(), rgba=rgba
    )
    user_scn.ngeom += 1

    bar_center = pos_world + R @ np.array([0.0, bar_off_y, 0.0], dtype=float)
    geom = user_scn.geoms[user_scn.ngeom]
    mujoco.mjv_initGeom(
        geom, mujoco.mjtGeom.mjGEOM_BOX,
        size=bar_half, pos=bar_center, mat=R.flatten(), rgba=rgba
    )
    user_scn.ngeom += 1


# ===============================
# Main
# ===============================


def load_scene(xml_path: str):
    """Load MuJoCo model and data."""
    mj_model = mujoco.MjModel.from_xml_path(xml_path)
    mj_data = mujoco.MjData(mj_model)

    key_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_KEY, "home")
    if key_id >= 0:
        mujoco.mj_resetDataKeyframe(mj_model, mj_data, key_id)

    print(f"Loading model from: {xml_path}")
    cyl_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, "push_cylinder_visual")
    print(f"push_cylinder_visual geom id: {cyl_id} (−1 = not found)")
    return mj_model, mj_data

#################################################################

def run_realtime(ctrl, theta, mj_model, mj_data, ik_cache,
                 optitrack_client, robot_planner, method="sir", temperature=None,
                 log_path=None, trial=0, seed=0, spec_init=None):
    """Closed-loop realtime push: Optitrack → PFMP → robot controller."""
    from push_tee_optitrack import tee_pose
    from pub_sub.protocol import Command
    import mujoco.viewer as mjviewer

    key = jax.random.key(0)
    gripper_id = theta.gripper_site_id

    def goal_marker_callback(user_scn, th):
        user_scn.ngeom = 0
        add_goal_T(user_scn, np.array(th.goal_xy), float(th.goal_angle),
                   rgba=(0.2, 0.8, 0.2, 0.5))
        knots = np.array(th.mean)
        z = float(EE_HEIGHT)
        for k in range(len(knots) - 1):
            if user_scn.ngeom >= user_scn.maxgeom:
                break
            p0 = np.array([knots[k,0], knots[k,1], z], dtype=np.float64)
            p1 = np.array([knots[k+1,0], knots[k+1,1], z], dtype=np.float64)
            geom = user_scn.geoms[user_scn.ngeom]
            mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_LINE, 2.0, p0, p1)
            geom.rgba[:] = [1.0, 0.5, 0.0, 1.0]
            user_scn.ngeom += 1

    from pfmp.algorithm import ps_update, mppi_update
    if method == "mppi":
        replan = lambda key, data, th: ctrl.run_sis(key, data, th)
        def _mppi_update(th, tr, co):
            import jax.numpy as jnp
            finite = jnp.isfinite(jnp.sum(co, axis=1))
            co = jnp.where(finite[:, None], co, jnp.zeros_like(co))
            return mppi_update(th, tr.transpose(1, 0, 2), co.T, max(float(temperature), 1e-6))
        update = _mppi_update
    elif method == "ps":
        replan = lambda key, data, th: ctrl.run_sis(key, data, th)
        update = ps_update
    elif method == "cem":
        from pfmp.algorithm import cem_update
        num_elites = max(1, PARTICLES // 10)
        replan = lambda key, data, th: ctrl.run_sis(key, data, th)
        update = lambda th, tr, co: cem_update(th, tr, jnp.sum(co, axis=1),
                                               elite_frac=num_elites / PARTICLES)
    elif method == "dial":
        from pfmp import dial_sigma
        replan = lambda key, data, th: ctrl.run_sis(key, data, th)
        def _dial_update(th, tr, co):
            theta_out = th
            for i in range(5, 0, -1):
                sigma = float(np.mean([dial_sigma(i, h, th.u_stdev, 5, NUM_KNOTS, 1.0, 1.0)
                                       for h in range(NUM_KNOTS)]))
                total = jnp.sum(co, axis=1)
                temp = float(jnp.maximum(jnp.std(total), 1e-6))
                theta_out, _ = mppi_update(theta_out, tr.transpose(1, 0, 2), co.T, temp)
            return theta_out, jnp.zeros(PARTICLES)
        update = _dial_update
    else:  # sir
        replan = lambda key, data, th: ctrl.run_sir(key, data, th)
        update = ps_update

    print("JIT warm-up…", end=" ", flush=True)
    mjx_data = mjx.put_data(mj_model, mj_data)
    _ = replan(key, mjx_data, theta)
    print("done")

    # Per-trial log record. Written once when the trial ends (goal reached, or
    # the viewer is closed to abort). `n_cmd_failures` > 0 flags a run
    # contaminated by non-deterministic controller/robot aborts.
    t_start = time.time()
    # `spec_*` is what tee_problem_gen prescribed for this (seed, trial);
    # `realized_*` is what OptiTrack actually saw. They differ by however
    # accurately the tee was placed by hand, so placement slop is recorded as
    # data rather than assumed away.
    rec = {
        "ts": datetime.datetime.now().isoformat(timespec="seconds"),
        "trial": int(trial), "seed": int(seed), "method": method,
        "spec_init": ([float(spec_init[0]), float(spec_init[1]), float(spec_init[2])]
                      if spec_init is not None else None),
        "spec_goal": [float(theta.goal_xy[0]), float(theta.goal_xy[1]),
                      float(theta.goal_angle)],
        "realized_init": [float(mj_data.qpos[T_QPOS_OFFSET]),
                          float(mj_data.qpos[T_QPOS_OFFSET + 1]),
                          float(mj_data.qpos[T_QPOS_OFFSET + 2])],
        "realized_final": None,
        "placement_err": None,
        "n_replans": 0, "n_cmd_failures": 0, "success": False, "timeout": False,
        # Tracking health: replans driven by a stale tee pose, and the longest
        # single dropout. Both flag contamination the same way n_cmd_failures does.
        "n_track_dropouts": 0, "n_stale_replans": 0, "max_consec_misses": 0,
        "final_pos_err": None, "final_angle_err": None, "duration_s": None,
    }
    if spec_init is not None:
        rec["placement_err"] = round(float(np.linalg.norm(
            np.array(rec["realized_init"][:2]) - np.asarray(spec_init[:2]))), 4)

    def _write_record():
        rec["duration_s"] = round(time.time() - t_start, 2)
        rec["realized_final"] = [round(float(mj_data.qpos[T_QPOS_OFFSET]), 4),
                                 round(float(mj_data.qpos[T_QPOS_OFFSET + 1]), 4),
                                 round(float(mj_data.qpos[T_QPOS_OFFSET + 2]), 4)]
        if log_path:
            with open(log_path, "a") as f:
                f.write(json.dumps(rec) + "\n")
        print(f"\n[log] trial {rec['trial']} method={rec['method']} success={rec['success']} "
              f"pos_err={rec['final_pos_err']} angle_err={rec['final_angle_err']} "
              f"placement_err={rec['placement_err']} "
              f"aborts={rec['n_cmd_failures']} "
              f"dropouts={rec['n_track_dropouts']} (max {rec['max_consec_misses']} consec, "
              f"{rec['n_stale_replans']}/{rec['n_replans']} stale replans) -> {log_path}")

    tracking = {"misses": 0}

    with mjviewer.launch_passive(mj_model, mj_data) as viewer:
        print(f"Realtime loop running (timeout {TRIAL_TIMEOUT_S:.0f}s). "
              f"Close viewer to stop (logs an abort).")
        while viewer.is_running():
            if time.time() - t_start > TRIAL_TIMEOUT_S:
                print(f"\n✗ Timeout after {TRIAL_TIMEOUT_S:.0f}s — trial abandoned.")
                rec["timeout"] = True
                break
            rec["n_replans"] += 1
            # 1. Get real tee pose from Optitrack
            rbs = []
            for frame in optitrack_client.MoCap(timeout=None):
                rbs = [rb for rb in frame.rigid_body_data.rigid_bodies if rb.tracking]
                break
            # Tracking health. `rbs` empty means Motive has no fix on the tee
            # (arm occlusion, or a camera dead zone); the pose below is then
            # whatever was last seen, and planning silently continues on it.
            if rbs:
                if tracking["misses"] >= TRACK_STALE_MISSES:
                    print(f"[track] reacquired after {tracking['misses']} missed detections")
                tracking["misses"] = 0
            else:
                tracking["misses"] += 1
                rec["n_stale_replans"] += 1
                rec["max_consec_misses"] = max(rec["max_consec_misses"],
                                               tracking["misses"])
                if tracking["misses"] == TRACK_STALE_MISSES:
                    rec["n_track_dropouts"] += 1
                if tracking["misses"] >= TRACK_STALE_MISSES:
                    print(f"[track] WARNING: tee not detected for "
                          f"{tracking['misses']} consecutive replans "
                          f"— planning on a stale pose "
                          f"({rec['n_track_dropouts']} dropouts this trial)")

            if rbs:
                xy, yaw = tee_pose(rbs[0])
                mj_data.qpos[T_QPOS_OFFSET]     = float(xy[0])
                mj_data.qpos[T_QPOS_OFFSET + 1] = float(xy[1])
                mj_data.qpos[T_QPOS_OFFSET + 2] = float(yaw)
                mujoco.mj_forward(mj_model, mj_data)
                # Show the freshly tracked pose *before* the ~1s replan blocks, so
                # the tee visibly follows the real block instead of appearing to
                # jump at the same instant the arm moves. Makes a stale pose
                # (frozen tee while the real one moves) obvious at a glance.
                viewer.sync()

                dist_to_goal  = float(np.linalg.norm(xy - np.array(theta.goal_xy)))
                angle_err     = float(abs(((yaw - float(theta.goal_angle) + np.pi) % (2 * np.pi)) - np.pi))
                rec["final_pos_err"]   = round(dist_to_goal, 4)
                rec["final_angle_err"] = round(angle_err, 4)
                if dist_to_goal < SUCCESS_POS_RADIUS and angle_err < SUCCESS_ANGLE_TOL:
                    print(f"\n✓ Goal reached (dist={dist_to_goal*100:.1f}cm, angle={np.degrees(angle_err):.1f}°).")
                    rec["success"] = True
                    break

            # 2. Run PFMP
            key, subkey = jax.random.split(key)
            if not np.all(np.isfinite(mj_data.qpos)) or not np.all(np.isfinite(mj_data.ctrl)):
                raise RuntimeError(
                    f"NaN/Inf in mj_data (replan {rec['n_replans']}, method={method}): "
                    f"qpos finite={np.all(np.isfinite(mj_data.qpos))}, "
                    f"ctrl finite={np.all(np.isfinite(mj_data.ctrl))}")
            mjx_data = mjx.put_data(mj_model, mj_data)
            trajs, costs = replan(subkey, mjx_data, theta)
            theta, _ = update(theta, trajs, costs)
            if not np.all(np.isfinite(np.array(theta.mean))):
                raise RuntimeError(
                    f"NaN in theta.mean after update (replan {rec['n_replans']}, "
                    f"method={method}). A control left the reachable sector.")

            # 3. Execute one knot before replanning
            q_prev  = np.array(mj_data.qpos[:7]).copy()   # last commanded arm pose
            # Project, not just clip: the update averages in Cartesian, so a mean
            # of two in-sector knots can fall outside it (e.g. +/-105 deg averages
            # to near the base). The rectangle clip alone lets that reach the IK.
            next_wp = np.asarray(_project_reachable(jnp.asarray(theta.mean[1]), theta))

            goal_marker_callback(viewer.user_scn, theta)
            try:
                reply = robot_planner.send_command(Command(xy=next_wp.tolist()))
                actual_xy = reply.get("actual_xy")
                ee_xy = np.array(actual_xy) if actual_xy else next_wp
                q_cmd = np.asarray(ik_from_cache(jnp.array(ee_xy), ik_cache), dtype=float)
                mj_data.ctrl[:7] = q_cmd
                # Smooth viz: glide the sim arm q_prev -> q_cmd across sub-frames so
                # the display updates independently of the (slow) replan rate instead
                # of snapping between IK solutions. Purely visual — the real command
                # was already sent above; the tee (qpos[7:]) is untouched.
                for s in range(1, VIZ_INTERP_STEPS + 1):
                    a = s / VIZ_INTERP_STEPS
                    mj_data.qpos[:7] = (1.0 - a) * q_prev + a * q_cmd
                    mujoco.mj_forward(mj_model, mj_data)
                    viewer.sync()
                    time.sleep(VIZ_INTERP_DT)
            except Exception as e:
                print(f"[planner] command failed: {e}")
                rec["n_cmd_failures"] += 1           # contamination flag
                mj_data.qpos[:7] = q_prev            # hold pose on failure
                mujoco.mj_forward(mj_model, mj_data)
                viewer.sync()

            # Shift mean by 1 knot
            new_mean = jnp.concatenate([theta.mean[1:], theta.mean[-1:]])
            theta = theta._replace(mean=new_mean)

    _write_record()



#################################################################

def main():
    parser = argparse.ArgumentParser(description="T-block pushing with PFMP")
    parser.add_argument('--cache-resolution', type=int, default=IK_CACHE_RESOLUTION,
                        help='IK cache grid resolution')
    parser.add_argument('--init-xy',  type=float, nargs=2, default=[0.4, 0.0], metavar=('X', 'Y'))
    # Manual init/goal, used only when --trial is omitted. With --trial, start and
    # goal come from the (seed, trial) spec instead.
    parser.add_argument('--goal-x',   type=float, default=0.45)
    parser.add_argument('--goal-y',   type=float, default=-0.35)
    parser.add_argument('--goal-yaw', type=float, default=float(4 * np.pi / 3))
    parser.add_argument('--log-path', type=str,   default='realtime_log.jsonl',
                        help='JSONL per-trial log for realtime runs (pass before the subcommand)')
    parser.add_argument('--seed',     type=int,   default=42)

    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument('--trial', type=int, default=None,
                        help='Trial index. When set, start and goal come from the '
                             '(seed, trial) spec and the placement view opens first; '
                             'when omitted, --init-xy and --goal-* are used.')

    subparsers = parser.add_subparsers(dest="algorithm", required=True)
    subparsers.add_parser("ps",  parents=[shared], help="Predictive Sampling")
    subparsers.add_parser("sir", parents=[shared], help="Sequential Importance Resampling")
    mppi_p = subparsers.add_parser("mppi", parents=[shared], help="MPPI baseline")
    mppi_p.add_argument("--temperature", type=float, default=1.0)
    cem_p = subparsers.add_parser("cem", parents=[shared], help="Cross-Entropy Method")
    cem_p.add_argument("--num-elites", type=int, default=13)
    dial_p = subparsers.add_parser("dial", parents=[shared], help="DIAL-MPC baseline")
    dial_p.add_argument("--temperature",  type=float, default=1.0)
    dial_p.add_argument("--beta1",        type=float, default=1.0)
    dial_p.add_argument("--beta2",        type=float, default=1.0)
    dial_p.add_argument("--n-inner",      type=int,   default=5)
    rt = subparsers.add_parser("realtime", parents=[shared], help="Closed-loop real robot")
    rt.add_argument("--server",      default="127.0.0.1", help="Motive IP")
    rt.add_argument("--local",       default="127.0.0.1", help="Local IP")
    rt.add_argument("--server-host", default="localhost",  help="Controller host")
    rt.add_argument("--cmd-port",    type=int, default=5555)
    rt.add_argument("--method",         default="sir", choices=["sir", "ps", "mppi", "cem", "dial"])
    rt.add_argument("--temperature",    type=float, default=1.0, help="MPPI temperature")
    args = parser.parse_args()

    # Load scene
    mj_model, mj_data = load_scene(SCENE_XML_PATH)

    # Build IK cache (one-time cost)
    print("Building IK cache...")
    ik_cache = get_or_build_cache(
        xy_min=WORKSPACE_XY_MIN,
        xy_max=WORKSPACE_XY_MAX,
        resolution=args.cache_resolution,
        z=EE_HEIGHT,
    )

    # With --trial set, tee_problem_gen is the single source of truth for
    # (start, goal): the placement view shows the start ghost to position the
    # block on, and the goal the planner optimises for is the matching spec goal,
    # so the two can never disagree. Without --trial, fall back to --init-xy and
    # --goal-* for one-off manual runs.
    from tee_problem_gen import sample_problem
    trial = getattr(args, "trial", None)
    use_spec = trial is not None
    if use_spec:
        spec_init_xy, spec_init_yaw, spec_goal_xy, spec_goal_yaw = sample_problem(
            args.seed, trial, ik_cache)
        goal_x, goal_y, goal_yaw = float(spec_goal_xy[0]), float(spec_goal_xy[1]), float(spec_goal_yaw)
        init_x, init_y, init_yaw = float(spec_init_xy[0]), float(spec_init_xy[1]), float(spec_init_yaw)
    else:
        spec_init_xy = spec_init_yaw = None
        goal_x, goal_y, goal_yaw = args.goal_x, args.goal_y, args.goal_yaw
        init_x, init_y = float(args.init_xy[0]), float(args.init_xy[1])
        init_yaw = 0.0
    print(f"Problem (seed {args.seed}, trial {trial}):"
          f"\n  init = ({init_x:+.3f}, {init_y:+.3f}, {np.degrees(init_yaw):+.1f}°)"
          f"\n  goal = ({goal_x:+.3f}, {goal_y:+.3f}, {np.degrees(goal_yaw):+.1f}°)"
          + ("  [per-trial spec]" if use_spec else "  [manual]"))

    # Task configuration
    horizon   = HORIZON
    num_knots = NUM_KNOTS
    particles = PARTICLES

    # Initial end-effector position
    init_xy = jnp.array(args.init_xy)
    q0 = ik_from_cache(init_xy, ik_cache)
    mj_data.qpos[:7] = np.array(q0)
    # ctrl is the ramp's starting reference in `dynamics`, so it must begin at the
    # current pose. Left at zeros, every rollout ramps from the zero configuration
    # (arm extended horizontally) and sweeps through the table.
    mj_data.ctrl[:7] = np.array(q0)

    # Initialize mean trajectory: linear interp from EE to T-block position.
    gripper_site_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SITE, "gripper")
    mujoco.mj_fwdPosition(mj_model, mj_data)
    r_xy = jnp.array(mj_data.site_xpos[gripper_site_id, :2])

    t_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "t_block")
    t_world_xy = mj_data.xpos[t_id, :2]
    t_qpos_xy  = mj_data.qpos[T_QPOS_OFFSET:T_QPOS_OFFSET + 2]
    print(f"T world xy (xpos):  {t_world_xy}")
    print(f"T qpos[7:9]:        {t_qpos_xy}")
    # Initial T pose comes from the same problem spec as the goal (or --init-xy in
    # manual mode), so a sim run and a real run of trial N start from the same place.
    mj_data.qpos[T_QPOS_OFFSET]     = init_x
    mj_data.qpos[T_QPOS_OFFSET + 1] = init_y
    mj_data.qpos[T_QPOS_OFFSET + 2] = init_yaw
    mujoco.mj_forward(mj_model, mj_data)

    t_xy = jnp.array(mj_data.qpos[T_QPOS_OFFSET:T_QPOS_OFFSET + 2])
    alphas = jnp.linspace(0.0, 1.0, num_knots)
    init_mean = (1.0 - alphas[:, None]) * r_xy + alphas[:, None] * t_xy

    # Create parameters
    theta = PushParams(
        mean=init_mean,
        mjx_model=mjx.put_model(mj_model),
        ik_cache=ik_cache,
        goal_xy=jnp.array([goal_x, goal_y]),
        goal_angle=float(goal_yaw),
        pos_weight=15.0,
        angle_weight=0.3,
        u_stdev=0.025,
        gripper_site_id=gripper_site_id,
    )

    _common = dict(
        horizon=horizon, num_knots=num_knots, particles=particles,
        dt=mj_model.opt.timestep, ess_resample_thresh=0.5,
        # Hold each knot for its interval instead of splining between knots.
        # The proposal projects knots into the reachable sector, but a hermite
        # interpolant between two in-sector knots leaves it ~32% of the time —
        # the sector is non-convex in Cartesian and hermite overshoots its
        # endpoints — and those intermediate controls reach the IK cache as NaN.
        # ZOH also matches execution: one waypoint per knot, as the robot runs.
        interp="zoh",
    )
    ctrl = pfmp.PFMP(dynamics, running_cost, terminal_cost, init_control, proposal, **_common)
    ctrl_mppi = pfmp.PFMP(dynamics, running_cost, terminal_cost, init_control, mppi_proposal, **_common)

    z_traj = float(EE_HEIGHT)
    traj_rgba = np.array([1.0, 0.5, 0.0, 1.0], dtype=np.float32)

    def goal_marker_callback(user_scn, theta):
        add_goal_T(user_scn, np.array(theta.goal_xy), float(theta.goal_angle),
                rgba=(0.2, 0.8, 0.2, 0.5))
        # Draw planned EE trajectory as orange line segments between knots.
        knots = np.array(theta.mean)  # (num_knots, 2)
        for k in range(len(knots) - 1):
            if user_scn.ngeom >= user_scn.maxgeom:
                break
            p0 = np.array([knots[k, 0],   knots[k, 1],   z_traj], dtype=np.float64)
            p1 = np.array([knots[k+1, 0], knots[k+1, 1], z_traj], dtype=np.float64)
            geom = user_scn.geoms[user_scn.ngeom]
            mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_LINE, 2.0, p0, p1)
            geom.rgba[:] = traj_rgba
            user_scn.ngeom += 1

    if args.algorithm == "realtime":
        from new_natnet_client.Client import NatNetClient, NatNetParams
        from pub_sub.planner import PlannerClient

        # With --trial, open the placement view first so the block can be set on
        # the spec start (and the spec goal) before the trial runs. Launched as a
        # subprocess so its OptiTrack client fully closes before the planner opens
        # its own — two live NatNet clients to Motive conflict.
        if use_spec:
            import subprocess
            print("Opening placement view — position the block on the ghost, "
                  "then close the window to start the trial.")
            subprocess.run(
                [sys.executable, os.path.join(os.path.dirname(__file__), "tee_problem_gen.py"),
                 "--server", str(args.server), "--local", str(args.local),
                 "--seed", str(args.seed), "--trial", str(trial)],
                check=False)

        # Goal already resolved from the problem spec above.
        theta = theta._replace(
            goal_xy=jnp.array([goal_x, goal_y]),
            goal_angle=jnp.array(goal_yaw),
        )

        robot_planner = PlannerClient(
            command_addr=f"tcp://{args.server_host}:{args.cmd_port}",
            response_addr="tcp://localhost:5556",
            conflate=False, command_timeout_ms=30000,
        )
        params = NatNetParams(server_address=args.server, local_ip_address=args.local,
                              use_multicast=False)
        print(f"Connecting to Motive at {args.server}...")
        with NatNetClient(params) as optitrack_client:
            if optitrack_client is None:
                print("Failed to connect."); return
            print("Connected. Sampling initial tee pose...")

            # Seed tee position from Optitrack before starting
            from push_tee_optitrack import tee_pose
            for frame in optitrack_client.MoCap(timeout=None):
                rbs = [rb for rb in frame.rigid_body_data.rigid_bodies if rb.tracking]
                if rbs:
                    xy, yaw = tee_pose(rbs[0])
                    mj_data.qpos[T_QPOS_OFFSET]     = float(xy[0])
                    mj_data.qpos[T_QPOS_OFFSET + 1] = float(xy[1])
                    mj_data.qpos[T_QPOS_OFFSET + 2] = float(yaw)
                    t_xy = jnp.array(xy)
                    alphas = jnp.linspace(0.0, 1.0, num_knots)
                    r_xy   = jnp.array(mj_data.qpos[
                        mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SITE, "gripper")
                    ] if False else [init_xy[0], init_xy[1]])
                    theta = theta._replace(
                        mean=(1.0 - alphas[:, None]) * r_xy + alphas[:, None] * t_xy)
                    print(f"  tee at ({xy[0]:+.3f}, {xy[1]:+.3f}, {np.degrees(yaw):+.1f}°)")
                    break

            try:
                rt_method = getattr(args, "method", "sir")
                temperature = args.temperature if rt_method == "mppi" else None
                active_ctrl = ctrl_mppi if rt_method in ("ps", "mppi") else ctrl
                run_realtime(active_ctrl, theta, mj_model, mj_data, ik_cache,
                             optitrack_client, robot_planner,
                             method=rt_method, temperature=temperature,
                             log_path=args.log_path,
                             trial=(trial if trial is not None else -1), seed=args.seed,
                             spec_init=((spec_init_xy[0], spec_init_xy[1], spec_init_yaw)
                                        if use_spec else None))
            finally:
                robot_planner.close()
    else:
        from pfmp.algorithm import cem_update, mppi_update

        active_ctrl = ctrl_mppi if args.algorithm in ("mppi", "ps", "cem", "dial") else ctrl
        update_fn = None

        if args.algorithm == "cem":
            num_elites = args.num_elites
            update_fn = lambda th, tr, co: cem_update(th, tr, jnp.sum(co, axis=1),
                                                       elite_frac=num_elites / PARTICLES)
        elif args.algorithm == "dial":
            from pfmp import dial_sigma

            n_inner     = args.n_inner
            beta1       = args.beta1
            beta2       = args.beta2
            temperature = args.temperature

            def dial_update(th, tr, co):
                theta_out = th
                for i in range(n_inner, 0, -1):
                    sigma = float(np.mean([dial_sigma(i, h, th.u_stdev, n_inner, NUM_KNOTS, beta1, beta2)
                                           for h in range(NUM_KNOTS)]))
                    total = jnp.sum(co, axis=1)
                    temp = float(jnp.maximum(jnp.std(total), 1e-6)) * temperature
                    theta_out, _ = mppi_update(theta_out, tr.transpose(1, 0, 2), co.T, temp)
                return theta_out, jnp.zeros(PARTICLES)
            update_fn = dial_update

        pfmp.run_interactive(
            active_ctrl,
            theta,
            mj_model,
            mj_data,
            use_dynamics=True,
            frequency=RUN_INTERACTIVE_FREQUENCY,
            method=args.algorithm if args.algorithm not in ("cem", "dial") else "sis",
            fixed_camera_id=None,
            record_video=False,
            record_traj=False,
            max_step=8000,
            goal_marker_fn=goal_marker_callback,
            update_fn=update_fn,
        )


if __name__ == "__main__":
    main()