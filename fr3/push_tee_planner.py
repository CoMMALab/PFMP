"""
T-block pushing task using PFMP with Panda robot.

Real world workspace is 53' 1/8" x 23' 1/4"
"""

import argparse
import datetime
import json
import os
import threading
import time
from dataclasses import dataclass
from typing import Callable, NamedTuple, Tuple

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from mujoco import mjx

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))
from geom_tee import STEM_HALF, CAP_HALF, CAP_CTR
from env import load_env
from push_tee_2d import sample_mixture

import pfmp

from ik_cache import (
    IKCache, ik_from_cache, ik_solve, get_or_build_cache, reachable_r_center, HOME_Q,
    WORKSPACE_XY_MIN, WORKSPACE_XY_MAX, IK_CACHE_RESOLUTION, EE_HEIGHT,
)
from push_tee_controller import VELOCITY_SCALE

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
    stdev: float
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
PARTICLES = 256

KNOT_INTERVAL = HORIZON / (NUM_KNOTS - 1)
RUN_INTERACTIVE_FREQUENCY = 1.0 / KNOT_INTERVAL

VIZ_FPS          = 60.0     # render-thread refresh, decoupled from the replan rate
TRACK_STALE_AGE  = 0.25     # a live tee pose older than this (s) counts as stale

SUCCESS_POS_RADIUS = 0.05   # metres
SUCCESS_ANGLE_TOL  = 0.35   # radians (~20°)
TRIAL_TIMEOUT_S    = 300.0  # abandon a trial after 5 min, logged as timeout

TRACK_STALE_MISSES = 2    # consecutive missed detections before warning

PROPOSAL_MIX    = 0.4   # directed toward tee
ORBITAL_MIX     = 0.2   # orbit around tee at push radius
STEP_LEN_VEL    = 0.05
PUSH_RADIUS     = 0.16  # just outside tee cap half-extent
ORBITAL_Y_OFFSET = 0.04  # metres up the tee stem for the orbital centre

CEM_STDEV_FLOOR = 0.02

# DIAL_N=2 is inspired by the canonical dial-mpc repo's steady-state Ndiffuse for
# its SIM config (their real-hardware deploy config drops to Ndiffuse=1).
# DIAL_N_INIT=10 (Ndiffuse_init) only applies to a trial's first replan, a
# one-time higher-fidelity solve from a cold start.
DIAL_N, DIAL_N_INIT, DIAL_B1, DIAL_B2 = 2, 10, 1.0, 1.0

# Max Velocity
FR3_DQ_MAX     = jnp.array([2.62, 2.62, 2.62, 2.62, 5.26, 4.18, 5.26])   # rad/s


def dynamics(x: mjx.Data, u: jax.Array, th: PushParams, t: float) -> Tuple[mjx.Data, jax.Array]:
    """Step MuJoCo, ramping the joint reference toward the waypoint's IK solution."""
    dt      = th.mjx_model.opt.timestep
    dq_max  = FR3_DQ_MAX * VELOCITY_SCALE * dt

    target_qpos = ik_from_cache(u, th.ik_cache)
    delta = target_qpos - x.ctrl[:7]
    scale = jnp.minimum(1.0, jnp.min(dq_max / (jnp.abs(delta) + 1e-9)))
    step  = delta * scale

    ctrl = x.ctrl.at[:7].set(x.ctrl[:7] + step)
    x_next = mjx.step(th.mjx_model, x.replace(ctrl=ctrl))
    return x_next, u


TEE_ANNULUS_CENTER = 0.549   # reachable radial midpoint; main pins this exactly
                             # from the cache via reachable_r_center()
TEE_ANNULUS_WEIGHT = 0.3   # radial shaping toward TEE_ANNULUS_CENTER (keeps tee in recoverable band)

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


def _sigma_at(th: PushParams, idx) -> jax.Array:
    """Sampling noise for knot `idx`.

    Supports both conventions: a scalar `stdev` (fixed-noise methods — PS, MPPI,
    SIR/SIS, and DIAL, which overwrites it per annealing level) and a per-knot
    array of shape (num_knots, 2) (CEM, whose covariance is refit from the elites
    each replan). `jnp.ndim` is static, so this costs nothing under jit and the
    scalar path is bit-identical to reading `th.stdev` directly.
    """
    return th.stdev if jnp.ndim(th.stdev) == 0 else th.stdev[idx]


def _knot_index(t: float) -> jax.Array:
    """Knot whose interval contains time `t` (shared by the proposals)."""
    step_len = HORIZON / (NUM_KNOTS - 1)
    return jnp.clip(jnp.floor(t / step_len).astype(jnp.int32) + 1, 0, NUM_KNOTS - 1)


def init_control(key: jax.Array, x: mjx.Data, th: PushParams) -> jax.Array:
    """`proposal` at the first knot, seeded from the warm-started mean.

    The orbital branch is anchored to the tee rather than the mean, so particles
    stay feasible even if the mean has drifted outside the reachable set.
    """
    return proposal(key, x, th.mean[0], th, 0.0, idx=0)


def mppi_proposal(key: jax.Array, _x: mjx.Data, _u: jax.Array, th: PushParams, t: float) -> jax.Array:
    """Gaussian proposal centred on the mean knot at time t (for PS/MPPI baselines). """
    idx = _knot_index(t)
    return project_reachable(
        th.mean[idx] + jax.random.normal(key, (2,)) * _sigma_at(th, idx), th)


def project_reachable(u: jax.Array, th: PushParams) -> jax.Array:
    """Snap an EE waypoint to the nearest reachable grid cell. """
    cache = th.ik_cache
    Nx, Ny = cache.reachable.shape
    xs = jnp.linspace(cache.xy_min[0], cache.xy_max[0], Nx)
    ys = jnp.linspace(cache.xy_min[1], cache.xy_max[1], Ny)
    X, Y = jnp.meshgrid(xs, ys, indexing="ij")            # (Nx, Ny)
    d2 = (X - u[0]) ** 2 + (Y - u[1]) ** 2
    d2 = jnp.where(cache.reachable > 0.5, d2, jnp.inf)     # only reachable cells
    flat = jnp.argmin(d2)
    return jnp.array([X.reshape(-1)[flat], Y.reshape(-1)[flat]])


def proposal(key: jax.Array, x: mjx.Data, u: jax.Array, th: PushParams, t: float,
             idx=None) -> jax.Array:
    """One draw from the three-way mixture, snapped to the reachable set.

    The mixture itself lives in examples/push_tee_2d.py so the simulated and
    hardware push-T share one proposal and the paper describes it once. Here it
    is bound to the FR3's state layout, tee geometry, and reachable set. The
    constants are unchanged, so the draw is identical to the local copy of the
    mixture this replaced.

    `idx` overrides which knot's noise width to use. `init_control` passes 0,
    which `_knot_index` never returns for t >= 0; it only matters under CEM,
    whose stdev is per-knot.
    """
    idx = _knot_index(t) if idx is None else idx
    return project_reachable(sample_mixture(
        key, u,
        x.qpos[T_QPOS_OFFSET:T_QPOS_OFFSET + 2], x.qpos[T_QPOS_OFFSET + 2],
        _sigma_at(th, idx),
        directed_frac=PROPOSAL_MIX, orbital_frac=ORBITAL_MIX,
        step_len=STEP_LEN_VEL,
        push_radius=PUSH_RADIUS, orbital_y_offset=ORBITAL_Y_OFFSET), th)


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


def _sample_traj(traj: np.ndarray, alpha: float) -> np.ndarray:
    """Sample a dense trajectory (N, d) at fraction alpha in [0, 1], linearly
    interpolating between the two bracketing samples."""
    n = len(traj)
    if n <= 1:
        return traj[0]
    f = alpha * (n - 1)
    i = min(int(f), n - 2)
    w = f - i
    return traj[i] * (1.0 - w) + traj[i + 1] * w


def draw_goal_and_plan(user_scn, th) -> None:
    """Viewer overlay: translucent goal T-marker plus the planned EE path as
    orange line segments between knots. Resets user_scn's geom count first,
    so it's safe to call once per frame regardless of what drew last frame."""
    user_scn.ngeom = 0
    add_goal_T(user_scn, np.array(th.goal_xy), float(th.goal_angle),
               rgba=(0.2, 0.8, 0.2, 0.5))
    knots = np.array(th.mean)
    z = float(EE_HEIGHT)
    for k in range(len(knots) - 1):
        if user_scn.ngeom >= user_scn.maxgeom:
            break
        p0 = np.array([knots[k, 0],   knots[k, 1],   z], dtype=np.float64)
        p1 = np.array([knots[k + 1, 0], knots[k + 1, 1], z], dtype=np.float64)
        geom = user_scn.geoms[user_scn.ngeom]
        mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_LINE, 2.0, p0, p1)
        geom.rgba[:] = [1.0, 0.5, 0.0, 1.0]
        user_scn.ngeom += 1


class LiveVis:
    """Optional MuJoCo viewer + render thread that animates the plan's
    predicted push, fully decoupled from the realtime planning loop: when
    `enabled=False` (headless collection), every method is a no-op and no
    viewer/thread is ever created, so the planning loop calls the same
    interface either way and never has to branch on whether anyone's
    watching.

    This is a class, not a dataclass, despite the rest of this file preferring
    dataclasses for state: it owns a thread and a viewer handle with real
    setup/teardown, not just data.

    Glide state below is written by `update()` (called from the planning
    loop after each replan) and read by the render thread — always under
    `self._lock`, separate from the tee-tracking lock the caller passes in
    for `live` (a `LiveTee`), since planning and rendering write/read each
    from different threads.

    glide_start/glide_dur: wall-clock start time and duration (s) of the
        current glide. alpha = (now - glide_start) / glide_dur.
    arm_traj: (dense, 7) predicted arm joint trajectory over the glide.
    tee_traj: (dense, 3) predicted tee (x, y, yaw) over the same glide.
    theta: latest planning params, for the goal/knot-trajectory overlay.
    """

    TRACKER_Z = 0.020

    def __init__(self, enabled: bool, mj_model, mj_data, theta, live,
                tee_lock: threading.Lock, timeout: float):
        self.enabled = enabled
        self._mj_model = mj_model
        self._mj_data = mj_data
        self._live = live
        self._tee_lock = tee_lock
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self.viewer = None
        self._thread = None
        self._tracker_mocapid = -1
        if not enabled:
            print(f"Realtime loop running headless (timeout {timeout:.0f}s). Ctrl-C to abort.")
            return

        import mujoco.viewer as mjviewer

        # The blue translucent t_tracker (mocap body) shows the live OptiTrack tee.
        _tb = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "t_tracker")
        self._tracker_mocapid = int(mj_model.body_mocapid[_tb]) if _tb >= 0 else -1

        q0_arm = np.array(mj_data.qpos[:7], float)
        tee0 = np.array([float(live.xy[0]), float(live.xy[1]), live.yaw], float)
        self.glide_start = time.time()
        self.glide_dur = 0.0
        self.theta = theta
        self.arm_traj = q0_arm[None].copy()
        self.tee_traj = tee0[None].copy()

        self.viewer = mjviewer.launch_passive(mj_model, mj_data)
        print(f"Realtime loop running (timeout {timeout:.0f}s). Close viewer to stop (logs an abort).")
        self._thread = threading.Thread(target=self._render_loop, daemon=True)

    def start(self):
        """Launch the render thread. No-op when disabled (there is none)."""
        if not self.enabled:
            return
        self._thread.start()

    def is_running(self) -> bool:
        """True while the planning loop should keep going: always True
        headless, tied to the viewer window otherwise."""
        return (not self.enabled) or self.viewer.is_running()

    def update(self, theta, glide_dur: float, arm_traj=None, tee_traj=None):
        """Publish the latest replan's predicted push. No-op when disabled.
        arm_traj/tee_traj are optional: the headless glide-duration fallback
        doesn't compute a predicted trajectory to show."""
        if not self.enabled:
            return
        with self._lock:
            self.glide_start = time.time()
            self.glide_dur = glide_dur
            self.theta = theta
            if arm_traj is not None:
                self.arm_traj = arm_traj
                self.tee_traj = tee_traj

    def close(self):
        """Stop the render thread and close the viewer. No-op when disabled."""
        if not self.enabled:
            return
        self._stop.set()
        self._thread.join(timeout=1.0)
        self.viewer.close()

    def _render_loop(self):
        """Sole owner of the display MjData + viewer. Tee follows the live
        block straight from OptiTrack; the arm glides along the latest
        predicted push, so both keep moving even while the main thread is
        blocked in send_command."""
        dt = 1.0 / VIZ_FPS
        while not self._stop.is_set() and self.viewer.is_running():
            with self._tee_lock:
                xy = self._live.xy.copy()
                yaw = self._live.yaw
            with self._lock:
                glide_start, glide_dur = self.glide_start, self.glide_dur
                th = self.theta
                arm_traj, tee_traj = self.arm_traj, self.tee_traj
            alpha = (1.0 if glide_dur <= 1e-6
                     else min(1.0, (time.time() - glide_start) / glide_dur))

            self._mj_data.qpos[:7] = _sample_traj(arm_traj, alpha)
            tee = _sample_traj(tee_traj, alpha)
            tee_xy, tee_yaw = tee[:2], tee[2]
            self._mj_data.qpos[T_QPOS_OFFSET]     = float(tee_xy[0])
            self._mj_data.qpos[T_QPOS_OFFSET + 1] = float(tee_xy[1])
            self._mj_data.qpos[T_QPOS_OFFSET + 2] = float(tee_yaw)
            if self._tracker_mocapid >= 0:
                self._mj_data.mocap_pos[self._tracker_mocapid] = [
                    float(xy[0]), float(xy[1]), self.TRACKER_Z]
                self._mj_data.mocap_quat[self._tracker_mocapid] = [
                    float(np.cos(yaw / 2)), 0.0, 0.0, float(np.sin(yaw / 2))]
            mujoco.mj_forward(self._mj_model, self._mj_data)
            draw_goal_and_plan(self.viewer.user_scn, th)
            self.viewer.sync()
            time.sleep(dt)


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


@dataclass
class MethodConfig:
    """Resolved replan/update behavior for one `--method` choice.

    replan: (key, mjx_data, theta) -> (trajs, costs). Used directly by every
        method except dial (which drives its own loop via dial_step below),
        and always used once for JIT warm-up regardless of method.
    update: (theta, trajs, costs) -> (theta, aux). None for dial.
    dial_step: (key, mjx_data, theta, n) -> theta. None for every other method.
    axis_proposal, axis_update: logged separately per trial so the analysis can
        isolate each design axis. The `biased_` prefix only swaps the proposal
        (handled by which `ctrl` the caller passes in) — the update rule is
        always its plain counterpart, hence axis_update = base_method.
    """
    replan: Callable
    update: Callable | None
    is_dial: bool
    dial_step: Callable | None
    needs_per_knot_stdev: bool
    axis_proposal: str
    axis_update: str


def build_method_config(method: str, ctrl, temperature: float | None) -> MethodConfig:
    """Resolve a `--method` string into the callables run_realtime needs."""
    from pfmp.algorithm import ps_update, mppi_update

    axis_proposal = "custom" if method in ("sir", "sis", "biased_mppi", "biased_cem") else "gaussian"
    base_method = method[len("biased_"):] if method.startswith("biased_") else method
    is_dial = False
    dial_step_fn = None
    needs_per_knot_stdev = False

    if base_method == "mppi":
        replan = lambda key, data, th: ctrl.run_sis(key, data, th)
        def update(th, tr, co):
            finite = jnp.isfinite(jnp.sum(co, axis=1))
            co = jnp.where(finite[:, None], co, jnp.zeros_like(co))
            return mppi_update(th, tr, co, max(float(temperature), 1e-6))
    elif base_method == "ps":
        replan = lambda key, data, th: ctrl.run_sis(key, data, th)
        update = ps_update
    elif base_method == "sis":
        # SIR with resampling turned off (ρ=0): same object proposal (via
        # active_ctrl) and same ps_update as sir, only run_sis vs run_sir.
        replan = lambda key, data, th: ctrl.run_sis(key, data, th)
        update = ps_update
    elif base_method == "cem":
        from pfmp.algorithm import cem_update
        num_elites = max(1, PARTICLES // 10)
        replan = lambda key, data, th: ctrl.run_sis(key, data, th)
        # CEM refits both the mean and the covariance from the elites.
        update = lambda th, tr, co: cem_update(th, tr, co,
                                               elite_frac=num_elites / PARTICLES,
                                               stdev_floor=CEM_STDEV_FLOOR)
        needs_per_knot_stdev = True
    elif base_method == "dial":
        from pfmp import dial_step as _dial_step
        is_dial = True
        replan = lambda key, data, th: ctrl.run_sis(key, data, th)  # for JIT warm-up
        update = None   # dial drives its own re-sampling, via dial_step below
        # Thin binding of the shared pfmp.dial_step to this ctrl/beta config, so
        # the call site reads the same as every other method's dispatch.
        dial_step_fn = lambda key, data, th, n: _dial_step(
            key, ctrl, data, th, n=n, beta1=DIAL_B1, beta2=DIAL_B2,
            temp_sample=1.0)   # not dial-mpc's 0.06; preserves logged trials
    else:  # sir
        replan = lambda key, data, th: ctrl.run_sir(key, data, th)
        update = ps_update

    return MethodConfig(replan=replan, update=update, is_dial=is_dial,
                        dial_step=dial_step_fn, needs_per_knot_stdev=needs_per_knot_stdev,
                        axis_proposal=axis_proposal, axis_update=base_method)


def run_realtime(ctrl, theta, mj_model, mj_data, ik_cache,
                 optitrack_client, robot_planner, method="sir", temperature=None,
                 log_path=None, trial=0, seed=0, spec_init=None,
                 timeout=TRIAL_TIMEOUT_S, viz=False):
    """Closed-loop realtime push: Optitrack → PFMP → robot controller.

    viz: launch the MuJoCo viewer with a render thread. The orange simulated tee
    follows the plan's predicted push (ctrl.predict) and the blue t_tracker mocap
    body shows the live OptiTrack observation, so MPC relocalization is visible.
    Off = fully headless (no viewer, no prediction), for collection. The mocap
    tracker thread always runs for state estimation.
    """
    from push_tee_optitrack import LiveTee, track_tee
    from pub_sub.protocol import Command

    key = jax.random.key(0)

    cfg = build_method_config(method, ctrl, temperature)

    # CEM's covariance is per-knot, so widen the scalar stdev to (num_knots, 2).
    # Every other method keeps the scalar, for which _sigma_at is a no-op.
    if cfg.needs_per_knot_stdev and jnp.ndim(theta.stdev) == 0:
        theta = theta._replace(
            stdev=jnp.full((NUM_KNOTS, 2), float(theta.stdev)))

    print("JIT warm-up…", end=" ", flush=True)
    mjx_data = mjx.put_data(mj_model, mj_data)
    _ = cfg.replan(key, mjx_data, theta)
    print("done")

    # Per-trial log record. Written once when the trial ends.
    t_start = time.time()
    # `spec_*` is what tee_problem_gen prescribed for this (seed, trial);
    # `realized_*` is the measured OptiTrack value.
    rec = {
        "ts": datetime.datetime.now().isoformat(timespec="seconds"),
        "trial": int(trial), "seed": int(seed), "method": method,
        "proposal": cfg.axis_proposal, "update": cfg.axis_update,
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
        "n_track_dropouts": 0, "n_stale_replans": 0, "max_consec_misses": 0,
        "final_pos_err": None, "final_angle_err": None, "duration_s": None,
    }
    if spec_init is not None:
        rec["placement_err"] = round(float(np.linalg.norm(
            np.array(rec["realized_init"][:2]) - np.asarray(spec_init[:2]))), 4)

    # Fallback for realized_final below if the loop body never runs (e.g.
    # timeout <= 0 on the very first check).
    xy, yaw = np.array(rec["realized_init"][:2], float), rec["realized_init"][2]

    # Descend onto the pushing plane before planning begins.
    print("Descending onto pushing plane…")
    robot_planner.lifecycle("descend")

    consec_misses = 0   # consecutive stale/missed tee-pose reads

    # ── Display / tracking decoupled from planning ────────────────────────────
    # send_command blocks until franky finishes the push. The planner runs on its
    # own MjData copy so there are no races.
    st_lock = threading.Lock()
    stop_event = threading.Event()
    live = LiveTee(xy=np.array(mj_data.qpos[T_QPOS_OFFSET:T_QPOS_OFFSET + 2], float),
                   yaw=float(mj_data.qpos[T_QPOS_OFFSET + 2]))
    live_vis = LiveVis(viz, mj_model, mj_data, theta, live, st_lock, timeout)

    mj_data_plan = mujoco.MjData(mj_model)        # planner-only state (main thread)
    mj_data_plan.qpos[:] = mj_data.qpos
    mj_data_plan.ctrl[:] = mj_data.ctrl

    q_prev = np.array(mj_data.qpos[:7], float)   # last commanded arm pose
    threading.Thread(target=track_tee, args=(optitrack_client, live, st_lock, stop_event),
                     daemon=True).start()  # begin mocap
    live_vis.start()

    try:
        while live_vis.is_running():
            # trial < 0: manual/exploratory run (no --trial) — no timeout either.
            if trial >= 0 and time.time() - t_start > timeout:
                print(f"\n✗ Timeout after {timeout:.0f}s — trial abandoned.")
                rec["timeout"] = True
                break
            rec["n_replans"] += 1

            # 1. Latest tracked tee pose (from the tracker thread). A pose that
            #    is missing or older than TRACK_STALE_AGE counts as stale, the
            #    same contamination signal as a missed detection before.
            with st_lock:
                xy = live.xy.copy()
                yaw = live.yaw
                stale = live.is_stale(TRACK_STALE_AGE)
            if stale:
                consec_misses += 1
                rec["n_stale_replans"] += 1
                rec["max_consec_misses"] = max(rec["max_consec_misses"], consec_misses)
                if consec_misses == TRACK_STALE_MISSES:
                    rec["n_track_dropouts"] += 1
                if consec_misses >= TRACK_STALE_MISSES:
                    print(f"[track] WARNING: tee pose stale for "
                          f"{consec_misses} consecutive replans "
                          f"({rec['n_track_dropouts']} dropouts this trial)")
            else:
                if consec_misses >= TRACK_STALE_MISSES:
                    print(f"[track] reacquired after {consec_misses} stale replans")
                consec_misses = 0

            dist_to_goal  = float(np.linalg.norm(xy - np.array(theta.goal_xy)))
            angle_err     = float(abs(((yaw - float(theta.goal_angle) + np.pi) % (2 * np.pi)) - np.pi))
            rec["final_pos_err"]   = round(dist_to_goal, 4)
            rec["final_angle_err"] = round(angle_err, 4)
            # Live progress on one rewritten line (goal tolerances in brackets).
            print(f"\r  [{time.time() - t_start:5.1f}s] "
                  f"dist {dist_to_goal * 100:5.1f}cm [<{SUCCESS_POS_RADIUS * 100:.0f}]  "
                  f"angle {np.degrees(angle_err):5.1f}° [<{np.degrees(SUCCESS_ANGLE_TOL):.0f}]  "
                  f"replan {rec['n_replans']}   ", end="", flush=True)
            if dist_to_goal < SUCCESS_POS_RADIUS and angle_err < SUCCESS_ANGLE_TOL:
                if not rec["success"]:
                    print(f"\n✓ Goal reached (dist={dist_to_goal*100:.1f}cm, "
                          f"angle={np.degrees(angle_err):.1f}°).")
                rec["success"] = True
                if trial >= 0:
                    break

            # 2. Run PFMP on the planner-private MjData (tee = live pose, arm =
            #    last commanded pose), so the render thread never races us.
            mj_data_plan.qpos[:7] = q_prev
            mj_data_plan.ctrl[:7] = q_prev   # rollout ramp starts from x.ctrl
            mj_data_plan.qpos[T_QPOS_OFFSET]     = float(xy[0])
            mj_data_plan.qpos[T_QPOS_OFFSET + 1] = float(xy[1])
            mj_data_plan.qpos[T_QPOS_OFFSET + 2] = float(yaw)
            if not np.all(np.isfinite(mj_data_plan.qpos)):
                raise RuntimeError(
                    f"NaN/Inf in mj_data_plan (replan {rec['n_replans']}, method={method})")
            mujoco.mj_forward(mj_model, mj_data_plan)
            key, subkey = jax.random.split(key)
            mjx_data = mjx.put_data(mj_model, mj_data_plan)
            if cfg.is_dial:
                # DIAL runs its own annealed re-sampling loop over one replan.
                # A higher-fidelity solve on the trial's first replan only,
                # matching canonical dial-mpc's Ndiffuse_init warm-up.
                theta = cfg.dial_step(subkey, mjx_data, theta,
                                      n=DIAL_N_INIT if rec["n_replans"] == 1 else DIAL_N)
            else:
                trajs, costs = cfg.replan(subkey, mjx_data, theta)
                theta, _ = cfg.update(theta, trajs, costs)
            if not np.all(np.isfinite(np.array(theta.mean))):
                raise RuntimeError(
                    f"NaN in theta.mean after update (replan {rec['n_replans']}, "
                    f"method={method}). A control left the reachable sector.")

            # 3. Execute one knot before replanning, projecting in-bounds because
            #    some updates can land in infeasible regions.
            next_wp = np.asarray(project_reachable(jnp.asarray(theta.mean[1]), theta))

            # q_target also backstops the seed refinement below if ik_solve on
            # the realized ee_xy fails, so it's needed regardless of viz.
            q_target = ik_solve(next_wp, q_prev)
            q_target = np.asarray(q_target if q_target is not None else q_prev, dtype=float)

            if live_vis.enabled:
                hold     = jnp.broadcast_to(jnp.asarray(next_wp), (ctrl.num_knots,) + next_wp.shape)
                pred_q   = np.asarray(ctrl.predict(mjx_data, hold, theta).qpos)
                arm_traj = pred_q[:, :7]                                 # (dense, 7)
                tee_traj = pred_q[:, T_QPOS_OFFSET:T_QPOS_OFFSET + 3]    # (dense, 3)
                glide_dur = (len(arm_traj) - 1) * float(ctrl.dt)
            else:
                # Match franky's velocity-limited move time; nothing renders
                # this glide headless, but glide_dur is written below regardless.
                arm_traj = tee_traj = None
                glide_dur = float(np.max(np.abs(q_target - q_prev) /
                                         (np.asarray(FR3_DQ_MAX) * VELOCITY_SCALE)))
            new_mean = jnp.concatenate([theta.mean[1:], theta.mean[-1:]])
            theta = theta._replace(mean=new_mean)
            live_vis.update(theta, glide_dur, arm_traj, tee_traj)

            try:
                reply = robot_planner.send_command(Command(xy=next_wp.tolist()))
                actual_xy = reply.get("actual_xy")
                ee_xy = np.array(actual_xy) if actual_xy else next_wp
                # Refine the planning seed with where the arm actually landed.
                q_cmd = ik_solve(ee_xy, q_prev)
                q_prev = np.asarray(q_cmd if q_cmd is not None else q_target, dtype=float)
            except Exception as e:
                print(f"[planner] command failed: {e}")
                rec["n_cmd_failures"] += 1           # contamination flag
    finally:
        stop_event.set()
        live_vis.close()

    rec["duration_s"] = round(time.time() - t_start, 2)
    rec["realized_final"] = [round(float(xy[0]), 4), round(float(xy[1]), 4), round(float(yaw), 4)]
    # trial < 0 means --trial was never passed (manual/exploratory run) — don't
    # pollute the trial log with it.
    logged = bool(log_path) and trial >= 0
    if logged:
        with open(log_path, "a") as f:
            f.write(json.dumps(rec) + "\n")
    print(f"\n[log] trial {rec['trial']} method={rec['method']} success={rec['success']} "
          f"pos_err={rec['final_pos_err']} angle_err={rec['final_angle_err']} "
          f"placement_err={rec['placement_err']} "
          f"aborts={rec['n_cmd_failures']} "
          f"dropouts={rec['n_track_dropouts']} (max {rec['max_consec_misses']} consec, "
          f"{rec['n_stale_replans']}/{rec['n_replans']} stale replans)"
          + (f" -> {log_path}" if logged else " (not saved: no --trial)"))

    # Planning finished — return the arm to home.
    print("Returning to home…")
    robot_planner.lifecycle("home")



#################################################################

def main():
    env = load_env()
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
    rt.add_argument("--mocap-ip",     default=env["mocap_ip"], help="Motive IP")
    rt.add_argument("--local-ip",     default=env["local_ip"], help="Local IP")
    rt.add_argument("--controller-ip", default=env["controller_ip"], help="Controller host")
    rt.add_argument("--cmd-port",    type=int, default=5555)
    rt.add_argument("--timeout",     type=float, default=TRIAL_TIMEOUT_S,
                    help="seconds before abandoning a trial (default 300; use ~120 for stuck-prone baselines)")
    rt.add_argument("--method",         default="sis",
                    choices=["sir", "sis", "ps", "mppi", "biased_mppi", "cem", "biased_cem", "dial"])
    rt.add_argument("--temperature",    type=float, default=1.0, help="MPPI temperature")
    rt.add_argument("--viz",            action="store_true",
                    help="launch the MuJoCo viewer: orange simulated tee follows the "
                         "plan's predicted push, blue mocap body shows the live tee. "
                         "Omit to run fully headless (no viewer). Demo only.")
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
    # Pin the annulus cost center to the arm's actual reachable midpoint, so the
    # radial shaping tracks the robot geometry instead of a hand-picked radius.
    global TEE_ANNULUS_CENTER
    TEE_ANNULUS_CENTER = reachable_r_center(ik_cache)

    # With --trial set, tee_problem_gen sets (start, goal) and displays
    # an interface for user to position the tee
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

    # Initial end-effector position
    init_xy = jnp.array(args.init_xy)
    q0 = ik_solve(np.asarray(args.init_xy, float), np.asarray(HOME_Q))
    if q0 is None:
        raise RuntimeError(f"init xy {np.asarray(args.init_xy)} is unreachable")
    mj_data.qpos[:7] = np.array(q0)
    mj_data.ctrl[:7] = np.array(q0)

    gripper_site_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SITE, "gripper")
    mujoco.mj_fwdPosition(mj_model, mj_data)
    r_xy = jnp.array(mj_data.site_xpos[gripper_site_id, :2])

    t_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "t_block")
    t_world_xy = mj_data.xpos[t_id, :2]
    t_qpos_xy  = mj_data.qpos[T_QPOS_OFFSET:T_QPOS_OFFSET + 2]
    print(f"T world xy (xpos):  {t_world_xy}")
    print(f"T qpos[7:9]:        {t_qpos_xy}")
    mj_data.qpos[T_QPOS_OFFSET]     = init_x
    mj_data.qpos[T_QPOS_OFFSET + 1] = init_y
    mj_data.qpos[T_QPOS_OFFSET + 2] = init_yaw
    mujoco.mj_forward(mj_model, mj_data)

    t_xy = jnp.array(mj_data.qpos[T_QPOS_OFFSET:T_QPOS_OFFSET + 2])
    alphas = jnp.linspace(0.0, 1.0, NUM_KNOTS)
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
        stdev=0.025,
        gripper_site_id=gripper_site_id,
    )

    _common = dict(
        horizon=HORIZON, num_knots=NUM_KNOTS, particles=PARTICLES,
        dt=mj_model.opt.timestep, ess_resample_thresh=0.5,
        interp="zoh",
    )
    ctrl = pfmp.PFMP(dynamics, running_cost, terminal_cost, init_control, proposal, **_common)
    ctrl_mppi = pfmp.PFMP(dynamics, running_cost, terminal_cost, init_control, mppi_proposal, **_common)

    if args.algorithm == "realtime":
        from new_natnet_client.Client import NatNetClient, NatNetParams
        from pub_sub.planner import PlannerClient

        if use_spec:
            import subprocess
            print("Opening placement view — position the block on the ghost, "
                  "then close the window to start the trial.")
            subprocess.run(
                [sys.executable, os.path.join(os.path.dirname(__file__), "tee_problem_gen.py"),
                 "--mocap-ip", str(args.mocap_ip), "--local-ip", str(args.local_ip),
                 "--seed", str(args.seed), "--trial", str(trial)],
                check=False)

        # Goal already resolved from the problem spec above.
        theta = theta._replace(
            goal_xy=jnp.array([goal_x, goal_y]),
            goal_angle=jnp.array(goal_yaw),
        )

        robot_planner = PlannerClient(
            command_addr=f"tcp://{args.controller_ip}:{args.cmd_port}",
            command_timeout_ms=30000,
        )
        params = NatNetParams(server_address=args.mocap_ip, local_ip_address=args.local_ip,
                              use_multicast=False)
        print(f"Connecting to Motive at {args.mocap_ip}...")
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
                    alphas = jnp.linspace(0.0, 1.0, NUM_KNOTS)
                    r_xy = jnp.array([init_xy[0], init_xy[1]])
                    theta = theta._replace(
                        mean=(1.0 - alphas[:, None]) * r_xy + alphas[:, None] * t_xy)
                    print(f"  tee at ({xy[0]:+.3f}, {xy[1]:+.3f}, {np.degrees(yaw):+.1f}°)")
                    break

            try:
                rt_method = getattr(args, "method", "sir")
                temperature = args.temperature if rt_method in ("mppi", "biased_mppi") else None
                _custom_proposal = rt_method in ("sir", "sis", "biased_mppi", "biased_cem")
                active_ctrl = ctrl if _custom_proposal else ctrl_mppi
                run_realtime(active_ctrl, theta, mj_model, mj_data, ik_cache,
                             optitrack_client, robot_planner,
                             method=rt_method, temperature=temperature,
                             log_path=args.log_path,
                             trial=(trial if trial is not None else -1), seed=args.seed,
                             spec_init=((spec_init_xy[0], spec_init_xy[1], spec_init_yaw)
                                        if use_spec else None),
                             timeout=args.timeout, viz=args.viz)
            finally:
                robot_planner.close()
        os._exit(0)
    else:
        from pfmp.algorithm import cem_update, mppi_update

        active_ctrl = ctrl_mppi if args.algorithm in ("mppi", "ps", "cem", "dial") else ctrl
        update_fn = None

        if args.algorithm == "cem":
            num_elites = args.num_elites
            update_fn = lambda th, tr, co: cem_update(th, tr, co,
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
                    sigma = float(np.mean([dial_sigma(i, h, th.stdev, n_inner, NUM_KNOTS, beta1, beta2)
                                           for h in range(NUM_KNOTS)]))
                    total = jnp.sum(co, axis=1)
                    temp = float(jnp.maximum(jnp.std(total), 1e-6)) * temperature
                    theta_out, _ = mppi_update(theta_out, tr, co, temp)
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
            goal_marker_fn=draw_goal_and_plan,
            update_fn=update_fn,
        )


if __name__ == "__main__":
    main()