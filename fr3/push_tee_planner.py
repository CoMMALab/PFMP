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
    IKCache, ik_from_cache, reachable_frac, get_or_build_cache,
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


HORIZON   = 2.0
NUM_KNOTS = 5
PARTICLES = 2048

VIZ_INTERP_STEPS = 6        # sub-frames to glide the sim arm between replan poses
VIZ_INTERP_DT    = 0.01     # seconds per sub-frame (~60ms glide at defaults)

SUCCESS_POS_RADIUS = 0.05   # metres
SUCCESS_ANGLE_TOL  = 0.35   # radians (~20°)

PROPOSAL_MIX    = 0.4   # directed toward tee
ORBITAL_MIX     = 0.2   # orbit around tee at push radius
STEP_LEN_VEL    = 0.05
PUSH_OVERSHOOT  = 1.0
PUSH_RADIUS     = 0.16  # just outside tee cap half-extent
ORBITAL_Y_OFFSET = 0.04  # metres up the tee stem for the orbital centre


def dynamics(x: mjx.Data, u: jax.Array, th: PushParams, t: float) -> Tuple[mjx.Data, jax.Array]:
    """Step MuJoCo with IK-resolved joint targets for the given EE waypoint."""
    target_qpos = ik_from_cache(u, th.ik_cache)
    ctrl = x.ctrl.at[:7].set(target_qpos)
    x_next = mjx.step(th.mjx_model, x.replace(ctrl=ctrl))
    return x_next, u


TEE_ANNULUS_CENTER = (0.25 + 0.78) / 2   # ideal radial distance from base
TEE_ANNULUS_WEIGHT = 20.0

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


def init_control(key: jax.Array, _x: mjx.Data, th: PushParams) -> jax.Array:
    """Initialize at current EE position (from warm-started mean) with small noise."""
    r_xy = th.mean[0]
    return r_xy + jax.random.normal(key, (2,)) * th.u_stdev * 0.1


def mppi_proposal(key: jax.Array, _x: mjx.Data, _u: jax.Array, th: PushParams, t: float) -> jax.Array:
    """Gaussian proposal centred on the mean knot at time t (for MPPI baseline)."""
    step_len = HORIZON / (NUM_KNOTS - 1)
    idx = jnp.clip(jnp.floor(t / step_len).astype(jnp.int32) + 1, 0, NUM_KNOTS - 1)
    return th.mean[idx] + jax.random.normal(key, (2,)) * th.u_stdev


N_REJECT_CANDIDATES = 8      # mask-aware rejection: candidates drawn per proposal call
REACHABLE_THRESH    = 0.75   # min bilinear reachability to accept a candidate


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
    """Three-way mixture with mask-aware rejection: draw N candidates from the
    mixture, keep the first one that lands in the cache's reachable component (so
    the IK-resolved joint motion tracks the intended EE path instead of arcing
    across a wrist-flip / unreachable cell). If none clear the threshold, keep the
    most-reachable candidate and clamp it into the reachable annulus as a fallback.
    """
    keys = jax.random.split(key, N_REJECT_CANDIDATES)
    cands = jax.vmap(lambda k: _sample_mixture(k, x, u, th))(keys)       # (N, 2)
    fracs = jax.vmap(lambda c: reachable_frac(c, th.ik_cache))(cands)    # (N,)

    ok  = fracs >= REACHABLE_THRESH
    # First accepted candidate, else the most-reachable one.
    idx = jnp.where(jnp.any(ok), jnp.argmax(ok), jnp.argmax(fracs))
    sample = cands[idx]

    # Fallback safety: if even the best candidate is unreachable, project into the
    # reachable annulus so a degenerate query never reaches the IK cache.
    accepted = fracs[idx] >= REACHABLE_THRESH
    EE_INNER, EE_OUTER = 0.30, 0.775
    d = jnp.linalg.norm(sample) + 1e-7
    clamped = sample
    clamped = jnp.where(d < EE_INNER, clamped * (EE_INNER / d), clamped)
    clamped = jnp.where(d > EE_OUTER, clamped * (EE_OUTER / d), clamped)
    return jnp.where(accepted, sample, clamped)


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
                 log_path=None, trial=0, seed=0):
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
    rec = {
        "ts": datetime.datetime.now().isoformat(timespec="seconds"),
        "trial": int(trial), "seed": int(seed), "method": method,
        "goal_xy": [float(theta.goal_xy[0]), float(theta.goal_xy[1])],
        "goal_yaw": float(theta.goal_angle),
        "init_tee": [float(mj_data.qpos[T_QPOS_OFFSET]),
                     float(mj_data.qpos[T_QPOS_OFFSET + 1]),
                     float(mj_data.qpos[T_QPOS_OFFSET + 2])],
        "n_replans": 0, "n_cmd_failures": 0, "success": False,
        "final_pos_err": None, "final_angle_err": None, "duration_s": None,
    }

    def _write_record():
        rec["duration_s"] = round(time.time() - t_start, 2)
        if log_path:
            with open(log_path, "a") as f:
                f.write(json.dumps(rec) + "\n")
        print(f"\n[log] trial {rec['trial']} success={rec['success']} "
              f"pos_err={rec['final_pos_err']} angle_err={rec['final_angle_err']} "
              f"aborts={rec['n_cmd_failures']} -> {log_path}")

    with mjviewer.launch_passive(mj_model, mj_data) as viewer:
        print("Realtime loop running. Close viewer to stop (logs an abort).")
        while viewer.is_running():
            rec["n_replans"] += 1
            # 1. Get real tee pose from Optitrack
            rbs = []
            for frame in optitrack_client.MoCap(timeout=None):
                rbs = [rb for rb in frame.rigid_body_data.rigid_bodies if rb.tracking]
                break
            if rbs:
                xy, yaw = tee_pose(rbs[0])
                mj_data.qpos[T_QPOS_OFFSET]     = float(xy[0])
                mj_data.qpos[T_QPOS_OFFSET + 1] = float(xy[1])
                mj_data.qpos[T_QPOS_OFFSET + 2] = float(yaw)
                mujoco.mj_forward(mj_model, mj_data)

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
                print("[warn] NaN/Inf in mj_data — resetting ctrl to zero")
                mj_data.ctrl[:] = 0.0
                mujoco.mj_forward(mj_model, mj_data)
            mjx_data = mjx.put_data(mj_model, mj_data)
            trajs, costs = replan(subkey, mjx_data, theta)
            theta, _ = update(theta, trajs, costs)
            if not np.all(np.isfinite(np.array(theta.mean))):
                print("[warn] NaN in theta.mean after update — resetting to zeros")
                theta = theta._replace(mean=jnp.zeros_like(theta.mean))

            # 3. Execute one knot before replanning
            q_prev  = np.array(mj_data.qpos[:7]).copy()   # last commanded arm pose
            next_wp = np.clip(np.array(theta.mean[1]),
                              np.array(WORKSPACE_XY_MIN), np.array(WORKSPACE_XY_MAX))
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
    parser.add_argument('--goal-x',   type=float, default=0.45)
    parser.add_argument('--goal-y',   type=float, default=0.15)
    parser.add_argument('--goal-yaw', type=float, default=float(4*np.pi/3))
    parser.add_argument('--log-path', type=str,   default='realtime_log.jsonl',
                        help='JSONL per-trial log for realtime runs (pass before the subcommand)')
    parser.add_argument('--seed',     type=int,   default=42)
    parser.add_argument('--sample-x', type=float, nargs=2, default=[0.30, 0.55])
    parser.add_argument('--sample-y', type=float, nargs=2, default=[-0.25, 0.25])

    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument('--trial', type=int, default=0,
                        help='Trial index — matches view_start_region.py seed')

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

    # Task configuration
    horizon   = HORIZON
    num_knots = NUM_KNOTS
    particles = PARTICLES

    # Initial end-effector position
    init_xy = jnp.array(args.init_xy)
    q0 = ik_from_cache(init_xy, ik_cache)
    mj_data.qpos[:7] = np.array(q0)

    # Initialize mean trajectory: linear interp from EE to T-block position.
    gripper_site_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SITE, "gripper")
    mujoco.mj_fwdPosition(mj_model, mj_data)
    r_xy = jnp.array(mj_data.site_xpos[gripper_site_id, :2])

    t_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "t_block")
    t_world_xy = mj_data.xpos[t_id, :2]
    t_qpos_xy  = mj_data.qpos[T_QPOS_OFFSET:T_QPOS_OFFSET + 2]
    print(f"T world xy (xpos):  {t_world_xy}")
    print(f"T qpos[7:9]:        {t_qpos_xy}")
    # Sample initial T pose using same RNG as view_start_region.py
    trial = getattr(args, 'trial', 0)
    rng = np.random.default_rng([args.seed, trial])
    init_x   = rng.uniform(*args.sample_x)
    init_y   = rng.uniform(*args.sample_y)
    init_yaw = rng.uniform(-np.pi, np.pi)
    print(f"Trial {trial} (seed={args.seed}): T init=({init_x:.3f}, {init_y:.3f}, {np.degrees(init_yaw):.1f}°)")
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
        goal_xy=jnp.array([args.goal_x, args.goal_y]),
        goal_angle=float(args.goal_yaw),
        pos_weight=15.0,
        angle_weight=1.5,
        u_stdev=0.025,
        gripper_site_id=gripper_site_id,
    )

    _common = dict(
        horizon=horizon, num_knots=num_knots, particles=particles,
        dt=mj_model.opt.timestep, ess_resample_thresh=0.5,
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

        # Override goal from args
        goal_yaw = args.goal_yaw if args.goal_yaw is not None else float(theta.goal_angle)
        theta = theta._replace(
            goal_xy=jnp.array([args.goal_x, args.goal_y]),
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
                             trial=getattr(args, "trial", 0), seed=args.seed)
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
            ik_interp=lambda target, theta, mjx_data: ik_from_cache(target, ik_cache),
            frequency=0.5,
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