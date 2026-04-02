"""
push_t_real.py — PFMP push-T on the real FR3, no perception.

T-block state lives in MuJoCo sim.  The real robot executes IK-cache joint
targets while the sim is stepped forward using the actual EE position as the
robot state — closing the loop on the robot side while keeping the T estimate
open-loop.

When perception (UKF + AprilTag) is ready, swap `sim_t_state()` for the UKF
estimate and nothing else changes.

Usage:
    python examples/fr3/push_t_real.py --dry-run            # sim only, no robot
    python examples/fr3/push_t_real.py --ip 172.16.0.2
    python examples/fr3/push_t_real.py --ip 172.16.0.2 --goal-xy 0.4 0.1
"""

import argparse
import os
import time
import pickle
from typing import NamedTuple, Tuple

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from mujoco import mjx

import pfmp
from pfmp.interpolate import Jet2

from ik_cache import (
    IKCache, ik_from_cache, get_or_build_cache,
    WORKSPACE_XY_MIN, WORKSPACE_XY_MAX, IK_CACHE_RESOLUTION, EE_HEIGHT,
    HOME_Q,
)

# ── Config ────────────────────────────────────────────────────────────────────

SCENE_XML_PATH  = os.path.join(os.path.dirname(__file__), "..", "..", "models", "mjx_t_push.xml")
T_QPOS_OFFSET   = 7          # T-block free-joint offset in MuJoCo qpos
ROBOT_IP        = "172.16.0.2"
VELOCITY_SCALE  = 0.08

HORIZON         = 3.0
NUM_KNOTS       = 6
PARTICLES       = 2048
ESS_THRESH      = 0.5
REPLAN_HZ       = 2.0        # replanning frequency (Hz)
MAX_REPLANS     = 40

POS_WEIGHT      = 3.0
ANGLE_WEIGHT    = 0.5
U_STDEV         = 0.05
DU_STDEV        = 0.1

MAX_JUMP_DEG    = 20.0


# ── Params ────────────────────────────────────────────────────────────────────

class PushParams(NamedTuple):
    mean:       jax.Array
    mjx_model:  mjx.Model
    ik_cache:   IKCache
    goal_xy:    jax.Array
    goal_angle: float
    pos_weight: float
    angle_weight: float
    u_stdev:    float
    du_stdev:   float


# ── Task functions (shared with pushT.py) ─────────────────────────────────────

@jax.jit
def _quat_yaw(q):
    w, x, y, z = q
    return jnp.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


@jax.jit
def _angle_wrap(a):
    return (a + jnp.pi) % (2.0 * jnp.pi) - jnp.pi


def dynamics(x: mjx.Data, u: Jet2, th: PushParams, t: float):
    target_qpos = ik_from_cache(u.u, th)
    x_next = mjx.step(th.mjx_model, x.replace(ctrl=target_qpos))
    return x_next, u.u


def running_cost(x: mjx.Data, u: Jet2, th: PushParams, t: float) -> jax.Array:
    t_qpos  = x.qpos[T_QPOS_OFFSET:T_QPOS_OFFSET + 7]
    err_pos = jnp.linalg.norm(t_qpos[:2] - th.goal_xy)
    err_ang = jnp.abs(_angle_wrap(_quat_yaw(t_qpos[3:7]) - th.goal_angle))
    return th.pos_weight * err_pos ** 2 + th.angle_weight * err_ang ** 2


def terminal_cost(x: mjx.Data, th: PushParams) -> jax.Array:
    return 10.0 * running_cost(x, Jet2(u=th.goal_xy, du=jnp.zeros(2)), th, 0.0)


def init_control(key, x: mjx.Data, th: PushParams) -> Jet2:
    k1, k2 = jax.random.split(key)
    t_xy = x.qpos[T_QPOS_OFFSET:T_QPOS_OFFSET + 2]
    return Jet2(u=t_xy + jax.random.normal(k1, (2,)) * th.u_stdev,
                du=jax.random.normal(k2, (2,)) * th.du_stdev)


def proposal(key, x: mjx.Data, u: Jet2, th: PushParams, t: float) -> Jet2:
    k1, k2 = jax.random.split(key)
    return Jet2(u=u.u + jax.random.normal(k1, (2,)) * th.u_stdev,
                du=u.du + jax.random.normal(k2, (2,)) * th.du_stdev)


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_scene(t_xy: np.ndarray, t_angle: float):
    mj_model = mujoco.MjModel.from_xml_path(SCENE_XML_PATH)
    mj_data  = mujoco.MjData(mj_model)
    key_id   = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_KEY, "home")
    if key_id >= 0:
        mujoco.mj_resetDataKeyframe(mj_model, mj_data, key_id)
    # Set T-block initial pose
    mj_data.qpos[T_QPOS_OFFSET:T_QPOS_OFFSET + 2] = t_xy
    mj_data.qpos[T_QPOS_OFFSET + 3]               = np.cos(t_angle / 2)  # quat w
    mj_data.qpos[T_QPOS_OFFSET + 6]               = np.sin(t_angle / 2)  # quat z
    mujoco.mj_forward(mj_model, mj_data)
    return mj_model, mj_data


def check_jump(q_prev: np.ndarray, q_next: np.ndarray) -> bool:
    diff = np.abs((q_next - q_prev + np.pi) % (2 * np.pi) - np.pi)
    worst = float(np.degrees(diff.max()))
    if worst > MAX_JUMP_DEG:
        print(f"  [WARN] Joint jump {worst:.1f}° exceeds limit — skipping waypoint")
        return False
    return True


def ee_xy_from_robot(robot) -> np.ndarray:
    """Read current EE xy from franky robot state."""
    state = robot.state
    pos   = np.array(state.O_T_EE)[12:15]   # 4x4 col-major → last col = translation
    return pos[:2]


def teach_t_pose(robot) -> Tuple[np.ndarray, float]:
    """Kinesthetic teach: user touches two cap-bar corners to recover T pose.

    Touch order:
      1. Right cap corner  → body frame [ 0.0675,  0.0900]
      2. Left  cap corner  → body frame [-0.0675,  0.0900]

    Returns (t_xy, t_angle) in robot base frame.
    """
    cap_local = np.array([0.0, 0.0900])  # midpoint of cap corners in body frame

    robot.set_guiding_mode(True)
    print("\n--- Kinesthetic teach ---")
    print("Orient the T with stem pointing away from you, cap bar across the top.")
    print("Touch the cap corner on YOUR RIGHT, then press Enter.")
    input()
    right_world = ee_xy_from_robot(robot)
    print(f"  Right corner: {right_world}")

    print("Touch the cap corner on YOUR LEFT, then press Enter.")
    input()
    left_world = ee_xy_from_robot(robot)
    print(f"  Left corner:  {left_world}")
    robot.set_guiding_mode(False)

    # Angle: vector from left→right in world matches left→right in body (along +x)
    vec_world = right_world - left_world
    t_angle   = float(np.arctan2(vec_world[1], vec_world[0]))

    # T origin: midpoint of the two corners, then subtract cap offset in body frame
    cap_world = (right_world + left_world) / 2.0
    c, s      = np.cos(t_angle), np.sin(t_angle)
    R         = np.array([[c, -s], [s, c]])
    t_xy      = cap_world - R @ cap_local

    print(f"  → T pose: xy={t_xy}  angle={np.degrees(t_angle):.1f}°")
    return t_xy, t_angle


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run",    action="store_true", help="Sim only, no robot")
    parser.add_argument("--ip",         default=ROBOT_IP)
    parser.add_argument("--speed",      type=float, default=VELOCITY_SCALE)
    parser.add_argument("--cache",      default=None)
    parser.add_argument("--goal-xy",    type=float, nargs=2, default=[0.40, 0.0])
    parser.add_argument("--goal-angle", type=float, default=0.0)
    parser.add_argument("--t-xy",       type=float, nargs=2, default=[0.25, 0.0],
                        help="Initial T-block position (metres, robot frame)")
    parser.add_argument("--t-angle",    type=float, default=0.0)
    parser.add_argument("--teach",      action="store_true",
                        help="Kinesthetic teach: touch two cap corners to set T pose")
    args = parser.parse_args()

    # ── IK cache ──────────────────────────────────────────────────────────────
    cache_path = args.cache or os.path.join(os.path.dirname(__file__), "..", "ik_cache.pkl")
    ik_cache   = get_or_build_cache(path=cache_path)

    # ── Robot (needed early for --teach) ──────────────────────────────────────
    robot = None
    if not args.dry_run:
        import franky
        print(f"\nConnecting to {args.ip}…")
        robot = franky.Robot(args.ip)
        robot.relative_dynamics_factor = args.speed
        robot.recover_from_errors()

    # ── T initial pose ────────────────────────────────────────────────────────
    if args.teach:
        if robot is None:
            parser.error("--teach requires a robot connection (remove --dry-run)")
        t_xy_0, t_angle_0 = teach_t_pose(robot)
    else:
        t_xy_0    = np.array(args.t_xy)
        t_angle_0 = args.t_angle
    print(f"T initial:  xy={t_xy_0}  angle={np.degrees(t_angle_0):.1f}°")
    print(f"Goal:       xy={args.goal_xy}  angle={np.degrees(args.goal_angle):.1f}°")

    mj_model, mj_data = load_scene(t_xy_0, t_angle_0)
    mjx_model = mjx.put_model(mj_model)
    mjx_data  = mjx.put_data(mj_model, mj_data)

    # ── PFMP ──────────────────────────────────────────────────────────────────
    theta = PushParams(
        mean=jnp.zeros((NUM_KNOTS, 2)),
        mjx_model=mjx_model,
        ik_cache=ik_cache,
        goal_xy=jnp.array(args.goal_xy),
        goal_angle=float(args.goal_angle),
        pos_weight=POS_WEIGHT,
        angle_weight=ANGLE_WEIGHT,
        u_stdev=U_STDEV,
        du_stdev=DU_STDEV,
    )

    ctrl = pfmp.PFMP(
        dynamics, running_cost, terminal_cost, init_control, proposal,
        horizon=HORIZON, num_knots=NUM_KNOTS, particles=PARTICLES,
        dt=mj_model.opt.timestep, ess_resample_thresh=ESS_THRESH,
    )

    print("JIT warm-up…", end=" ", flush=True)
    key = jax.random.key(0)
    key, sk = jax.random.split(key)
    _ = ctrl.run_sis(sk, mjx_data, theta)
    _ = ctrl.run_sis(sk, mjx_data, theta)
    print("done")

    # ── Move to home ──────────────────────────────────────────────────────────
    if robot is not None:
        print("Moving to HOME_Q…")
        robot.move(franky.JointMotion(HOME_Q.tolist()))
        input("At home. Press Enter to start pushing… ")

    # ── Control loop ──────────────────────────────────────────────────────────
    step_dt  = 1.0 / REPLAN_HZ
    q_prev   = np.array(HOME_Q)

    for replan_i in range(MAX_REPLANS):
        # -- If real robot: read EE position and inject into sim ---------------
        if robot is not None:
            ee_xy = ee_xy_from_robot(robot)
            # Set robot qpos in sim to match real EE
            real_q = ik_from_cache(jnp.array(ee_xy), theta)
            mj_data.qpos[:7] = np.array(real_q)
            mujoco.mj_forward(mj_model, mj_data)
            mjx_data = mjx.put_data(mj_model, mj_data)

        # -- Plan --------------------------------------------------------------
        key, sk = jax.random.split(key)
        trajs, costs = ctrl.run_sis(sk, mjx_data, theta)
        theta, _     = pfmp.algorithm.ps_update(theta, trajs, costs)

        best_xy = np.array(theta.mean[0])   # first knot = immediate waypoint
        best_q  = np.array(ik_from_cache(jnp.array(best_xy), theta))

        t_qpos = np.array(mjx_data.qpos)[T_QPOS_OFFSET:T_QPOS_OFFSET + 7]
        err_pos = float(np.linalg.norm(t_qpos[:2] - args.goal_xy))
        print(f"[{replan_i+1:3d}/{MAX_REPLANS}] "
              f"T=({t_qpos[0]:.3f},{t_qpos[1]:.3f})  "
              f"target_xy=({best_xy[0]:.3f},{best_xy[1]:.3f})  "
              f"pos_err={err_pos:.3f}m")

        # -- Execute on robot --------------------------------------------------
        if robot is not None and check_jump(q_prev, best_q):
            robot.move(franky.JointMotion(best_q.tolist()))
            q_prev = best_q
        elif args.dry_run:
            time.sleep(step_dt)

        # -- Step sim forward --------------------------------------------------
        target_q = ik_from_cache(jnp.array(best_xy), theta)
        n_steps  = max(1, int(round(step_dt / mj_model.opt.timestep)))
        for _ in range(n_steps):
            mj_data.ctrl[:7] = np.array(target_q)
            mujoco.mj_step(mj_model, mj_data)
        mjx_data = mjx.put_data(mj_model, mj_data)

        if err_pos < 0.02:
            print("Goal reached.")
            break

    if robot is not None:
        print("Returning to HOME_Q…")
        robot.move(franky.JointMotion(HOME_Q.tolist()))
    print("Done.")


if __name__ == "__main__":
    main()
