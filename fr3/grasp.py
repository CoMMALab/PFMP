"""Cube grasp-and-lift task for the FR3, defined for the PFMP kernel.

The robot samples control sequences that pick a cube off the table and lift it
`LIFT_HEIGHT` metres. This module is the single source of truth for the task
(dynamics, cost, proposal); `fr3/sim_grasp.py` imports it rather than
re-deriving, the same arrangement push-T uses between planner and simulator.

Control space. A knot is `u = (theta, r, z, grip)`:

    theta   base rotation, rad        r  radius from the base, m
    z       fingertip height, m       grip  commanded finger opening, m

which is go_fr3's cylindrical chart, so `fr3/go_fr3/ik.py` maps a knot straight
to joint angles with no numerical IK and no redundancy resolution. Sampling in
these coordinates matters: box bounds on (theta, r, z) are very nearly the
reachable set, so particles land on feasible poses instead of being rejected.
The wrist angle q7 is not sampled; it is solved to align the jaws with the
cube's yaw, which a cube's quarter-turn symmetry makes unambiguous.

The gripper is sampled like any other dimension. What makes the task tractable
is the proposal, not a scripted open/close: `proposal` reads which phase of the
grasp the state is in and anchors its draws on the matching cube-relative pose.
That is the same object-centric-proposal axis as push-T's orbital branch.
"""

import os
import sys

import jax
import jax.numpy as jnp
from typing import NamedTuple, Tuple

from mujoco import mjx

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "go_fr3"))

import ik   # go_fr3's analytic FR3 kinematics, traced here via solve_clamped


SCENE_XML_PATH = os.path.join(os.path.dirname(__file__), "..", "models",
                              "mjx_cube_grasp.xml")

# Cube free joint in the MuJoCo state: 7 arm joints + 2 finger joints precede it.
CUBE_QPOS_OFFSET = 9
FINGER_QPOS_OFFSET = 7        # the two finger slide joints

CUBE_HALF = 0.0225          # 45 mm cube, inside the hand's 80 mm opening
LIFT_HEIGHT = 0.50          # metres off the table, the task's whole point
TARGET_Z = CUBE_HALF + LIFT_HEIGHT

GRIP_OPEN, GRIP_CLOSED = 0.04, 0.0
GRIP_RATE = 0.1             # m/s of commanded finger travel

# The 'gripper' site in models/mjx_panda_hand.xml is 0.1 m up the hand, while
# go_fr3's TCP is the fingertip at 0.1034 m. Measured against mujoco's own FK,
# not assumed; tests/test_grasp.py pins it.
SITE_Z_OFFSET = 0.0034

# Fingertips stop this far below the cube's centre, so the pads straddle it
# rather than resting on the top face.
GRASP_Z_OFFSET = -0.010
PREGRASP_HEIGHT = 0.12      # metres above the cube for the approach pose


# ===============================
# Task Parameters
# ===============================


class GraspParams(NamedTuple):
    """Parameters for the cube grasp-and-lift task."""
    mean: jax.Array           # (num_knots, 4) control knots
    mjx_model: mjx.Model
    stdev: jax.Array          # scalar, (4,) per-dim, or (num_knots, 4) for CEM
    gripper_site_id: int
    lift_weight: float = 12.0
    reach_weight: float = 6.0
    ik_weight: float = 20.0


# ===============================
# PFMP Task Functions
# ===============================


HORIZON   = 8.0
# 6 intervals over the ~7 s a grasp takes at VELOCITY_SCALE. That is one interval
# per phase (approach, descend, close, lift) plus slack, because reaching the
# pre-grasp pose from home takes about 2 s, longer than a single interval, and
# the cube then has 0.5 m to travel. Without the slack every phase has to land
# first try and almost no particle completes the sequence.
NUM_KNOTS = 7
PARTICLES = 256

KNOT_INTERVAL = HORIZON / (NUM_KNOTS - 1)
RUN_INTERACTIVE_FREQUENCY = 1.0 / KNOT_INTERVAL

SUCCESS_LIFT_TOL = 0.05     # metres; "approximately 50 cm" either side

# Control bounds. theta/r/z are go_fr3's cylindrical limits narrowed to the
# table in front of the robot; z reaches above TARGET_Z so the lift is inside
# the box rather than pinned against it.
U_MIN = jnp.array([-1.20, 0.28, 0.004, GRIP_CLOSED])
U_MAX = jnp.array([ 1.20, 0.72, 0.640, GRIP_OPEN])

# Proposal mixture weights. The remainder is a random walk around the previous
# knot, which is what keeps the proposal from being a scripted grasp.
PHASE_MIX    = 0.55    # draw anchored on the current phase's cube-relative pose
DIRECTED_MIX = 0.20    # step from the previous knot toward that pose
STEP_LEN     = 0.10    # metres/radians of a directed step

# Phase predicates. Deliberately loose: they only steer the proposal, and the
# cost, not these thresholds, decides which particle wins.
NEAR_XY_TOL  = 0.030
NEAR_Z_TOL   = 0.030
CLOSED_CTRL  = 0.020   # commanded opening below this counts as closing
# The jaws cannot close past whatever is between them, so a shut command plus a
# finger gap this wide means the cube is in the grasp. 0.6x the cube's width:
# empty jaws close to ~0, a held cube pins them at ~0.0443.
JAWS_BLOCKED_MIN = 0.6 * 2 * CUBE_HALF

# Per-dimension sampling width: rad, m, m, m. Sized against the grasp's actual
# tolerance, which is what the first version of this task got wrong. The hand
# opens to 0.04 m on a 0.0225 m half-cube, leaving 0.0175 m of clearance per
# side, so a lateral 1-sigma of 0.04 m (the natural-looking number) misses the
# cube more often than it hits it. At r = 0.5 m, 0.02 rad of theta is 0.01 m.
STDEV = jnp.array([0.020, 0.012, 0.020, 0.008])

# Phase-dependent tightening of the informed branch. Approaching is coarse, so
# it explores; closing on the cube has centimetre tolerance, so it is precise.
# This is the point of an object-centric proposal: match the draw's width to the
# phase's tolerance instead of using one width everywhere.
SIGMA_SCALE_PREGRASP = 1.00
SIGMA_SCALE_DESCEND  = 0.40
SIGMA_SCALE_CLOSE    = 0.25
SIGMA_SCALE_LIFT     = 0.35

# Joint velocity ceiling, as the hardware controller runs it
# (fr3/push_tee_controller.py's VELOCITY_SCALE, not imported: that module pins
# JAX to CPU and needs franky at import time).
FR3_DQ_MAX     = jnp.array([2.62, 2.62, 2.62, 2.62, 5.26, 4.18, 5.26])   # rad/s
VELOCITY_SCALE = 0.2


def _yaw_from_quat(quat: jax.Array) -> jax.Array:
    """World z rotation of a (w, x, y, z) quaternion, as mujoco stores it."""
    w, x, y, z = quat[0], quat[1], quat[2], quat[3]
    return jnp.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def cube_pose(x: mjx.Data) -> Tuple[jax.Array, jax.Array]:
    """(xyz, yaw) of the cube."""
    q = x.qpos[CUBE_QPOS_OFFSET:CUBE_QPOS_OFFSET + 7]
    return q[:3], _yaw_from_quat(q[3:7])


def fingertip(x: mjx.Data, th: GraspParams) -> jax.Array:
    """Fingertip position, the point go_fr3's IK actually solves for.

    The 'gripper' site sits SITE_Z_OFFSET above it.
    """
    return x.site_xpos[th.gripper_site_id] - jnp.array([0.0, 0.0, SITE_Z_OFFSET])


def ik_target(u: jax.Array, x: mjx.Data) -> Tuple[jax.Array, jax.Array]:
    """Knot -> (joint angles, IK violation), with the jaws aligned to the cube.

    Shared by `dynamics` and `running_cost` so the pose the rollout executes and
    the pose the cost scores cannot diverge.
    """
    _cube_xyz, cube_yaw = cube_pose(x)
    q7 = ik.q7_for_jaw_yaw(u[0], cube_yaw)
    return ik.solve_clamped(u[0], u[1], u[2], q7)


def dynamics(x: mjx.Data, u: jax.Array, th: GraspParams, t: float) -> Tuple[mjx.Data, jax.Array]:
    """Step MuJoCo, ramping the joint and gripper references toward the knot.

    Both references move at bounded rate rather than snapping to the target, so
    the rollout scores the trajectory the velocity-limited robot actually runs.
    """
    dt     = th.mjx_model.opt.timestep
    dq_max = FR3_DQ_MAX * VELOCITY_SCALE * dt

    target_q, _violation = ik_target(u, x)
    delta = target_q - x.ctrl[:7]
    scale = jnp.minimum(1.0, jnp.min(dq_max / (jnp.abs(delta) + 1e-9)))
    ctrl  = x.ctrl.at[:7].set(x.ctrl[:7] + delta * scale)

    grip = jnp.clip(u[3], GRIP_CLOSED, GRIP_OPEN)
    dg   = jnp.clip(grip - x.ctrl[7], -GRIP_RATE * dt, GRIP_RATE * dt)
    ctrl = ctrl.at[7].set(x.ctrl[7] + dg)

    return mjx.step(th.mjx_model, x.replace(ctrl=ctrl)), u


def running_cost(x: mjx.Data, u: jax.Array, th: GraspParams, t: float) -> jax.Array:
    """Lift the cube to TARGET_Z, keep the fingers on it, stay reachable."""
    cube_xyz, _yaw = cube_pose(x)
    lift_err  = TARGET_Z - cube_xyz[2]
    reach_err = jnp.linalg.norm(fingertip(x, th) - cube_xyz)
    _q, violation = ik_target(u, x)
    return (th.lift_weight * lift_err ** 2
            + th.reach_weight * reach_err ** 2
            + th.ik_weight * violation)


def terminal_cost(x: mjx.Data, th: GraspParams) -> jax.Array:
    cube_xyz, _yaw = cube_pose(x)
    lift_err  = jnp.abs(TARGET_Z - cube_xyz[2])
    reach_err = jnp.linalg.norm(fingertip(x, th) - cube_xyz)
    return 10.0 * (th.lift_weight * lift_err + th.reach_weight * reach_err)


def _sigma_at(th: GraspParams, idx) -> jax.Array:
    """Sampling noise for knot `idx`.

    Three conventions, all static under jit so this costs nothing: a scalar, a
    per-dimension (4,) vector (the default, since theta is radians while r, z
    and grip are metres), and a per-knot (num_knots, 4) array, which is what CEM
    refits from its elites each replan.
    """
    ndim = jnp.ndim(th.stdev)
    if ndim == 0 or ndim == 1:
        return th.stdev
    return th.stdev[idx]


def _knot_index(t: float) -> jax.Array:
    """Knot whose interval contains time `t`."""
    return jnp.clip(jnp.floor(t / KNOT_INTERVAL).astype(jnp.int32) + 1, 0, NUM_KNOTS - 1)


def clip_control(u: jax.Array) -> jax.Array:
    """Snap a knot into the control box. The IK's violation term handles the
    corners of the box that fall outside the arm's actual reachable shell."""
    return jnp.clip(u, U_MIN, U_MAX)


def phase_anchor(x: mjx.Data, th: GraspParams) -> Tuple[jax.Array, jax.Array]:
    """The cube-relative knot the current phase of the grasp calls for, and how
    tightly to sample around it.

    Four poses, selected by where the hand is relative to the cube and whether
    it is closing:

        pre-grasp  above the cube, jaws open
        descend    down onto the cube, jaws open
        close      held at the cube, jaws shut
        lift       raised to TARGET_Z, jaws shut

    This reads the state, so a rollout advances through the phases on its own as
    the simulated hand arrives, closes and lifts. It anchors on the cube rather
    than on `theta.mean`, which keeps a particle useful when the mean has
    drifted somewhere it cannot grasp from.
    """
    cube_xyz, _yaw = cube_pose(x)
    c_theta = jnp.arctan2(cube_xyz[1], cube_xyz[0])
    c_r     = jnp.hypot(cube_xyz[0], cube_xyz[1])
    grasp_z = cube_xyz[2] + GRASP_Z_OFFSET

    tip   = fingertip(x, th)
    d_xy  = jnp.linalg.norm(tip[:2] - cube_xyz[:2])
    d_z   = jnp.abs(tip[2] - grasp_z)
    closing = x.ctrl[7] < CLOSED_CTRL

    at_cube = (d_xy < NEAR_XY_TOL) & (d_z < NEAR_Z_TOL)

    # `held` deliberately does NOT use at_cube. A grasped cube creeps downward
    # through the pads by a few mm/s, and keying the phase off that geometry made
    # the creep self-amplifying: once d_z passed NEAR_Z_TOL the proposal decided
    # it was not holding and confidently proposed GRIP_OPEN, dropping a cube it
    # was still gripping. The finger gap has no such feedback, because the jaws
    # stay blocked for exactly as long as the cube is actually between them.
    finger_gap = (x.qpos[FINGER_QPOS_OFFSET] + x.qpos[FINGER_QPOS_OFFSET + 1])
    held = closing & (finger_gap > JAWS_BLOCKED_MIN)

    pregrasp = jnp.array([c_theta, c_r, cube_xyz[2] + PREGRASP_HEIGHT, GRIP_OPEN])
    descend  = jnp.array([c_theta, c_r, grasp_z, GRIP_OPEN])
    close    = jnp.array([c_theta, c_r, grasp_z, GRIP_CLOSED])
    lift     = jnp.array([c_theta, c_r, TARGET_Z + GRASP_Z_OFFSET, GRIP_CLOSED])

    aligned = d_xy < NEAR_XY_TOL
    anchor = jnp.where(held, lift,
             jnp.where(at_cube, close,
             jnp.where(aligned, descend, pregrasp)))
    scale = jnp.where(held, SIGMA_SCALE_LIFT,
            jnp.where(at_cube, SIGMA_SCALE_CLOSE,
            jnp.where(aligned, SIGMA_SCALE_DESCEND, SIGMA_SCALE_PREGRASP)))
    return anchor, scale


def sample_mixture(key: jax.Array, u: jax.Array, anchor: jax.Array,
                   sigma: jax.Array, scale: jax.Array) -> jax.Array:
    """One draw from the three-way mixture: phase / directed / random walk.

    Anchored on `u`, the previous knot, so this is a proper q(u_k | u_{k-1}, x).
    The grip dimension is drawn like every other one; the phase branch only
    proposes a sensible value for it, it does not impose one.
    """
    key_mix, key_a, key_b, key_c = jax.random.split(key, 4)
    r = jax.random.uniform(key_mix)

    # Branch 1: the phase's cube-relative pose, jittered at the phase's width.
    phase = anchor + jax.random.normal(key_a, (4,)) * sigma * scale

    # Branch 2: a bounded step from the previous knot toward that pose.
    direction = anchor - u
    dist      = jnp.linalg.norm(direction) + 1e-7
    directed  = u + (direction / dist) * jnp.minimum(dist, STEP_LEN)
    directed  = directed + jax.random.normal(key_b, (4,)) * sigma * scale

    # Branch 3: random walk, the uninformed fallback. It respects `scale` too.
    # Leaving it unscaled is a trap: while the cube is held, sigma*2 on the grip
    # dimension reopens the jaws in ~10% of walk draws, which is ~10 open-jaw
    # knots per replan at 64 particles. A MAP update only has to prefer one of
    # them (cheap if the release lands late in the horizon, where the cube has
    # barely fallen by the terminal cost) and the cube is dropped in execution.
    walk = u + jax.random.normal(key_c, (4,)) * sigma * 2.0 * scale

    return jnp.where(r < PHASE_MIX, phase,
           jnp.where(r < PHASE_MIX + DIRECTED_MIX, directed, walk))


def proposal(key: jax.Array, x: mjx.Data, u: jax.Array, th: GraspParams,
             t: float, idx=None) -> jax.Array:
    """One draw from the grasp-aware mixture, clipped to the control box.

    `idx` overrides which knot's noise width to use. `init_control` passes 0,
    which `_knot_index` never returns for t >= 0; it only matters under CEM,
    whose stdev is per-knot.
    """
    idx = _knot_index(t) if idx is None else idx
    anchor, scale = phase_anchor(x, th)
    return clip_control(sample_mixture(key, u, anchor, _sigma_at(th, idx), scale))


def gaussian_proposal(key: jax.Array, _x: mjx.Data, _u: jax.Array,
                      th: GraspParams, t: float) -> jax.Array:
    """Uninformed proposal centred on the mean knot, for the PS/MPPI/CEM/DIAL
    baselines. This is the q axis's other corner: same update rules, no
    knowledge of where the cube is."""
    idx = _knot_index(t)
    return clip_control(th.mean[idx] + jax.random.normal(key, (4,)) * _sigma_at(th, idx))


def init_control(key: jax.Array, x: mjx.Data, th: GraspParams) -> jax.Array:
    """`proposal` at the first knot, seeded from the warm-started mean."""
    return proposal(key, x, th.mean[0], th, 0.0, idx=0)


def initial_mean(x: mjx.Data, th: GraspParams) -> jax.Array:
    """Warm-start plan: a straight ramp from the hand to a pose above the cube.

    Deliberately neutral, and it carries exactly the knowledge push-T's warm
    start does ("move toward the object") and no more: the gripper stays open at
    every knot and there is no descend/close/lift structure in it.

    That structure has to come from the proposal, which is the axis this task
    exists to measure. Seeding the mean with a scripted grasp instead would hand
    the uninformed gaussian baselines the answer, and `sir` beating `mppi` would
    then say nothing about proposals.
    """
    cube_xyz, _yaw = cube_pose(x)
    tip = fingertip(x, th)
    start = jnp.array([jnp.arctan2(tip[1], tip[0]),
                       jnp.hypot(tip[0], tip[1]), tip[2]])
    goal  = jnp.array([jnp.arctan2(cube_xyz[1], cube_xyz[0]),
                       jnp.hypot(cube_xyz[0], cube_xyz[1]),
                       cube_xyz[2] + PREGRASP_HEIGHT])
    alphas = jnp.linspace(0.0, 1.0, NUM_KNOTS)[:, None]
    ramp = (1.0 - alphas) * start + alphas * goal
    return jnp.concatenate([ramp, jnp.full((NUM_KNOTS, 1), GRIP_OPEN)], axis=1)


def load_scene(xml_path: str = SCENE_XML_PATH):
    """MjModel/MjData for the grasp scene, reset to its home keyframe."""
    import mujoco
    mj_model = mujoco.MjModel.from_xml_path(os.path.abspath(xml_path))
    mj_data = mujoco.MjData(mj_model)
    mujoco.mj_resetDataKeyframe(mj_model, mj_data, 0)
    mujoco.mj_forward(mj_model, mj_data)
    return mj_model, mj_data
