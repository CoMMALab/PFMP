"""Tests for the FR3 cube grasp-and-lift task and the kinematics under it.

Two things are pinned here.

**go_fr3's IK stays one implementation.** `fr3/go_fr3/ik.py` is dispatched on
its inputs' array namespace so the same source runs under numpy for the CLI and
traces under JAX inside a rollout. The risk that buys is a silent divergence
between the two paths, so `solve` and `solve_clamped` are checked against each
other and against mujoco's own forward kinematics, not just against themselves.

**Geometry the task assumes is measured, not asserted.** The fingertip offset
and the jaw-yaw formula are read off mujoco's FK; if the hand's mounting or the
gripper site moves, these fail rather than quietly mis-aiming every grasp.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "fr3"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "fr3", "go_fr3"))

import ik


# ── go_fr3's IK: numpy and traced paths agree ───────────────────────────────

REACHABLE = [(0.0, 0.307, 0.487), (0.0, 0.5, 0.2), (0.5, 0.45, 0.06),
             (-0.4, 0.6, 0.35), (0.9, 0.4, 0.5)]


@pytest.mark.parametrize("theta,r,z", REACHABLE)
def test_solve_clamped_matches_solve_where_reachable(theta, r, z):
    """The branchless path must be the raising path, minus the raise."""
    q = ik.solve(theta, r, z, q7=0.0)
    q_c, violation = ik.solve_clamped(theta, r, z, q7=0.0)
    np.testing.assert_allclose(q_c, q, atol=1e-12)
    assert violation == 0.0


def test_solve_clamped_flags_unreachable_instead_of_raising():
    """A sampler will propose these constantly; they must cost, not crash."""
    with pytest.raises(ik.OutOfReach):
        ik.solve(0.0, 2.0, 0.2)
    q, violation = ik.solve_clamped(0.0, 2.0, 0.2)
    assert violation > 0.0
    assert np.all(np.isfinite(q))
    # Still a configuration the robot could hold.
    assert not ik.out_of_limits(q)


def test_solve_clamped_random_agreement():
    """Sweep the workspace: identical where solve succeeds, flagged where not."""
    rng = np.random.default_rng(0)
    agreed = flagged = 0
    for _ in range(500):
        theta, r = rng.uniform(-2.8, 2.8), rng.uniform(0.0, 0.9)
        z, q7 = rng.uniform(-0.4, 0.85), rng.uniform(-2.9, 2.9)
        q_c, violation = ik.solve_clamped(theta, r, z, q7)
        try:
            q = ik.solve(theta, r, z, q7)
        except ik.OutOfReach:
            assert violation > 0.0
            flagged += 1
            continue
        np.testing.assert_allclose(q_c, q, atol=1e-12)
        assert violation == 0.0
        agreed += 1
    assert agreed > 100 and flagged > 100, "sweep should cover both outcomes"


def test_ik_traces_under_jit_and_vmap():
    """The whole point of the namespace dispatch: no separate JAX port."""
    jax = pytest.importorskip("jax")
    jnp = jax.numpy

    q_np = ik.solve(0.0, 0.5, 0.2, q7=-0.785398)
    q_jit, violation = jax.jit(ik.solve_clamped)(
        jnp.float32(0.0), jnp.float32(0.5), jnp.float32(0.2), jnp.float32(-0.785398))
    np.testing.assert_allclose(np.asarray(q_jit), q_np, atol=1e-5)
    assert float(violation) == 0.0

    qs, violations = jax.jit(jax.vmap(ik.solve_clamped))(
        jnp.zeros(4), jnp.array([0.4, 0.5, 0.6, 2.0]), jnp.full(4, 0.2), jnp.zeros(4))
    assert qs.shape == (4, 7)
    assert float(violations[-1]) > 0.0, "r=2.0 m is far outside the arm's shell"
    assert np.all(np.asarray(violations[:3]) == 0.0)


def test_close_in_target_is_flagged():
    """Unreachable is not only about distance. (r=0.30, z=0.20) sits well inside
    the arm's shell but needs joint 4 at -3.00 rad, past its -2.98 limit, and
    `violation` has to account for joint limits and not just over-reach."""
    _q, violation = ik.solve_clamped(0.0, 0.30, 0.2)
    assert violation > 0.0
    with pytest.raises(ik.OutOfReach):
        ik.solve(0.0, 0.30, 0.2)


# ── geometry, measured against mujoco's forward kinematics ──────────────────

@pytest.fixture(scope="module")
def mj():
    mujoco = pytest.importorskip("mujoco")
    import grasp
    model, data = grasp.load_scene()
    site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "gripper")
    return mujoco, model, data, site, grasp


@pytest.mark.parametrize("theta,r,z", REACHABLE)
def test_ik_agrees_with_mujoco_fk(mj, theta, r, z):
    """go_fr3's chart and the MuJoCo model must describe the same arm."""
    mujoco, model, data, site, grasp = mj
    data.qpos[:7] = ik.solve(theta, r, z, q7=0.0)
    mujoco.mj_forward(model, data)
    p = data.site_xpos[site]
    assert np.hypot(p[0], p[1]) == pytest.approx(r, abs=1e-4)
    assert np.arctan2(p[1], p[0]) == pytest.approx(theta, abs=1e-4)
    # The site sits SITE_Z_OFFSET above the fingertip the IK solves for.
    assert p[2] - z == pytest.approx(grasp.SITE_Z_OFFSET, abs=1e-4)


def test_approach_is_straight_down(mj):
    """The gauge q3=q5=0, q6=q2-q4 is what makes a top-down grasp well posed."""
    mujoco, model, data, site, _ = mj
    data.qpos[:7] = ik.solve(0.3, 0.5, 0.25, q7=0.4)
    mujoco.mj_forward(model, data)
    approach = data.site_xmat[site].reshape(3, 3)[:, 2]
    np.testing.assert_allclose(approach, [0.0, 0.0, -1.0], atol=1e-6)


@pytest.mark.parametrize("theta", [0.0, 0.5, -0.4])
@pytest.mark.parametrize("jaw_yaw", [0.0, 0.3, -0.6])
def test_q7_aims_the_jaws(mj, theta, jaw_yaw):
    """q7_for_jaw_yaw must actually align the jaws, modulo the cube's symmetry."""
    mujoco, model, data, site, _ = mj
    q7 = ik.q7_for_jaw_yaw(theta, jaw_yaw)
    data.qpos[:7] = ik.solve(theta, 0.5, 0.2, q7=q7)
    mujoco.mj_forward(model, data)
    jaw = data.site_xmat[site].reshape(3, 3)[:, 1]     # finger slide axis
    got = np.arctan2(jaw[1], jaw[0])
    # A cube is symmetric under quarter turns, so compare modulo pi/2.
    err = (got - jaw_yaw + np.pi / 4) % (np.pi / 2) - np.pi / 4
    assert err == pytest.approx(0.0, abs=1e-6)


# ── the task's shape contracts ──────────────────────────────────────────────

def test_control_bounds_and_plan_line_up():
    import grasp
    assert grasp.U_MIN.shape == grasp.U_MAX.shape == (4,)
    assert np.all(np.asarray(grasp.U_MIN) < np.asarray(grasp.U_MAX))
    # The lift target must be inside the control box, not pinned against it.
    assert grasp.TARGET_Z + grasp.GRASP_Z_OFFSET < float(grasp.U_MAX[2])


def test_warm_start_does_not_encode_the_grasp(mj):
    """The warm start must not hand the uninformed baselines a scripted grasp.

    It may ramp toward the cube, but the gripper stays open throughout and no
    knot descends to grasp depth; otherwise `sir` beating `mppi` would only show
    that both can follow a plan they were given.
    """
    pytest.importorskip("jax")
    from mujoco import mjx
    import numpy as np

    _mujoco, model, data, site, grasp = mj
    x0 = mjx.put_data(model, data)
    theta = grasp.GraspParams(mean=np.zeros((grasp.NUM_KNOTS, 4)),
                              mjx_model=mjx.put_model(model),
                              stdev=grasp.STDEV, gripper_site_id=site)
    mean = np.asarray(grasp.initial_mean(x0, theta))
    assert mean.shape == (grasp.NUM_KNOTS, 4)
    assert np.all(mean[:, 3] == grasp.GRIP_OPEN), "warm start must never close the gripper"
    cube_z = float(np.asarray(x0.qpos)[grasp.CUBE_QPOS_OFFSET + 2])
    assert mean[:, 2].min() > cube_z + 0.5 * grasp.PREGRASP_HEIGHT, \
        "warm start must not descend to grasp depth"


@pytest.mark.slow
def test_unroll_shapes_and_lifts(mj):
    """(P, K) cost convention, and the task is actually solvable by sampling."""
    jax = pytest.importorskip("jax")
    jnp = jax.numpy
    from mujoco import mjx
    import pfmp

    _mujoco, model, data, site, grasp = mj
    model.opt.timestep = 0.02
    mjx_model = mjx.put_model(model)
    particles = 64
    ctrl = pfmp.PFMP(grasp.dynamics, grasp.running_cost, grasp.terminal_cost,
                     grasp.init_control, grasp.proposal,
                     horizon=grasp.HORIZON, num_knots=grasp.NUM_KNOTS,
                     particles=particles, dt=0.02, interp="zoh")
    x0 = mjx.put_data(model, data)
    theta = grasp.GraspParams(mean=jnp.zeros((grasp.NUM_KNOTS, 4)),
                              mjx_model=mjx_model, stdev=grasp.STDEV,
                              gripper_site_id=site)
    theta = theta._replace(mean=grasp.initial_mean(x0, theta))

    us, costs, xs = ctrl.unroll(jax.random.key(0), x0, theta)
    assert us.shape == (particles, grasp.NUM_KNOTS, 4)
    assert costs.shape == (particles, grasp.NUM_KNOTS)
    assert np.all(np.isfinite(np.asarray(costs)))

    # The grasp-aware proposal must find the cube, or the task is unusable.
    cube_z = np.asarray(xs.qpos)[:, :, grasp.CUBE_QPOS_OFFSET + 2]
    lifted = (cube_z.max(axis=1) - grasp.CUBE_HALF) > 0.10
    assert lifted.sum() >= particles // 8, (
        f"only {lifted.sum()}/{particles} particles lifted the cube 10 cm")
