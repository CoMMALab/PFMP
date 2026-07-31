"""Tests for the batched headless rollout driver (`pfmp.run_batch`).

This is the path that produces the simulation benchmark numbers, so the emphasis
is on the shape contract and the replan/step arithmetic:

    sim_steps_per_replan = int((1/frequency) / model.timestep)   (min 1)
    num_replans          = max_step // sim_steps_per_replan      (floor!)
    T                    = num_replans * sim_steps_per_replan

The floor division is the sharp edge: `max_step` smaller than one replan period
yields an empty trajectory rather than a partial one.

Full `run_batch` calls JIT-compile an MJX rollout (~10s each), so those are
marked `slow` and share module-scoped fixtures. Run just the fast ones with
`pytest tests/test_run_batch.py -m "not slow"`.
"""
import jax
import jax.numpy as jnp
import mujoco
import numpy as np
import pytest
from mujoco import mjx

from pfmp import run_batch
from pfmp.algorithm import PFMP, GenericMJXTaskParams, mjx_physics, ps_update
from pfmp.run_batch import Trajectory, _batch_scenes


# ── a minimal 1-DoF cart, so MJX compiles fast ───────────────────────────────

CART_XML = """
<mujoco>
  <option timestep="0.01"/>
  <worldbody>
    <body name="cart" pos="0 0 0">
      <joint name="slide" type="slide" axis="1 0 0"/>
      <geom type="box" size=".1 .1 .1" mass="1"/>
    </body>
  </worldbody>
  <actuator><motor joint="slide" ctrlrange="-1 1"/></actuator>
</mujoco>
"""

NUM_KNOTS, PARTICLES = 4, 8
FREQUENCY, TIMESTEP = 10.0, 0.01
SIM_STEPS_PER_REPLAN = int((1.0 / FREQUENCY) / TIMESTEP)      # = 10


@pytest.fixture(scope="module")
def model_data():
    m = mujoco.MjModel.from_xml_string(CART_XML)
    return m, mujoco.MjData(m)


@pytest.fixture(scope="module")
def controller():
    def running_cost(x, u, th, t):
        return jnp.sum(x.qpos ** 2)

    def terminal_cost(x, th):
        return jnp.sum(x.qpos ** 2)

    def init(key, x, th):
        return jax.random.normal(key, (1,)) * th.noise_level + th.mean[0]

    def proposal(key, x, u, th, t):
        return u + jax.random.normal(key, (1,)) * th.noise_level

    return PFMP(mjx_physics, running_cost, terminal_cost, init, proposal,
                horizon=0.2, num_knots=NUM_KNOTS, particles=PARTICLES,
                dt=TIMESTEP, interp="zoh")


@pytest.fixture(scope="module")
def theta(model_data):
    m, _ = model_data
    return GenericMJXTaskParams(mjx_model=mjx.put_model(m),
                                mean=jnp.zeros((NUM_KNOTS, m.nu)),
                                noise_level=0.3)


# ── Trajectory container ─────────────────────────────────────────────────────

def test_trajectory_save_load_roundtrip(tmp_path):
    traj = Trajectory(
        ctrl=np.random.rand(2, 5, 1), cost=np.random.rand(2, 5),
        qpos=np.random.rand(2, 5, 1), qvel=np.random.rand(2, 5, 1),
        replan_ms=1.25,
    )
    path = tmp_path / "traj.npz"
    traj.save(str(path))
    out = Trajectory.load(str(path))
    for field in ("ctrl", "cost", "qpos", "qvel"):
        assert np.allclose(getattr(out, field), getattr(traj, field))
    assert out.replan_ms == pytest.approx(1.25)


def test_trajectory_load_without_replan_ms_defaults_to_zero(tmp_path):
    """Older archives predate replan_ms; loading must not raise."""
    path = tmp_path / "old.npz"
    np.savez(str(path), ctrl=np.zeros((1, 2, 1)), cost=np.zeros((1, 2)),
             qpos=np.zeros((1, 2, 1)), qvel=np.zeros((1, 2, 1)))
    assert Trajectory.load(str(path)).replan_ms == 0.0


# ── _batch_scenes ────────────────────────────────────────────────────────────

def test_batch_scenes_shared_scene_replicates(model_data, theta):
    """Without sample_problem every batch element is the same scene."""
    m, d = model_data
    mjx_data, theta_b = _batch_scenes(None, m, d, theta, num=4)
    assert mjx_data.qpos.shape == (4, m.nq)
    assert theta_b.mean.shape == (4, NUM_KNOTS, m.nu)
    assert np.allclose(np.asarray(theta_b.mean[0]), np.asarray(theta_b.mean[-1]))


def test_batch_scenes_uses_sample_problem_per_element(model_data, theta):
    """With sample_problem each element gets its own initial state."""
    m, _ = model_data

    def sample_problem(i):
        d_i = mujoco.MjData(m)
        d_i.qpos[0] = 0.1 * i
        mujoco.mj_forward(m, d_i)
        return d_i, theta

    mjx_data, theta_b = _batch_scenes(sample_problem, m, mujoco.MjData(m), theta, num=3)
    assert mjx_data.qpos.shape == (3, m.nq)
    qpos = np.asarray(mjx_data.qpos).ravel()
    assert np.allclose(qpos, [0.0, 0.1, 0.2])          # distinct scenes, in order


@pytest.mark.parametrize("num", [1, 5])
def test_batch_scenes_respects_batch_size(model_data, theta, num):
    m, d = model_data
    mjx_data, theta_b = _batch_scenes(None, m, d, theta, num=num)
    assert mjx_data.qpos.shape[0] == num
    assert theta_b.mean.shape[0] == num


# ── run_batch: shapes and replan arithmetic ──────────────────────────────────

@pytest.fixture(scope="module")
def sis_run(controller, theta, model_data):
    """One compiled reference run, shared by the shape assertions."""
    m, d = model_data
    return run_batch(controller, theta, m, d, frequency=FREQUENCY,
                     algorithm="sis", max_step=2 * SIM_STEPS_PER_REPLAN, num=3)


@pytest.mark.slow
def test_run_batch_shapes(sis_run, model_data):
    m, _ = model_data
    T = 2 * SIM_STEPS_PER_REPLAN
    assert sis_run.ctrl.shape == (3, T, m.nu)
    assert sis_run.cost.shape == (3, T)
    assert sis_run.qpos.shape == (3, T, m.nq)
    assert sis_run.qvel.shape == (3, T, m.nv)


@pytest.mark.slow
def test_run_batch_output_is_finite(sis_run):
    for field in ("ctrl", "cost", "qpos", "qvel"):
        assert np.all(np.isfinite(getattr(sis_run, field))), field


@pytest.mark.slow
def test_run_batch_reports_replan_time(sis_run):
    assert sis_run.replan_ms > 0.0


@pytest.mark.slow
def test_horizon_is_num_replans_times_steps_per_replan(controller, theta, model_data):
    """T must be a whole number of replan periods, floor-divided from max_step.
    max_step=25 with a 10-step period gives 2 replans (20 steps), not 2.5."""
    m, d = model_data
    traj = run_batch(controller, theta, m, d, frequency=FREQUENCY,
                     algorithm="sis", max_step=25, num=1)
    assert traj.ctrl.shape[1] == 2 * SIM_STEPS_PER_REPLAN == 20


@pytest.mark.slow
def test_max_step_below_one_replan_yields_empty_trajectory(controller, theta, model_data):
    """Floor division boundary: fewer steps than one replan period => no rollout."""
    m, d = model_data
    traj = run_batch(controller, theta, m, d, frequency=FREQUENCY,
                     algorithm="sis", max_step=SIM_STEPS_PER_REPLAN - 1, num=1)
    assert traj.ctrl.shape[1] == 0


@pytest.mark.slow
@pytest.mark.parametrize("num", [1, 4])
def test_batch_dimension_propagates(controller, theta, model_data, num):
    m, d = model_data
    traj = run_batch(controller, theta, m, d, frequency=FREQUENCY,
                     algorithm="sis", max_step=SIM_STEPS_PER_REPLAN, num=num)
    assert traj.ctrl.shape[0] == num
    assert traj.qpos.shape[0] == num


# ── algorithm dispatch (the proposal x update x resampling axes) ─────────────

@pytest.mark.slow
@pytest.mark.parametrize("algorithm", ["sis", "sir", "biased_mppi"])
def test_pfmp_algorithms_run_and_agree_on_shape(controller, theta, model_data, algorithm):
    m, d = model_data
    traj = run_batch(controller, theta, m, d, frequency=FREQUENCY,
                     algorithm=algorithm, max_step=SIM_STEPS_PER_REPLAN, num=2)
    assert traj.ctrl.shape == (2, SIM_STEPS_PER_REPLAN, m.nu)
    assert np.all(np.isfinite(traj.ctrl))


@pytest.mark.slow
def test_update_fn_overrides_the_default_rule(controller, theta, model_data):
    """A caller-supplied update must actually be invoked (sweeps rely on this)."""
    m, d = model_data
    calls = []

    def counting_update(th, trajs, costs):
        calls.append(1)
        return ps_update(th, trajs, costs)

    traj = run_batch(controller, theta, m, d, frequency=FREQUENCY,
                     algorithm="sis", max_step=SIM_STEPS_PER_REPLAN, num=1,
                     update_fn=counting_update)
    assert calls, "update_fn was never called"
    assert np.all(np.isfinite(traj.ctrl))


def test_hydrax_family_requires_theta_none(model_data, theta):
    """The hydrax branch plans from its own params, so being handed a pfmp theta
    is a wiring error and must fail loudly rather than silently ignoring it.

    Uses a stub controller because the guard sits after `controller.optimize` is
    resolved — a pfmp controller would raise AttributeError first, which would
    make this test pass for the wrong reason.
    """
    m, d = model_data

    class HydraxStub:
        num_knots, ctrl_steps, plan_horizon = NUM_KNOTS, 10, 0.2
        def optimize(self, *a, **k): raise AssertionError("should not be reached")
        def interp_func(self, *a, **k): raise AssertionError("should not be reached")

    with pytest.raises(AssertionError):
        run_batch(HydraxStub(), theta, m, d, frequency=FREQUENCY,
                  algorithm="ps", max_step=SIM_STEPS_PER_REPLAN, num=1)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
