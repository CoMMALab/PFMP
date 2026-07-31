"""API-surface tests for the PFMP controller.

Covers the pieces `test_update.py` (update rules) and `test_dial.py` (annealing
schedule) do not: the knot grid, interval integration, rollouts, resampling, and
ancestor tracing. The emphasis is on **shapes and off-by-one boundaries**, since
those are silent failure modes — a transposed cost array or a knot-grid that is
one element short still runs, it just optimizes the wrong thing.

Shape conventions pinned here (P particles, K knots, A action dim):

    tk(endpoint=True)   (K,)         knot times, 0 .. horizon inclusive
    tk(endpoint=False)  (K-1,)       interval start times
    unroll  -> us       (P, K, A)    control knots
               costs    (P, K)       K-1 interval costs + 1 terminal cost
               xs       (P, K-1, .)  state at the end of each interval
    run_sis -> (us, costs)           same as unroll without xs
    run_sir -> (us, costs)           same shapes, with resampling
    predict -> ((K-1)*steps_per_knot + 1, ...)   dense, x0 first
    _sir    -> states     (K, P, A)
               ancestors  (K-1, P)   one row per transition
"""
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pfmp.algorithm import PFMP, ess, get_field_info


# ── fixtures: a tiny analytic controller with no MuJoCo dependency ───────────

PARTICLES, NUM_KNOTS, ACTION_DIM = 6, 5, 2
HORIZON, DT = 0.4, 0.01
STEPS_PER_KNOT = int((HORIZON / (NUM_KNOTS - 1)) / DT)   # = 10
NUM_INTERVALS = NUM_KNOTS - 1


class Theta(NamedTuple):
    mean: jax.Array
    stdev: float


def _dynamics(x, u, th, t):
    return x + u * 0.01, None


def _running_cost(x, u, th, t):
    return jnp.sum(x ** 2)


def _terminal_cost(x, th):
    return jnp.sum(x ** 2)


def _init(key, x, th):
    return jax.random.normal(key, (ACTION_DIM,)) * th.stdev


def _proposal(key, x, u, th, t):
    return u + jax.random.normal(key, (ACTION_DIM,)) * th.stdev


def make_ctrl(interp="zoh", particles=PARTICLES, ess_thresh=0.5):
    return PFMP(
        _dynamics, _running_cost, _terminal_cost, _init, _proposal,
        horizon=HORIZON, num_knots=NUM_KNOTS, particles=particles,
        dt=DT, interp=interp, ess_resample_thresh=ess_thresh,
    )


@pytest.fixture
def ctrl():
    return make_ctrl()


@pytest.fixture
def theta():
    return Theta(mean=jnp.zeros((NUM_KNOTS, ACTION_DIM)), stdev=0.1)


@pytest.fixture
def x0():
    return jnp.zeros(ACTION_DIM)


@pytest.fixture
def key():
    return jax.random.key(0)


# ── knot grid: the classic off-by-one ────────────────────────────────────────

def test_tk_endpoint_spans_horizon_inclusive(ctrl):
    tk = ctrl.tk(endpoint=True)
    assert tk.shape == (NUM_KNOTS,)
    assert float(tk[0]) == 0.0
    assert np.isclose(float(tk[-1]), HORIZON)          # inclusive of horizon


def test_tk_no_endpoint_drops_exactly_the_last_knot(ctrl):
    """Interval starts: one per transition, i.e. K-1 — not K, not K-2."""
    assert ctrl.tk(endpoint=False).shape == (NUM_INTERVALS,)
    assert np.allclose(np.asarray(ctrl.tk(endpoint=False)),
                       np.asarray(ctrl.tk(endpoint=True))[:-1])


def test_tk_spacing_is_uniform(ctrl):
    d = np.diff(np.asarray(ctrl.tk(endpoint=True)))
    assert np.allclose(d, HORIZON / NUM_INTERVALS)


# ── unroll / run_sis / run_sir shapes ────────────────────────────────────────

def test_unroll_shapes(ctrl, key, x0, theta):
    us, costs, xs = ctrl.unroll(key, x0, theta)
    assert us.shape == (PARTICLES, NUM_KNOTS, ACTION_DIM)
    assert costs.shape == (PARTICLES, NUM_KNOTS)
    # one state per *interval*, so K-1 — this is the off-by-one that matters
    assert xs.shape == (PARTICLES, NUM_INTERVALS, ACTION_DIM)


def test_costs_are_interval_costs_plus_one_terminal(ctrl, key, x0, theta):
    """K cost columns = (K-1) interval costs + 1 terminal cost."""
    _, costs, _ = ctrl.unroll(key, x0, theta)
    assert costs.shape[1] == NUM_INTERVALS + 1 == NUM_KNOTS


@pytest.mark.parametrize("run", ["run_sis", "run_sir"])
def test_rollout_shapes(ctrl, key, x0, theta, run):
    trajs, costs = getattr(ctrl, run)(key, x0, theta)
    assert trajs.shape == (PARTICLES, NUM_KNOTS, ACTION_DIM)
    assert costs.shape == (PARTICLES, NUM_KNOTS)


@pytest.mark.parametrize("run", ["run_sis", "run_sir"])
def test_rollout_is_finite(ctrl, key, x0, theta, run):
    trajs, costs = getattr(ctrl, run)(key, x0, theta)
    assert np.all(np.isfinite(np.asarray(trajs)))
    assert np.all(np.isfinite(np.asarray(costs)))


def test_run_sis_equals_unroll_without_states(ctrl, key, x0, theta):
    """SIS is exactly `unroll` with the per-knot states dropped."""
    us, costs, _ = ctrl.unroll(key, x0, theta)
    trajs, sis_costs = ctrl.run_sis(key, x0, theta)
    assert np.allclose(np.asarray(us), np.asarray(trajs))
    assert np.allclose(np.asarray(costs), np.asarray(sis_costs))


@pytest.mark.parametrize("interp", ["zoh", "hermite"])
def test_both_interpolation_modes_run(interp, key, x0, theta):
    trajs, costs = make_ctrl(interp=interp).run_sis(key, x0, theta)
    assert trajs.shape == (PARTICLES, NUM_KNOTS, ACTION_DIM)
    assert np.all(np.isfinite(np.asarray(costs)))


@pytest.mark.parametrize("particles", [1, 3, 32])
def test_particle_count_propagates(particles, key, x0, theta):
    trajs, costs = make_ctrl(particles=particles).run_sis(key, x0, theta)
    assert trajs.shape[0] == particles
    assert costs.shape[0] == particles


# ── predict: dense deterministic rollout ─────────────────────────────────────

def test_predict_dense_length(ctrl, x0, theta):
    """(K-1) intervals x steps_per_knot substeps, +1 for the x0 row."""
    xs = ctrl.predict(x0, jnp.ones((NUM_KNOTS, ACTION_DIM)) * 0.5, theta)
    assert xs.shape == (NUM_INTERVALS * STEPS_PER_KNOT + 1, ACTION_DIM)


def test_predict_starts_at_x0(ctrl, theta):
    start = jnp.array([0.3, -0.2])
    xs = ctrl.predict(start, jnp.ones((NUM_KNOTS, ACTION_DIM)) * 0.5, theta)
    assert np.allclose(np.asarray(xs[0]), np.asarray(start))


def test_predict_is_deterministic(ctrl, x0, theta):
    """No key argument: the same plan must always give the same rollout."""
    plan = jnp.ones((NUM_KNOTS, ACTION_DIM)) * 0.5
    a = ctrl.predict(x0, plan, theta)
    b = ctrl.predict(x0, plan, theta)
    assert np.array_equal(np.asarray(a), np.asarray(b))


def test_predict_advances_under_constant_positive_control(ctrl, x0, theta):
    """Integrating +u must move the state monotonically (catches a dead scan)."""
    xs = np.asarray(ctrl.predict(x0, jnp.ones((NUM_KNOTS, ACTION_DIM)) * 0.5, theta))
    assert np.all(np.diff(xs[:, 0]) >= -1e-9)
    assert xs[-1, 0] > xs[0, 0]


def test_predict_handles_zero_sized_leaves(ctrl, theta):
    """mjx.Data carries empty contact buffers; a reshape(-1) on those divides by
    zero. The dense concat must survive a leaf with a zero-length axis."""
    state = {"q": jnp.zeros(ACTION_DIM), "empty": jnp.zeros((0,))}

    def dyn(x, u, th, t):
        return {"q": x["q"] + u * 0.01, "empty": x["empty"]}, None

    c = ctrl._replace(mpc_dynamics=dyn,
                      mpc_running_cost=lambda x, u, th, t: jnp.sum(x["q"] ** 2),
                      mpc_terminal_cost=lambda x, th: jnp.sum(x["q"] ** 2))
    xs = c.predict(state, jnp.ones((NUM_KNOTS, ACTION_DIM)) * 0.5, theta)
    assert xs["q"].shape == (NUM_INTERVALS * STEPS_PER_KNOT + 1, ACTION_DIM)
    assert xs["empty"].shape == (NUM_INTERVALS * STEPS_PER_KNOT + 1, 0)


# ── resampling / ancestors ───────────────────────────────────────────────────

def test_sir_internal_shapes(ctrl, key, x0, theta):
    states, ancestors, costs, final_xs = ctrl._sir(key, x0, theta)
    assert states.shape == (NUM_KNOTS, PARTICLES, ACTION_DIM)
    # one ancestor row per *transition*, so K-1
    assert ancestors.shape == (NUM_INTERVALS, PARTICLES)
    assert costs.shape == (PARTICLES, NUM_KNOTS)
    assert final_xs.shape == (PARTICLES, ACTION_DIM)


def test_ancestors_are_valid_particle_indices(ctrl, key, x0, theta):
    _, ancestors, _, _ = ctrl._sir(key, x0, theta)
    a = np.asarray(ancestors)
    assert a.min() >= 0 and a.max() < PARTICLES


def test_zero_threshold_never_resamples(key, x0, theta):
    """ess_resample_thresh=0 is SIS: every ancestor is the identity map."""
    _, ancestors, _, _ = make_ctrl(ess_thresh=0.0)._sir(key, x0, theta)
    expected = np.tile(np.arange(PARTICLES), (NUM_INTERVALS, 1))
    assert np.array_equal(np.asarray(ancestors), expected)


def test_high_threshold_always_resamples(key, x0, theta):
    """thresh > 1 forces resampling at every step, so ancestors leave identity."""
    _, ancestors, _, _ = make_ctrl(ess_thresh=10.0)._sir(key, x0, theta)
    identity = np.tile(np.arange(PARTICLES), (NUM_INTERVALS, 1))
    assert not np.array_equal(np.asarray(ancestors), identity)


@pytest.mark.parametrize("n", [1, 3, PARTICLES])
def test_choose_paths_shape(ctrl, key, x0, theta, n):
    states, ancestors, _, _ = ctrl._sir(key, x0, theta)
    paths = ctrl.choose_paths(states, ancestors, jnp.arange(n))
    assert paths.shape == (n, NUM_KNOTS, ACTION_DIM)


def test_choose_paths_without_resampling_is_the_identity(key, x0, theta):
    """With no resampling, tracing ancestors must return each particle's own
    trajectory unchanged — i.e. states transposed, not shuffled."""
    c = make_ctrl(ess_thresh=0.0)
    states, ancestors, _, _ = c._sir(key, x0, theta)
    paths = c.choose_paths(states, ancestors, jnp.arange(PARTICLES))
    assert np.allclose(np.asarray(paths),
                       np.asarray(states).transpose(1, 0, 2))


# ── determinism ──────────────────────────────────────────────────────────────

@pytest.mark.parametrize("run", ["run_sis", "run_sir"])
def test_same_key_same_result(ctrl, x0, theta, run):
    f = getattr(ctrl, run)
    a, ca = f(jax.random.key(7), x0, theta)
    b, cb = f(jax.random.key(7), x0, theta)
    assert np.array_equal(np.asarray(a), np.asarray(b))
    assert np.array_equal(np.asarray(ca), np.asarray(cb))


@pytest.mark.parametrize("run", ["run_sis", "run_sir"])
def test_different_key_different_samples(ctrl, x0, theta, run):
    f = getattr(ctrl, run)
    a, _ = f(jax.random.key(1), x0, theta)
    b, _ = f(jax.random.key(2), x0, theta)
    assert not np.array_equal(np.asarray(a), np.asarray(b))


# ── ess ──────────────────────────────────────────────────────────────────────

def test_ess_uniform_costs_is_one():
    assert np.isclose(float(ess(jnp.zeros(8))), 1.0)


def test_ess_degenerate_is_one_over_n():
    """One dominant particle -> normalized ESS = 1/n."""
    costs = jnp.array([0.0, 1e6, 1e6, 1e6])
    assert np.isclose(float(ess(costs)), 1.0 / 4)


def test_ess_accepts_per_knot_costs_and_sums_over_knots():
    """(P, K) input must reduce over knots, matching the update rules."""
    flat = ess(jnp.array([0.0, 3.0]))
    per_knot = ess(jnp.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]))
    assert np.isclose(float(flat), float(per_knot))


def test_ess_is_bounded():
    for c in [jnp.zeros(5), jnp.arange(5.0), jnp.array([0.0, 1e3, 1e3, 1e3, 1e3])]:
        v = float(ess(c))
        assert 1.0 / 5 - 1e-6 <= v <= 1.0 + 1e-6


# ── get_field_info: container round-trips used by ancestor tracing ───────────

def test_get_field_info_namedtuple_roundtrip():
    class S(NamedTuple):
        a: jax.Array
        b: jax.Array
    s = S(jnp.ones(3), jnp.zeros(3))
    names, accessor, builder = get_field_info(s)
    assert names == ("a", "b")
    rebuilt = builder([accessor(s, n) for n in names])
    assert isinstance(rebuilt, S) and np.allclose(np.asarray(rebuilt.a), 1.0)


def test_get_field_info_array_is_single_field():
    x = jnp.ones((2, 3))
    names, accessor, builder = get_field_info(x)
    assert len(names) == 1
    assert np.allclose(np.asarray(builder([accessor(x, names[0])])), np.asarray(x))


def test_get_field_info_tuple_roundtrip():
    t = (jnp.ones(2), jnp.zeros(2))
    names, accessor, builder = get_field_info(t)
    rebuilt = builder([accessor(t, n) for n in names])
    assert isinstance(rebuilt, tuple) and len(rebuilt) == 2


def test_get_field_info_rejects_unsupported():
    with pytest.raises(TypeError):
        get_field_info("not a state")


# ── jit / vmap compatibility ─────────────────────────────────────────────────

def test_rollout_is_jittable(ctrl, key, x0, theta):
    trajs, costs = jax.jit(ctrl.run_sis)(key, x0, theta)
    assert trajs.shape == (PARTICLES, NUM_KNOTS, ACTION_DIM)


def test_rollout_vmaps_over_keys(ctrl, x0, theta):
    keys = jax.random.split(jax.random.key(0), 4)
    trajs, costs = jax.vmap(ctrl.run_sis, in_axes=(0, None, None))(keys, x0, theta)
    assert trajs.shape == (4, PARTICLES, NUM_KNOTS, ACTION_DIM)
    assert costs.shape == (4, PARTICLES, NUM_KNOTS)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
