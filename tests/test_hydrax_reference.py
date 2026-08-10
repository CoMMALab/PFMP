"""Differential tests of our update rules against hydrax's implementations.

These call hydrax's *actual* methods rather than transcriptions, so the tests
fail if either implementation drifts. Hydrax is used as the reference for the
standard, well-specified algorithms (CEM, MPPI), and for DIAL's *annealing
schedule*, which hydrax and the original agree on.

Hydrax is NOT the reference for DIAL's *update*: hydrax uses a plain
fixed-lambda softmax where the original (LeCAR-Lab/dial-mpc) normalises by the
return std and divides by `temp_sample`. Treating hydrax as authoritative for
all of DIAL is how our `temp_sample` came to sit at 1.0 against a reference
default of 0.06 without any test failing. That half is pinned against an import
of dial-mpc in `test_dial_reference.py`.

`update_params` reads only a couple of scalar attributes on the controller, so
the controllers are built with `object.__new__` — a real one would drag in a
MuJoCo Task for no added coverage.

Note both hydrax updates index knots as (rollouts, num_knots, nu) and hydrax's
MPPI hardcodes that rank (`weights[:, None, None]`), so the shared tests here
use a vector action space. Our `mppi_update` contracts the particle axis rank-
agnostically and additionally handles scalar actions; that extra generality is
covered in `test_update.py`, not here.
"""
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from hydrax.alg_base import Trajectory
from hydrax.algs.cem import CEM, CEMParams
from hydrax.algs.dial import DIAL, DIALParams
from hydrax.algs.mppi import MPPI, MPPIParams

from pfmp.algorithm import cem_update, dial_sigma, mppi_update


class Theta(NamedTuple):
    mean: jax.Array
    stdev: jax.Array


class ThetaMean(NamedTuple):
    mean: jax.Array


def _hydrax_cem(num_elites, sigma_min):
    """A CEM whose update_params is callable without a Task."""
    c = object.__new__(CEM)
    c.num_elites = num_elites
    c.sigma_min = sigma_min
    return c


def _hydrax_mppi(temperature):
    """An MPPI whose update_params is callable without a Task."""
    c = object.__new__(MPPI)
    c.temperature = temperature
    return c


def _rollouts(knots, costs):
    return Trajectory(controls=None, knots=knots, costs=costs, trace_sites=None)


def _fixture(P, K, A, seed=0, cost_scale=1.0):
    rng = np.random.default_rng(seed)
    knots = jnp.asarray(rng.normal(size=(P, K, A)))
    # Both implementations reduce costs with sum(axis=1), so the same (P, K)
    # array is a valid input to each despite hydrax naming that axis H+1.
    costs = jnp.asarray(rng.uniform(0.0, cost_scale, size=(P, K)))
    return knots, costs


@pytest.mark.parametrize("P,K,A,num_elites", [(20, 5, 2, 5), (64, 4, 1, 6), (13, 3, 3, 1)])
def test_cem_mean_matches_hydrax(P, K, A, num_elites):
    knots, costs = _fixture(P, K, A)
    sigma_min = 0.01

    ref = _hydrax_cem(num_elites, sigma_min).update_params(
        CEMParams(tk=jnp.zeros(K), mean=jnp.zeros((K, A)),
                  rng=jax.random.key(0), cov=jnp.full((K, A), 0.5)),
        _rollouts(knots, costs),
    )
    ours, _ = cem_update(Theta(jnp.zeros((K, A)), jnp.full((K, A), 0.5)),
                         knots, costs, elite_frac=num_elites / P,
                         stdev_floor=sigma_min)

    assert np.allclose(np.asarray(ours.mean), np.asarray(ref.mean)), "elite mean differs"


@pytest.mark.parametrize("P,K,A,num_elites", [(20, 5, 2, 5), (64, 4, 1, 6)])
def test_cem_covariance_matches_hydrax(P, K, A, num_elites):
    """The refitted covariance — including the sigma_min floor — must match."""
    knots, costs = _fixture(P, K, A, seed=1)
    sigma_min = 0.02

    ref = _hydrax_cem(num_elites, sigma_min).update_params(
        CEMParams(tk=jnp.zeros(K), mean=jnp.zeros((K, A)),
                  rng=jax.random.key(0), cov=jnp.full((K, A), 0.5)),
        _rollouts(knots, costs),
    )
    ours, _ = cem_update(Theta(jnp.zeros((K, A)), jnp.full((K, A), 0.5)),
                         knots, costs, elite_frac=num_elites / P,
                         stdev_floor=sigma_min)

    assert np.allclose(np.asarray(ours.stdev), np.asarray(ref.cov)), "elite covariance differs"


def test_cem_floor_binds_identically_when_elites_coincide():
    """Degenerate case: identical elites give zero spread, so both must clamp to
    the floor rather than collapsing to zero."""
    P, K, A, num_elites, sigma_min = 12, 3, 2, 4, 0.05
    knots = jnp.tile(jnp.ones((1, K, A)), (P, 1, 1))
    costs = jnp.tile(jnp.arange(P, dtype=float)[:, None], (1, K))

    ref = _hydrax_cem(num_elites, sigma_min).update_params(
        CEMParams(tk=jnp.zeros(K), mean=jnp.zeros((K, A)),
                  rng=jax.random.key(0), cov=jnp.full((K, A), 0.5)),
        _rollouts(knots, costs),
    )
    ours, _ = cem_update(Theta(jnp.zeros((K, A)), jnp.full((K, A), 0.5)),
                         knots, costs, elite_frac=num_elites / P,
                         stdev_floor=sigma_min)

    assert np.allclose(np.asarray(ours.stdev), sigma_min)
    assert np.allclose(np.asarray(ours.stdev), np.asarray(ref.cov))


def test_cem_elite_selection_matches_hydrax_on_ties():
    """Equal total costs make elite selection order-dependent; both must resolve
    it the same way (jnp.argsort is stable), or the means silently diverge."""
    P, K, A, num_elites = 10, 3, 2, 3
    rng = np.random.default_rng(2)
    knots = jnp.asarray(rng.normal(size=(P, K, A)))
    costs = jnp.ones((P, K))                          # every particle identical

    ref = _hydrax_cem(num_elites, 0.01).update_params(
        CEMParams(tk=jnp.zeros(K), mean=jnp.zeros((K, A)),
                  rng=jax.random.key(0), cov=jnp.full((K, A), 0.5)),
        _rollouts(knots, costs),
    )
    ours, _ = cem_update(Theta(jnp.zeros((K, A)), jnp.full((K, A), 0.5)),
                         knots, costs, elite_frac=num_elites / P, stdev_floor=0.01)

    assert np.allclose(np.asarray(ours.mean), np.asarray(ref.mean))


# ─────────────────────────── MPPI vs hydrax ───────────────────────────
# hydrax: weights = softmax(-sum(costs)/T); mean = sum(weights * knots).
# Ours subtracts the min total before exponentiating instead of relying on
# softmax's internal baseline subtraction — mathematically identical, and the
# tests below are what pin that claim.

@pytest.mark.parametrize("P,K,A", [(20, 5, 2), (64, 4, 3), (7, 2, 1)])
@pytest.mark.parametrize("temperature", [0.1, 1.0, 25.0])
def test_mppi_mean_matches_hydrax(P, K, A, temperature):
    knots, costs = _fixture(P, K, A, seed=5)

    ref = _hydrax_mppi(temperature).update_params(
        MPPIParams(tk=jnp.zeros(K), mean=jnp.zeros((K, A)), rng=jax.random.key(0)),
        _rollouts(knots, costs),
    )
    ours, _ = mppi_update(ThetaMean(jnp.zeros((K, A))), knots, costs, temperature)

    assert np.allclose(np.asarray(ours.mean), np.asarray(ref.mean), atol=1e-6), \
        "weighted mean differs from hydrax"


def test_mppi_weights_match_hydrax():
    """Our returned weights must equal hydrax's softmax weights, not just the mean."""
    P, K, A, temperature = 32, 4, 2, 0.7
    knots, costs = _fixture(P, K, A, seed=6)
    _, w_ours = mppi_update(ThetaMean(jnp.zeros((K, A))), knots, costs, temperature)
    w_ref = jax.nn.softmax(-jnp.sum(costs, axis=1) / temperature, axis=0)
    assert np.allclose(np.asarray(w_ours), np.asarray(w_ref), atol=1e-6)


def test_mppi_agrees_with_hydrax_on_large_costs():
    """Both must survive costs large enough to underflow a naive exp(-c/T)."""
    P, K, A, temperature = 16, 3, 2, 1.0
    rng = np.random.default_rng(7)
    knots = jnp.asarray(rng.normal(size=(P, K, A)))
    costs = jnp.full((P, K), 1e4).at[3].set(1e4 - 5.0)

    ref = _hydrax_mppi(temperature).update_params(
        MPPIParams(tk=jnp.zeros(K), mean=jnp.zeros((K, A)), rng=jax.random.key(0)),
        _rollouts(knots, costs),
    )
    ours, _ = mppi_update(ThetaMean(jnp.zeros((K, A))), knots, costs, temperature)

    assert np.all(np.isfinite(np.asarray(ours.mean)))
    assert np.allclose(np.asarray(ours.mean), np.asarray(ref.mean), atol=1e-6)


def test_mppi_agrees_with_hydrax_when_one_particle_dominates():
    """Low temperature drives the softmax to a near-delta; both must land on the
    same particle rather than one of them averaging."""
    P, K, A, temperature = 12, 3, 2, 1e-3
    knots, costs = _fixture(P, K, A, seed=8, cost_scale=10.0)

    ref = _hydrax_mppi(temperature).update_params(
        MPPIParams(tk=jnp.zeros(K), mean=jnp.zeros((K, A)), rng=jax.random.key(0)),
        _rollouts(knots, costs),
    )
    ours, _ = mppi_update(ThetaMean(jnp.zeros((K, A))), knots, costs, temperature)
    assert np.allclose(np.asarray(ours.mean), np.asarray(ref.mean), atol=1e-6)


# ===========================================================================
# DIAL: annealing schedule
#
# Only the schedule is checked against hydrax; hydrax and dial-mpc agree here
# (their exponential and power forms are the same family, see
# test_dial_reference.test_sigma_schedule_matches_dial_mpc_power_form).
# DIAL's UPDATE is checked against dial-mpc, not hydrax, in that same file.
# ===========================================================================


def hydrax_noise_level(opt_iter, K, sigma0, b1, b2, N, nu=1):
    """Per-knot sigma from Hydrax's OWN `sample_knots`, recovered by driving it.

    `sample_knots` returns `mean + noise_level[None, :, None] * noise`, so with
    mean = 0 and noise = 1 the controls are exactly the per-knot noise level.
    Binding the unbound method to a stand-in avoids constructing a Task (which
    would need a MuJoCo model) while still running Hydrax's real code.
    """
    class FakeModel:
        pass
    FakeModel.nu = nu

    class FakeTask:
        model = FakeModel

    class Fake:
        num_samples = 1
        num_knots = K
        task = FakeTask
        noise_level = sigma0
        beta_opt_iter = b1
        beta_horizon = b2
        iterations = N

    params = DIALParams(
        tk=jnp.zeros(K),
        mean=jnp.zeros((K, nu)),
        rng=jax.random.key(0),
        opt_iteration=opt_iter,
    )

    orig_normal = jax.random.normal
    try:
        jax.random.normal = lambda key, shape, *a, **k: jnp.ones(shape)
        controls, _ = DIAL.sample_knots(Fake(), params)
    finally:
        jax.random.normal = orig_normal
    return np.asarray(controls[0, :, 0])


@pytest.mark.parametrize("sigma0,b1,b2,N,K",
                         [(0.025, 1.0, 1.0, 5, 5), (0.1, 2.0, 0.5, 4, 6)])
def test_dial_sigma_matches_hydrax_schedule(sigma0, b1, b2, N, K):
    """Our per-knot schedule must equal what Hydrax's sample_knots produces."""
    for i in range(N, 0, -1):                       # our loop order
        ours = np.array([dial_sigma(i, h, sigma0, N, K, b1, b2) for h in range(K)])
        ref = hydrax_noise_level(N - i, K, sigma0, b1, b2, N)
        np.testing.assert_allclose(ours, ref, rtol=1e-6, atol=1e-9)


def test_hydrax_opt_iteration_wraps():
    """Our loop runs i = N..1, mapping to Hydrax opt_iteration 0..N-1. Confirm
    Hydrax wraps at N, so the mapping stays valid across a full cycle."""
    K, N = 4, 3
    params = DIALParams(tk=jnp.zeros(K), mean=jnp.zeros((K, 1)),
                        rng=jax.random.key(0), opt_iteration=N - 1)

    class FakeModel:
        nu = 1

    class FakeTask:
        model = FakeModel

    class Fake:
        num_samples = 1
        num_knots = K
        task = FakeTask
        noise_level = 0.1
        beta_opt_iter = 1.0
        beta_horizon = 1.0
        iterations = N

    _, new_params = DIAL.sample_knots(Fake(), params)
    assert int(new_params.opt_iteration) == 0, "opt_iteration must wrap at N"


def test_dial_sigma_reaches_full_noise_at_terminal_knot():
    """At the widest optimization step the terminal knot samples at the full
    sigma0 (exp(0)). A shrunk terminal knot means the horizon offset is wrong."""
    N = K = 5
    sigma0 = 0.025
    first_step = np.array([dial_sigma(N, h, sigma0, N, K, 1.0, 1.0) for h in range(K)])
    assert np.isclose(first_step[-1], sigma0), (
        f"terminal-knot noise {first_step[-1]} != sigma0 {sigma0}")


def test_dial_sigma_anneals_over_iterations():
    """Noise must shrink as the optimization loop progresses (i: N -> 1)."""
    N = K = 5
    per_iter_max = [max(dial_sigma(i, h, 0.025, N, K, 1.0, 1.0) for h in range(K))
                    for i in range(N, 0, -1)]
    assert all(a >= b for a, b in zip(per_iter_max, per_iter_max[1:])), per_iter_max


def test_dial_sigma_varies_across_horizon():
    """Dual-loop: noise must depend on the knot h (action-level annealing), not
    collapse to a constant across the horizon."""
    N = K = 5
    row = [dial_sigma(N, h, 0.025, N, K, 1.0, 1.0) for h in range(K)]
    assert len(set(np.round(row, 8))) > 1, "sigma is constant across knots"

if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
