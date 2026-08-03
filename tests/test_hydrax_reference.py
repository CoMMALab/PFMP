"""Differential tests of our update rules against hydrax's implementations.

These call hydrax's *actual* methods rather than transcriptions, so the tests
fail if either implementation drifts. Hydrax is used as the reference for the
standard, well-specified algorithms (CEM, MPPI). It is deliberately **not** the
reference for DIAL — hydrax's DIAL is itself a reimplementation, so that one is
checked against the canonical LeCAR-Lab repo instead (see `test_dial.py`).

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
from hydrax.algs.mppi import MPPI, MPPIParams

from pfmp.algorithm import cem_update, mppi_update


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


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
