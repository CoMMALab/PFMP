"""Unit tests for the SMC update rules (U axis).

All three share one signature and convention:

    update(theta, trajs, costs, *hp) -> (theta, aux)
    trajs : PyTree, each leaf (P, K, A)   particles × knots × action
    costs : (P, K)                        per-knot cost, particle-major

Each rule reduces the knot axis itself (total return = sum over K). These tests
pin the canonical definitions and the runtime shape contract so an axis/shape
slip — like the per-knot MPPI bug — fails loudly instead of silently.
"""
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pfmp.algorithm import ps_update, mppi_update, cem_update


class Theta(NamedTuple):
    mean: jax.Array


class ThetaStd(NamedTuple):
    mean: jax.Array
    stdev: jax.Array


def _rand(shape, seed=0, lo=0.0, hi=1.0):
    return jnp.asarray(np.random.default_rng(seed).uniform(lo, hi, size=shape))


def _zeros_mean(K, A):
    return Theta(mean=jnp.zeros((K, A)))


# ─────────────────────────── ps_update ───────────────────────────

def test_ps_update_picks_argmin_total_cost():
    P, K, A = 4, 3, 2
    trajs = _rand((P, K, A), seed=1)
    costs = jnp.asarray([[5, 5, 5],     # total 15
                         [1, 1, 1],     # total  3  <- best
                         [2, 4, 2],     # total  8
                         [9, 0, 0]], float)  # total 9
    th, aux = ps_update(_zeros_mean(K, A), trajs, costs)
    assert np.allclose(np.asarray(th.mean), np.asarray(trajs[1]))
    assert np.allclose(np.asarray(aux), np.asarray(costs[1]))  # chosen traj's per-knot costs
    assert aux.shape == (K,)


def test_ps_update_mean_shape():
    P, K, A = 6, 4, 2
    th, _ = ps_update(_zeros_mean(K, A), _rand((P, K, A)), _rand((P, K)))
    assert th.mean.shape == (K, A)


# ─────────────────────────── cem_update ───────────────────────────

def test_cem_update_is_mean_of_lowest_total_cost_elites():
    P, K, A = 10, 3, 2
    trajs = _rand((P, K, A), seed=2)
    # per-knot costs whose per-particle total is monotonic in the index
    costs = jnp.tile(jnp.arange(P, dtype=float)[:, None], (1, K))
    th, _ = cem_update(_zeros_mean(K, A), trajs, costs, 0.3)   # top 30% -> idx 0,1,2
    assert np.allclose(np.asarray(th.mean), np.mean(np.asarray(trajs[:3]), axis=0))


def test_cem_update_adapts_covariance_when_stdev_is_per_knot():
    # per-knot stdev (ndim > 0): CEM fits the elite std (covariance adaptation on)
    P, K, A = 10, 3, 2
    trajs = _rand((P, K, A), seed=3)
    costs = jnp.tile(jnp.arange(P, dtype=float)[:, None], (1, K))
    th, _ = cem_update(ThetaStd(jnp.zeros((K, A)), jnp.ones((K, A))), trajs, costs, 0.5)
    assert np.allclose(np.asarray(th.stdev), np.std(np.asarray(trajs[:5]), axis=0))


def test_cem_update_respects_stdev_floor():
    """Identical elites give zero std, which would freeze exploration; the floor
    is what keeps CEM sampling (controls snap to a discrete grid, so exact ties
    are reachable in practice)."""
    P, K, A = 10, 3, 2
    trajs = jnp.tile(jnp.ones((1, K, A)), (P, 1, 1))    # every particle identical
    costs = jnp.tile(jnp.arange(P, dtype=float)[:, None], (1, K))
    th, _ = cem_update(ThetaStd(jnp.zeros((K, A)), jnp.ones((K, A))), trajs, costs,
                       0.5, stdev_floor=0.01)
    assert np.allclose(np.asarray(th.stdev), 0.01)


def test_cem_update_covariance_concentrates_on_elites():
    """The refitted std must be the elite spread, not the population spread —
    that shrinkage is the mechanism that makes CEM converge."""
    P, K, A = 20, 2, 1
    # elites (lowest cost) tightly clustered, the rest spread wide
    tight = jnp.full((5, K, A), 1.0) + jnp.linspace(0, 0.01, 5)[:, None, None]
    wide = jnp.full((15, K, A), 1.0) + jnp.linspace(0, 10.0, 15)[:, None, None]
    trajs = jnp.concatenate([tight, wide], axis=0)
    costs = jnp.tile(jnp.arange(P, dtype=float)[:, None], (1, K))
    th, _ = cem_update(ThetaStd(jnp.zeros((K, A)), jnp.ones((K, A))), trajs, costs, 0.25)
    assert float(jnp.max(th.stdev)) < float(jnp.std(trajs))   # concentrated


def test_cem_update_leaves_scalar_stdev_untouched():
    # scalar stdev (ndim 0): a scalar-noise proposal can't take a per-knot std,
    # so the gate keeps covariance adaptation OFF — only the mean updates.
    P, K, A = 10, 3, 2
    trajs = _rand((P, K, A), seed=3)
    costs = jnp.tile(jnp.arange(P, dtype=float)[:, None], (1, K))
    th, _ = cem_update(ThetaStd(jnp.zeros((K, A)), jnp.asarray(0.5)), trajs, costs, 0.5)
    assert float(np.asarray(th.stdev)) == 0.5           # unchanged


def test_cem_update_mean_shape():
    P, K, A = 8, 4, 2
    th, _ = cem_update(_zeros_mean(K, A), _rand((P, K, A)), _rand((P, K)), 0.25)
    assert th.mean.shape == (K, A)


# ─────────────────────────── mppi_update ───────────────────────────
# Canonical MPPI: one weight per TRAJECTORY from its total return, applied to
# every knot. A per-knot softmax (different weights per timestep) is NOT MPPI.

def test_mppi_update_weights_whole_trajectory_by_total_return():
    P, K, A = 4, 3, 2
    trajs = _rand((P, K, A), seed=4)
    costs = _rand((P, K), seed=5, lo=1.0, hi=5.0)
    temp = 1.0
    th, w = mppi_update(_zeros_mean(K, A), trajs, costs, temp)

    total = jnp.sum(costs, axis=1)
    wexp = jnp.exp(-(total - total.min()) / temp); wexp = wexp / wexp.sum()
    expected = jnp.sum(trajs * wexp[:, None, None], axis=0)
    assert np.allclose(np.asarray(th.mean), np.asarray(expected))
    assert np.allclose(np.asarray(w), np.asarray(wexp))
    assert w.shape == (P,)                       # one weight per trajectory, not per knot


def test_mppi_update_stable_softmax_no_nan_on_large_costs():
    P, K, A = 5, 3, 1
    trajs = _rand((P, K, A), seed=6)
    costs = jnp.full((P, K), 1e4).at[0].set(1e4 - 1)
    th, _ = mppi_update(_zeros_mean(K, A), trajs, costs, 1.0)
    assert np.all(np.isfinite(np.asarray(th.mean)))


def test_mppi_update_mean_shape():
    P, K, A = 6, 4, 2
    th, _ = mppi_update(_zeros_mean(K, A), _rand((P, K, A)), _rand((P, K)), 1.0)
    assert th.mean.shape == (K, A)


# ─────────────────────────── shape contract ───────────────────────────
# The runtime check must reject the exact mistake that caused the MPPI bug:
# handing an update a transposed (K, P) cost array.

@pytest.mark.parametrize("update", [ps_update, mppi_update, cem_update])
def test_update_rejects_transposed_costs(update):
    P, K, A = 4, 3, 2
    trajs = _rand((P, K, A))
    bad_costs = _rand((K, P))                    # transposed — must be caught
    hp = (1.0,) if update is not ps_update else ()
    with pytest.raises(AssertionError):
        update(_zeros_mean(K, A), trajs, bad_costs, *hp)


@pytest.mark.parametrize("update", [ps_update, mppi_update, cem_update])
def test_update_rejects_1d_costs(update):
    P, K, A = 4, 3, 2
    trajs = _rand((P, K, A))
    hp = (1.0,) if update is not ps_update else ()
    with pytest.raises(AssertionError):
        update(_zeros_mean(K, A), trajs, _rand((P,)), *hp)   # pre-summed no longer allowed


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
