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

from pfmp.algorithm import ps_update, mppi_update, cem_update, sample_update


class Theta(NamedTuple):
    mean: jax.Array


class ThetaStd(NamedTuple):
    mean: jax.Array
    stdev: jax.Array


def _rand(shape, seed=0, lo=0.0, hi=1.0):
    return jnp.asarray(np.random.default_rng(seed).uniform(lo, hi, size=shape))


def _zeros_mean(K, A):
    return Theta(mean=jnp.zeros((K, A)))


def _hp(update):
    """The hyperparameters each rule takes after (theta, trajs, costs)."""
    if update is ps_update:
        return ()
    if update is sample_update:
        return (jax.random.key(0),)          # Piche-style draw needs a key
    return (1.0,)                            # mppi temperature / cem elite_frac


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


@pytest.mark.parametrize("update", [ps_update, mppi_update, cem_update, sample_update])
def test_updates_handle_scalar_action_spaces(update):
    """A scalar action gives (P, K) traj leaves, not (P, K, A). A rank-3
    broadcast would silently produce (P, P, K) rather than erroring."""
    P, K = 6, 4
    trajs = _rand((P, K), seed=9)
    costs = _rand((P, K), seed=10)
    hp = _hp(update)
    th, _ = update(Theta(mean=jnp.zeros(K)), trajs, costs, *hp)
    assert th.mean.shape == (K,), f"{update.__name__} gave {th.mean.shape}"


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


# ─────────────────────────── sample_update ───────────────────────────
# Piché et al.: draw from the weighted particle set instead of summarising it.
# It is the stochastic counterpart of ps_update (MAP), so it must return an
# actual particle — never a blend — and must collapse onto the MAP as T -> 0.

def test_sample_update_returns_an_actual_particle():
    P, K, A = 8, 3, 2
    trajs = _rand((P, K, A), seed=11)
    costs = _rand((P, K), seed=12)
    th, _ = sample_update(_zeros_mean(K, A), trajs, costs, jax.random.key(0))
    rows = [np.asarray(trajs[i]) for i in range(P)]
    assert any(np.allclose(np.asarray(th.mean), r) for r in rows), "not a particle"


def test_sample_update_collapses_to_map_at_low_temperature():
    """T -> 0 makes the categorical draw deterministic on the argmin, so it must
    agree with ps_update."""
    P, K, A = 10, 3, 2
    trajs = _rand((P, K, A), seed=13)
    costs = _rand((P, K), seed=14)
    lo, _ = sample_update(_zeros_mean(K, A), trajs, costs, jax.random.key(3), 1e-6)
    mapp, _ = ps_update(_zeros_mean(K, A), trajs, costs)
    assert np.allclose(np.asarray(lo.mean), np.asarray(mapp.mean))


def test_sample_update_is_stochastic_across_keys():
    """High temperature -> near-uniform selection, so different keys must pick
    different particles (otherwise the draw is ignoring the key)."""
    P, K, A = 32, 3, 2
    trajs = _rand((P, K, A), seed=15)
    costs = _rand((P, K), seed=16)
    picks = {tuple(np.asarray(
        sample_update(_zeros_mean(K, A), trajs, costs, jax.random.key(s), 10.0)[0].mean.ravel()))
        for s in range(25)}
    assert len(picks) > 1


def test_sample_update_favours_low_cost_particles():
    """One clearly-best particle must dominate the draws."""
    P, K, A = 12, 2, 1
    trajs = _rand((P, K, A), seed=17)
    costs = jnp.full((P, K), 100.0).at[7].set(0.0)      # particle 7 far better
    hits = sum(
        int(np.allclose(np.asarray(
            sample_update(_zeros_mean(K, A), trajs, costs, jax.random.key(s))[0].mean),
            np.asarray(trajs[7])))
        for s in range(20))
    assert hits == 20


def test_sample_update_aux_is_the_chosen_particles_costs():
    P, K, A = 6, 4, 2
    trajs = _rand((P, K, A), seed=18)
    costs = _rand((P, K), seed=19)
    th, aux = sample_update(_zeros_mean(K, A), trajs, costs, jax.random.key(1))
    assert aux.shape == (K,)


# ─────────────────────────── shape contract ───────────────────────────
# The runtime check must reject the exact mistake that caused the MPPI bug:
# handing an update a transposed (K, P) cost array.

@pytest.mark.parametrize("update", [ps_update, mppi_update, cem_update, sample_update])
def test_update_rejects_transposed_costs(update):
    P, K, A = 4, 3, 2
    trajs = _rand((P, K, A))
    bad_costs = _rand((K, P))                    # transposed — must be caught
    hp = _hp(update)
    with pytest.raises(AssertionError):
        update(_zeros_mean(K, A), trajs, bad_costs, *hp)


@pytest.mark.parametrize("update", [ps_update, mppi_update, cem_update, sample_update])
def test_update_rejects_1d_costs(update):
    P, K, A = 4, 3, 2
    trajs = _rand((P, K, A))
    hp = _hp(update)
    with pytest.raises(AssertionError):
        update(_zeros_mean(K, A), trajs, _rand((P,)), *hp)   # pre-summed no longer allowed


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
