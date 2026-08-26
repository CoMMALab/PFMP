""" Update Rules

Every update rule is  update(theta, trajs, costs, *hyperparams) -> (theta, aux)
with the SAME convention:
    trajs : PyTree, each leaf (P, K, ...)   particles × knots × action
    costs : (P, K)                          per-knot cost, particle-major

This is the "outer loop"/update axis of the proposal x resampling x update
decomposition: these functions only ever read a completed batch of rollouts,
never the PFMP sampler that produced them (see algorithm.py for that half).
"""
import math

import jax
import jax.numpy as jnp
from jaxtyping import PyTree, PRNGKeyArray


def ess(costs: jax.Array) -> jax.Array:
    """Effective Sample Size normalised to [0, 1] from costs shaped (particles,) or (particles, horizon).

    1.0 = uniform weights (maximum diversity).
    1/N = fully degenerate (one particle dominates).

    """
    total = jnp.sum(costs, axis=1) if costs.ndim == 2 else costs
    n = total.shape[0]
    ws = jnp.exp(-(total - jnp.min(total)))
    ps = ws / jnp.sum(ws)
    return (1.0 / jnp.sum(ps ** 2)) / n


def _check_update_shapes(trajs, costs):
    """Runtime verification shared by all update rules: costs (P, K), trajs with leaf shape
    (P, K, ...). Shapes are static under jit/vmap."""
    assert costs.ndim == 2, f"costs must be (P, K); got {costs.shape}"
    P, K = costs.shape
    for u in jax.tree.leaves(trajs):
        assert u.ndim >= 2 and u.shape[0] == P and u.shape[1] == K, (
            f"trajs leaf {u.shape} inconsistent with costs (P={P}, K={K})")


# Predictive Sampling: keep the single lowest-total-cost trajectory (MAP).
def ps_update(theta: PyTree, trajs: PyTree, costs: jax.Array, *_):
    _check_update_shapes(trajs, costs)
    best = jnp.argmin(jnp.sum(costs, axis=1))             # total return per particle
    new_mean = jax.tree.map(lambda u: u[best], trajs)
    return theta._replace(mean=new_mean), costs[best]     # (K,) costs of the chosen traj


# Posterior sampling: draw one trajectory from the weighted particle set.
def sample_update(theta: PyTree, trajs: PyTree, costs: jax.Array,
                  key: PRNGKeyArray, temperature: float = 1.0):
    _check_update_shapes(trajs, costs)
    total = jnp.sum(costs, axis=1)                        # (P,) total return
    logits = -(total - jnp.min(total)) / temperature      # shift: softmax-invariant
    idx = jax.random.categorical(key, logits)
    new_mean = jax.tree.map(lambda u: u[idx], trajs)
    return theta._replace(mean=new_mean), costs[idx]


# Model Predictive Path Integral: softmax over trajectories by TOTAL return, one
# weight per trajectory applied to every knot.
def mppi_update(theta: PyTree, trajs: PyTree, costs: jax.Array,
                temperature: float = 1.0):
    _check_update_shapes(trajs, costs)
    total = jnp.sum(costs, axis=1)                        # (P,) total return
    # Uniform weights if all particles are NaN
    total = jnp.where(jnp.any(jnp.isfinite(total)), total, 0.0)
    total = total - jnp.min(total)
    weights = jnp.exp(-total / temperature)
    weights = weights / jnp.sum(weights)                  # (P,) one weight per trajectory
    # Contract the particle axis, whatever the leaf's rank.
    new_mean = jax.tree.map(lambda u: jnp.tensordot(weights, u, axes=1), trajs)
    return theta._replace(mean=new_mean), weights


# Cross-Entropy Method: mean (and std) of the lowest-total-cost elite fraction.
def cem_update(theta: PyTree, trajs: PyTree, costs: jax.Array,
               elite_frac: float = 0.1, stdev_floor: float = 0.0):
    _check_update_shapes(trajs, costs)
    total = jnp.sum(costs, axis=1)                        # (P,) total return
    num_elite = max(1, int(elite_frac * total.shape[0]))
    elite = jnp.argsort(total)[:num_elite]
    new_mean = jax.tree.map(lambda u: jnp.mean(u[elite], axis=0), trajs)
    updates = {"mean": new_mean}
    # Refit the sampling covariance from the elites.
    if hasattr(theta, "stdev") and jnp.ndim(theta.stdev) > 0:
        updates["stdev"] = jax.tree.map(
            lambda u: jnp.maximum(jnp.std(u[elite], axis=0), stdev_floor), trajs)
    return theta._replace(**updates), jnp.mean(total[elite])


# DIAL-MPC comes in two pieces. `dial_update` is a normal update rule and obeys
# the convention above. `dial_step` is not: it owns an outer optimizing loop that
# calls the sampler itself rather than just consuming a single batch.
# DIAL's mechanism is N annealing levels, each sampling at a shrinking
# per-knot sigma (`dial_sigma`) around the current mean.
def dial_sigma(i: int, h: int, sigma_max: float, N: int, H: int,
               beta1: float, beta2: float) -> float:
    """DIAL-MPC annealing schedule (matches Hydrax `DIAL.sample_knots`)"""
    return sigma_max * math.exp(-(N - i) / (beta1 * N) - (H - 1 - h) / (beta2 * H))

# one annealing step
def dial_update(theta: PyTree, trajs: PyTree, costs: jax.Array,
                temp_sample: float = 1.0):
    """DIAL's update: MPPI whose temperature is the spread of the returns.
    This is the only thing separating DIAL's update from plain MPPI.
    temp_sample: scales that temperature. Lower is more peaked. Reference
        default is 0.06, we set to 1.0 for simplicity.
    """
    total = jnp.sum(costs, axis=1)
    # Zero out non-finite returns (infinite collision cost) from the std.
    temp = jnp.maximum(
        jnp.std(jnp.where(jnp.isfinite(total), total, 0.0)) * temp_sample, 1e-6)
    return mppi_update(theta, trajs, costs, temp)

# outer loop
def dial_step(key: PRNGKeyArray, ctrl, x0: PyTree, theta: PyTree,
              n: int = 2, beta1: float = 1.0, beta2: float = 1.0,
              temp_sample: float = 0.06, return_costs: bool = False) -> PyTree:
    """One DIAL-MPC replan: `n` annealing levels, each re-sampling at a
    shrinking per-knot sigma (`dial_sigma`) and folding the batch in with
    `dial_update`.

    n: annealing levels. Callers pass dial-mpc's Ndiffuse_init on a cold start
        and Ndiffuse afterwards; this function is stateless between calls.
    return_costs: also return the costs from the FINAL annealing level, so a
        caller can report the same diagnostics (ESS) it computes for the
        single-pass methods. Off by default to keep the return type unchanged.

    Verified against the reference in tests/test_dial_reference.py.
    """
    base = theta.stdev
    num_knots = ctrl.num_knots
    for i in range(n, 0, -1):
        sigma = jnp.stack([dial_sigma(i, h, base, n, num_knots, beta1, beta2)
                           for h in range(num_knots)])
        key, sub = jax.random.split(key)
        trajs, costs = ctrl.run_sis(sub, x0, theta._replace(stdev=sigma))
        theta, _ = dial_update(theta, trajs, costs, temp_sample)
    return (theta, costs) if return_costs else theta
