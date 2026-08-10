"""Update rules (U) and the annealing/degeneracy helpers they share.

Every update rule is  update(theta, trajs, costs, *hyperparams) -> (theta, aux)
with the SAME convention:
    trajs : PyTree, each leaf (P, K, ...)   particles × knots × action
    costs : (P, K)                          per-knot cost, particle-major
Reducing the knot axis (summing to a total return) is each rule's own job, so
call sites just pass what run_sir/run_sis return — no transpose, no pre-sum.

This is the "outer loop" axis of the proposal x resampling x update
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
    """Runtime contract shared by all update rules: costs (P, K), trajs leaves
    (P, K, ...). Shapes are static under jit/vmap, so this fails at trace time on
    a mismatched call instead of silently reducing the wrong axis."""
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


# Posterior sampling: draw one trajectory from the weighted particle set rather
# than summarising it. This is the update used by Piché et al. (2018) — the
# particle set approximates the controlled posterior, so sampling from it returns
# a control distributed according to that posterior instead of its MAP
# (`ps_update`) or its mean (`mppi_update`). Needs a key, passed as the first
# hyperparameter, so it keeps the shared update signature.
def sample_update(theta: PyTree, trajs: PyTree, costs: jax.Array,
                  key: PRNGKeyArray, temperature: float = 1.0):
    _check_update_shapes(trajs, costs)
    total = jnp.sum(costs, axis=1)                        # (P,) total return
    logits = -(total - jnp.min(total)) / temperature      # shift: softmax-invariant
    idx = jax.random.categorical(key, logits)
    new_mean = jax.tree.map(lambda u: u[idx], trajs)
    return theta._replace(mean=new_mean), costs[idx]


# Model Predictive Path Integral: softmax over trajectories by TOTAL return, one
# weight per trajectory applied to every knot. Stable softmax (shift by the min
# total) so exp() cannot underflow to all-zeros -> 0/0 = NaN.
def mppi_update(theta: PyTree, trajs: PyTree, costs: jax.Array, temperature: float):
    _check_update_shapes(trajs, costs)
    total = jnp.sum(costs, axis=1)                        # (P,) total return
    total = total - jnp.min(total)
    weights = jnp.exp(-total / temperature)
    weights = weights / jnp.sum(weights)                  # (P,) one weight per trajectory
    # Contract the particle axis, whatever the leaf's rank. A scalar action space
    # gives (P, K) leaves, where a hardcoded weights[:, None, None] would silently
    # broadcast to (P, P, K) instead of erroring.
    new_mean = jax.tree.map(lambda u: jnp.tensordot(weights, u, axes=1), trajs)
    return theta._replace(mean=new_mean), weights


# Cross-Entropy Method: mean (and std) of the lowest-total-cost elite fraction.
def cem_update(theta: PyTree, trajs: PyTree, costs: jax.Array, elite_frac: float,
               stdev_floor: float = 0.0):
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
# calls the SAMPLER itself rather than just consuming a single batch, which
# is DIAL's whole mechanism. N annealing levels, each sampling at a shrinking
# PER-KNOT sigma (`dial_sigma`) around the current mean. Per-knot, not averaged
# across knots: DIAL anneals wider->narrower across annealing levels AND across
# the horizon (canonical dial-mpc's sigma_control is a per-knot vector);
# averaging collapses the second (action-level) half of that dual-loop schedule.
def dial_sigma(i: int, h: int, sigma_max: float, N: int, H: int,
               beta1: float, beta2: float) -> float:
    """DIAL-MPC annealing schedule (matches Hydrax `DIAL.sample_knots`):
    σ_{i,h} = σ_max · exp(−(N−i)/(β₁N) − (H−1−h)/(β₂H)). The (H−1−h) term (not
    H−h) is what lets the terminal knot sample at the full σ_max, exp(0)."""
    return sigma_max * math.exp(-(N - i) / (beta1 * N) - (H - 1 - h) / (beta2 * H))

# inner loop
def dial_update(theta: PyTree, trajs: PyTree, costs: jax.Array,
                temp_sample: float = 1.0):
    """DIAL's update: MPPI whose temperature is the spread of the returns.

    This is the only thing separating DIAL's update from plain MPPI. Setting
    the temperature from `std(returns)` makes it adaptive, so one `temp_sample`
    transfers across tasks and cost scales where a fixed lambda would not.
    (Hydrax's DIAL omits this and uses a fixed lambda; the original does not.)

    temp_sample: scales that temperature. Lower is more peaked. Reference
        default is 0.06 as opposed to our 1.0.
    """
    total = jnp.sum(costs, axis=1)
    # Drop non-finite returns (infinite collision cost) from the std, else one
    # collided particle makes the temperature NaN and poisons the update. They
    # still get zero weight inside mppi_update, as they should.
    temp = jnp.maximum(
        jnp.std(jnp.where(jnp.isfinite(total), total, 0.0)) * temp_sample, 1e-6)
    return mppi_update(theta, trajs, costs, temp)

# outer loop
def dial_step(key: PRNGKeyArray, ctrl, x0: PyTree, theta: PyTree,
              n: int = 2, beta1: float = 1.0, beta2: float = 1.0,
              temp_sample: float = 1.0) -> PyTree:
    """One DIAL-MPC replan: `n` annealing levels, each re-sampling at a
    shrinking per-knot sigma (`dial_sigma`) and folding the batch in with
    `dial_update`.

    n: annealing levels. Callers pass dial-mpc's Ndiffuse_init on a cold start
        and Ndiffuse afterwards; this function is stateless between calls.

    Verified against the reference in tests/test_dial_reference.py.
    """
    base = theta.stdev
    num_knots = ctrl.num_knots
    for i in range(n, 0, -1):
        sigma = jnp.stack([dial_sigma(i, h, base, n, num_knots, beta1, beta2)
                           for h in range(num_knots)])
        key, sub = jax.random.split(key)

        # Run **Sequential Importance Sampling** inside the **inner** loop
        trajs, costs = ctrl.run_sis(sub, x0, theta._replace(stdev=sigma))

        theta, _ = dial_update(theta, trajs, costs, temp_sample)
    return theta
