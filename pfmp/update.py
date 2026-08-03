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


def dial_sigma(i: int, h: int, sigma_max: float, N: int, H: int,
               beta1: float, beta2: float) -> float:
    """DIAL-MPC annealing schedule (matches Hydrax `DIAL.sample_knots`):
    σ_{i,h} = σ_max · exp(−(N−i)/(β₁N) − (H−1−h)/(β₂H)). The (H−1−h) term (not
    H−h) is what lets the terminal knot sample at the full σ_max, exp(0)."""
    return sigma_max * math.exp(-(N - i) / (beta1 * N) - (H - 1 - h) / (beta2 * H))


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
    # Refit the sampling covariance from the elites — the "CE" in CEM, and the
    # mechanism that lets the distribution concentrate. Only when `stdev` is
    # per-knot: the elite std is (K, ...), which a scalar-stdev proposal cannot
    # consume. `stdev_floor` guards the degenerate case where every elite is
    # identical (controls are projected onto a discrete reachable grid, so exact
    # ties happen) and the std collapses to zero, freezing exploration.
    if hasattr(theta, "stdev") and jnp.ndim(theta.stdev) > 0:
        updates["stdev"] = jax.tree.map(
            lambda u: jnp.maximum(jnp.std(u[elite], axis=0), stdev_floor), trajs)
    return theta._replace(**updates), jnp.mean(total[elite])


# DIAL-MPC: unlike the four update rules above, this is not a pure readout over
# a completed rollout batch — it owns a multi-round re-sampling loop, calling
# the SAMPLER itself (not just consuming its output). That is DIAL's whole
# mechanism: N annealing levels, each re-samples the proposal at a shrinking
# PER-KNOT sigma (dial_sigma) around the current mean, folding each round in
# with an MPPI update. Per-knot, not averaged across knots: DIAL anneals
# wider->narrower both across annealing levels AND across the horizon
# (canonical dial-mpc's sigma_control is a per-knot vector); averaging
# collapses the second (action-level) half of that dual-loop schedule.
#
# This lives here rather than as a closure in any one driver so every
# entry point (real-time, batch, interactive, sim) imports the same
# implementation instead of hand-rolling its own — the annealing/temperature
# math is easy to get subtly wrong in each retelling.
def dial_step(key: PRNGKeyArray, ctrl, x0: PyTree, theta: PyTree,
              n: int = 2, beta1: float = 1.0, beta2: float = 1.0) -> PyTree:
    """One DIAL-MPC replan.

    ctrl: a pfmp.PFMP sampler; only `ctrl.run_sis` and `ctrl.num_knots` are used.
    x0: initial state passed through to `ctrl.run_sis`.
    theta: current params; must have `.mean` and `.stdev` (the pre-annealing
        scalar sigma_max — fixed at its value on entry, since mppi_update never
        writes `stdev`, so it cannot drift across annealing levels or replans).
    n: annealing levels this call runs. The caller decides whether this is a
        cold-start warm-up (canonical dial-mpc's Ndiffuse_init, e.g. on a
        trial's first replan) or the steady-state budget (Ndiffuse) — dial_step
        itself is stateless between calls.
    """
    base = float(theta.stdev)
    num_knots = ctrl.num_knots
    for i in range(n, 0, -1):
        sigma = jnp.asarray([dial_sigma(i, h, base, n, num_knots, beta1, beta2)
                             for h in range(num_knots)])
        key, sub = jax.random.split(key)
        trajs, costs = ctrl.run_sis(sub, x0, theta._replace(stdev=sigma))
        temp = float(jnp.maximum(jnp.std(jnp.sum(costs, axis=1)), 1e-6))
        theta, _ = mppi_update(theta, trajs, costs, temp)
    return theta
