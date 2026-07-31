"""Reference tests for the DIAL-MPC annealing schedule.

Pinned to the authoritative Hydrax implementation (`hydrax/algs/dial.py`,
`DIAL.sample_knots`), whose per-knot sampling noise is

    sigma[i, h] = sigma0 * exp(-i / (b1 * N) - (K - 1 - h) / (b2 * K))

with i the optimization iteration in [0, N) and h the knot index in [0, K).
Our `dial_step` iterates i = N, N-1, ..., 1, so our loop-step i maps to Hydrax
opt_iteration (N - i).
"""
import numpy as np
import pytest

from pfmp.algorithm import dial_sigma


def hydrax_noise_level(opt_iter, K, sigma0, b1, b2, N):
    """Reference schedule, transcribed from hydrax/algs/dial.py."""
    h = np.arange(K)
    return sigma0 * np.exp(-opt_iter / (b1 * N) - (K - 1 - h) / (b2 * K))


@pytest.mark.parametrize("sigma0,b1,b2,N,K",
                         [(0.025, 1.0, 1.0, 5, 5), (0.1, 2.0, 0.5, 4, 6)])
def test_dial_sigma_matches_hydrax_schedule(sigma0, b1, b2, N, K):
    """Our per-knot schedule must equal Hydrax's, step for step."""
    for i in range(N, 0, -1):                       # our loop order
        ours = np.array([dial_sigma(i, h, sigma0, N, K, b1, b2) for h in range(K)])
        ref = hydrax_noise_level(N - i, K, sigma0, b1, b2, N)
        assert np.allclose(ours, ref), (
            f"iteration i={i} (hydrax opt_iter={N-i}): {ours} != {ref}")


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
