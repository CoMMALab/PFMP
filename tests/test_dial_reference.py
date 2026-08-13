"""Regression tests for `dial_step` against an IMPORT of canonical DIAL-MPC.

`tests/test_hydrax_reference.py` pins `dial_sigma` against *Hydrax's*
`DIAL.sample_knots`. That is a second-hand reference: Hydrax's DIAL uses a plain
fixed-lambda softmax, whereas the original (LeCAR-Lab/dial-mpc) normalises by the
return std and then divides by `temp_sample`.

These tests import the reference rather than transcribing it, so an upstream
change or a mistake in our reading shows up as a failure instead of silent
disagreement. `dial_mpc.core.dial_core` pulls brax / jax_cosmo / scienceplots
at import time, none of which the weighting math touches, so those are stubbed;
`MBDPI.reverse_once` is then bound to a minimal fake so its real body runs on
arrays we control.

The source is vendored verbatim under `tests/vendor/dial_mpc/` (Apache-2.0,
commit 871c84f), so these tests run on any machine without needing a dial-mpc
checkout or its dependencies, which include a `numpy<2.0.0` pin that conflicts
with ours. Set `DIAL_MPC_PATH` to a real checkout to additionally verify the
vendored copy has not drifted from upstream.
"""

import os
import pathlib
import sys
import types
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pfmp.update import dial_sigma, dial_step

# Vendored verbatim copy (see tests/vendor/README.md); always available.
_VENDOR_PATH = str(pathlib.Path(__file__).parent / "vendor")
# Optional real checkout, used only to detect drift from upstream.
_UPSTREAM_PATH = os.environ.get("DIAL_MPC_PATH")


def _import_dial_mpc():
    """Import dial-mpc's real dial_core from the vendored copy, stubbing only
    non-mathematical dependencies (brax, jax_cosmo, plotting)."""
    if _VENDOR_PATH not in sys.path:
        sys.path.insert(0, _VENDOR_PATH)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.style.use = lambda *a, **k: None       # dial_core does plt.style.use("science")

    for name in ("scienceplots", "art", "emoji", "jax_cosmo", "jax_cosmo.scipy",
                 "jax_cosmo.scipy.interpolate", "brax", "brax.io", "brax.envs",
                 "dial_mpc.envs", "dial_mpc.examples"):
        if name not in sys.modules:
            mod = types.ModuleType(name)
            mod.__path__ = []
            sys.modules[name] = mod
    sys.modules["jax_cosmo.scipy.interpolate"].InterpolatedUnivariateSpline = object
    sys.modules["brax.io"].html = None
    sys.modules["dial_mpc.examples"].examples = {}

    try:
        from dial_mpc.core.dial_core import MBDPI
        from dial_mpc.core.dial_config import DialConfig
    except Exception as e:                                  # pragma: no cover
        pytest.skip(f"could not import dial-mpc: {type(e).__name__}: {e}")
    return MBDPI, DialConfig


class Theta(NamedTuple):
    mean: jnp.ndarray
    stdev: jnp.ndarray


def _reference_weights(costs_total, temp_sample):
    """Weights computed by dial-mpc's OWN `reverse_once`, run on our arrays.

    `reverse_once` is a method that would normally need a full brax env, so its
    unwrapped function is bound to a stand-in exposing only the attributes its
    body touches. The weights are captured straight out of the call it makes to
    `self.update_fn(weights, ...)`, so what is compared is the reference's
    actual softmax output, not a re-derivation of it.

    dial-mpc maximises reward where we minimise cost, hence rews = -costs. It
    also appends the nominal as one extra sample, so Nsample = P-1 gives P
    returns total, matching the P particles we hand our own implementation.
    """
    MBDPI, _ = _import_dial_mpc()
    P = len(costs_total)
    rews = -jnp.asarray(costs_total, dtype=jnp.float32)
    captured = {}

    class Args:
        Nsample = P - 1
        Hnode = 0
        temp_sample = None
    Args.temp_sample = temp_sample

    def _rollout(state, us):
        states = types.SimpleNamespace(
            q=jnp.zeros((P, 1, 1)), qd=jnp.zeros((P, 1, 1)),
            x=types.SimpleNamespace(pos=jnp.zeros((P, 1, 1, 3))))
        return rews[:, None], states          # rewss: (P, T=1)

    def _update_fn(weights, Y0s, noise_scale, Ybar_i):
        captured["weights"] = np.asarray(weights)
        return Ybar_i, noise_scale

    class Fake:
        args = Args
        nu = 1
        node2u_vvmap = staticmethod(lambda Y0s: Y0s)
        rollout_us_vmap = staticmethod(_rollout)
        update_fn = staticmethod(_update_fn)

    MBDPI.reverse_once.__wrapped__(
        Fake(), None, jax.random.key(0),
        jnp.zeros((1, 1)),        # Ybar_i: (Hnode+1, nu)
        jnp.zeros(1),             # noise_scale: (Hnode+1,)
    )
    return captured["weights"]


def _ours_weights(costs_total, temp_sample):
    """Weights implied by the REAL `dial_step`, recovered end-to-end.

    `dial_step` with n=1 over a fixed batch reduces to one weighted average of
    the knots, so one-hot knots make the resulting mean *be* the weight vector.
    Any change to how dial_step derives or applies its temperature shows here.
    """
    P = len(costs_total)
    knots = jnp.eye(P)[:, :, None]                     # (P, K, A) with K=P, A=1
    costs = jnp.zeros((P, P)).at[:, 0].set(jnp.asarray(costs_total))

    class FakeCtrl:
        num_knots = P
        def run_sis(self, key, x0, theta):
            return knots, costs

    th = Theta(jnp.zeros((P, 1)), jnp.asarray(0.1))
    out = dial_step(jax.random.key(0), FakeCtrl(), None, th,
                    n=1, temp_sample=temp_sample)
    return np.asarray(out.mean[:, 0])


@pytest.mark.parametrize("H,N", [(6, 2), (5, 5), (4, 3)])
def test_sigma_schedule_matches_dial_mpc_power_form(H, N):
    """Our exponential `dial_sigma` must reproduce dial-mpc's power-form
    schedule exactly, under the parameterisation conversion

        beta1 = -1 / (N * ln traj_diffuse_factor)
        beta2 = -1 / (H * ln horizon_diffuse_factor)

    dial-mpc builds sigma as (dial_core.MBDPI.__init__ and the caller):

        sigma_control       = horizon_diffuse_factor ** arange(Hnode+1)[::-1] * sigma_scale
        traj_diffuse_factors = sigma_control * traj_diffuse_factor ** arange(n_diffuse)[:, None]

    with Hnode + 1 == num_knots. Pinning the conversion matters because our
    DIAL_BETA1/BETA2 default to 1.0, which is NOT dial-mpc's default schedule.
    """
    _, DialConfig = _import_dial_mpc()
    hdf = DialConfig.horizon_diffuse_factor
    tdf = DialConfig.traj_diffuse_factor
    sigma_scale = DialConfig.sigma_scale
    Hnode = H - 1

    beta1 = -1.0 / (N * np.log(tdf))
    beta2 = -1.0 / (H * np.log(hdf))

    for j, i in enumerate(range(N, 0, -1)):        # j = dial-mpc's iteration index
        theirs = np.array([sigma_scale * hdf ** (Hnode - h) * tdf ** j
                           for h in range(H)])
        ours = np.array([dial_sigma(i, h, sigma_scale, N, H, beta1, beta2)
                         for h in range(H)])
        np.testing.assert_allclose(ours, theirs, rtol=1e-9, atol=1e-12)


def test_dial_does_not_adapt_its_covariance_from_samples():
    """dial-mpc's `softmax_update` returns sigma UNCHANGED: DIAL anneals its
    noise on a fixed schedule and never refits it from the elites the way CEM
    does. Our dial_step likewise re-derives sigma from `theta.stdev` each level
    rather than from the samples, so `stdev` must survive a call untouched."""
    _import_dial_mpc()          # installs the stubs dial_core needs
    from dial_mpc.core.dial_core import softmax_update

    w = jnp.array([0.2, 0.3, 0.5])
    Y0s = jnp.arange(3 * 2 * 1, dtype=jnp.float32).reshape(3, 2, 1)
    sigma_in = jnp.array([0.7, 0.7])
    mu_out, sigma_out = softmax_update(w, Y0s, sigma_in, jnp.zeros((2, 1)))
    np.testing.assert_allclose(np.asarray(sigma_out), np.asarray(sigma_in))
    # and the mean is the plain weighted average of the knots
    np.testing.assert_allclose(
        np.asarray(mu_out), np.asarray(jnp.einsum("n,nij->ij", w, Y0s)), rtol=1e-6)

    # ours: stdev unchanged across a dial_step call
    P, K = 8, 3
    knots = jnp.zeros((P, K, 1))
    costs = jnp.zeros((P, K)).at[:, 0].set(jnp.arange(P, dtype=jnp.float32))

    class FakeCtrl:
        num_knots = K
        def run_sis(self, key, x0, theta):
            return knots, costs

    th = Theta(jnp.zeros((K, 1)), jnp.asarray(0.1))
    out = dial_step(jax.random.key(0), FakeCtrl(), None, th, n=3, temp_sample=0.06)
    assert float(out.stdev) == float(th.stdev), (
        "dial_step must not write stdev; the schedule owns it")


def test_vendored_copy_matches_upstream():
    """When a real dial-mpc checkout is available, the vendored files must be
    byte-identical to it. This is what keeps vendoring from silently going
    stale; without a checkout there is nothing to compare, so it skips."""
    if not _UPSTREAM_PATH or not os.path.isdir(_UPSTREAM_PATH):
        pytest.skip("set DIAL_MPC_PATH to a dial-mpc checkout to check for drift")
    for rel in ("dial_mpc/core/dial_core.py", "dial_mpc/core/dial_config.py"):
        vendored = pathlib.Path(_VENDOR_PATH) / rel
        upstream = pathlib.Path(_UPSTREAM_PATH) / rel
        assert upstream.is_file(), f"missing upstream file: {upstream}"
        assert vendored.read_bytes() == upstream.read_bytes(), (
            f"{rel} has drifted from upstream. Re-copy it into tests/vendor/ and "
            "update the commit hash in tests/vendor/README.md, then re-run these "
            "tests: a failure afterwards means upstream changed the algorithm.")


def test_reference_defaults_are_what_we_assume():
    """Pin the upstream config defaults our Push-T settings are derived from.
    If dial-mpc changes these, our DIAL_TEMP_SAMPLE / DIAL_N_INNER are stale."""
    _, DialConfig = _import_dial_mpc()
    assert DialConfig.temp_sample == 0.06
    assert DialConfig.Ndiffuse == 2
    assert DialConfig.Ndiffuse_init == 10


def test_reference_weighting_source_is_unchanged():
    """The weighting line we match, read from the installed source. Guards
    against upstream silently changing the weighting scheme under us."""
    MBDPI, _ = _import_dial_mpc()
    import inspect
    src = inspect.getsource(MBDPI.reverse_once)
    assert "logp0 = (rews - rew_Ybar_i) / rews.std(axis=-1) / self.args.temp_sample" in src
    assert "weights = jax.nn.softmax(logp0)" in src


@pytest.mark.parametrize("temp_sample", [0.04, 0.06, 0.1, 1.0])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_weights_match_dial_mpc_reverse_once(temp_sample, seed):
    """dial_step's weights must equal those of dial-mpc's own reverse_once."""
    rng = np.random.default_rng(seed)
    costs_total = rng.normal(50.0, 8.0, 16)
    np.testing.assert_allclose(
        _ours_weights(costs_total, temp_sample),
        _reference_weights(costs_total, temp_sample),
        rtol=1e-4, atol=1e-6,
    )


def test_temp_sample_controls_concentration():
    """Lower temp_sample must concentrate the weights. Untestable while
    temp_sample did not exist as a parameter."""
    rng = np.random.default_rng(0)
    costs_total = rng.normal(50.0, 8.0, 64)
    ess = [1.0 / np.sum(_ours_weights(costs_total, ts) ** 2)
           for ts in (1.0, 0.3, 0.1, 0.06)]
    assert ess == sorted(ess, reverse=True), f"ESS should fall as temp drops: {ess}"
    assert ess[-1] < 5.0, "at the reference default the update should be sharply peaked"


def test_reference_default_is_far_more_peaked_than_unit_temperature():
    """Guards the specific regression: temp_sample=1.0 (our old pinned value)
    against 0.06 (dial-mpc's config default)."""
    rng = np.random.default_rng(0)
    costs_total = rng.normal(50.0, 8.0, 64)
    ess_unit = 1.0 / np.sum(_ours_weights(costs_total, 1.0) ** 2)
    ess_ref = 1.0 / np.sum(_ours_weights(costs_total, 0.06) ** 2)
    assert ess_unit > 10 * ess_ref, (
        f"unit temperature averages {ess_unit:.1f} particles vs {ess_ref:.1f} at "
        "the reference default; these are materially different algorithms")
