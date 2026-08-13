# Vendored reference implementations

Third-party source vendored **verbatim** so the reference tests execute the
original authors' code rather than our re-reading of it, without taking a hard
dependency on repos whose own requirements conflict with ours.

Nothing here is imported by `pfmp/`. It exists only for `tests/`.

## dial_mpc/

DIAL-MPC, used by `tests/test_dial_reference.py` to pin our `dial_step`
weighting and annealing schedule.

| | |
|---|---|
| Upstream | https://github.com/LeCAR-Lab/dial-mpc |
| Commit | `871c84f1cefbe9fa9d83a06b05ab4f0243c1a05c` (2025-05-28) |
| License | Apache-2.0 (see `dial_mpc/LICENSE`) |
| Vendored | `dial_mpc/core/dial_core.py`, `dial_mpc/core/dial_config.py` |

**Why vendored rather than a dev dependency.** Installing dial-mpc pulls
`numpy<2.0.0` (we run numpy 2.x), plus `brax`, `jax_cosmo`, `art`, `emoji`, and
`scienceplots`. That is a forced downgrade of a core dependency and a full RL
framework in exchange for a handful of tests. Vendoring two files avoids all of
it and, unlike a transcription, still runs their actual code.

**What is not upstream.** `dial_mpc/utils/io_utils.py` is a stub: `dial_core`
imports two helpers from it that serve its CLI entry point and are never reached
by the code under test. The `__init__.py` files are empty. Everything else is
byte-identical to the commit above, which
`test_vendored_copy_matches_upstream` verifies whenever a real checkout is
present (set `DIAL_MPC_PATH`), so upstream drift is still detected on machines
that have one.

**Updating.** Re-copy the two files from a newer commit, update the commit hash
above, then run the reference tests. Failures there are the point: they mean
upstream changed something we depend on.
