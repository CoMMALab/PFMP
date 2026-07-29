"""IK cache: build and look up a grid of joint solutions, backed by GeoFIK.

The grid is built analytically from GeoFIK's closed-form Franka IK (`geofikinpy`),
in-memory at startup (~0.1s for 128x128) — there is no pickle to save or load.

Usage:
    python ik_cache.py               # build and validate a cache
    python ik_cache.py --validate    # build and report smoothness only
"""

import argparse
import os
import time
from collections import deque
from contextlib import contextmanager
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

import geofikinpy as g

ROBOT_URDF_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "panda_spherized.urdf")

# FR3 joint position limits (rad), from franka_description robots/fr3/joint_limits.yaml.
# GeoFIK solves full-range Panda IK; we reject any branch that lands outside the FR3
# limits. No extra inset margin: real-robot testing (go_fr3) reaches cleanly down to
# ~0.295m, where the near-base elbow (joint 4) sits ~2° inside its hard stop, and the
# cond(J) branch penalty already steers away from ill-conditioned near-singular poses.
FR3_JOINT_LIMITS = {
    "panda_joint1": (-2.9007,  2.9007),
    "panda_joint2": (-1.8361,  1.8361),
    "panda_joint3": (-2.9007,  2.9007),
    "panda_joint4": (-3.0770, -0.1169),
    "panda_joint5": (-2.8763,  2.8763),
    "panda_joint6": ( 0.4398,  4.6216),
    "panda_joint7": (-3.0508,  3.0508),
}
FR3_LIMIT_MARGIN = 0.0  # rad, kept inside each bound (0 = raw FR3 limits, matches go_fr3)

_FR3_LO = np.array([FR3_JOINT_LIMITS[f"panda_joint{i}"][0] for i in range(1, 8)]) + FR3_LIMIT_MARGIN
_FR3_HI = np.array([FR3_JOINT_LIMITS[f"panda_joint{i}"][1] for i in range(1, 8)]) - FR3_LIMIT_MARGIN

EE_ID     = 12
EE_HEIGHT = 0.060

HOME_Q = jnp.array([0., -0.785398, 0., -2.356194, 0., 1.570796, 0.785398])
_HOME_Q_NP = np.asarray(HOME_Q)
Q7_HOME = float(_HOME_Q_NP[6])   # fix joint-7 redundancy so the branch field is smooth

# Straight-down end-effector orientation (EE z-axis pointing at the table).
EE_ROTATION = jnp.array([
    [1.,  0.,  0.],
    [0., -1.,  0.],
    [0.,  0., -1.],
])
_ROE_NP = np.asarray(EE_ROTATION)

# Branch-selection conditioning penalty (rad per log-unit of cond(J)). Ranking
# GeoFIK's branches by continuity alone can pick one that is locally close in
# joint space but passes near a wrist/elbow singularity (cond(J) blows up). This
# term pulls solutions back into the well-conditioned, de-singularized posture
# family — the role the numeric solver's rest_cost toward HOME_Q used to play.
# Value matches go_fr3's reference geofik solve (ik_inspector_geofik.py).
COND_PENALTY = 0.03

_CACHE_MAX_JUMP = np.pi / 3   # ~60° — larger neighbour jumps indicate a wrist flip

WORKSPACE_XY_MIN = jnp.array([-0.16, -0.85])
WORKSPACE_XY_MAX = jnp.array([0.85,  0.85])
IK_CACHE_RESOLUTION = 128

# Clip cells closer to the base than this radius. The folded near-base configs are
# where the FR3 self-collision-avoidance reflex trips; excluding them is cheaper and
# far more robust than a self-collision cost. With GeoFIK's cond-penalized branch
# selection, real-robot testing (go_fr3) confirmed clean pushes down to ~0.295m, so
# the clip sits just inside that — the near-base solves are ~2x worse conditioned
# (cond ~19-22 vs ~8-10 further out) but still drive reliably.
WORKSPACE_R_MIN = 0.29

# Clip cells beyond this |y| off the workspace: the far large-|y| corners are
# kinematically reachable but the arm swings into the physical cage sides there.
WORKSPACE_Y_ABS_MAX = 0.78


import sys as _sys
_sys.modules.setdefault('ik_cache', _sys.modules[__name__])


# ── Cache type ────────────────────────────────────────────────────────────────

class IKCache(NamedTuple):
    xy_min:    jax.Array   # (2,)
    xy_max:    jax.Array   # (2,)
    q_grid:    jax.Array   # (Nx, Ny, 7)
    z:         float
    reachable: jax.Array   # (Nx, Ny) float 1.0/0.0 — IK actually reaches the cell

IKCache.__module__ = 'ik_cache'


# ── GeoFIK analytic solve ───────────────────────────────────────────────────────

import ctypes as _ctypes
try:
    _libc = _ctypes.CDLL(None)   # for flushing C stdio buffers (fflush(NULL))
except Exception:
    _libc = None


@contextmanager
def _silence_stdio():
    """GeoFIK prints 'unable to assembly kinematic chain' to fd 1/2 on every
    unreachable pose. Redirecting the fds isn't enough on its own: geofik's C
    stdio is block-buffered when fd 1 is not a tty, so the bytes flush to whatever
    fd 1 points at *when the buffer flushes* — after we restore, i.e. the real
    terminal. So flush the C buffers (fflush(NULL)) while still pointed at devnull."""
    o1, o2 = os.dup(1), os.dup(2)
    dn = os.open(os.devnull, os.O_WRONLY)
    os.dup2(dn, 1); os.dup2(dn, 2); os.close(dn)
    try:
        yield
    finally:
        if _libc is not None:
            _libc.fflush(None)   # flush any buffered geofik output to devnull
        os.dup2(o1, 1); os.dup2(o2, 2); os.close(o1); os.close(o2)


def _wrap(d: np.ndarray) -> np.ndarray:
    """Wrap joint-angle differences to (-pi, pi]."""
    return (d + np.pi) % (2 * np.pi) - np.pi


def _rotz(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])


def cond_number(q) -> float:
    """Condition number of the geometric Jacobian at q (sigma_max / sigma_min)."""
    J = np.asarray(g.J_from_q(np.asarray(q, float))).T   # (6, 7)
    s = np.linalg.svd(J, compute_uv=False)
    s = s[s > 1e-9]
    return float(s.max() / s.min()) if len(s) else float("inf")


def _valid_branches(x, y, z, q7=Q7_HOME):
    """All GeoFIK IK branches for a straight-down tool at (x, y, z) within the
    tightened FR3 joint limits. Empty list ⇒ unreachable.

    Only the tool z-axis is constrained (roll about it is a free gauge DOF), so we
    rotate the target roll with the target's polar angle — R = Rz(atan2 y, x) @ EE_ROTATION.
    This lets q1 absorb the azimuth (q1 ≈ theta) instead of forcing GeoFIK to fight
    it with the elbow/wrist, matching go_fr3's production gauge.
    """
    R_target = _rotz(np.arctan2(y, x)) @ _ROE_NP
    with _silence_stdio():
        sols = g.ik_q7(np.array([x, y, z], float), R_target, float(q7))
    out = []
    for s in sols:
        s = np.asarray(s, float)
        if np.all(np.isfinite(s)) and np.all(s >= _FR3_LO) and np.all(s <= _FR3_HI):
            out.append(s)
    return out


def _best_branch(branches, seed_q):
    """Pick the branch minimizing continuity distance to seed_q plus a
    log(cond(J)) conditioning penalty (keeps the elbow/wrist out of near-singular
    postures). Returns (branch, max_per_joint_jump_vs_seed)."""
    seed = np.asarray(seed_q, float)
    best, best_score, best_jump = None, np.inf, np.inf
    for b in branches:
        w = _wrap(b - seed)
        score = float(np.linalg.norm(w)) + COND_PENALTY * np.log1p(cond_number(b))
        if score < best_score:
            best, best_score, best_jump = b, score, float(np.abs(w).max())
    return best, best_jump


def ik_solve(xy, seed_q=_HOME_Q_NP, z=EE_HEIGHT, q7=Q7_HOME):
    """Exact analytic IK for a single (x, y): the best conditioned branch
    continuous with seed_q, or None if unreachable. This is the host-side path —
    no grid, no interpolation."""
    branches = _valid_branches(float(xy[0]), float(xy[1]), z, q7)
    if not branches:
        return None
    return _best_branch(branches, seed_q)[0]


def ik_step_from_xy(prev_q, xy, z=EE_HEIGHT):
    """Analytic IK for `xy`, continuous with `prev_q`. Returns NaNs if unreachable."""
    q = ik_solve(xy, np.asarray(prev_q, float), z)
    return jnp.asarray(q) if q is not None else jnp.full(7, jnp.nan)


# ── Cache lookup (JAX, used by planner) ───────────────────────────────────────

@jax.jit
def ik_from_cache(xy: jax.Array, cache: IKCache) -> jax.Array:
    uv = jnp.clip((xy - cache.xy_min) / (cache.xy_max - cache.xy_min), 0.0, 1.0)
    Nx, Ny = cache.q_grid.shape[:2]
    gx, gy = uv[0] * (Nx - 1), uv[1] * (Ny - 1)
    ix = jnp.clip(jnp.floor(gx).astype(jnp.int32), 0, Nx - 2)
    iy = jnp.clip(jnp.floor(gy).astype(jnp.int32), 0, Ny - 2)
    fx, fy = gx - ix, gy - iy

    # Bilinear weights, then zero out unreachable corners and renormalize so a
    # reachable query never blends in a garbage (unreachable) corner solution.
    w = jnp.array([(1 - fx) * (1 - fy), (1 - fx) * fy, fx * (1 - fy), fx * fy])
    r = jnp.array([cache.reachable[ix,   iy  ], cache.reachable[ix,   iy+1],
                   cache.reachable[ix+1, iy  ], cache.reachable[ix+1, iy+1]])
    qs = jnp.stack([cache.q_grid[ix,   iy  ], cache.q_grid[ix,   iy+1],
                    cache.q_grid[ix+1, iy  ], cache.q_grid[ix+1, iy+1]])
    wm = w * r
    wsum = wm.sum()
    # No reachable corner means there is nothing valid to interpolate from, so
    # reject: return NaN rather than a blend of failed IK solves. Callers check.
    q_masked = (wm[:, None] * qs).sum(0) / jnp.where(wsum > 0, wsum, 1.0)
    return jnp.where(wsum > 0, q_masked, jnp.nan)


@jax.jit
def reachable_frac(xy: jax.Array, cache: IKCache) -> jax.Array:
    """Bilinearly-interpolated reachability of `xy` in [0, 1].

    1.0 = all four surrounding grid corners have a smooth IK solution; 0.0 = none
    do (out-of-range queries clamp to the grid edge, which is unreachable, so they
    score ~0). Used by the planner's proposal to reject EE waypoints that fall
    outside the connected reachable component before they are ever sampled.
    """
    uv = jnp.clip((xy - cache.xy_min) / (cache.xy_max - cache.xy_min), 0.0, 1.0)
    Nx, Ny = cache.reachable.shape
    gx, gy = uv[0] * (Nx - 1), uv[1] * (Ny - 1)
    ix = jnp.clip(jnp.floor(gx).astype(jnp.int32), 0, Nx - 2)
    iy = jnp.clip(jnp.floor(gy).astype(jnp.int32), 0, Ny - 2)
    fx, fy = gx - ix, gy - iy
    return (cache.reachable[ix,   iy  ] * (1 - fx) * (1 - fy) +
            cache.reachable[ix,   iy+1] * (1 - fx) *      fy  +
            cache.reachable[ix+1, iy  ] *      fx  * (1 - fy) +
            cache.reachable[ix+1, iy+1] *      fx  *      fy)


# ── Target validation ─────────────────────────────────────────────────────────

def _ee_position(q) -> np.ndarray:
    """World EE (x, y, z) for a joint config, via GeoFIK forward kinematics."""
    return np.asarray(g.franka_fk(np.asarray(q, float)))[:3, 3]


def validate_xy(xy, cache: IKCache, tol: float = 0.01, z: float = EE_HEIGHT):
    """Check whether `xy` is actually in the cache's solution set.

    `ik_from_cache` always returns *something* (it clamps out-of-range targets to
    the grid edge and blindly interpolates across discontinuities), so a returned
    `q` is not proof the target is reachable. This runs forward kinematics on the
    interpolated `q` and compares the achieved EE position to the request.

    Returns (ok, info) with in_bounds / reachable / residual / z_error / q / achieved.
    """
    xy = np.asarray(xy, dtype=float)
    xy_min = np.asarray(cache.xy_min); xy_max = np.asarray(cache.xy_max)
    in_bounds = bool(np.all(xy >= xy_min) and np.all(xy <= xy_max))

    r = np.asarray(cache.reachable); Nx, Ny = r.shape
    uv = np.clip((xy - xy_min) / (xy_max - xy_min), 0.0, 1.0)
    ix = int(np.clip(np.floor(uv[0] * (Nx - 1)), 0, Nx - 2))
    iy = int(np.clip(np.floor(uv[1] * (Ny - 1)), 0, Ny - 2))
    reachable = bool(r[ix:ix+2, iy:iy+2].sum() > 0.5)

    q = ik_from_cache(jnp.asarray(xy), cache)
    achieved = _ee_position(q)
    residual = float(np.linalg.norm(achieved[:2] - xy))
    z_error = float(abs(achieved[2] - z))
    ok = in_bounds and reachable and residual <= tol and z_error <= tol
    return ok, {
        "in_bounds": in_bounds,
        "reachable": reachable,
        "residual": residual,
        "z_error": z_error,
        "q": np.asarray(q, dtype=float),
        "achieved": achieved,
    }


# ── Build ──────────────────────────────────────────────────────────────────────

def build_ik_cache(xy_min, xy_max, resolution=IK_CACHE_RESOLUTION,
                   z=EE_HEIGHT, verbose=True) -> IKCache:
    """Build a smooth, connected grid of Franka IK solutions from GeoFIK.

    For every cell we take GeoFIK's exact analytic branches (empty ⇒ unreachable),
    then flood-fill from the workspace centre picking, at each step, the branch
    nearest the parent cell's solution. This yields a globally consistent branch
    field (safe to bilinearly interpolate) and, as a by-product, an exact reachable
    mask that is the single connected component around the centre — replacing the
    old FK-residual reachability + flip-repair + seam + island heuristics wholesale.
    """
    xy_min, xy_max = jnp.asarray(xy_min), jnp.asarray(xy_max)
    xs = np.linspace(float(xy_min[0]), float(xy_max[0]), resolution)
    ys = np.linspace(float(xy_min[1]), float(xy_max[1]), resolution)
    N = resolution

    if verbose:
        print(f"Building IK cache (GeoFIK): {N}×{N} = {N**2} points")

    t0 = time.perf_counter()
    branches = np.empty((N, N), dtype=object)
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            branches[i, j] = _valid_branches(x, y, z)
    if verbose:
        elapsed = time.perf_counter() - t0
        has_sol = sum(1 for i in range(N) for j in range(N) if branches[i, j])
        print(f"  solved in {elapsed*1000:.0f}ms ({N**2/elapsed:.0f} solves/s); "
              f"{has_sol}/{N**2} cells have ≥1 branch")

    # Near-base disk clip + far-side cage clip on |y|.
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    workspace_ok = (np.sqrt(X**2 + Y**2) >= WORKSPACE_R_MIN) & (np.abs(Y) <= WORKSPACE_Y_ABS_MAX)

    def usable(i, j):
        return workspace_ok[i, j] and len(branches[i, j]) > 0

    # Seed at the cell nearest the workspace centre that is usable.
    ci = int(np.argmin(np.abs(xs - float((xy_min[0] + xy_max[0]) / 2))))
    cj = int(np.argmin(np.abs(ys - float((xy_min[1] + xy_max[1]) / 2))))
    if not usable(ci, cj):
        cand = [(i, j) for i in range(N) for j in range(N) if usable(i, j)]
        if not cand:
            raise RuntimeError("GeoFIK found no reachable cell in the workspace")
        ci, cj = min(cand, key=lambda ij: (ij[0] - ci) ** 2 + (ij[1] - cj) ** 2)

    q_grid = np.zeros((N, N, 7))
    reach = np.zeros((N, N), bool)
    q_grid[ci, cj] = _best_branch(branches[ci, cj], _HOME_Q_NP)[0]
    reach[ci, cj] = True

    # Flood-fill the connected reachable component, choosing at each cell the
    # branch continuous with its (already-solved) parent. Cells whose best branch
    # would still jump more than _CACHE_MAX_JUMP are treated as a branch-cut
    # boundary and left unreachable, so interpolation never crosses a seam.
    dq = deque([(ci, cj)])
    while dq:
        i, j = dq.popleft()
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ni, nj = i + di, j + dj
            if not (0 <= ni < N and 0 <= nj < N) or reach[ni, nj] or not usable(ni, nj):
                continue
            b, jump = _best_branch(branches[ni, nj], q_grid[i, j])
            if jump <= _CACHE_MAX_JUMP:
                q_grid[ni, nj] = b
                reach[ni, nj] = True
                dq.append((ni, nj))

    if verbose:
        print(f"  reachable (connected, smooth): {int(reach.sum())}/{reach.size}")

    return IKCache(xy_min=xy_min, xy_max=xy_max, q_grid=jnp.array(q_grid), z=z,
                   reachable=jnp.asarray(reach, dtype=jnp.float32))


def reachable_r_center(cache: IKCache) -> float:
    """Radial midpoint of the reachable region: (min_reachable_r + max_reachable_r)/2.

    The natural 'center' of the arm's workspace, equidistant from the near-base
    and far-reach boundaries. Used to pin the task's annulus cost and sampling
    band to the robot's actual geometry rather than hand-picked radii.
    """
    r = np.asarray(cache.reachable) > 0.5
    xs = np.linspace(float(cache.xy_min[0]), float(cache.xy_max[0]), r.shape[0])
    ys = np.linspace(float(cache.xy_min[1]), float(cache.xy_max[1]), r.shape[1])
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    rad = np.hypot(X, Y)[r]
    return float((rad.min() + rad.max()) / 2)


def validate_cache(cache: IKCache) -> bool:
    """Report the worst neighbour jump among *reachable* cell pairs only —
    discontinuities across the unreachable mask boundary are expected and handled
    by mask-aware interpolation, so they don't count against the cache.
    """
    q = np.array(cache.q_grid)
    r = np.array(cache.reachable) > 0.5
    def _wrapped(diff):
        return np.abs((diff + np.pi) % (2 * np.pi) - np.pi).max(-1)
    di = _wrapped(np.diff(q, axis=0)); both_i = r[:-1] & r[1:]
    dj = _wrapped(np.diff(q, axis=1)); both_j = r[:, :-1] & r[:, 1:]
    worst = max(float(di[both_i].max()) if both_i.any() else 0.0,
                float(dj[both_j].max()) if both_j.any() else 0.0)
    ok = worst < _CACHE_MAX_JUMP
    status = "valid" if ok else "INVALID"
    print(f"  Cache {status}: max reachable neighbour jump = {np.degrees(worst):.1f}° "
          f"(limit {np.degrees(_CACHE_MAX_JUMP):.0f}°), "
          f"{int(r.sum())}/{r.size} reachable")
    return ok


def get_or_build_cache(xy_min=WORKSPACE_XY_MIN, xy_max=WORKSPACE_XY_MAX,
                       resolution=IK_CACHE_RESOLUTION, z=EE_HEIGHT,
                       path=None) -> IKCache:
    """Build the IK cache in memory. GeoFIK makes this a sub-second operation, so
    there is no persistence — `path` is accepted for call-site compatibility and
    ignored."""
    cache = build_ik_cache(xy_min, xy_max, resolution=resolution, z=z)
    if not validate_cache(cache):
        print("WARNING: built cache still has discontinuities")
    return cache


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate", action="store_true",
                        help="Build and report smoothness only")
    parser.add_argument("--resolution", type=int, default=IK_CACHE_RESOLUTION)
    args = parser.parse_args()

    cache = build_ik_cache(WORKSPACE_XY_MIN, WORKSPACE_XY_MAX, resolution=args.resolution)
    validate_cache(cache)
