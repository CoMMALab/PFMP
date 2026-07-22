"""IK cache: build, validate, and look up a precomputed grid of joint solutions.

Usage:
    python examples/fr3/ik_cache.py               # build and save cache
    python examples/fr3/ik_cache.py --validate    # load and validate existing cache
"""

import argparse
import os
import pickle
import time
from typing import NamedTuple, Tuple
from io import StringIO
from xml.etree import ElementTree as ET

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import jaxlie
import jaxls
import numpy as np
import pyroki as pk
import yourdfpy

ROBOT_URDF_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "panda_spherized.urdf")
CACHE_PATH      = os.path.join(os.path.dirname(__file__), "ik_cache.pkl")

# FR3 joint position limits (rad), from franka_description robots/fr3/joint_limits.yaml.
# The spherized URDF carries Panda limits, which are looser than the FR3 firmware's on
# joints 4/5 and let IK solutions drift past the FR3 boundary — where the FR3's
# position-dependent velocity ceiling collapses, tripping joint_velocity_violation.
# We override with FR3 limits and pull in by FR3_LIMIT_MARGIN so cached solutions keep
# clear of the boundary.
FR3_JOINT_LIMITS = {
    "panda_joint1": (-2.9007,  2.9007),
    "panda_joint2": (-1.8361,  1.8361),
    "panda_joint3": (-2.9007,  2.9007),
    "panda_joint4": (-3.0770, -0.1169),
    "panda_joint5": (-2.8763,  2.8763),
    "panda_joint6": ( 0.4398,  4.6216),
    "panda_joint7": (-3.0508,  3.0508),
}
FR3_LIMIT_MARGIN = 0.1  # rad, kept inside each bound

EE_ID     = 12
EE_HEIGHT = 0.060

HOME_Q = jnp.array([0., -0.785398, 0., -2.356194, 0., 1.570796, 0.785398])
REST_WEIGHT = jnp.array([0.0, 0.1, 0.1, 0.1, 1.0, 1.0, 1.0])
REST_Q = HOME_Q

EE_ROTATION = jnp.array([
    [1.,  0.,  0.],
    [0., -1.,  0.],
    [0.,  0., -1.],
])

_CACHE_MAX_JUMP = np.pi / 3   # ~60° — larger jumps indicate a wrist flip

WORKSPACE_XY_MIN = jnp.array([-0.16, -0.85])
WORKSPACE_XY_MAX = jnp.array([0.85,  0.85])
IK_CACHE_RESOLUTION = 128

# Clip cells closer to the base than this radius. The folded near-base configs are
# where the FR3 self-collision-avoidance reflex trips; excluding them is cheaper and
# far more robust than a self-collision cost (which shattered the solution field).
WORKSPACE_R_MIN = 0.33


# ── Robot ─────────────────────────────────────────────────────────────────────

def load_panda_robot():
    urdf = yourdfpy.URDF.load(ROBOT_URDF_PATH)
    xml_tree = urdf.write_xml()
    for joint in xml_tree.findall('.//joint[@type="prismatic"]'):
        joint.set("type", "fixed")
        for tag in ("axis", "limit", "dynamics"):
            child = joint.find(tag)
            if child is not None:
                joint.remove(child)
    # Override Panda limits with (margin-tightened) FR3 limits so pyroki's limit_cost
    # keeps IK solutions inside the FR3 firmware's valid range.
    for joint in xml_tree.findall('.//joint[@type="revolute"]'):
        bounds = FR3_JOINT_LIMITS.get(joint.get("name"))
        if bounds is None:
            continue
        lower, upper = bounds[0] + FR3_LIMIT_MARGIN, bounds[1] - FR3_LIMIT_MARGIN
        limit = joint.find("limit")
        if limit is not None:
            limit.set("lower", f"{lower:.4f}")
            limit.set("upper", f"{upper:.4f}")
        safety = joint.find("safety_controller")
        if safety is not None:
            safety.set("soft_lower_limit", f"{lower:.4f}")
            safety.set("soft_upper_limit", f"{upper:.4f}")
    xml_str = ET.tostring(xml_tree.getroot(), encoding="unicode")
    return pk.Robot.from_urdf(yourdfpy.URDF.load(StringIO(xml_str)))


PANDA = load_panda_robot()


# ── Cache type ────────────────────────────────────────────────────────────────

class IKCache(NamedTuple):
    xy_min:    jax.Array   # (2,)
    xy_max:    jax.Array   # (2,)
    q_grid:    jax.Array   # (Nx, Ny, 7)
    z:         float
    reachable: jax.Array   # (Nx, Ny) float 1.0/0.0 — IK actually reaches the cell

import sys as _sys
_sys.modules.setdefault('ik_cache', _sys.modules[__name__])
IKCache.__module__ = 'ik_cache'


# ── IK solver ─────────────────────────────────────────────────────────────────

def _ee_z_axis_residual(vals, robot, joint_var, target_link_index, weight):
    Ts   = robot.forward_kinematics(vals[joint_var])
    R    = jaxlie.SE3(Ts[..., target_link_index, :]).rotation().as_matrix()
    ee_z = R @ jnp.array([0., 0., 1.])
    return ((ee_z - jnp.array([0., 0., -1.])) * weight).flatten()

@jdc.jit
def _make_ik_problem(target_mat4, pos_weight=10.0, ori_weight=5.0, limit_weight=100.0):
    R    = target_mat4[:3, :3]
    t    = target_mat4[:3,  3]
    q    = jaxlie.SO3.from_matrix(R).wxyz
    T_target   = jaxlie.SE3.from_rotation_and_translation(jaxlie.SO3(q), t)
    joint_var  = PANDA.joint_var_cls(0)
    factors = [
        pk.costs.pose_cost_analytic_jac(PANDA, joint_var, T_target,
                                         target_link_index=jnp.array(EE_ID),
                                         pos_weight=pos_weight, ori_weight=0.0),
        jaxls.Cost(_ee_z_axis_residual, (PANDA, joint_var, jnp.array(EE_ID), ori_weight)),
        pk.costs.limit_cost(PANDA, joint_var, weight=limit_weight),
        pk.costs.rest_cost(joint_var, REST_Q, REST_WEIGHT * 0.1),
    ]
    return jaxls.LeastSquaresProblem(factors, [joint_var]).analyze(), joint_var


def _mat4_from_xy(xy, z=EE_HEIGHT):
    xyz = jnp.array([xy[0], xy[1], z])
    T   = jnp.eye(4).at[:3, :3].set(EE_ROTATION).at[:3, 3].set(xyz)
    return T


@jax.jit
def ik_step_from_mat4(prev_q, target_mat4):
    prob, joint_var = _make_ik_problem(target_mat4)
    sol = prob.solve(
        initial_vals=jaxls.VarValues.make([joint_var.with_value(prev_q)]),
        verbose=False,
        linear_solver="dense_cholesky",
        trust_region=jaxls.TrustRegionConfig(lambda_initial=1.),
    )
    q = sol[joint_var]
    return q, q


def ik_step_from_xy(prev_q, xy, z=EE_HEIGHT):
    return ik_step_from_mat4(prev_q, _mat4_from_xy(xy, z))[0]


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
    # reject: return NaN rather than a blend of failed IK solves (which put the
    # EE up to 19cm off target height). Callers must check before commanding.
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

@jax.jit
def _ee_position(q: jax.Array) -> jax.Array:
    """World EE (x, y, z) for a joint config, via forward kinematics."""
    Ts = PANDA.forward_kinematics(q)
    return jaxlie.SE3(Ts[..., EE_ID, :]).translation()


def validate_xy(xy, cache: IKCache, tol: float = 0.01, z: float = EE_HEIGHT):
    """Check whether `xy` is actually in the cache's solution set.

    `ik_from_cache` always returns *something* (it clamps out-of-range targets to
    the grid edge and blindly interpolates across discontinuities), so a returned
    `q` is not proof the target is reachable. This runs forward kinematics on the
    interpolated `q` and compares the achieved EE position to the request.

    Returns (ok, info) where info has:
      in_bounds:  xy within [xy_min, xy_max] before clamping
      residual:   ||achieved_xy - target_xy|| (m)
      z_error:    |achieved_z - z| (m)
      q:          the interpolated joint config
      achieved:   achieved EE (x, y, z)
    Both the residual and z_error must be within `tol` for ok=True. A large
    residual with in_bounds=True flags a discontinuity/wrist-flip cell.
    """
    xy = np.asarray(xy, dtype=float)
    xy_min = np.asarray(cache.xy_min); xy_max = np.asarray(cache.xy_max)
    in_bounds = bool(np.all(xy >= xy_min) and np.all(xy <= xy_max))

    # Reachable support: are any of the four surrounding grid corners reachable?
    r = np.asarray(cache.reachable); Nx, Ny = r.shape
    uv = np.clip((xy - xy_min) / (xy_max - xy_min), 0.0, 1.0)
    ix = int(np.clip(np.floor(uv[0] * (Nx - 1)), 0, Nx - 2))
    iy = int(np.clip(np.floor(uv[1] * (Ny - 1)), 0, Ny - 2))
    reachable = bool(r[ix:ix+2, iy:iy+2].sum() > 0.5)

    q = ik_from_cache(jnp.asarray(xy), cache)
    achieved = np.asarray(_ee_position(q), dtype=float)
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


# ── Build & validate ───────────────────────────────────────────────────────────

_REACH_TOL = 0.005   # m — FK residual above this means the cell isn't reachable


def _compute_reachable(q_grid, xs, ys, z) -> np.ndarray:
    """Boolean (Nx, Ny) mask: does each cell's solution actually reach its target?

    Clips the unreachable regions — the near-base blob (too close for a straight-
    down EE) and the far corners (beyond max reach).
    """
    Nx, Ny = q_grid.shape[:2]
    reach = np.zeros((Nx, Ny), bool)
    for i in range(Nx):
        ee  = np.asarray(_ee_position(jnp.asarray(q_grid[i])))          # (Ny, 3)
        tgt = np.stack([np.full(Ny, xs[i]), ys, np.full(Ny, z)], axis=1)
        reach[i] = np.linalg.norm(ee - tgt, axis=1) < _REACH_TOL
    return reach


def _neighbour_flips(q_grid, reach, i, j, thresh) -> int:
    """Count reachable neighbours cell (i,j) disagrees with by > thresh (rad)."""
    Nx, Ny = reach.shape
    c = 0
    for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        ni, nj = i + di, j + dj
        if 0 <= ni < Nx and 0 <= nj < Ny and reach[ni, nj]:
            d = (q_grid[i, j] - q_grid[ni, nj] + np.pi) % (2 * np.pi) - np.pi
            if np.abs(d).max() > thresh:
                c += 1
    return c


def _repair_flips(q_grid, reach, xs, ys, z, n_sweeps=6, verbose=True) -> int:
    """Re-solve reachable cells that flip vs their reachable neighbours, seeding
    from a consistent neighbour. Accept only if it lowers local disagreement, so
    the already-smooth interior is never disturbed. Returns remaining flip edges.
    """
    thresh = _CACHE_MAX_JUMP
    Nx, Ny = reach.shape
    for _ in range(n_sweeps):
        changed = 0
        flips = [(i, j) for i in range(Nx) for j in range(Ny)
                 if reach[i, j] and _neighbour_flips(q_grid, reach, i, j, thresh) > 0]
        if not flips:
            break
        for i, j in flips:
            # Seed from the reachable neighbour that is most consistent with the bulk.
            best, seed_ij = 99, None
            for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                ni, nj = i + di, j + dj
                if 0 <= ni < Nx and 0 <= nj < Ny and reach[ni, nj]:
                    fc = _neighbour_flips(q_grid, reach, ni, nj, thresh)
                    if fc < best:
                        best, seed_ij = fc, (ni, nj)
            if seed_ij is None:
                continue
            before = _neighbour_flips(q_grid, reach, i, j, thresh)
            new_q = np.asarray(ik_step_from_xy(jnp.asarray(q_grid[seed_ij]),
                                               jnp.array([float(xs[i]), float(ys[j])]), z))
            old_q = q_grid[i, j].copy()
            q_grid[i, j] = new_q
            if _neighbour_flips(q_grid, reach, i, j, thresh) < before:
                changed += 1
            else:
                q_grid[i, j] = old_q
        if changed == 0:
            break
    remaining = sum(_neighbour_flips(q_grid, reach, i, j, thresh)
                    for i in range(Nx) for j in range(Ny) if reach[i, j]) // 2
    if verbose:
        print(f"  repair: {remaining} reachable flip edge(s) remaining")
    return remaining


def build_ik_cache(xy_min, xy_max, resolution=IK_CACHE_RESOLUTION,
                   z=EE_HEIGHT, verbose=True) -> IKCache:
    xy_min, xy_max = jnp.asarray(xy_min), jnp.asarray(xy_max)
    xs = np.linspace(float(xy_min[0]), float(xy_max[0]), resolution)
    ys = np.linspace(float(xy_min[1]), float(xy_max[1]), resolution)

    if verbose:
        print(f"Building IK cache: {resolution}×{resolution} = {resolution**2} points")

    t0     = time.perf_counter()
    q_grid = np.zeros((resolution, resolution, 7))
    home   = jnp.array(HOME_Q)

    for i, x in enumerate(xs):
        if verbose and i % 10 == 0:
            print(f"  row {i}/{resolution}…")
        for j, y in enumerate(ys):
            q_grid[i, j] = np.array(ik_step_from_xy(home, jnp.array([float(x), float(y)]), z))

    if verbose:
        elapsed = time.perf_counter() - t0
        print(f"  done in {elapsed:.1f}s ({resolution**2 / elapsed:.0f} solves/s)")

    # Clip the near-base disk (self-collision-prone folds).
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    radius_ok = np.sqrt(X**2 + Y**2) >= WORKSPACE_R_MIN
    reach = _compute_reachable(q_grid, xs, ys, z) & radius_ok
    if verbose:
        print(f"  reachable cells (r >= {WORKSPACE_R_MIN}): {int(reach.sum())}/{reach.size}")
    _repair_flips(q_grid, reach, xs, ys, z, verbose=verbose)
    reach = _compute_reachable(q_grid, xs, ys, z) & radius_ok  # repair may shift reachability

    # Mask out the residual branch-cut seam: any reachable cell that still flips
    # against a reachable neighbour is unusable (interpolation across it is
    # meaningless), so fold it into the unreachable mask. Iterate until the
    # remaining reachable region is fully smooth.
    Nx, Ny = reach.shape
    while True:
        seam = [(i, j) for i in range(Nx) for j in range(Ny)
                if reach[i, j] and _neighbour_flips(q_grid, reach, i, j, _CACHE_MAX_JUMP) > 0]
        if not seam:
            break
        for i, j in seam:
            reach[i, j] = False
    if verbose:
        print(f"  reachable after seam mask: {int(reach.sum())}/{reach.size}")

    # Drop islands: keep only the connected reachable component containing the
    # working center, so ik_from_cache never lands in a small isolated pocket.
    from collections import deque
    ci = int(np.argmin(np.abs(xs - float((xy_min[0] + xy_max[0]) / 2))))
    cj = int(np.argmin(np.abs(ys - float((xy_min[1] + xy_max[1]) / 2))))
    if not reach[ci, cj]:  # center unreachable → fall back to largest component's seed
        idx = np.argwhere(reach)
        ci, cj = idx[len(idx) // 2]
    keep = np.zeros_like(reach)
    dq = deque([(ci, cj)]); keep[ci, cj] = True
    while dq:
        i, j = dq.popleft()
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ni, nj = i + di, j + dj
            if 0 <= ni < Nx and 0 <= nj < Ny and reach[ni, nj] and not keep[ni, nj]:
                keep[ni, nj] = True
                dq.append((ni, nj))
    dropped = int(reach.sum() - keep.sum())
    reach = keep
    if verbose:
        print(f"  reachable after island removal: {int(reach.sum())}/{reach.size} "
              f"({dropped} island cell(s) dropped)")

    return IKCache(xy_min=xy_min, xy_max=xy_max, q_grid=jnp.array(q_grid), z=z,
                   reachable=jnp.asarray(reach, dtype=jnp.float32))


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
                       path=CACHE_PATH) -> IKCache:
    if os.path.exists(path):
        print(f"Loading cache from {path}…")
        with open(path, "rb") as f:
            cache = pickle.load(f)
        if validate_cache(cache):
            return cache
        print("  Validation failed — using cache anyway (run ik_cache.py directly to rebuild).")
        return cache

    cache = build_ik_cache(xy_min, xy_max, resolution=resolution, z=z)
    if not validate_cache(cache):
        print("WARNING: rebuilt cache still has discontinuities")
    with open(path, "wb") as f:
        pickle.dump(cache, f)
    print(f"  Saved to {path}")
    return cache


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate", action="store_true", help="Validate existing cache only")
    parser.add_argument("--cache", default=CACHE_PATH)
    parser.add_argument("--resolution", type=int, default=IK_CACHE_RESOLUTION)
    args = parser.parse_args()

    if args.validate:
        with open(args.cache, "rb") as f:
            cache = pickle.load(f)
        validate_cache(cache)
    else:
        get_or_build_cache(resolution=args.resolution, path=args.cache)
