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

EE_ID     = 12
EE_HEIGHT = 0.075

HOME_Q = jnp.array([0., -0.785398, 0., -2.356194, 0., 1.570796, 0.785398])
REST_WEIGHT = jnp.array([0.0, 0.1, 0.1, 0.1, 1.0, 1.0, 1.0])

EE_ROTATION = jnp.array([
    [1.,  0.,  0.],
    [0., -1.,  0.],
    [0.,  0., -1.],
])

_CACHE_MAX_JUMP = np.pi / 3   # ~60° — larger jumps indicate a wrist flip

WORKSPACE_XY_MIN = jnp.array([0.175, -0.675])
WORKSPACE_XY_MAX = jnp.array([0.715,  0.675])
IK_CACHE_RESOLUTION = 64


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
    xml_str = ET.tostring(xml_tree.getroot(), encoding="unicode")
    return pk.Robot.from_urdf(yourdfpy.URDF.load(StringIO(xml_str)))


PANDA = load_panda_robot()


# ── Cache type ────────────────────────────────────────────────────────────────

class IKCache(NamedTuple):
    xy_min:  jax.Array   # (2,)
    xy_max:  jax.Array   # (2,)
    q_grid:  jax.Array   # (Nx, Ny, 7)
    z:       float

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
        pk.costs.rest_cost(joint_var, HOME_Q, REST_WEIGHT * 0.1),
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
    q = (cache.q_grid[ix,   iy  ] * (1 - fx) * (1 - fy) +
         cache.q_grid[ix,   iy+1] * (1 - fx) *      fy  +
         cache.q_grid[ix+1, iy  ] *      fx  * (1 - fy) +
         cache.q_grid[ix+1, iy+1] *      fx  *      fy)
    return q


# ── Build & validate ───────────────────────────────────────────────────────────

def build_ik_cache(xy_min, xy_max, resolution=IK_CACHE_RESOLUTION,
                   z=EE_HEIGHT, verbose=True) -> IKCache:
    xy_min, xy_max = jnp.asarray(xy_min), jnp.asarray(xy_max)
    xs = jnp.linspace(xy_min[0], xy_max[0], resolution)
    ys = jnp.linspace(xy_min[1], xy_max[1], resolution)

    if verbose:
        print(f"Building IK cache: {resolution}×{resolution} = {resolution**2} points")

    t0     = time.perf_counter()
    q_grid = np.zeros((resolution, resolution, 7))
    home   = np.array(HOME_Q)

    for i, x in enumerate(xs):
        if verbose and i % 10 == 0:
            print(f"  row {i}/{resolution}…")
        for j, y in enumerate(ys):
            q_grid[i, j] = np.array(ik_step_from_xy(jnp.array(home), jnp.array([float(x), float(y)]), z))

    if verbose:
        elapsed = time.perf_counter() - t0
        print(f"  done in {elapsed:.1f}s ({resolution**2 / elapsed:.0f} solves/s)")

    return IKCache(xy_min=xy_min, xy_max=xy_max, q_grid=jnp.array(q_grid), z=z)


def validate_cache(cache: IKCache) -> bool:
    q = np.array(cache.q_grid)
    def _wrapped_max(diff):
        d = (diff + np.pi) % (2 * np.pi) - np.pi
        return float(np.max(np.abs(d)))
    worst = max(_wrapped_max(np.diff(q, axis=0)), _wrapped_max(np.diff(q, axis=1)))
    ok = worst < _CACHE_MAX_JUMP
    status = "valid" if ok else "INVALID"
    print(f"  Cache {status}: max neighbour jump = {np.degrees(worst):.1f}° "
          f"(limit {np.degrees(_CACHE_MAX_JUMP):.0f}°)")
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
