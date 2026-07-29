"""trace_bounds.py — Trace the boundary of the IK-cache reachable region.

Follows the outer contour of the cache's connected reachable component (Moore-
neighbour tracing), converts each boundary cell to an XY target, and drives the
robot around the perimeter. --dry-run prints the path without a robot.

Usage:
    python src/trace_bounds.py --ip 172.16.0.3
    python src/trace_bounds.py --dry-run
"""

import os
os.environ["JAX_PLATFORMS"] = "cpu"

import argparse

import numpy as np
import jax.numpy as jnp

from ik_cache import get_or_build_cache, ik_from_cache, validate_xy, HOME_Q

# 8-connected neighbour offsets, clockwise starting from "up" (−i).
_DIRS = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]


def trace_boundary(mask: np.ndarray):
    """Ordered list of (i, j) cells forming the outer contour of the foreground
    (True) region, via Moore-neighbour tracing with Jacob's stopping criterion.
    """
    Nx, Ny = mask.shape

    def fg(i, j):
        return 0 <= i < Nx and 0 <= j < Ny and mask[i, j]

    pts = np.argwhere(mask)
    start = tuple(pts[np.lexsort((pts[:, 1], pts[:, 0]))[0]])  # min i, then min j

    boundary = [start]
    cur = start
    # Backtrack pixel: the cell we "came from" — start just outside, above start.
    back = (start[0] - 1, start[1])
    while True:
        bi = _DIRS.index((back[0] - cur[0], back[1] - cur[1]))
        nxt = None
        for k in range(1, 9):
            idx = (bi + k) % 8
            ni, nj = cur[0] + _DIRS[idx][0], cur[1] + _DIRS[idx][1]
            if fg(ni, nj):
                nxt = (ni, nj)
                back = (cur[0] + _DIRS[(idx - 1) % 8][0], cur[1] + _DIRS[(idx - 1) % 8][1])
                break
        if nxt is None:          # isolated pixel
            break
        cur = nxt
        if cur == start:         # returned to start — loop closed
            break
        boundary.append(cur)
    return boundary


def boundary_xy(cache, step: int = 1):
    """Boundary of the reachable component as ordered XY waypoints."""
    r = np.array(cache.reachable) > 0.5
    Nx, Ny = r.shape
    xs = np.linspace(float(cache.xy_min[0]), float(cache.xy_max[0]), Nx)
    ys = np.linspace(float(cache.xy_min[1]), float(cache.xy_max[1]), Ny)
    loop = trace_boundary(r)
    loop = loop[::step] if step > 1 else loop
    return np.array([[xs[i], ys[j]] for i, j in loop]), loop


def main(args):
    cache = get_or_build_cache(path=args.cache)
    waypoints, cells = boundary_xy(cache, step=args.step)
    print(f"Boundary: {len(waypoints)} waypoints (every {args.step} cell(s)).")

    # Sanity: every waypoint should validate (it's a reachable boundary cell).
    bad = [tuple(w) for w in waypoints if not validate_xy(w, cache)[0]]
    if bad:
        print(f"  WARNING: {len(bad)} boundary waypoints failed validation")

    if args.dry_run:
        for (x, y), (i, j) in zip(waypoints, cells):
            print(f"  cell ({i:2d},{j:2d}) → xy ({x:+.3f}, {y:+.3f})")
        return

    import franky
    print(f"Connecting to {args.ip}…")
    robot = franky.Robot(args.ip)
    robot.relative_dynamics_factor = args.speed

    # Precompute joint configs, then issue a single continuous multi-waypoint
    # motion. Separate per-waypoint JointMotions each ramp to zero velocity, and
    # starting the next before the robot settles causes an acceleration
    # discontinuity ("still moving") reflex — JointWaypointMotion blends instead.
    qs = [np.asarray(ik_from_cache(jnp.array([x, y]), cache), dtype=float) for x, y in waypoints]
    lap_wps = [franky.JointWaypoint(q.tolist()) for q in qs]

    robot.move(franky.JointMotion(HOME_Q.tolist()))
    print("Tracing boundary (Ctrl-C to stop)…")
    try:
        for _ in range(args.laps):
            robot.move(franky.JointWaypointMotion(lap_wps))
    except KeyboardInterrupt:
        pass
    finally:
        robot.move(franky.JointMotion(HOME_Q.tolist()))
        print("Returned to home.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", default="172.16.0.3")
    parser.add_argument("--speed", type=float, default=0.1)
    parser.add_argument("--cache", default=os.path.join(os.path.dirname(__file__), "ik_cache.pkl"))
    parser.add_argument("--step", type=int, default=1, help="use every Nth boundary cell")
    parser.add_argument("--laps", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    main(args)
