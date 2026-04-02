"""
IK cache smoke test on the real FR3 robot.

Loads a pre-built IK cache (no GPU needed) and traces a figure-8
through the workspace to verify IK smoothness before running push-T.

Usage:
    python examples/fr3/test_real_pbc.py                    # connect and run
    python examples/fr3/test_real_pbc.py --dry-run          # solve + print, no robot
    python examples/fr3/test_real_pbc.py --cache path/to/ik_cache.pkl
    python examples/fr3/test_real_pbc.py --ip 172.16.0.2 --speed 0.05
"""

import argparse
import pickle
import sys
from typing import NamedTuple
import numpy as np


# Must match pushT.IKCache exactly so pickle can deserialise the cache file.
class IKCache(NamedTuple):
    xy_min: object
    xy_max: object
    q_grid: object
    z: float


class _IKCacheUnpickler(pickle.Unpickler):
    """Redirect any module's IKCache to our local definition."""
    def find_class(self, module, name):
        if name == "IKCache":
            return IKCache
        return super().find_class(module, name)

# ── constants (duplicated from pushT.py to avoid importing JAX) ──────────────

HOME_Q = np.array([0., -0.785398, 0., -2.356194, 0., 1.570796, 0.785398])

DEFAULT_CACHE_PATH = "examples/ik_cache.pkl"
ROBOT_IP = "172.16.0.2"
VELOCITY_SCALE = 0.1

# Figure-8 path (center + radii in metres)
PATH_CENTER = np.array([0.40, 0.00])
PATH_RADIUS = np.array([0.30, 0.6])   # actual reach: ±10 cm X, ±30 cm Y
N_WAYPOINTS = 40
MAX_JUMP_DEG = 20.0


# ── cache lookup (pure numpy bilinear interpolation) ─────────────────────────

def cache_lookup(xy: np.ndarray, cache) -> np.ndarray:
    """Bilinear interpolation into the IK cache grid."""
    xy_min = np.array(cache.xy_min)
    xy_max = np.array(cache.xy_max)
    q_grid = np.array(cache.q_grid)   # (Nx, Ny, 7)
    Nx, Ny = q_grid.shape[:2]

    uv = np.clip((xy - xy_min) / (xy_max - xy_min), 0.0, 1.0)
    gx, gy = uv[0] * (Nx - 1), uv[1] * (Ny - 1)
    ix = int(np.clip(np.floor(gx), 0, Nx - 2))
    iy = int(np.clip(np.floor(gy), 0, Ny - 2))
    fx, fy = gx - ix, gy - iy

    q = (q_grid[ix,   iy  ] * (1 - fx) * (1 - fy) +
         q_grid[ix,   iy+1] * (1 - fx) *      fy  +
         q_grid[ix+1, iy  ] *      fx  * (1 - fy) +
         q_grid[ix+1, iy+1] *      fx  *      fy)
    return q


# ── path generation ───────────────────────────────────────────────────────────

def make_figure8_path(n: int) -> np.ndarray:
    """XY waypoints for a figure-8 (Lissajous 1:2)."""
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    # Rotated 90°: spine along Y, loops left/right
    x = PATH_CENTER[0] + PATH_RADIUS[0] * np.sin(2 * t) / 2
    y = PATH_CENTER[1] + PATH_RADIUS[1] * np.sin(t)
    return np.stack([x, y], axis=1)   # (N, 2)


def solve_trajectory(path_xy: np.ndarray, cache) -> np.ndarray:
    """Look up IK for every waypoint from the cache."""
    qs = np.array([cache_lookup(xy, cache) for xy in path_xy])
    return qs


def check_jumps(qs: np.ndarray, path_xy: np.ndarray = None) -> bool:
    diff = np.diff(qs, axis=0)
    diff = (diff + np.pi) % (2 * np.pi) - np.pi
    abs_diff = np.abs(diff)
    idx = np.unravel_index(np.argmax(abs_diff), abs_diff.shape)
    worst_deg = float(np.degrees(abs_diff[idx]))
    step = idx[0]
    loc = f" xy=({path_xy[step,0]:.3f}, {path_xy[step,1]:.3f})→({path_xy[step+1,0]:.3f}, {path_xy[step+1,1]:.3f})" if path_xy is not None else ""
    print(f"Max jump: {worst_deg:.1f}° at step {step+1}→{step+2}, j{idx[1]+1}{loc}")
    if worst_deg > MAX_JUMP_DEG:
        print(f"ABORT: exceeds {MAX_JUMP_DEG}° limit")
        return False
    return True


# ── visualisation ─────────────────────────────────────────────────────────────

def _plot(path_xy, qs, xy_min, xy_max):
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    fig, (ax_xy, ax_q) = plt.subplots(1, 2, figsize=(13, 5))

    # ── left: XY path ─────────────────────────────────────────────────────────
    # workspace rectangle
    w = xy_max[0] - xy_min[0]
    h = xy_max[1] - xy_min[1]
    ax_xy.add_patch(mpatches.Rectangle(
        (xy_min[0], xy_min[1]), w, h,
        linewidth=1.5, edgecolor="steelblue", facecolor="steelblue", alpha=0.12,
        label="cache bounds",
    ))
    # smooth path (dense)
    dense = make_figure8_path(400)
    ax_xy.plot(dense[:, 0], dense[:, 1], "steelblue", lw=1.5, alpha=0.5, label="path")
    # waypoints, coloured by sequence
    sc = ax_xy.scatter(
        path_xy[:, 0], path_xy[:, 1],
        c=np.arange(len(path_xy)), cmap="viridis", s=60, zorder=3, label="waypoints",
    )
    # number first and last
    ax_xy.annotate("0", path_xy[0],  fontsize=8, ha="right")
    ax_xy.annotate(f"{len(path_xy)-1}", path_xy[-1], fontsize=8, ha="left")
    # robot base
    ax_xy.plot(0, 0, "k^", ms=10, label="base")
    ax_xy.set_xlabel("X (m)"); ax_xy.set_ylabel("Y (m)")
    ax_xy.set_aspect("equal"); ax_xy.legend(fontsize=8)
    ax_xy.set_title("Workspace path")
    plt.colorbar(sc, ax=ax_xy, label="waypoint index")

    # ── right: joint angles along path ────────────────────────────────────────
    qs_deg = np.degrees(qs)
    for j in range(7):
        ax_q.plot(qs_deg[:, j], marker=".", ms=4, label=f"j{j+1}")
    # mark jumps > threshold
    diff = np.abs((np.diff(qs, axis=0) + np.pi) % (2 * np.pi) - np.pi)
    bad_steps = np.where(np.any(diff > np.radians(MAX_JUMP_DEG), axis=1))[0]
    for s in bad_steps:
        ax_q.axvline(s + 0.5, color="red", lw=1, alpha=0.6)
    ax_q.set_xlabel("waypoint"); ax_q.set_ylabel("angle (°)")
    ax_q.legend(fontsize=7, ncol=2); ax_q.grid(True, alpha=0.3)
    ax_q.set_title(f"Joint angles  (red = jump > {MAX_JUMP_DEG}°)")

    plt.tight_layout()
    plt.show()


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--cache", default=DEFAULT_CACHE_PATH)
    parser.add_argument("--ip", default=ROBOT_IP)
    parser.add_argument("--speed", type=float, default=VELOCITY_SCALE)
    args = parser.parse_args()

    print(f"Loading cache from {args.cache}...")
    with open(args.cache, "rb") as f:
        cache = _IKCacheUnpickler(f).load()
    xy_min = np.array(cache.xy_min)
    xy_max = np.array(cache.xy_max)
    print(f"  Grid: {cache.q_grid.shape}  "
          f"x=[{xy_min[0]:.3f}, {xy_max[0]:.3f}]  "
          f"y=[{xy_min[1]:.3f}, {xy_max[1]:.3f}]")

    print(f"Path: figure-8  center={PATH_CENTER}  radii={PATH_RADIUS}  n={N_WAYPOINTS}")
    path_xy = make_figure8_path(N_WAYPOINTS)

    # Sanity-check path is within cache bounds
    if np.any(path_xy[:, 0] < xy_min[0]) or np.any(path_xy[:, 0] > xy_max[0]):
        print("WARNING: path X exceeds cache bounds — will be clipped")
    if np.any(path_xy[:, 1] < xy_min[1]) or np.any(path_xy[:, 1] > xy_max[1]):
        print("WARNING: path Y exceeds cache bounds — will be clipped")

    print("Looking up IK from cache...")
    qs = solve_trajectory(path_xy, cache)

    if not check_jumps(qs, path_xy):
        sys.exit(1)

    if args.dry_run:
        _plot(path_xy, qs, xy_min, xy_max)
        return

    import franky

    print(f"\nConnecting to {args.ip}...")
    robot = franky.Robot(args.ip)
    robot.relative_dynamics_factor = args.speed
    robot.recover_from_errors()
    print("Connected.")

    print("Moving to HOME_Q...")
    robot.move(franky.JointMotion(HOME_Q.tolist()))
    input("At home. Press Enter to move to figure-8 start... ")

    print("Moving to figure-8 start (waypoint 0)...")
    robot.move(franky.JointMotion(qs[0].tolist()))
    input("At start. Press Enter to run figure-8... ")

    print(f"Running ({N_WAYPOINTS} waypoints)...")
    waypoints = [franky.JointWaypoint(q.tolist()) for q in qs]
    robot.move(franky.JointWaypointMotion(waypoints))

    print("Returning to HOME_Q...")
    robot.move(franky.JointMotion(HOME_Q.tolist()))
    print("Done.")


if __name__ == "__main__":
    main()
