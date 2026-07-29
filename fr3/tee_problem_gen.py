#!/usr/bin/env python3
"""
tee_problem_gen.py — Push-T problem specification and placement aid.

This module is the single source of truth for what a trial *is*: `sample_problem`
maps (seed, trial) to a start pose and a goal pose, both on the reachable
annulus. The planner imports it so the spec can never drift from what gets run;
running this file as a script opens a live OptiTrack view for placing the real
T-block on the sampled start pose.

Start and goal are drawn on the annulus rather than from an axis-aligned box so
both endpoints are reachable by construction, and the goal is placed at a
*controlled* separation from the start rather than sampled independently — that
keeps per-trial difficulty roughly constant instead of ranging from trivial to
impossible, which would otherwise swamp the method comparison in variance.

Usage:
    python fr3/tee_problem_gen.py --server 192.168.50.181 --local <ip> --trial 0
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

_XML = os.path.join(os.path.dirname(__file__), "..", "models", "mjx_t_push.xml")

# Annulus band the tee is sampled from, centered on the arm's reachable radial
# midpoint (ik_cache.reachable_r_center) so the task lives in the middle of the
# workspace. sample_problem pins the center from the passed cache; these are the
# fallback values (center 0.549) when no cache is given.
SAMPLE_R_HALFWIDTH = 0.09
SAMPLE_R_CENTER    = 0.549
SAMPLE_R_MIN   = SAMPLE_R_CENTER - SAMPLE_R_HALFWIDTH
SAMPLE_R_MAX   = SAMPLE_R_CENTER + SAMPLE_R_HALFWIDTH
SAMPLE_PHI_MAX = np.deg2rad(55.0)   # half-angle of the frontal wedge, about +x

# Controlled start->goal separation.
PUSH_DIST_MIN  = 0.15
PUSH_DIST_MAX  = 0.25
GOAL_DYAW_MAX  = np.pi              # required yaw change, symmetric

REACHABLE_THRESH = 0.75             # min reachable_frac for both endpoints
_MAX_RESAMPLE    = 200

# The arm descends to this EE position (matches push_tee_planner's --init-xy
# default) before planning. Reject block inits whose actual footprint (T-shape
# SDF, orientation-aware) sits under the descending end-effector — a fixed
# bounding radius would carve out far too much of the workspace.
EE_START      = np.array([0.4, 0.0])
EE_CLEARANCE  = 0.03   # pusher radius (~0.02) + ~1cm margin from the tee surface


def _on_annulus(rng, r_min, r_max):
    r   = rng.uniform(r_min, r_max)
    phi = rng.uniform(-SAMPLE_PHI_MAX, SAMPLE_PHI_MAX)
    return np.array([r * np.cos(phi), r * np.sin(phi)])


def sample_problem(seed, trial, cache=None):
    """Deterministically map (seed, trial) -> (init_xy, init_yaw, goal_xy, goal_yaw).

    Start is uniform on the frontal annulus wedge; the goal is placed a
    controlled distance away and re-projected onto the annulus, with a required
    yaw change drawn independently. If `cache` is given, both endpoints are
    rejection-sampled against the IK cache's reachable mask so a trial can never
    prescribe a pose the arm cannot service.

    The same (seed, trial) yields the same problem for every method, which is
    what makes the across-method comparison paired.
    """
    rng = np.random.default_rng([seed, trial])

    frac = None
    r_min, r_max = SAMPLE_R_MIN, SAMPLE_R_MAX
    if cache is not None:
        import jax.numpy as jnp
        from ik_cache import reachable_frac, reachable_r_center
        frac = lambda xy: float(reachable_frac(jnp.asarray(xy), cache))
        # Pin the sampling band to the arm's reachable radial midpoint so the
        # task sits in the center of the workspace, tracking the same geometry
        # as the annulus cost in push_tee_planner.
        rc = reachable_r_center(cache)
        r_min, r_max = rc - SAMPLE_R_HALFWIDTH, rc + SAMPLE_R_HALFWIDTH

    for _ in range(_MAX_RESAMPLE):
        init_xy = _on_annulus(rng, r_min, r_max)
        d       = rng.uniform(PUSH_DIST_MIN, PUSH_DIST_MAX)
        heading = rng.uniform(-np.pi, np.pi)
        goal_xy = init_xy + d * np.array([np.cos(heading), np.sin(heading)])

        # Re-project the goal onto the annulus band so it stays reachable, then
        # keep the pair only if the push is still a meaningful distance.
        r = np.linalg.norm(goal_xy)
        if r < 1e-9:
            continue
        goal_xy = goal_xy * (np.clip(r, r_min, r_max) / r)
        if np.linalg.norm(goal_xy - init_xy) < PUSH_DIST_MIN * 0.6:
            continue

        if frac is not None and (frac(init_xy) < REACHABLE_THRESH
                                 or frac(goal_xy) < REACHABLE_THRESH):
            continue

        init_yaw = rng.uniform(-np.pi, np.pi)
        # Orientation-aware: reject if the descending EE lands within EE_CLEARANCE
        # of the actual T-shape surface at this init pose (uses the tee SDF, not a
        # bounding radius, so only genuinely-fouled placements are dropped).
        from geom_tee import sdf_world
        if sdf_world(EE_START, init_xy, init_yaw) < EE_CLEARANCE:
            continue
        goal_yaw = _wrap(init_yaw + rng.uniform(-GOAL_DYAW_MAX, GOAL_DYAW_MAX))
        return init_xy, float(init_yaw), goal_xy, float(goal_yaw)

    raise RuntimeError(
        f"sample_problem(seed={seed}, trial={trial}) failed to find a reachable "
        f"start/goal pair in {_MAX_RESAMPLE} draws; check the IK cache mask."
    )


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi



def _workspace_bounds():
    import xml.etree.ElementTree as ET

    try:
        tree = ET.parse(_XML)
        for geom in tree.iter('geom'):
            if geom.get('name') == 'workspace':
                pos  = [float(v) for v in geom.get('pos').split()]
                size = [float(v) for v in geom.get('size').split()]
                return np.array([[pos[0]-size[0], pos[0]+size[0]],
                                 [pos[1]-size[1], pos[1]+size[1]]])
    except Exception:
        pass
    return np.array([[0.08, 0.63], [-0.68, 0.68]])


def main():
    import threading
    import time

    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.patches import Polygon as MplPolygon

    from push_tee_optitrack import tee_pose
    from geom_tee import verts_world
    from new_natnet_client.Client import NatNetClient, NatNetParams

    parser = argparse.ArgumentParser()
    parser.add_argument("--server",    default="127.0.0.1")
    parser.add_argument("--local",     default="127.0.0.1")
    parser.add_argument("--seed",     type=int, default=42)
    parser.add_argument("--trial",    type=int, default=0)
    parser.add_argument("--tol",      type=float, default=0.03, help="Position tolerance (m)")
    parser.add_argument("--no-mask",  action="store_true",
                        help="Skip the IK-cache reachability check when sampling")
    args = parser.parse_args()

    ws = _workspace_bounds()
    state = {"xy": np.array([0.5, 0.0]), "yaw": 0.0}
    lock  = threading.Lock()

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_xlim(*ws[1][::-1])
    ax.set_ylim(*ws[0])
    ax.set_aspect("equal")
    ax.set_xlabel("y (m)")
    ax.set_ylabel("x (m)")
    ax.set_title("Place tee at ghost  |  orange=live  |  gray=target init  |  blue=sampling region")
    ax.grid(True, alpha=0.3)

    cache = None
    if not args.no_mask:
        from ik_cache import get_or_build_cache
        cache = get_or_build_cache()

    # IK reachable region (axes swapped: y→x-axis, x→y-axis). Shaded underneath
    # everything so the sampling band and ghosts read against the true workspace.
    r_min, r_max = SAMPLE_R_MIN, SAMPLE_R_MAX
    if cache is not None:
        from ik_cache import reachable_r_center
        reach = np.asarray(cache.reachable) > 0.5
        xs = np.linspace(float(cache.xy_min[0]), float(cache.xy_max[0]), reach.shape[0])
        ys = np.linspace(float(cache.xy_min[1]), float(cache.xy_max[1]), reach.shape[1])
        X, Y = np.meshgrid(xs, ys, indexing="ij")
        ax.contourf(Y, X, reach.astype(float), levels=[0.5, 1.5],
                    colors=["darkseagreen"], alpha=0.18, zorder=0)
        ax.contour(Y, X, reach.astype(float), levels=[0.5],
                   colors="darkseagreen", linewidths=1.2, alpha=0.6, zorder=1)
        rc = reachable_r_center(cache)
        r_min, r_max = rc - SAMPLE_R_HALFWIDTH, rc + SAMPLE_R_HALFWIDTH

    # Sampling annulus (axes swapped: y→x-axis, x→y-axis)
    arc = np.linspace(-SAMPLE_PHI_MAX, SAMPLE_PHI_MAX, 200)
    for r in (r_min, r_max):
        ax.plot(r * np.sin(arc), r * np.cos(arc),
                color="steelblue", lw=1.5, alpha=0.7, zorder=2)
    ax.fill(np.concatenate([r_max * np.sin(arc), r_min * np.sin(arc[::-1])]),
            np.concatenate([r_max * np.cos(arc), r_min * np.cos(arc[::-1])]),
            color="steelblue", alpha=0.12, label="Sampling annulus", zorder=2)

    init_xy, init_yaw, goal_xy, goal_yaw = sample_problem(args.seed, args.trial, cache)
    print(f"Trial {args.trial} (seed {args.seed}):")
    print(f"  init = ({init_xy[0]:.3f}, {init_xy[1]:.3f}, {np.degrees(init_yaw):+.1f}°)")
    print(f"  goal = ({goal_xy[0]:.3f}, {goal_xy[1]:.3f}, {np.degrees(goal_yaw):+.1f}°)")
    print(f"  push = {np.linalg.norm(goal_xy - init_xy)*100:.1f} cm, "
          f"dyaw = {np.degrees(_wrap(goal_yaw - init_yaw)):+.1f}°")
    print(f"\n  Pass these to the planner:\n"
          f"  --goal-x {goal_xy[0]:.4f} --goal-y {goal_xy[1]:.4f} --goal-yaw {goal_yaw:.4f}\n")

    # Target initial pose (ghost) and goal pose (ghost)
    init_patch = MplPolygon(verts_world(init_xy, init_yaw)[:, ::-1], closed=True,
                            facecolor="gray", edgecolor="dimgray",
                            alpha=0.35, linewidth=2, linestyle="--",
                            label="Target init", zorder=3)
    ax.add_patch(init_patch)
    goal_patch = MplPolygon(verts_world(goal_xy, goal_yaw)[:, ::-1], closed=True,
                            facecolor="mediumseagreen", edgecolor="darkgreen",
                            alpha=0.30, linewidth=2, linestyle="--",
                            label="Goal", zorder=3)
    ax.add_patch(goal_patch)
    ax.annotate("", xy=goal_xy[::-1], xytext=init_xy[::-1],
                arrowprops=dict(arrowstyle="->", color="darkgreen", lw=1.5, alpha=0.7))

    # Tolerance circle around target
    theta_c = np.linspace(0, 2 * np.pi, 200)
    ax.plot(init_xy[1] + args.tol * np.cos(theta_c),
            init_xy[0] + args.tol * np.sin(theta_c),
            "gray", linewidth=1, linestyle=":", alpha=0.6)

    # Live tee overlay
    tee_patch = MplPolygon(np.zeros((8, 2)), closed=True,
                           facecolor="darkorange", edgecolor="saddlebrown",
                           alpha=0.6, linewidth=1.5, label="Tee (live)", zorder=5)
    ax.add_patch(tee_patch)

    status_text = ax.text(0.5, 0.97, "", transform=ax.transAxes,
                          ha="center", va="top", fontsize=11, fontweight="bold")

    ax.legend(loc="upper right", fontsize=8)

    def optitrack_loop(client):
        for frame in client.MoCap(timeout=None):
            rbs = [rb for rb in frame.rigid_body_data.rigid_bodies if rb.tracking]
            if not rbs:
                continue
            xy, yaw = tee_pose(rbs[0])
            with lock:
                state["xy"]  = xy
                state["yaw"] = yaw

    def plot_loop():
        while plt.fignum_exists(fig.number):
            with lock:
                xy  = state["xy"].copy()
                yaw = state["yaw"]
            verts  = verts_world(xy, yaw)
            pverts = verts[:, ::-1]
            tee_patch.set_xy(pverts)
            dist = np.linalg.norm(xy - init_xy)
            if dist < args.tol:
                status_text.set_text("✓ READY")
                status_text.set_color("green")
            else:
                status_text.set_text(f"Δ = {dist*100:.1f} cm")
                status_text.set_color("firebrick")
            fig.canvas.draw_idle()
            time.sleep(0.05)

    params = NatNetParams(server_address=args.server, local_ip_address=args.local,
                          use_multicast=False)
    print(f"Connecting to Motive at {args.server}...")
    with NatNetClient(params) as client:
        if client is None:
            print("Failed to connect."); return
        print("Connected.")
        threading.Thread(target=optitrack_loop, args=(client,), daemon=True).start()
        threading.Thread(target=plot_loop, daemon=True).start()
        plt.show()


if __name__ == "__main__":
    main()
