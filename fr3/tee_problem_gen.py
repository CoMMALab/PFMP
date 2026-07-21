#!/usr/bin/env python3
"""
tee_problem_gen.py — Visualize live tee pose, workspace, and sampling region.

Usage:
    python fr3/view_start_region.py --server 192.168.50.181 --local <ip>
"""

import argparse
import os
import sys
import threading
import time

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from matplotlib.patches import Polygon as MplPolygon

sys.path.insert(0, os.path.dirname(__file__))
from push_tee_optitrack import tee_pose
from geom_tee import verts_world
from new_natnet_client.Client import NatNetClient, NatNetParams

_XML = os.path.join(os.path.dirname(__file__), "..", "models", "mjx_t_push.xml")



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
    parser = argparse.ArgumentParser()
    parser.add_argument("--server",    default="127.0.0.1")
    parser.add_argument("--local",     default="127.0.0.1")
    parser.add_argument("--seed",     type=int, default=42)
    parser.add_argument("--trial",    type=int, default=0)
    parser.add_argument("--tol",      type=float, default=0.03, help="Position tolerance (m)")
    parser.add_argument("--sample-x", type=float, nargs=2, default=[0.30, 0.55], metavar=("X_MIN", "X_MAX"))
    parser.add_argument("--sample-y", type=float, nargs=2, default=[-0.25, 0.25], metavar=("Y_MIN", "Y_MAX"))
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

    # Sampling region (axes swapped: y→x-axis, x→y-axis)
    sx0, sx1 = args.sample_x
    sy0, sy1 = args.sample_y
    ax.add_patch(patches.Rectangle(
        (sy0, sx0), sy1 - sy0, sx1 - sx0,
        linewidth=1.5, edgecolor="steelblue", facecolor="steelblue",
        alpha=0.12, label="Sampling region", zorder=2,
    ))

    # Sample initial pose deterministically from region
    rng = np.random.default_rng([args.seed, args.trial])
    sx0, sx1 = args.sample_x
    sy0, sy1 = args.sample_y
    init_x   = rng.uniform(sx0, sx1)
    init_y   = rng.uniform(sy0, sy1)
    init_yaw = rng.uniform(-np.pi, np.pi)
    init_xy  = np.array([init_x, init_y])
    print(f"Trial {args.trial}: init=({init_x:.3f}, {init_y:.3f}, {np.degrees(init_yaw):.1f}°)")

    # Target initial pose (ghost)
    init_verts = verts_world(init_xy, init_yaw)
    init_patch = MplPolygon(init_verts[:, ::-1], closed=True,
                            facecolor="gray", edgecolor="dimgray",
                            alpha=0.35, linewidth=2, linestyle="--",
                            label="Target init", zorder=3)
    ax.add_patch(init_patch)

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
