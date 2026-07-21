"""
mbm_figure.py — viser snapshot of an SMC-MPC plan on an MBM problem.

Plans a path with SMC-MPC, then opens a viser scene showing the obstacle
environment plus ghosted robot poses along the plan (start -> intermediates
-> goal) and the end-effector trace. Position the camera in the browser and
click "Save Screenshot" to write a PNG to figures/.

Usage:
    python experiments/mbm_figure.py
    python experiments/mbm_figure.py --dataset models/problems/cage_panda --problem 0
"""

import sys
import time
import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import viser
import yourdfpy
from viser.extras import ViserUrdf
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

from fr3.mbm import (
    load_robot, load_mbm_dataset, Params, Q_MIN, Q_MAX,
    forward_kinematics, dynamics, cost_fn, terminal_cost, init, proposal,
    add_environment_to_viser,
    NUM_KNOTS, HORIZON, ESS_THRESH,
)
from pfmp import PFMP
from pfmp.interpolate import hermite_spline


OUTPUT_DIR = Path(__file__).parent.parent / "figures"
URDF_PATH = "models/panda_spherized.urdf"
NUM_WAYPOINTS = 3   # intermediate poses shown between start and goal
FINGER_OPEN = 0.04  # finger joint value for display only


def plan(dataset, problem_idx, seed=0):
    robot = load_robot(URDF_PATH)
    problems = load_mbm_dataset(dataset, max_problems=problem_idx + 1)
    problem = problems[problem_idx]

    pfmp = PFMP(dynamics, cost_fn, terminal_cost, init, proposal,
                horizon=HORIZON, num_knots=NUM_KNOTS, particles=1000,
                ess_resample_thresh=ESS_THRESH)

    theta = Params(goal=problem['q_goal'], env=problem['env'], q_min=Q_MIN, q_max=Q_MAX, robot=robot)

    @jax.jit
    def run_sir(key, q_start, theta):
        traj, scores = pfmp.run_sir(key, q_start, theta)
        return traj[jnp.argmin(jnp.sum(scores, axis=-1))]

    @jax.jit
    def extract_path(knots, q_start, theta):
        tq = jnp.linspace(0, pfmp.horizon, int(pfmp.horizon / pfmp.dt))
        us = hermite_spline(tq, pfmp.tk(), knots)
        _, path = jax.lax.scan(lambda x, u: (dynamics(x, u, theta, 0.0)[0], x), q_start, us)
        return path

    knots = run_sir(jax.random.key(seed), problem['q_start'], theta)
    path = np.array(extract_path(knots, problem['q_start'], theta))
    return robot, problem, path


def cfg_with_finger(q):
    return np.append(np.array(q), FINGER_OPEN)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="models/problems/table_pick_panda")
    parser.add_argument("--problem", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default="fig_mbm")
    args = parser.parse_args()

    robot, problem, path = plan(args.dataset, args.problem, args.seed)

    server = viser.ViserServer()
    add_environment_to_viser(server, problem['env'], prefix="/env")

    urdf = yourdfpy.URDF.load(URDF_PATH)

    idxs = np.linspace(0, len(path) - 1, NUM_WAYPOINTS + 2).round().astype(int)
    viridis = (np.array([
        [68, 1, 84], [59, 82, 139], [33, 145, 140], [94, 201, 98], [253, 231, 37],
    ]))
    cmap_idx = np.linspace(0, len(viridis) - 1, NUM_WAYPOINTS).round().astype(int)

    ViserUrdf(server, urdf, root_node_name="/robot_start",
              mesh_color_override=(140, 140, 140, 0.5)).update_cfg(cfg_with_finger(path[idxs[0]]))
    for i, ci in enumerate(cmap_idx):
        color = (*viridis[ci].tolist(), 0.35)
        ViserUrdf(server, urdf, root_node_name=f"/robot_mid_{i}",
                  mesh_color_override=color).update_cfg(cfg_with_finger(path[idxs[1 + i]]))
    ViserUrdf(server, urdf, root_node_name="/robot_goal",
              mesh_color_override=(255, 193, 7, 0.8)).update_cfg(cfg_with_finger(path[idxs[-1]]))

    # Live robot, driven by the frame slider / play button
    live = ViserUrdf(server, urdf, root_node_name="/robot_live")
    live.update_cfg(cfg_with_finger(path[0]))

    # End-effector trace
    ee_trace = np.array([forward_kinematics(jnp.array(q), robot)[-1, :3, 3] for q in path])
    server.scene.add_spline_catmull_rom("/ee_trace", ee_trace, color=(220, 20, 60), line_width=3.0)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"{args.out}.png"

    frame_slider = server.gui.add_slider("Frame", min=0, max=len(path) - 1, step=1, initial_value=0)
    play_button = server.gui.add_button("Play")
    playing = {"on": False}

    @frame_slider.on_update
    def _(_) -> None:
        live.update_cfg(cfg_with_finger(path[frame_slider.value]))

    @play_button.on_click
    def _(_) -> None:
        playing["on"] = not playing["on"]
        play_button.label = "Pause" if playing["on"] else "Play"

    @server.on_client_connect
    def _(client: viser.ClientHandle) -> None:
        @client.gui.add_button("Save Screenshot").on_click
        def _(_) -> None:
            img = client.get_render(height=1080, width=1920, transport_format="png")
            Image.fromarray(img).save(out_path)
            print(f"Saved {out_path}")

    print(f"Loaded plan: {problem.get('name', 'problem')} from {args.dataset}")
    print("Open http://localhost:8080. Use the Frame slider or Play to preview the "
          "trajectory; position the camera and click 'Save Screenshot' to export a PNG.")
    while True:
        if playing["on"]:
            frame_slider.value = (frame_slider.value + 1) % len(path)
            live.update_cfg(cfg_with_finger(path[frame_slider.value]))
            time.sleep(0.02)
        else:
            time.sleep(0.05)


if __name__ == "__main__":
    main()
