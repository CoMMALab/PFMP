"""IK Inspector — interactive Viser tool for debugging IK solutions.

Run from repo root:
  python examples/fr3/ik_inspector.py

Open http://localhost:8080 in your browser.

Controls:
  - Drag X / Y sliders to move the EE target around the workspace
  - "Build heatmap" builds a 16x16 IK grid and colours workspace
    cells green (smooth) → red (discontinuous)
"""

import sys
import time
import numpy as np

sys.path.insert(0, ".")
from examples.fr3.ik_cache import (
    ik_step_from_xy,
    HOME_Q,
    EE_HEIGHT,
    WORKSPACE_XY_MIN,
    WORKSPACE_XY_MAX,
    ROBOT_URDF_PATH,
    _CACHE_MAX_JUMP,
)
import jax.numpy as jnp
import yourdfpy
import viser
from viser.extras import ViserUrdf

HOME_Q_NP = np.array(HOME_Q)
# panda_joint1 … panda_joint7 — order matches HOME_Q indices
ARM_JOINT_NAMES = [f"panda_joint{i+1}" for i in range(7)]


def main() -> None:
    server = viser.ViserServer()

    # ----- scene -----
    server.scene.add_grid("/grid", width=2, height=2)
    urdf = yourdfpy.URDF.load(ROBOT_URDF_PATH)
    viser_urdf = ViserUrdf(server, urdf_or_path=urdf)
    ee_frame = server.scene.add_frame(
        "/ee_target", axes_length=0.08, axes_radius=0.006,
        position=(0.3, 0.0, EE_HEIGHT),
    )

    # ----- GUI -----
    with server.gui.add_folder("Target position"):
        x_slider = server.gui.add_slider(
            "X",
            min=float(WORKSPACE_XY_MIN[0]),
            max=float(WORKSPACE_XY_MAX[0]),
            step=0.005,
            initial_value=0.3,
        )
        y_slider = server.gui.add_slider(
            "Y",
            min=float(WORKSPACE_XY_MIN[1]),
            max=float(WORKSPACE_XY_MAX[1]),
            step=0.005,
            initial_value=0.0,
        )

    with server.gui.add_folder("IK solution"):
        info_md = server.gui.add_markdown("*Move the sliders to solve IK.*")

    with server.gui.add_folder("Workspace heatmap"):
        res_slider = server.gui.add_slider(
            "Resolution", min=8, max=32, step=1, initial_value=16,
        )
        build_btn = server.gui.add_button("Build heatmap")

    # ----- solve IK and refresh -----
    def solve_and_update() -> None:
        xy = np.array([x_slider.value, y_slider.value])
        q = np.array(ik_step_from_xy(jnp.array(HOME_Q_NP), jnp.array(xy)))

        # Update robot URDF
        cfg = {name: float(q[k]) for k, name in enumerate(ARM_JOINT_NAMES)}
        viser_urdf.update_cfg(cfg)

        # Move EE frame
        ee_frame.position = (float(xy[0]), float(xy[1]), float(EE_HEIGHT))

        # Deviation from HOME_Q
        diff = ((q - HOME_Q_NP) + np.pi) % (2 * np.pi) - np.pi
        worst_idx = int(np.argmax(np.abs(diff)))
        worst_deg = float(np.degrees(np.abs(diff[worst_idx])))
        ok = worst_deg < np.degrees(_CACHE_MAX_JUMP)

        rows = ["| jt | angle | Δ home |", "|:--:|------:|------:|"]
        for k in range(7):
            flag = " ⚠" if abs(diff[k]) >= _CACHE_MAX_JUMP else ""
            rows.append(
                f"| {k+1} | {np.degrees(q[k]):+.1f}° "
                f"| {np.degrees(diff[k]):+.1f}°{flag} |"
            )
        status = "🟢 OK" if ok else f"🔴 BAD — j{worst_idx+1} off by {worst_deg:.0f}°"
        info_md.content = "\n".join(rows) + f"\n\n**{status}**"

    @x_slider.on_update
    def _(_): solve_and_update()

    @y_slider.on_update
    def _(_): solve_and_update()

    # ----- heatmap -----
    @build_btn.on_click
    def _(_):
        build_btn.disabled = True
        resolution = int(res_slider.value)
        xs = np.linspace(float(WORKSPACE_XY_MIN[0]), float(WORKSPACE_XY_MAX[0]), resolution)
        ys = np.linspace(float(WORKSPACE_XY_MIN[1]), float(WORKSPACE_XY_MAX[1]), resolution)

        q_grid = np.zeros((resolution, resolution, 7))
        for i, x in enumerate(xs):
            for j, y in enumerate(ys):
                q_grid[i, j] = np.array(
                    ik_step_from_xy(jnp.array(HOME_Q_NP), jnp.array([x, y]))
                )

        def wd(arr, axis):
            d = np.diff(arr, axis=axis)
            return (d + np.pi) % (2 * np.pi) - np.pi

        dx = np.abs(wd(q_grid, 0))   # (N-1, N, 7)
        dy = np.abs(wd(q_grid, 1))   # (N, N-1, 7)
        jump = np.zeros((resolution, resolution))
        jump[:-1, :] = np.maximum(jump[:-1, :], np.max(dx, axis=-1))
        jump[:, :-1] = np.maximum(jump[:, :-1], np.max(dy, axis=-1))

        # Point cloud: green = smooth, red = discontinuous
        pts, colors = [], []
        for i, x in enumerate(xs):
            for j, y in enumerate(ys):
                pts.append([x, y, 1e-3])
                t = float(np.clip(jump[i, j] / (_CACHE_MAX_JUMP * 2), 0.0, 1.0))
                colors.append([t, 1.0 - t, 0.0])

        server.scene.add_point_cloud(
            "/jump_map",
            points=np.array(pts, dtype=np.float32),
            colors=np.array(colors, dtype=np.float32),
            point_size=0.05,
        )

        n_bad = int(np.sum(jump >= _CACHE_MAX_JUMP))
        print(f"Heatmap built: {n_bad}/{resolution**2} bad cells "
              f"(max jump {np.degrees(jump.max()):.1f}°)")
        build_btn.disabled = False

    # ----- init -----
    viser_urdf.update_cfg({n: float(v) for n, v in zip(ARM_JOINT_NAMES, HOME_Q_NP)})
    solve_and_update()

    print("Open http://localhost:8080")
    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    main()
