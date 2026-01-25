# arm2d_viser.py

import jax
import jax.numpy as jnp
import numpy as np
import time
import viser
import viser.transforms as tf
import trimesh

from typing import NamedTuple
from scipy.ndimage import label
from pfmp import PFMP
from pfmp.interpolate import hermite_spline


class Environment(NamedTuple):
    centers: jax.Array  # (N, 2)
    radii: jax.Array    # (N,)


class Params(NamedTuple):
    goal: jax.Array
    goal_ee: jax.Array
    env: Environment
    link_lengths: jax.Array
    q_min: jax.Array
    q_max: jax.Array
    u_stdev: float


def forward_kinematics(q: jax.Array, link_lengths: jax.Array) -> jax.Array:
    n = len(link_lengths)
    angles = jnp.cumsum(q)
    
    positions = [jnp.zeros(2)]
    pos = jnp.zeros(2)
    
    for i in range(n):
        pos = pos + link_lengths[i] * jnp.array([jnp.cos(angles[i]), jnp.sin(angles[i])])
        positions.append(pos)
    
    return jnp.stack(positions)


def arm_sdf_vmapped(q: jax.Array, link_lengths: jax.Array, env: Environment) -> float:
    positions = forward_kinematics(q, link_lengths)
    n_links = len(link_lengths)
    
    def segment_point_distance(p1, p2, center):
        v = p2 - p1
        w = center - p1
        t = jnp.clip(jnp.dot(w, v) / (jnp.dot(v, v) + 1e-8), 0.0, 1.0)
        closest = p1 + t * v
        return jnp.linalg.norm(center - closest)
    
    def link_sdf(i):
        p1 = positions[i]
        p2 = positions[i + 1]
        dists = jax.vmap(lambda c, r: segment_point_distance(p1, p2, c) - r)(
            env.centers, env.radii
        )
        return jnp.min(dists)
    
    link_sdfs = jax.vmap(link_sdf)(jnp.arange(n_links))
    return jnp.min(link_sdfs)


def collision_cost(sdf: float, margin: float = 0.1) -> float:
    return jnp.maximum(margin - sdf, 0.0) ** 2


def cost_fn(x: jax.Array, u: jax.Array, th: Params, t: float) -> float:
    sdf = arm_sdf_vmapped(x, th.link_lengths, th.env)
    col_cost = 1000.0 * collision_cost(sdf, margin=0.1)
    
    positions = forward_kinematics(x, th.link_lengths)
    ee = positions[-1]
    ee_cost = 0.1 * jnp.linalg.norm(ee - th.goal_ee)
    
    return col_cost + ee_cost


def terminal_cost(x: jax.Array, th: Params) -> float:
    sdf = arm_sdf_vmapped(x, th.link_lengths, th.env)
    col_cost = 1000.0 * collision_cost(sdf, margin=0.1)
    
    positions = forward_kinematics(x, th.link_lengths)
    ee = positions[-1]
    ee_cost = 50.0 * jnp.linalg.norm(ee - th.goal_ee) ** 2
    
    return col_cost + ee_cost


def angle_diff(target, current):
    diff = target - current
    return jnp.arctan2(jnp.sin(diff), jnp.cos(diff))


def dynamics(x: jax.Array, u: jax.Array, theta: Params, t: float):
    direction = jax.vmap(angle_diff)(u, x)
    dist = jnp.linalg.norm(direction)
    direction = direction / (dist + 1e-8)
    
    step = jnp.minimum(dist, 0.1)
    x_next = x + direction * step
    x_next = jnp.arctan2(jnp.sin(x_next), jnp.cos(x_next))
    
    return (x_next, x_next)


def init(key, x, theta: Params):
    k1, k2 = jax.random.split(key)
    r = jax.random.uniform(k1, ())
    target = jnp.where(
        r < 0.7,
        jax.random.uniform(k2, shape=x.shape, minval=theta.q_min, maxval=theta.q_max),
        theta.goal
    )
    return target


def proposal(key, x, u, theta: Params, t: float):
    k1, k2 = jax.random.split(key)
    r = jax.random.uniform(k1, ())
    target = jnp.where(
        r < 0.7,
        jax.random.uniform(k2, shape=x.shape, minval=theta.q_min, maxval=theta.q_max),
        theta.goal
    )
    return target


# ============================================================================
# C-Space Feasibility
# ============================================================================

def build_cspace(link_lengths, env, resolution=0.1):
    """Precompute C-space connected components."""
    n_dof = len(link_lengths)
    n_cells = int(2 * jnp.pi / resolution)
    
    axes = [jnp.linspace(-jnp.pi, jnp.pi, n_cells) for _ in range(n_dof)]
    grid_coords = jnp.stack(jnp.meshgrid(*axes, indexing='ij'), axis=-1)
    flat_coords = grid_coords.reshape(-1, n_dof)
    
    # Vectorized collision check
    sdfs = jax.vmap(lambda q: arm_sdf_vmapped(q, link_lengths, env))(flat_coords)
    free = (sdfs > 0.02).reshape(tuple([n_cells] * n_dof))
    
    # Connected components
    labeled, n_components = label(np.array(free))
    
    return labeled, resolution, n_cells, n_components


def is_feasible(q_start, q_goal, labeled, resolution, n_cells):
    """O(1) feasibility check via precomputed C-space."""
    def to_idx(q):
        return tuple(((np.array(q) + np.pi) / resolution).astype(int) % n_cells)
    
    start_label = labeled[to_idx(q_start)]
    goal_label = labeled[to_idx(q_goal)]
    
    return start_label > 0 and start_label == goal_label


# ============================================================================
# Environment Generation
# ============================================================================

def random_environment(key, n_obstacles=4, x_range=(-1.5, 2.0), y_range=(-1.5, 1.5), r_range=(0.15, 0.35)):
    k1, k2, k3 = jax.random.split(key, 3)
    
    centers_x = jax.random.uniform(k1, (n_obstacles,), minval=x_range[0], maxval=x_range[1])
    centers_y = jax.random.uniform(k2, (n_obstacles,), minval=y_range[0], maxval=y_range[1])
    radii = jax.random.uniform(k3, (n_obstacles,), minval=r_range[0], maxval=r_range[1])
    
    centers = jnp.stack([centers_x, centers_y], axis=-1)
    
    return Environment(centers=centers, radii=radii)


def find_valid_config(key, link_lengths, env, n_attempts=100):
    """Find a collision-free configuration."""
    for _ in range(n_attempts):
        key, subkey = jax.random.split(key)
        q = jax.random.uniform(subkey, (len(link_lengths),), minval=-jnp.pi, maxval=jnp.pi)
        sdf = arm_sdf_vmapped(q, link_lengths, env)
        if sdf > 0.1:
            return key, q
    return key, None


def find_feasible_problem(key, link_lengths, n_attempts=50, cspace_resolution=0.1):
    """Find a start/goal pair guaranteed to be feasible."""
    for _ in range(n_attempts):
        key, k1, k2, k3 = jax.random.split(key, 4)
        
        env = random_environment(k1)
        
        # Build C-space for this environment
        labeled, resolution, n_cells, n_components = build_cspace(
            link_lengths, env, resolution=cspace_resolution
        )
        
        # Skip environments with no free space
        if n_components == 0:
            continue
        
        # Find start config
        key, q_start = find_valid_config(k2, link_lengths, env)
        if q_start is None:
            continue
        
        # Find goal config in same component
        for _ in range(50):
            key, subkey = jax.random.split(key)
            q_goal = jax.random.uniform(subkey, (len(link_lengths),), minval=-jnp.pi, maxval=jnp.pi)
            
            if arm_sdf_vmapped(q_goal, link_lengths, env) > 0.1:
                if is_feasible(q_start, q_goal, labeled, resolution, n_cells):
                    # Also ensure they're not too close
                    if jnp.linalg.norm(q_start - q_goal) > 1.0:
                        return key, env, q_start, q_goal
        
    return key, None, None, None


# ============================================================================
# Visualization Helpers
# ============================================================================

def create_arm_with_links(server, name, positions, color, offset,
                          z_plane=0.0, joint_radius=0.08, link_radius=0.04):
    handles = {'joints': [], 'links': []}

    for i, pos in enumerate(positions):
        pos_3d = (pos[0] + offset[0], pos[1] + offset[1], z_plane)
        handles['joints'].append(
            server.scene.add_icosphere(
                f"/{name}/joint_{i}",
                radius=joint_radius,
                position=pos_3d,
                color=color,
            )
        )

    for i in range(len(positions) - 1):
        p1 = positions[i]
        p2 = positions[i + 1]

        center = (p1 + p2) / 2
        length = float(np.linalg.norm(p2 - p1))

        direction = (p2 - p1) / (length + 1e-8)
        angle = np.arctan2(direction[1], direction[0])

        rot_y = tf.SO3.from_y_radians(np.pi / 2)
        rot_z = tf.SO3.from_z_radians(angle)
        final_rot = rot_z @ rot_y

        mesh = trimesh.creation.cylinder(radius=link_radius, height=length)

        handles['links'].append(
            server.scene.add_mesh_trimesh(
                f"/{name}/link_{i}",
                mesh=mesh,
                position=(center[0] + offset[0], center[1] + offset[1], z_plane),
                wxyz=final_rot.wxyz,
            )
        )

    return handles


def update_arm_with_links(handles, positions, color, offset, z_plane=0.0):
    for handle, pos in zip(handles['joints'], positions):
        handle.position = (pos[0] + offset[0], pos[1] + offset[1], z_plane)
        handle.color = color

    for i, handle in enumerate(handles['links']):
        p1 = positions[i]
        p2 = positions[i + 1]

        center = (p1 + p2) / 2
        length = float(np.linalg.norm(p2 - p1))

        direction = (p2 - p1) / (length + 1e-8)
        angle = np.arctan2(direction[1], direction[0])

        rot_y = tf.SO3.from_y_radians(np.pi / 2)
        rot_z = tf.SO3.from_z_radians(angle)
        final_rot = rot_z @ rot_y

        handle.position = (center[0] + offset[0], center[1] + offset[1], z_plane)
        handle.wxyz = final_rot.wxyz
        handle.color = color


# ============================================================================
# Main Visualization
# ============================================================================

def run_visual(seed=42):
    n_dof = 3
    link_lengths = jnp.array([1.0, 1.0, 0.5])
    
    env = Environment(
        centers=jnp.array([
            [1.5, 0.8],
            [0.8, -0.4],
            [-0.5, 0.6],
            [1.2, -1.0],
        ]),
        radii=jnp.array([0.25, 0.25, 0.25, 0.25])
    )
    
    q_start = jnp.array([0., 0.5, 0.2])
    q_goal = jnp.array([1.2, -0.6, -0.2])
    
    # Check feasibility
    print("Building C-space...")
    labeled, resolution, n_cells, n_components = build_cspace(link_lengths, env)
    print(f"C-space: {n_cells}^{n_dof} cells, {n_components} components")
    
    feasible = is_feasible(q_start, q_goal, labeled, resolution, n_cells)
    print(f"Feasible: {feasible}")
    
    if not feasible:
        print("Warning: start and goal are not in the same C-space component!")
    
    goal_ee = forward_kinematics(q_goal, link_lengths)[-1]
    
    theta = Params(
        goal=q_goal,
        goal_ee=goal_ee,
        env=env,
        link_lengths=link_lengths,
        q_min=jnp.full(n_dof, -jnp.pi),
        q_max=jnp.full(n_dof, jnp.pi),
        u_stdev=0.05,
    )
    
    print(f"Start SDF: {arm_sdf_vmapped(q_start, link_lengths, env):.3f}")
    print(f"Goal SDF: {arm_sdf_vmapped(q_goal, link_lengths, env):.3f}")
    
    pfmp = PFMP(
        dynamics, cost_fn, terminal_cost, init, proposal,
        horizon=1.0, num_knots=5, particles=512,
        ess_resample_thresh=0.3
    )
    
    key = jax.random.key(seed)
    
    print("Running SIS...")
    traj_sis, costs_sis = pfmp.run_sis(key, q_start, theta)
    best_sis = jnp.argmin(jnp.sum(costs_sis, axis=-1))
    
    print("Running SIR...")
    traj_sir, costs_sir = pfmp.run_sir(key, q_start, theta)
    best_sir = jnp.argmin(jnp.sum(costs_sir, axis=-1))
    
    tq = jnp.linspace(0, 1.0, int(1.0 / pfmp.dt))
    
    us_sis = hermite_spline(tq, pfmp.tk(), traj_sis[best_sis])
    _, path_sis = jax.lax.scan(lambda x, u: (dynamics(x, u, theta, 0.0)[0], x), q_start, us_sis)
    path_sis = np.array(path_sis)
    
    us_sir = hermite_spline(tq, pfmp.tk(), traj_sir[best_sir])
    _, path_sir = jax.lax.scan(lambda x, u: (dynamics(x, u, theta, 0.0)[0], x), q_start, us_sir)
    path_sir = np.array(path_sir)
    
    # Precompute FK
    print("Precomputing FK...")
    fk_sis = [np.array(forward_kinematics(jnp.array(q), link_lengths)) for q in path_sis]
    fk_sir = [np.array(forward_kinematics(jnp.array(q), link_lengths)) for q in path_sir]
    col_sis = [float(arm_sdf_vmapped(jnp.array(q), link_lengths, env)) < 0 for q in path_sis]
    col_sir = [float(arm_sdf_vmapped(jnp.array(q), link_lengths, env)) < 0 for q in path_sir]
    
    fk_start = np.array(forward_kinematics(q_start, link_lengths))
    fk_goal = np.array(forward_kinematics(q_goal, link_lengths))
    
    print(f"SIS: collision={np.mean(col_sis):.0%}")
    print(f"SIR: collision={np.mean(col_sir):.0%}")
    
    # Viser
    server = viser.ViserServer()
    print("Open http://localhost:8080")
    
    offset_sis = np.array([-4.0, 0.0])
    offset_sir = np.array([4.0, 0.0])
    z_plane = 0.0
    
    # Obstacles
    for i, (c, r) in enumerate(zip(np.array(env.centers), np.array(env.radii))):
        server.scene.add_icosphere(
            f"/env_sis/obstacle_{i}",
            radius=float(r),
            position=(c[0] + offset_sis[0], c[1] + offset_sis[1], z_plane),
            color=(100, 100, 100),
        )
        server.scene.add_icosphere(
            f"/env_sir/obstacle_{i}",
            radius=float(r),
            position=(c[0] + offset_sir[0], c[1] + offset_sir[1], z_plane),
            color=(100, 100, 100),
        )
    
    # Ground
    server.scene.add_grid("/ground_sis", width=6, height=6, position=(offset_sis[0], offset_sis[1], -0.01), plane="xy")
    server.scene.add_grid("/ground_sir", width=6, height=6, position=(offset_sir[0], offset_sir[1], -0.01), plane="xy")
    
    # Static arms
    create_arm_with_links(server, "start_sis", fk_start, (180, 180, 180), offset_sis, z_plane, joint_radius=0.06, link_radius=0.03)
    create_arm_with_links(server, "start_sir", fk_start, (180, 180, 180), offset_sir, z_plane, joint_radius=0.06, link_radius=0.03)
    create_arm_with_links(server, "goal_sis", fk_goal, (255, 193, 7), offset_sis, z_plane, joint_radius=0.06, link_radius=0.03)
    create_arm_with_links(server, "goal_sir", fk_goal, (255, 193, 7), offset_sir, z_plane, joint_radius=0.06, link_radius=0.03)
    
    # Animated arms
    sis_handles = create_arm_with_links(server, "sis", fk_sis[0], (76, 175, 80), offset_sis, z_plane)
    sir_handles = create_arm_with_links(server, "sir", fk_sir[0], (76, 175, 80), offset_sir, z_plane)
    
    # EE traces
    ee_sis_points = np.array([[fk[-1][0] + offset_sis[0], fk[-1][1] + offset_sis[1], z_plane] for fk in fk_sis])
    ee_sir_points = np.array([[fk[-1][0] + offset_sir[0], fk[-1][1] + offset_sir[1], z_plane] for fk in fk_sir])
    
    ee_trace_sis = server.scene.add_spline_catmull_rom("/ee_trace_sis", positions=ee_sis_points[:2], color=(33, 150, 243), line_width=3.0)
    ee_trace_sir = server.scene.add_spline_catmull_rom("/ee_trace_sir", positions=ee_sir_points[:2], color=(33, 150, 243), line_width=3.0)
    
    # Labels
    server.scene.add_label("/label_sis", f"SIS (col: {np.mean(col_sis)*100:.0f}%)", position=(offset_sis[0], offset_sis[1], 2.5))
    server.scene.add_label("/label_sir", f"SIR (col: {np.mean(col_sir)*100:.0f}%)", position=(offset_sir[0], offset_sir[1], 2.5))
    
    # GUI
    with server.gui.add_folder("Controls"):
        play_button = server.gui.add_button("Play/Pause")
        reset_button = server.gui.add_button("Reset")
        frame_slider = server.gui.add_slider("Frame", min=0, max=len(path_sis)-1, step=1, initial_value=0)
        speed_slider = server.gui.add_slider("Speed", min=0.1, max=3.0, step=0.1, initial_value=1.0)
        loop_checkbox = server.gui.add_checkbox("Loop", initial_value=True)
    
    playing = True
    frame = 0
    
    @play_button.on_click
    def _(_):
        nonlocal playing
        playing = not playing
    
    @reset_button.on_click
    def _(_):
        nonlocal frame
        frame = 0
        frame_slider.value = 0
    
    @frame_slider.on_update
    def _(event):
        nonlocal frame
        frame = int(event.target.value)
    
    n_frames = len(path_sis)
    last_time = time.time()
    
    while True:
        current_time = time.time()
        dt = current_time - last_time
        
        if playing and dt > 0.033 / speed_slider.value:
            if loop_checkbox.value:
                frame = (frame + 1) % n_frames
            else:
                frame = min(frame + 1, n_frames - 1)
            frame_slider.value = frame
            last_time = current_time
        
        color_sis = (244, 67, 54) if col_sis[frame] else (76, 175, 80)
        color_sir = (244, 67, 54) if col_sir[frame] else (76, 175, 80)
        
        update_arm_with_links(sis_handles, fk_sis[frame], color_sis, offset_sis, z_plane)
        update_arm_with_links(sir_handles, fk_sir[frame], color_sir, offset_sir, z_plane)
        
        if frame >= 1:
            ee_trace_sis.positions = ee_sis_points[:frame+1]
            ee_trace_sir.positions = ee_sir_points[:frame+1]
        
        time.sleep(0.01)


def run_benchmark_visual(n_envs=20, seed=0):
    """Run benchmark with guaranteed feasible problems."""
    n_dof = 3
    link_lengths = jnp.array([1.0, 1.0, 0.5])
    
    server = viser.ViserServer()
    print("Open http://localhost:8080")
    
    offset_sis = np.array([-4.0, 0.0])
    offset_sir = np.array([4.0, 0.0])
    z_plane = 0.0
    
    server.scene.add_grid("/ground_sis", width=6, height=6, position=(offset_sis[0], offset_sis[1], -0.01), plane="xy")
    server.scene.add_grid("/ground_sir", width=6, height=6, position=(offset_sir[0], offset_sir[1], -0.01), plane="xy")
    
    label_sis = server.scene.add_label("/label_sis", "SIS: 0%", position=(offset_sis[0], offset_sis[1], 2.5))
    label_sir = server.scene.add_label("/label_sir", "SIR: 0%", position=(offset_sir[0], offset_sir[1], 2.5))
    
    with server.gui.add_folder("Stats"):
        env_text = server.gui.add_text("Environment", initial_value="0/0", disabled=True)
        sis_rate_text = server.gui.add_text("SIS Success", initial_value="0%", disabled=True)
        sir_rate_text = server.gui.add_text("SIR Success", initial_value="0%", disabled=True)
        feasibility_text = server.gui.add_text("Feasibility", initial_value="checking...", disabled=True)
    
    results = {'sis_success': [], 'sir_success': []}
    master_key = jax.random.key(seed)
    
    obstacle_handles = []
    sis_handles = None
    sir_handles = None
    start_handles_sis = None
    start_handles_sir = None
    goal_handles_sis = None
    goal_handles_sir = None
    ee_trace_sis = None
    ee_trace_sir = None
    
    def clear_scene():
        nonlocal obstacle_handles, sis_handles, sir_handles
        nonlocal start_handles_sis, start_handles_sir, goal_handles_sis, goal_handles_sir
        nonlocal ee_trace_sis, ee_trace_sir
        
        for h in obstacle_handles:
            h.remove()
        obstacle_handles.clear()
        
        for handles in [sis_handles, sir_handles, start_handles_sis, start_handles_sir, goal_handles_sis, goal_handles_sir]:
            if handles:
                for h in handles['joints']:
                    h.remove()
                for h in handles['links']:
                    h.remove()
        
        if ee_trace_sis:
            ee_trace_sis.remove()
        if ee_trace_sir:
            ee_trace_sir.remove()
    
    env_idx = 0
    while env_idx < n_envs:
        feasibility_text.value = "finding feasible problem..."
        
        # Find a guaranteed feasible problem
        master_key, env, q_start, q_goal = find_feasible_problem(master_key, link_lengths)
        
        if env is None:
            print(f"Could not find feasible problem, retrying...")
            continue
        
        feasibility_text.value = "feasible ✓"
        
        goal_ee = forward_kinematics(q_goal, link_lengths)[-1]
        
        theta = Params(
            goal=q_goal,
            goal_ee=goal_ee,
            env=env,
            link_lengths=link_lengths,
            q_min=jnp.full(n_dof, -jnp.pi),
            q_max=jnp.full(n_dof, jnp.pi),
            u_stdev=0.05,
        )
        
        pfmp = PFMP(
            dynamics, cost_fn, terminal_cost, init, proposal,
            horizon=1.0, num_knots=5, particles=512,
            ess_resample_thresh=0.3
        )
        
        master_key, run_key = jax.random.split(master_key)
        
        traj_sis, costs_sis = pfmp.run_sis(run_key, q_start, theta)
        traj_sir, costs_sir = pfmp.run_sir(run_key, q_start, theta)
        best_sis = jnp.argmin(jnp.sum(costs_sis, axis=-1))
        best_sir = jnp.argmin(jnp.sum(costs_sir, axis=-1))
        
        tq = jnp.linspace(0, 1.0, int(1.0 / pfmp.dt))
        
        us_sis = hermite_spline(tq, pfmp.tk(), traj_sis[best_sis])
        _, path_sis = jax.lax.scan(lambda x, u: (dynamics(x, u, theta, 0.0)[0], x), q_start, us_sis)
        path_sis = np.array(path_sis)
        
        us_sir = hermite_spline(tq, pfmp.tk(), traj_sir[best_sir])
        _, path_sir = jax.lax.scan(lambda x, u: (dynamics(x, u, theta, 0.0)[0], x), q_start, us_sir)
        path_sir = np.array(path_sir)
        
        # Precompute
        fk_sis = [np.array(forward_kinematics(jnp.array(q), link_lengths)) for q in path_sis]
        fk_sir = [np.array(forward_kinematics(jnp.array(q), link_lengths)) for q in path_sir]
        col_sis = [float(arm_sdf_vmapped(jnp.array(q), link_lengths, env)) < 0 for q in path_sis]
        col_sir = [float(arm_sdf_vmapped(jnp.array(q), link_lengths, env)) < 0 for q in path_sir]
        
        fk_start = np.array(forward_kinematics(q_start, link_lengths))
        fk_goal = np.array(forward_kinematics(q_goal, link_lengths))
        
        sis_success = not any(col_sis) and float(jnp.linalg.norm(path_sis[-1] - jnp.array(q_goal))) < 0.3
        sir_success = not any(col_sir) and float(jnp.linalg.norm(path_sir[-1] - jnp.array(q_goal))) < 0.3
        results['sis_success'].append(sis_success)
        results['sir_success'].append(sir_success)
        
        print(f"[{env_idx+1}/{n_envs}] SIS: {'✓' if sis_success else '✗'}, SIR: {'✓' if sir_success else '✗'}")
        
        # Clear and rebuild scene
        clear_scene()
        
        for i, (c, r) in enumerate(zip(np.array(env.centers), np.array(env.radii))):
            obstacle_handles.append(
                server.scene.add_icosphere(
                    f"/env_sis/obstacle_{i}",
                    radius=float(r),
                    position=(c[0] + offset_sis[0], c[1] + offset_sis[1], z_plane),
                    color=(100, 100, 100),
                )
            )
            obstacle_handles.append(
                server.scene.add_icosphere(
                    f"/env_sir/obstacle_{i}",
                    radius=float(r),
                    position=(c[0] + offset_sir[0], c[1] + offset_sir[1], z_plane),
                    color=(100, 100, 100),
                )
            )
        
        start_handles_sis = create_arm_with_links(server, "start_sis", fk_start, (180, 180, 180), offset_sis, z_plane, joint_radius=0.06, link_radius=0.03)
        start_handles_sir = create_arm_with_links(server, "start_sir", fk_start, (180, 180, 180), offset_sir, z_plane, joint_radius=0.06, link_radius=0.03)
        goal_handles_sis = create_arm_with_links(server, "goal_sis", fk_goal, (255, 193, 7), offset_sis, z_plane, joint_radius=0.06, link_radius=0.03)
        goal_handles_sir = create_arm_with_links(server, "goal_sir", fk_goal, (255, 193, 7), offset_sir, z_plane, joint_radius=0.06, link_radius=0.03)
        
        sis_handles = create_arm_with_links(server, "sis", fk_sis[0], (76, 175, 80), offset_sis, z_plane)
        sir_handles = create_arm_with_links(server, "sir", fk_sir[0], (76, 175, 80), offset_sir, z_plane)
        
        ee_sis_points = np.array([[fk[-1][0] + offset_sis[0], fk[-1][1] + offset_sis[1], z_plane] for fk in fk_sis])
        ee_sir_points = np.array([[fk[-1][0] + offset_sir[0], fk[-1][1] + offset_sir[1], z_plane] for fk in fk_sir])
        
        ee_trace_sis = server.scene.add_spline_catmull_rom("/ee_trace_sis", positions=ee_sis_points[:2], color=(33, 150, 243), line_width=3.0)
        ee_trace_sir = server.scene.add_spline_catmull_rom("/ee_trace_sir", positions=ee_sir_points[:2], color=(33, 150, 243), line_width=3.0)
        
        # Update stats
        sis_rate = np.mean(results['sis_success']) * 100
        sir_rate = np.mean(results['sir_success']) * 100
        label_sis.text = f"SIS: {sis_rate:.0f}%"
        label_sir.text = f"SIR: {sir_rate:.0f}%"
        env_text.value = f"{env_idx + 1}/{n_envs}"
        sis_rate_text.value = f"{sis_rate:.0f}%"
        sir_rate_text.value = f"{sir_rate:.0f}%"
        
        # Animate
        n_frames = len(path_sis)
        for _ in range(2):
            for frame in range(n_frames):
                color_sis = (244, 67, 54) if col_sis[frame] else (76, 175, 80)
                color_sir = (244, 67, 54) if col_sir[frame] else (76, 175, 80)
                
                update_arm_with_links(sis_handles, fk_sis[frame], color_sis, offset_sis, z_plane)
                update_arm_with_links(sir_handles, fk_sir[frame], color_sir, offset_sir, z_plane)
                
                if frame >= 1:
                    ee_trace_sis.positions = ee_sis_points[:frame+1]
                    ee_trace_sir.positions = ee_sir_points[:frame+1]
                
                time.sleep(0.02)
        
        env_idx += 1
    
    print(f"\nFinal: SIS={np.mean(results['sis_success'])*100:.0f}%, SIR={np.mean(results['sir_success'])*100:.0f}%")
    
    while True:
        time.sleep(1)


def main():
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "benchmark":
        n_envs = int(sys.argv[2]) if len(sys.argv) > 2 else 20
        run_benchmark_visual(n_envs=n_envs)
    else:
        run_visual()


if __name__ == "__main__":
    main()