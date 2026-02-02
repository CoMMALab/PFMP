"""
Motion planning benchmark with voxel grid collision.
Pure JAX - no MuJoCo dependency.
"""

import jax
import jax.numpy as jnp
import numpy as np
import yaml
import time
from pathlib import Path
from typing import NamedTuple
from scipy.spatial.transform import Rotation

# Voxel grid from common module
from voxels import (
    VoxelGrid,
    empty_voxel_grid,
    compute_distance_field,
    add_box,
    add_sphere,
    add_cylinder,
)


# ============================================================================
# SDF Query
# ============================================================================

@jax.jit
def point_sdf(point: jax.Array, origin: jax.Array, resolution: float, dist_field: jax.Array) -> jax.Array:
    voxel_coord = (point - origin) / resolution - 0.5
    idx = jnp.clip(jnp.floor(voxel_coord).astype(jnp.int32), 0, jnp.array(dist_field.shape, dtype=jnp.int32) - 2)
    frac = jnp.clip(voxel_coord - idx.astype(jnp.float32), 0.0, 1.0)
    i, j, k = idx[0], idx[1], idx[2]
    fx, fy, fz = frac[0], frac[1], frac[2]
    
    c000, c001 = dist_field[i, j, k], dist_field[i, j, k+1]
    c010, c011 = dist_field[i, j+1, k], dist_field[i, j+1, k+1]
    c100, c101 = dist_field[i+1, j, k], dist_field[i+1, j, k+1]
    c110, c111 = dist_field[i+1, j+1, k], dist_field[i+1, j+1, k+1]
    
    c00 = c000 + fx * (c100 - c000)
    c01 = c001 + fx * (c101 - c001)
    c10 = c010 + fx * (c110 - c010)
    c11 = c011 + fx * (c111 - c011)
    return (c00 + fy * (c10 - c00)) + fz * ((c01 + fy * (c11 - c01)) - (c00 + fy * (c10 - c00)))


# ============================================================================
# Robot
# ============================================================================

def parse_panda_urdf(urdf_path: str):
    import xml.etree.ElementTree as ET
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    
    joints_info = []
    for joint in root.findall('joint'):
        if joint.get('type') in ['revolute', 'continuous']:
            origin = joint.find('origin')
            xyz = [float(x) for x in origin.get('xyz', '0 0 0').split()] if origin is not None else [0,0,0]
            rpy = [float(x) for x in origin.get('rpy', '0 0 0').split()] if origin is not None else [0,0,0]
            axis_elem = joint.find('axis')
            axis = [float(x) for x in axis_elem.get('xyz', '0 0 1').split()] if axis_elem is not None else [0,0,1]
            joints_info.append({'name': joint.get('name'), 'child': joint.find('child').get('link'),
                               'xyz': xyz, 'rpy': rpy, 'axis': axis})
    
    spheres_info = []
    for link in root.findall('link'):
        link_name = link.get('name')
        for collision in link.findall('collision'):
            geom = collision.find('geometry')
            if geom is not None and geom.find('sphere') is not None:
                sphere = geom.find('sphere')
                origin = collision.find('origin')
                xyz = [float(x) for x in origin.get('xyz', '0 0 0').split()] if origin is not None else [0,0,0]
                spheres_info.append({'link': link_name, 'offset': xyz, 'radius': float(sphere.get('radius', '0.05'))})
    
    return {'joints': joints_info, 'spheres': spheres_info}


def build_robot_model(urdf_path: str):
    urdf_data = parse_panda_urdf(urdf_path)
    arm_joint_names = [f'panda_joint{i}' for i in range(1, 8)]
    arm_joints = sorted([j for j in urdf_data['joints'] if j['name'] in arm_joint_names],
                        key=lambda j: arm_joint_names.index(j['name']))
    
    link_to_joint_idx = {'panda_link0': -1}
    for i, j in enumerate(arm_joints):
        link_to_joint_idx[j['child']] = i
    
    sphere_data = [{'joint_idx': link_to_joint_idx[s['link']], 'offset': s['offset'], 'radius': s['radius']}
                   for s in urdf_data['spheres'] if s['link'] in link_to_joint_idx]
    
    return {
        'n_joints': len(arm_joints),
        'joint_xyz': jnp.array([j['xyz'] for j in arm_joints]),
        'joint_rpy': jnp.array([j['rpy'] for j in arm_joints]),
        'joint_axes': jnp.array([j['axis'] for j in arm_joints]),
        'sphere_joint_indices': jnp.array([s['joint_idx'] for s in sphere_data]),
        'sphere_offsets': jnp.array([s['offset'] for s in sphere_data]),
        'sphere_radii': jnp.array([s['radius'] for s in sphere_data]),
        'n_spheres': len(sphere_data),
    }


_ROBOT = None
def load_robot(urdf_path: str = 'models/panda_spherized.urdf'):
    global _ROBOT
    if _ROBOT is None:
        _ROBOT = build_robot_model(urdf_path)
        print(f"Loaded robot: {_ROBOT['n_joints']} joints, {_ROBOT['n_spheres']} spheres")
    return _ROBOT


def get_sphere_positions(q: jax.Array, robot: dict) -> jax.Array:
    joint_xyz, joint_rpy, joint_axes = robot['joint_xyz'], robot['joint_rpy'], robot['joint_axes']
    
    def compute_transform(T_prev, inputs):
        xyz, rpy, axis, theta = inputs
        r, p, y = rpy[0], rpy[1], rpy[2]
        cr, sr, cp, sp, cy, sy = jnp.cos(r), jnp.sin(r), jnp.cos(p), jnp.sin(p), jnp.cos(y), jnp.sin(y)
        R_fixed = jnp.array([[cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
                             [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr], [-sp, cp*sr, cp*cr]])
        T_fixed = jnp.eye(4).at[:3, :3].set(R_fixed).at[:3, 3].set(xyz)
        axis_n = axis / (jnp.linalg.norm(axis) + 1e-10)
        K = jnp.array([[0, -axis_n[2], axis_n[1]], [axis_n[2], 0, -axis_n[0]], [-axis_n[1], axis_n[0], 0]])
        R_joint = jnp.eye(3) + jnp.sin(theta) * K + (1 - jnp.cos(theta)) * (K @ K)
        T_joint = jnp.eye(4).at[:3, :3].set(R_joint)
        T_curr = T_prev @ T_fixed @ T_joint
        return T_curr, T_curr
    
    _, transforms = jax.lax.scan(compute_transform, jnp.eye(4), (joint_xyz, joint_rpy, joint_axes, q))
    all_transforms = jnp.concatenate([jnp.eye(4)[None], transforms], axis=0)
    
    def get_pos(idx, offset):
        return (all_transforms[idx + 1] @ jnp.append(offset, 1.0))[:3]
    
    return jax.vmap(get_pos)(robot['sphere_joint_indices'], robot['sphere_offsets'])


def make_robot_sdf_fn(robot: dict):
    joint_xyz, joint_rpy, joint_axes = robot['joint_xyz'], robot['joint_rpy'], robot['joint_axes']
    sphere_joint_indices, sphere_offsets, sphere_radii = (
        robot['sphere_joint_indices'], robot['sphere_offsets'], robot['sphere_radii'])
    
    @jax.jit
    def robot_sdf(q, origin, resolution, dist_field):
        def compute_transform(T_prev, inputs):
            xyz, rpy, axis, theta = inputs
            r, p, y = rpy[0], rpy[1], rpy[2]
            cr, sr, cp, sp, cy, sy = jnp.cos(r), jnp.sin(r), jnp.cos(p), jnp.sin(p), jnp.cos(y), jnp.sin(y)
            R_fixed = jnp.array([[cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
                                 [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr], [-sp, cp*sr, cp*cr]])
            T_fixed = jnp.eye(4).at[:3, :3].set(R_fixed).at[:3, 3].set(xyz)
            axis_n = axis / (jnp.linalg.norm(axis) + 1e-10)
            K = jnp.array([[0, -axis_n[2], axis_n[1]], [axis_n[2], 0, -axis_n[0]], [-axis_n[1], axis_n[0], 0]])
            R_joint = jnp.eye(3) + jnp.sin(theta) * K + (1 - jnp.cos(theta)) * (K @ K)
            T_joint = jnp.eye(4).at[:3, :3].set(R_joint)
            return T_prev @ T_fixed @ T_joint, T_prev @ T_fixed @ T_joint
        
        _, transforms = jax.lax.scan(compute_transform, jnp.eye(4), (joint_xyz, joint_rpy, joint_axes, q))
        all_transforms = jnp.concatenate([jnp.eye(4)[None], transforms], axis=0)
        
        def get_sphere_sdf(idx, offset, radius):
            pos = (all_transforms[idx + 1] @ jnp.append(offset, 1.0))[:3]
            return point_sdf(pos, origin, resolution, dist_field) - radius
        
        return jnp.min(jax.vmap(get_sphere_sdf)(sphere_joint_indices, sphere_offsets, sphere_radii))
    
    return robot_sdf


# ============================================================================
# MBM Dataset
# ============================================================================

def quat_to_rotmat(quat):
    return Rotation.from_quat(quat).as_matrix()


def load_mbm_problem(scene_path, request_path, grid_shape=(64, 64, 64),
                     grid_origin=(-0.5, -0.5, 0.0), grid_resolution=0.025):
    with open(scene_path, 'r') as f:
        scene = yaml.safe_load(f)
    with open(request_path, 'r') as f:
        request = yaml.safe_load(f)
    
    grid = empty_voxel_grid(shape=grid_shape, origin=grid_origin, resolution=grid_resolution)
    
    for obj in scene.get('world', {}).get('collision_objects', []):
        for prim, pose in zip(obj.get('primitives', []), obj.get('primitive_poses', [])):
            ptype, dims = prim.get('type', ''), prim.get('dimensions', [])
            pos = pose.get('position', [0, 0, 0])
            rot = quat_to_rotmat(pose.get('orientation', [0, 0, 0, 1]))
            
            if ptype == 'sphere':
                grid = add_sphere(grid, pos, dims[0])
            elif ptype == 'box':
                grid = add_box(grid, pos, [d/2 for d in dims], rot)
            elif ptype == 'cylinder':
                grid = add_cylinder(grid, pos, dims[1], dims[0]/2, rot)
    
    start_js = request.get('start_state', {}).get('joint_state', {})
    start_names, start_pos = start_js.get('name', []), start_js.get('position', [])
    arm_joints = [f'panda_joint{i}' for i in range(1, 8)]
    q_start = [start_pos[start_names.index(j)] if j in start_names else 0.0 for j in arm_joints]
    
    goal = request.get('goal_constraints', [{}])[0]
    q_goal = [0.0] * 7
    for c in goal.get('joint_constraints', []):
        if c.get('joint_name', '') in arm_joints:
            q_goal[arm_joints.index(c['joint_name'])] = c.get('position', 0.0)
    
    return {
        'grid': grid, 'dist_field': compute_distance_field(grid),
        'q_start': jnp.array(q_start), 'q_goal': jnp.array(q_goal),
    }


def load_mbm_dataset(dataset_dir, max_problems=None, **grid_kwargs):
    dataset_dir = Path(dataset_dir)
    problems = []
    for scene_file in sorted(dataset_dir.glob('scene*.yaml')):
        num = scene_file.stem.replace('scene', '')
        request_file = dataset_dir / f'request{num}.yaml'
        if request_file.exists():
            try:
                problem = load_mbm_problem(scene_file, request_file, **grid_kwargs)
                problem['name'] = f'problem_{num}'
                problems.append(problem)
                if max_problems and len(problems) >= max_problems:
                    break
            except Exception as e:
                print(f"Failed to load {scene_file}: {e}")
    return problems


# ============================================================================
# PFMP Integration
# ============================================================================

class Params(NamedTuple):
    goal: jax.Array
    origin: jax.Array
    resolution: float
    dist_field: jax.Array
    q_min: jax.Array
    q_max: jax.Array


Q_MIN = jnp.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
Q_MAX = jnp.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])


def make_planning_fns(robot: dict):
    robot_sdf = make_robot_sdf_fn(robot)
    
    def cost_fn(x, u, theta, t):
        sdf = robot_sdf(x, theta.origin, theta.resolution, theta.dist_field)
        return 1000.0 * jnp.maximum(0.05 - sdf, 0.0) ** 2 + 10.0 * jnp.linalg.norm(x - theta.goal)
    
    def terminal_cost(x, theta):
        sdf = robot_sdf(x, theta.origin, theta.resolution, theta.dist_field)
        return 1000.0 * jnp.maximum(0.05 - sdf, 0.0) ** 2 + 500.0 * jnp.linalg.norm(x - theta.goal) ** 2
    
    def dynamics(x, u, theta, t):
        diff = jax.vmap(lambda tgt, cur: jnp.arctan2(jnp.sin(tgt - cur), jnp.cos(tgt - cur)))(u, x)
        dist = jnp.linalg.norm(diff)
        x_next = jnp.clip(x + diff / (dist + 1e-8) * jnp.minimum(dist, 0.1), theta.q_min, theta.q_max)
        return x_next, x_next
    
    def init(key, x, theta):
        k1, k2, k3 = jax.random.split(key, 3)
        r = jax.random.uniform(k1)
        return jax.lax.switch(
            jnp.where(r < 0.05, 0, jnp.where(r < 0.1, 1, 2)),
            [lambda _: theta.goal,
             lambda _: jnp.clip(x + jax.random.normal(k2, x.shape) * 0.2, theta.q_min, theta.q_max),
             lambda _: jax.random.uniform(k3, x.shape, minval=theta.q_min, maxval=theta.q_max)], None)
    
    def proposal(key, x, u, theta, t):
        return init(key, x, theta)
    
    return cost_fn, terminal_cost, dynamics, init, proposal, robot_sdf


# ============================================================================
# Visualization
# ============================================================================

def visualize(grid, robot=None, q=None, dist_field=None, q_goal=None, paths=None):
    """
    Interactive visualization.
    
    Args:
        paths: dict with 'sis' and/or 'sir' keys, each containing (path, collision_mask)
    """
    import viser
    import trimesh
    
    server = viser.ViserServer()
    print("Open http://localhost:8080")
    
    if dist_field is None:
        dist_field = compute_distance_field(grid)
    
    occupied = np.array(grid.data > 0.5)
    indices = np.argwhere(occupied)
    voxel_positions = np.array(grid.origin) + (indices + 0.5) * grid.resolution
    
    voxel_handle = None
    
    def render_cubes():
        nonlocal voxel_handle
        if voxel_handle is not None:
            voxel_handle.remove()
        cube = trimesh.creation.box(extents=(grid.resolution * 0.9,) * 3)
        meshes = [cube.copy().apply_translation(p) for p in voxel_positions]
        voxel_handle = server.scene.add_mesh_trimesh("/voxels", mesh=trimesh.util.concatenate(meshes))
    
    def render_points():
        nonlocal voxel_handle
        if voxel_handle is not None:
            voxel_handle.remove()
        voxel_handle = server.scene.add_point_cloud(
            "/voxels", points=voxel_positions.astype(np.float32),
            colors=np.full((len(voxel_positions), 3), 100, dtype=np.uint8),
            point_size=0.01, point_shape="square")
    
    render_cubes()
    server.scene.add_grid("/ground", width=2, height=2, plane="xy")
    
    # Display controls
    with server.gui.add_folder("Display"):
        voxel_mode = server.gui.add_dropdown("Voxels", options=["cubes", "points"], initial_value="cubes")
        voxel_mode.on_update(lambda _: render_cubes() if voxel_mode.value == "cubes" else render_points())
    
    sphere_handles, goal_handles = [], []
    radii = np.array(robot['sphere_radii']) if robot else []
    
    if robot is not None:
        q = jnp.array(q) if q is not None else jnp.zeros(robot['n_joints'])
        positions = np.array(get_sphere_positions(q, robot))
        
        for i, (pos, r) in enumerate(zip(positions, radii)):
            sphere_handles.append(server.scene.add_icosphere(
                f"/robot/{i}", radius=float(r), position=tuple(pos)))
        
        if q_goal is not None:
            goal_positions = np.array(get_sphere_positions(jnp.array(q_goal), robot))
            for i, (pos, r) in enumerate(zip(goal_positions, radii)):
                goal_handles.append(server.scene.add_icosphere(
                    f"/goal/{i}", radius=float(r) * 0.8, position=tuple(pos)))
        
        robot_sdf = make_robot_sdf_fn(robot)
        
        def update_robot(q_new):
            positions = np.array(get_sphere_positions(q_new, robot))
            sdf = robot_sdf(q_new, grid.origin, grid.resolution, dist_field)
            color = (244, 67, 54) if sdf < 0 else (76, 175, 80)
            for h, pos in zip(sphere_handles, positions):
                h.position, h.color = tuple(pos), color
    
    # Animation state
    anim_state = {'playing': False, 'frame': 0, 'method': 'sir', 'speed': 1.0, 'updating': False}

    def set_frame(method, frame):
        if anim_state['updating']:
            return
        anim_state['updating'] = True
        
        data = path_data[method]
        frame = int(frame) % data['n_frames']
        positions = data['spheres'][frame]
        is_collision = bool(data['collision'][frame])
        color = (244, 67, 54) if is_collision else (76, 175, 80)
        for i, h in enumerate(sphere_handles):
            pos = positions[i]
            h.position = (float(pos[0]), float(pos[1]), float(pos[2]))
            h.color = color
        frame_slider.value = frame
        anim_state['frame'] = frame
        
        anim_state['updating'] = False

    if paths is not None:
        get_spheres_batch = jax.jit(jax.vmap(lambda q: get_sphere_positions(q, robot)))
        
        # Precompute sphere positions for all paths
        path_data = {}
        for method, (path, col_mask) in paths.items():
            path_data[method] = {
                'path': path,
                'spheres': np.array(get_spheres_batch(path)),
                'collision': np.array(col_mask),
                'n_frames': len(path),
            }

        path_data = {}
        for method, (path, col_mask) in paths.items():
            spheres_jax = get_spheres_batch(jnp.array(path))
            path_data[method] = {
                'path': np.asarray(path).copy(),
                'spheres': np.asarray(spheres_jax).copy(),  # .copy() ensures no JAX references
                'collision': np.asarray(col_mask).copy(),
                'n_frames': int(len(path)),
            }
        
        with server.gui.add_folder("Playback"):
            method_dropdown = server.gui.add_dropdown(
                "Method", options=list(paths.keys()), initial_value=anim_state['method'])
            frame_slider = server.gui.add_slider(
                "Frame", min=0, max=path_data[anim_state['method']]['n_frames'] - 1, 
                step=1, initial_value=0)
            play_button = server.gui.add_button("Play")
            speed_slider = server.gui.add_slider("Speed", min=0.1, max=3.0, step=0.1, initial_value=1.0)
            
            # Stats
            for method in paths.keys():
                col_pct = path_data[method]['collision'].mean() * 100
                server.gui.add_text(f"{method.upper()} collision", 
                                   initial_value=f"{col_pct:.1f}%", disabled=True)
        
        @method_dropdown.on_update
        def _(_):
            anim_state['method'] = method_dropdown.value
            frame_slider.max = path_data[method_dropdown.value]['n_frames'] - 1
            set_frame(method_dropdown.value, 0)
        
        @play_button.on_click
        def _(_):
            anim_state['playing'] = not anim_state['playing']
            play_button.name = "Pause" if anim_state['playing'] else "Play"
        
        @speed_slider.on_update
        def _(_):
            anim_state['speed'] = speed_slider.value
        
        set_frame(anim_state['method'], 0)
    
    elif robot is not None:
        # Manual joint control
        with server.gui.add_folder("Joints"):
            sliders = []
            for i in range(robot['n_joints']):
                s = server.gui.add_slider(f"q{i+1}", min=-3.14, max=3.14, step=0.01, initial_value=float(q[i]))
                s.on_update(lambda _, sl=sliders: update_robot(jnp.array([s.value for s in sl])))
                sliders.append(s)
    
    with server.gui.add_folder("Info"):
        server.gui.add_text("Resolution", initial_value=f"{grid.resolution:.3f}m", disabled=True)
        server.gui.add_text("Occupied", initial_value=str(len(voxel_positions)), disabled=True)
    
    # Main loop
    last_time = time.time()
    while True:
        if paths is not None and anim_state['playing']:
            now = time.time()
            if now - last_time > 0.03 / anim_state['speed']:
                anim_state['frame'] = (anim_state['frame'] + 1) % path_data[anim_state['method']]['n_frames']
                set_frame(anim_state['method'], anim_state['frame'])
                last_time = now
        time.sleep(0.01)


# ============================================================================
# Benchmark
# ============================================================================

def run_benchmark(dataset_dir, urdf_path='models/panda_spherized.urdf', n_problems=None, seed=0, visualize_results=False):
    from pfmp import PFMP
    from pfmp.interpolate import hermite_spline
    
    robot = load_robot(urdf_path)
    problems = load_mbm_dataset(dataset_dir, max_problems=n_problems)
    print(f"Loaded {len(problems)} problems")
    
    if not problems:
        return None
    
    cost_fn, terminal_cost, dynamics, init, proposal, robot_sdf = make_planning_fns(robot)
    
    pfmp = PFMP(dynamics, cost_fn, terminal_cost, init, proposal,
                horizon=1.0, num_knots=5, particles=512, ess_resample_thresh=0.1)
    
    @jax.jit
    def run_sis(key, q_start, theta):
        traj, scores = pfmp.run_sis(key, q_start, theta)
        return traj[jnp.argmin(jnp.sum(scores, axis=-1))]
    
    @jax.jit
    def run_sir(key, q_start, theta):
        traj, scores = pfmp.run_sir(key, q_start, theta)
        return traj[jnp.argmin(jnp.sum(scores, axis=-1))]
    
    @jax.jit
    def extract_path(traj, q_start, theta):
        tq = jnp.linspace(0, pfmp.horizon, int(pfmp.horizon / pfmp.dt))
        us = hermite_spline(tq, pfmp.tk(), traj)
        _, path = jax.lax.scan(lambda x, u: (dynamics(x, u, theta, 0.0)[0], x), q_start, us)
        return path
    
    keys = jax.random.split(jax.random.key(seed), len(problems))
    results = {k: [] for k in ['sis_success', 'sir_success', 'sis_collision', 'sir_collision',
                                'sis_dist', 'sir_dist', 'sis_time', 'sir_time']}
    
    print("\nRunning benchmark (SIS + SIR)...")
    
    for i, problem in enumerate(problems):
        theta = Params(goal=problem['q_goal'], origin=problem['grid'].origin,
                       resolution=problem['grid'].resolution, dist_field=problem['dist_field'],
                       q_min=Q_MIN, q_max=Q_MAX)
        
        # SIS
        t0 = time.time()
        traj_sis = run_sis(keys[i], problem['q_start'], theta)
        path_sis = extract_path(traj_sis, problem['q_start'], theta)
        sis_time = time.time() - t0
        
        # SIR
        t0 = time.time()
        traj_sir = run_sir(keys[i], problem['q_start'], theta)
        path_sir = extract_path(traj_sir, problem['q_start'], theta)
        sir_time = time.time() - t0
        
        # Evaluate
        col_sis = np.array(jax.vmap(lambda q: robot_sdf(q, theta.origin, theta.resolution, theta.dist_field) < 0)(path_sis))
        col_sir = np.array(jax.vmap(lambda q: robot_sdf(q, theta.origin, theta.resolution, theta.dist_field) < 0)(path_sir))
        
        sis_dist = float(jnp.linalg.norm(path_sis[-1] - problem['q_goal']))
        sir_dist = float(jnp.linalg.norm(path_sir[-1] - problem['q_goal']))
        
        results['sis_success'].append(not col_sis.any() and sis_dist < 0.5)
        results['sir_success'].append(not col_sir.any() and sir_dist < 0.5)
        results['sis_collision'].append(col_sis.mean())
        results['sir_collision'].append(col_sir.mean())
        results['sis_dist'].append(sis_dist)
        results['sir_dist'].append(sir_dist)
        results['sis_time'].append(sis_time)
        results['sir_time'].append(sir_time)
        
        if (i + 1) % 10 == 0 or i == len(problems) - 1:
            print(f"  [{i+1}/{len(problems)}] SIS: {np.mean(results['sis_success'])*100:.1f}%, SIR: {np.mean(results['sir_success'])*100:.1f}%")
        
        # Store paths for visualization
        if visualize_results and i == len(problems) - 1:
            problem['paths'] = {
                'sis': (path_sis, col_sis),
                'sir': (path_sir, col_sir),
            }
    
    print(f"\n{'='*60}")
    print(f"RESULTS ({len(problems)} problems)")
    print(f"{'='*60}")
    print(f"  {'Method':<8} {'Success':>10} {'Collision':>12} {'Dist':>10} {'Time (ms)':>12}")
    print(f"  {'-'*8} {'-'*10} {'-'*12} {'-'*10} {'-'*12}")
    for m in ['sis', 'sir']:
        print(f"  {m.upper():<8} {np.mean(results[f'{m}_success'])*100:>9.1f}% "
              f"{np.mean(results[f'{m}_collision'])*100:>11.1f}% "
              f"{np.mean(results[f'{m}_dist']):>10.3f} {np.mean(results[f'{m}_time'])*1000:>11.1f}")
    
    if visualize_results and problems:
        p = problems[-1]
        visualize(p['grid'], robot=robot, q=np.array(p['q_start']),
                 dist_field=p['dist_field'], q_goal=np.array(p['q_goal']), paths=p['paths'])
    
    return results


# ============================================================================
# Test Scene
# ============================================================================

def create_test_scene() -> tuple:
    grid = empty_voxel_grid(shape=(64, 64, 64), origin=(-0.5, -0.5, 0.0), resolution=0.025)
    grid = add_box(grid, [0.5, 0.0, 0.4], [0.3, 0.4, 0.02])
    grid = add_box(grid, [0.4, 0.1, 0.5], [0.05, 0.05, 0.1])
    grid = add_cylinder(grid, [0.5, -0.1, 0.5], 0.04, 0.08)
    grid = add_sphere(grid, [0.6, 0.0, 0.48], 0.05)
    return grid, compute_distance_field(grid)


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    import sys
    args = sys.argv[1:]
    
    if not args or args[0] == "viz":
        robot = load_robot()
        grid, dist_field = create_test_scene()
        visualize(grid, robot=robot, q=np.zeros(7), dist_field=dist_field)
    
    elif args[0] == "bench":
        dataset = args[1] if len(args) > 1 else "./models/problems/table_pick_panda"
        n = int(args[2]) if len(args) > 2 else None
        viz = "--viz" in args
        run_benchmark(dataset, n_problems=n, visualize_results=viz)
    
    elif args[0] == "problem":
        from pfmp import PFMP
        from pfmp.interpolate import hermite_spline
        
        dataset = args[1] if len(args) > 1 else "./models/problems/table_pick_panda"
        idx = int(args[2]) if len(args) > 2 else 0
        
        robot = load_robot()
        problems = load_mbm_dataset(dataset, max_problems=idx+1)
        
        if problems:
            p = problems[idx]
            cost_fn, terminal_cost, dynamics, init, proposal, robot_sdf = make_planning_fns(robot)
            
            pfmp = PFMP(dynamics, cost_fn, terminal_cost, init, proposal,
                        horizon=1.0, num_knots=5, particles=512, ess_resample_thresh=0.1)
            
            theta = Params(goal=p['q_goal'], origin=p['grid'].origin,
                          resolution=p['grid'].resolution, dist_field=p['dist_field'],
                          q_min=Q_MIN, q_max=Q_MAX)
            
            key = jax.random.key(0)
            
            # Run both methods
            traj_sis, scores_sis = pfmp.run_sis(key, p['q_start'], theta)
            traj_sir, scores_sir = pfmp.run_sir(key, p['q_start'], theta)
            
            best_sis = traj_sis[jnp.argmin(jnp.sum(scores_sis, axis=-1))]
            best_sir = traj_sir[jnp.argmin(jnp.sum(scores_sir, axis=-1))]
            
            # Extract paths
            tq = jnp.linspace(0, pfmp.horizon, int(pfmp.horizon / pfmp.dt))
            
            def extract(traj):
                us = hermite_spline(tq, pfmp.tk(), traj)
                _, path = jax.lax.scan(lambda x, u: (dynamics(x, u, theta, 0.0)[0], x), p['q_start'], us)
                return path
            
            path_sis, path_sir = extract(best_sis), extract(best_sir)
            
            col_sis = np.array(jax.vmap(lambda q: robot_sdf(q, theta.origin, theta.resolution, theta.dist_field) < 0)(path_sis))
            col_sir = np.array(jax.vmap(lambda q: robot_sdf(q, theta.origin, theta.resolution, theta.dist_field) < 0)(path_sir))
            
            print(f"SIS: {col_sis.mean()*100:.1f}% collision")
            print(f"SIR: {col_sir.mean()*100:.1f}% collision")
            
            visualize(p['grid'], robot=robot, q=np.array(p['q_start']),
                     dist_field=p['dist_field'], q_goal=np.array(p['q_goal']),
                     paths={'sis': (path_sis, col_sis), 'sir': (path_sir, col_sir)})
    
    else:
        print("Usage:")
        print("  python mbm.py viz                      # Visualize test scene")
        print("  python mbm.py bench [dataset] [n] [--viz]  # Run benchmark")
        print("  python mbm.py problem [dataset] [idx]  # Plan & visualize problem")