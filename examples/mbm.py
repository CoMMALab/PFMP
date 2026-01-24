"""
Simplified MBM benchmark with pure JAX forward kinematics.
No MuJoCo dependency - loads robot from panda_spherized.urdf.
"""

import jax
import jax.numpy as jnp
import numpy as np
import yaml
import time
from pathlib import Path
from typing import NamedTuple
from scipy.spatial.transform import Rotation


# ============================================================================
# Environment representation
# ============================================================================

class Environment(NamedTuple):
    """Environment with spheres, oriented boxes, and cylinders."""
    sphere_centers: jax.Array      # (N_s, 3)
    sphere_radii: jax.Array        # (N_s,)
    box_centers: jax.Array         # (N_b, 3)
    box_half_extents: jax.Array    # (N_b, 3)
    box_rotations: jax.Array       # (N_b, 3, 3)
    cylinder_centers: jax.Array    # (N_c, 3)
    cylinder_radii: jax.Array      # (N_c,)
    cylinder_half_heights: jax.Array  # (N_c,)
    cylinder_rotations: jax.Array  # (N_c, 3, 3)


def empty_environment() -> Environment:
    return Environment(
        sphere_centers=jnp.zeros((0, 3)),
        sphere_radii=jnp.zeros((0,)),
        box_centers=jnp.zeros((0, 3)),
        box_half_extents=jnp.zeros((0, 3)),
        box_rotations=jnp.zeros((0, 3, 3)),
        cylinder_centers=jnp.zeros((0, 3)),
        cylinder_radii=jnp.zeros((0,)),
        cylinder_half_heights=jnp.zeros((0,)),
        cylinder_rotations=jnp.zeros((0, 3, 3)),
    )


# ============================================================================
# SDF functions (all JIT-compatible)
# ============================================================================

def sdf_sphere(point: jax.Array, center: jax.Array, radius: float) -> float:
    """Signed distance to sphere surface."""
    return jnp.linalg.norm(point - center) - radius


def sdf_box_oriented(point: jax.Array, center: jax.Array, 
                     half_extents: jax.Array, rotation: jax.Array) -> float:
    """Signed distance to oriented box surface."""
    local_point = rotation.T @ (point - center)
    local = jnp.abs(local_point) - half_extents
    outside = jnp.linalg.norm(jnp.maximum(local, 0.0))
    inside = jnp.minimum(jnp.max(local), 0.0)
    return outside + inside


def sdf_cylinder(point: jax.Array, center: jax.Array, radius: float,
                 half_height: float, rotation: jax.Array) -> float:
    """Signed distance to oriented cylinder (axis along local Z)."""
    local_point = rotation.T @ (point - center)
    xy = local_point[:2]
    z = local_point[2]
    
    d_radial = jnp.linalg.norm(xy) - radius
    d_axial = jnp.abs(z) - half_height
    
    outside = jnp.sqrt(jnp.maximum(d_radial, 0.0)**2 + jnp.maximum(d_axial, 0.0)**2)
    inside = jnp.minimum(jnp.maximum(d_radial, d_axial), 0.0)
    
    return outside + inside


def point_sdf(point: jax.Array, env: Environment) -> float:
    """
    Minimum SDF from point to all obstacles.
    Handles empty arrays by padding with far-away dummy obstacles.
    """
    FAR = 1e10
    
    # Spheres: pad with far-away sphere
    sphere_centers = jnp.concatenate([env.sphere_centers, jnp.array([[FAR, FAR, FAR]])], axis=0)
    sphere_radii = jnp.concatenate([env.sphere_radii, jnp.array([1.0])], axis=0)
    sphere_sdfs = jax.vmap(lambda c, r: sdf_sphere(point, c, r))(sphere_centers, sphere_radii)
    
    # Boxes: pad with far-away box
    box_centers = jnp.concatenate([env.box_centers, jnp.array([[FAR, FAR, FAR]])], axis=0)
    box_half_extents = jnp.concatenate([env.box_half_extents, jnp.array([[1.0, 1.0, 1.0]])], axis=0)
    box_rotations = jnp.concatenate([env.box_rotations, jnp.eye(3)[None]], axis=0)
    box_sdfs = jax.vmap(lambda c, h, r: sdf_box_oriented(point, c, h, r))(
        box_centers, box_half_extents, box_rotations)
    
    # Cylinders: pad with far-away cylinder
    cyl_centers = jnp.concatenate([env.cylinder_centers, jnp.array([[FAR, FAR, FAR]])], axis=0)
    cyl_radii = jnp.concatenate([env.cylinder_radii, jnp.array([1.0])], axis=0)
    cyl_half_heights = jnp.concatenate([env.cylinder_half_heights, jnp.array([1.0])], axis=0)
    cyl_rotations = jnp.concatenate([env.cylinder_rotations, jnp.eye(3)[None]], axis=0)
    cyl_sdfs = jax.vmap(lambda c, r, hh, rot: sdf_cylinder(point, c, r, hh, rot))(
        cyl_centers, cyl_radii, cyl_half_heights, cyl_rotations)
    
    return jnp.minimum(jnp.minimum(jnp.min(sphere_sdfs), jnp.min(box_sdfs)), jnp.min(cyl_sdfs))


# ============================================================================
# Robot from URDF (pure JAX)
# ============================================================================

def parse_panda_urdf(urdf_path: str):
    """
    Parse panda_spherized.urdf to extract:
    - Joint axes and origins for FK
    - Collision sphere positions and radii
    """
    import xml.etree.ElementTree as ET
    
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    
    # Parse joints
    joints_info = []
    for joint in root.findall('joint'):
        jtype = joint.get('type')
        jname = joint.get('name')
        parent = joint.find('parent').get('link')
        child = joint.find('child').get('link')
        
        if jtype in ['revolute', 'continuous']:
            origin = joint.find('origin')
            if origin is not None:
                xyz = [float(x) for x in origin.get('xyz', '0 0 0').split()]
                rpy = [float(x) for x in origin.get('rpy', '0 0 0').split()]
            else:
                xyz, rpy = [0, 0, 0], [0, 0, 0]
            
            axis_elem = joint.find('axis')
            if axis_elem is not None:
                axis = [float(x) for x in axis_elem.get('xyz', '0 0 1').split()]
            else:
                axis = [0, 0, 1]
            
            joints_info.append({
                'name': jname,
                'parent': parent,
                'child': child,
                'xyz': xyz,
                'rpy': rpy,
                'axis': axis,
            })
    
    # Parse collision spheres from links
    spheres_info = []
    for link in root.findall('link'):
        link_name = link.get('name')
        
        for collision in link.findall('collision'):
            geom = collision.find('geometry')
            if geom is None:
                continue
                
            sphere = geom.find('sphere')
            if sphere is None:
                continue
            
            radius = float(sphere.get('radius', '0.05'))
            
            origin = collision.find('origin')
            if origin is not None:
                xyz = [float(x) for x in origin.get('xyz', '0 0 0').split()]
            else:
                xyz = [0, 0, 0]
            
            spheres_info.append({
                'link': link_name,
                'offset': xyz,
                'radius': radius,
            })
    
    return {'joints': joints_info, 'spheres': spheres_info}


def build_robot_model(urdf_path: str):
    """Build JAX-compatible robot model from URDF."""
    urdf_data = parse_panda_urdf(urdf_path)
    
    joints = urdf_data['joints']
    spheres = urdf_data['spheres']
    
    # Filter to arm joints (panda_joint1 through panda_joint7)
    arm_joint_names = [f'panda_joint{i}' for i in range(1, 8)]
    arm_joints = [j for j in joints if j['name'] in arm_joint_names]
    arm_joints.sort(key=lambda j: arm_joint_names.index(j['name']))
    
    # Build joint data
    joint_xyz = np.array([j['xyz'] for j in arm_joints])
    joint_rpy = np.array([j['rpy'] for j in arm_joints])
    joint_axes = np.array([j['axis'] for j in arm_joints])
    
    # Map link names to joint indices
    # Sphere on link X uses transform after joint that has X as child
    link_to_joint_idx = {'panda_link0': -1}  # Base link before any joints
    for i, j in enumerate(arm_joints):
        link_to_joint_idx[j['child']] = i
    
    # Build sphere data
    sphere_data = []
    for s in spheres:
        link = s['link']
        if link in link_to_joint_idx:
            sphere_data.append({
                'joint_idx': link_to_joint_idx[link],
                'offset': s['offset'],
                'radius': s['radius'],
            })
    
    if not sphere_data:
        raise ValueError("No collision spheres found in URDF")
    
    return {
        'n_joints': len(arm_joints),
        'joint_xyz': jnp.array(joint_xyz),
        'joint_rpy': jnp.array(joint_rpy),
        'joint_axes': jnp.array(joint_axes),
        'sphere_joint_indices': jnp.array([s['joint_idx'] for s in sphere_data]),
        'sphere_offsets': jnp.array([s['offset'] for s in sphere_data]),
        'sphere_radii': jnp.array([s['radius'] for s in sphere_data]),
        'n_spheres': len(sphere_data),
    }


def rpy_to_rotmat(rpy: jax.Array) -> jax.Array:
    """Convert roll-pitch-yaw to rotation matrix."""
    r, p, y = rpy[0], rpy[1], rpy[2]
    cr, sr = jnp.cos(r), jnp.sin(r)
    cp, sp = jnp.cos(p), jnp.sin(p)
    cy, sy = jnp.cos(y), jnp.sin(y)
    
    return jnp.array([
        [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
        [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
        [-sp,   cp*sr,            cp*cr],
    ])


def axis_angle_to_rotmat(axis: jax.Array, angle: float) -> jax.Array:
    """Convert axis-angle to rotation matrix (Rodrigues formula)."""
    axis = axis / (jnp.linalg.norm(axis) + 1e-10)
    K = jnp.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0],
    ])
    return jnp.eye(3) + jnp.sin(angle) * K + (1 - jnp.cos(angle)) * (K @ K)


def forward_kinematics(q: jax.Array, robot: dict) -> jax.Array:
    """
    Compute forward kinematics.
    
    Returns:
        Link transforms (n_joints, 4, 4) - transform from base to each link
    """
    def compute_transform(T_prev, inputs):
        xyz, rpy, axis, theta = inputs
        
        # Fixed transform from joint origin
        R_fixed = rpy_to_rotmat(rpy)
        T_fixed = jnp.eye(4).at[:3, :3].set(R_fixed).at[:3, 3].set(xyz)
        
        # Joint rotation
        R_joint = axis_angle_to_rotmat(axis, theta)
        T_joint = jnp.eye(4).at[:3, :3].set(R_joint)
        
        T_curr = T_prev @ T_fixed @ T_joint
        return T_curr, T_curr
    
    _, transforms = jax.lax.scan(
        compute_transform, 
        jnp.eye(4), 
        (robot['joint_xyz'], robot['joint_rpy'], robot['joint_axes'], q)
    )
    return transforms


def get_sphere_positions(q: jax.Array, robot: dict) -> jax.Array:
    """Get world positions of all collision spheres."""
    transforms = forward_kinematics(q, robot)
    
    # Add identity for base link (joint_idx = -1)
    all_transforms = jnp.concatenate([jnp.eye(4)[None], transforms], axis=0)
    
    def get_pos(idx, offset):
        T = all_transforms[idx + 1]  # Shift: -1 -> 0, 0 -> 1, etc.
        return (T @ jnp.append(offset, 1.0))[:3]
    
    return jax.vmap(get_pos)(robot['sphere_joint_indices'], robot['sphere_offsets'])


# Global robot model cache
_ROBOT = None


def load_robot(urdf_path: str = 'models/panda_spherized.urdf'):
    """Load robot model (cached)."""
    global _ROBOT
    if _ROBOT is None:
        _ROBOT = build_robot_model(urdf_path)
        print(f"Loaded robot: {_ROBOT['n_joints']} joints, {_ROBOT['n_spheres']} spheres")
    return _ROBOT


# ============================================================================
# Robot-Environment Collision
# ============================================================================

def robot_sdf(q: jax.Array, env: Environment, robot: dict) -> float:
    """Minimum signed distance from robot to environment."""
    positions = get_sphere_positions(q, robot)
    radii = robot['sphere_radii']
    
    sdfs = jax.vmap(lambda p, r: point_sdf(p, env) - r)(positions, radii)
    return jnp.min(sdfs)


# ============================================================================
# MBM Dataset Loading  
# ============================================================================

def quat_to_rotmat(quat):
    """Convert [x, y, z, w] quaternion to 3x3 rotation matrix."""
    return Rotation.from_quat(quat).as_matrix()


def parse_mbm_scene(scene_path):
    """Parse MotionBenchMaker scene YAML."""
    with open(scene_path, 'r') as f:
        scene = yaml.safe_load(f)
    
    spheres = {'centers': [], 'radii': []}
    boxes = {'centers': [], 'half_extents': [], 'rotations': []}
    cylinders = {'centers': [], 'radii': [], 'half_heights': [], 'rotations': []}
    
    for obj in scene.get('world', {}).get('collision_objects', []):
        for prim, pose in zip(obj.get('primitives', []), obj.get('primitive_poses', [])):
            prim_type = prim.get('type', '')
            dims = prim.get('dimensions', [])
            pos = pose.get('position', [0, 0, 0])
            quat = pose.get('orientation', [0, 0, 0, 1])
            rotmat = quat_to_rotmat(quat)
            
            if prim_type == 'sphere':
                spheres['centers'].append(pos)
                spheres['radii'].append(dims[0])
            elif prim_type == 'box':
                boxes['centers'].append(pos)
                boxes['half_extents'].append([d / 2 for d in dims])
                boxes['rotations'].append(rotmat)
            elif prim_type == 'cylinder':
                cylinders['centers'].append(pos)
                cylinders['half_heights'].append(dims[0] / 2)
                cylinders['radii'].append(dims[1])
                cylinders['rotations'].append(rotmat)
    
    return {'spheres': spheres, 'boxes': boxes, 'cylinders': cylinders}


def parse_mbm_request(request_path):
    """Parse MotionBenchMaker request YAML."""
    with open(request_path, 'r') as f:
        request = yaml.safe_load(f)
    
    start_js = request.get('start_state', {}).get('joint_state', {})
    start_names = start_js.get('name', [])
    start_pos = start_js.get('position', [])
    
    arm_joints = [f'panda_joint{i}' for i in range(1, 8)]
    
    q_start = [start_pos[start_names.index(j)] if j in start_names else 0.0 for j in arm_joints]
    
    goal = request.get('goal_constraints', [{}])[0]
    q_goal = [0.0] * 7
    for c in goal.get('joint_constraints', []):
        name = c.get('joint_name', '')
        if name in arm_joints:
            q_goal[arm_joints.index(name)] = c.get('position', 0.0)
    
    return {'q_start': np.array(q_start), 'q_goal': np.array(q_goal)}


def scene_to_environment(scene_data) -> Environment:
    """Convert parsed scene to Environment."""
    s, b, c = scene_data['spheres'], scene_data['boxes'], scene_data['cylinders']
    
    return Environment(
        sphere_centers=jnp.array(s['centers']) if s['centers'] else jnp.zeros((0, 3)),
        sphere_radii=jnp.array(s['radii']) if s['radii'] else jnp.zeros((0,)),
        box_centers=jnp.array(b['centers']) if b['centers'] else jnp.zeros((0, 3)),
        box_half_extents=jnp.array(b['half_extents']) if b['half_extents'] else jnp.zeros((0, 3)),
        box_rotations=jnp.array(b['rotations']) if b['rotations'] else jnp.zeros((0, 3, 3)),
        cylinder_centers=jnp.array(c['centers']) if c['centers'] else jnp.zeros((0, 3)),
        cylinder_radii=jnp.array(c['radii']) if c['radii'] else jnp.zeros((0,)),
        cylinder_half_heights=jnp.array(c['half_heights']) if c['half_heights'] else jnp.zeros((0,)),
        cylinder_rotations=jnp.array(c['rotations']) if c['rotations'] else jnp.zeros((0, 3, 3)),
    )


def load_mbm_problem(scene_path, request_path):
    """Load a single MBM problem."""
    scene_data = parse_mbm_scene(scene_path)
    request_data = parse_mbm_request(request_path)
    
    return {
        'env': scene_to_environment(scene_data),
        'q_start': jnp.array(request_data['q_start']),
        'q_goal': jnp.array(request_data['q_goal']),
    }


def load_mbm_dataset(dataset_dir, max_problems=None):
    """Load all problems from a dataset directory."""
    dataset_dir = Path(dataset_dir)
    problems = []
    
    for scene_file in sorted(dataset_dir.glob('scene*.yaml')):
        num = scene_file.stem.replace('scene', '')
        request_file = dataset_dir / f'request{num}.yaml'
        
        if request_file.exists():
            try:
                problem = load_mbm_problem(scene_file, request_file)
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
    env: Environment
    q_min: jax.Array
    q_max: jax.Array
    robot: dict


Q_MIN = jnp.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
Q_MAX = jnp.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])


def cost_fn(x, u, theta, t):
    sdf = robot_sdf(x, theta.env, theta.robot)
    col_cost = 1000.0 * jnp.maximum(0.05 - sdf, 0.0) ** 2
    goal_cost = 10 * jnp.linalg.norm(x - theta.goal)
    return col_cost + goal_cost


def terminal_cost(x, theta):
    sdf = robot_sdf(x, theta.env, theta.robot)
    col_cost = 1000.0 * jnp.maximum(0.05 - sdf, 0.0) ** 2
    goal_cost = 500.0 * jnp.linalg.norm(x - theta.goal) ** 2
    return col_cost + goal_cost


def angle_diff(target, current):
    diff = target - current
    return jnp.arctan2(jnp.sin(diff), jnp.cos(diff))


def dynamics(x, u, theta, t):
    direction = jax.vmap(angle_diff)(u, x)
    dist = jnp.linalg.norm(direction)
    direction = direction / (dist + 1e-8)
    step = jnp.minimum(dist, 0.1)
    x_next = x + direction * step
    x_next = jnp.clip(x_next, theta.q_min, theta.q_max)
    return (x_next, x_next)


def init(key, x, theta):
    k1, k2, k3 = jax.random.split(key, 3)
    r = jax.random.uniform(k1)
    
    branch = jnp.where(r < 0.05, 0, jnp.where(r < 0.1, 1, 2))
    return jax.lax.switch(branch, [
        lambda _: theta.goal,
        lambda _: jnp.clip(x + jax.random.normal(k2, x.shape) * 0.2, theta.q_min, theta.q_max),
        lambda _: jax.random.uniform(k3, x.shape, minval=theta.q_min, maxval=theta.q_max),
    ], None)


def proposal(key, x, u, theta, t):
    return init(key, x, theta)


# ============================================================================
# Benchmark Runners
# ============================================================================

def run_benchmark_fast(dataset_dir, urdf_path='models/panda_spherized.urdf', n_problems=None, seed=0):
    """Fast benchmark - no visualization, tests both SIS and SIR."""
    from pfmp import PFMP
    from pfmp.interpolate import hermite_spline
    
    robot = load_robot(urdf_path)
    problems = load_mbm_dataset(dataset_dir, max_problems=n_problems)
    n = len(problems)
    print(f"Loaded {n} problems from {dataset_dir}")
    
    if not problems:
        return None
    
    pfmp = PFMP(
        dynamics, cost_fn, terminal_cost, init, proposal,
        horizon=1.0, num_knots=5, particles=1000,
        ess_resample_thresh=0.1
    )
    
    @jax.jit
    def run_sis(key, q_start, theta):
        traj, scores = pfmp.run_sis(key, q_start, theta)
        best = jnp.argmin(jnp.sum(scores, axis=-1))
        return traj[best]
    
    @jax.jit
    def run_sir(key, q_start, theta):
        traj, scores = pfmp.run_sir(key, q_start, theta)
        best = jnp.argmin(jnp.sum(scores, axis=-1))
        return traj[best]
    
    @jax.jit
    def extract_path(traj_knots, q_start, theta):
        tq = jnp.linspace(0, pfmp.horizon, int(pfmp.horizon / pfmp.dt))
        us = hermite_spline(tq, pfmp.tk(), traj_knots)
        _, path = jax.lax.scan(lambda x, u: (dynamics(x, u, theta, 0.0)[0], x), q_start, us)
        return path
    
    @jax.jit
    def evaluate_path(path, theta):
        col = jax.vmap(lambda q: robot_sdf(q, theta.env, theta.robot) < 0)(path)
        return col.any(), col.mean(), jnp.linalg.norm(path[-1] - theta.goal)
    
    keys = jax.random.split(jax.random.key(seed), n)
    
    results = {k: [] for k in ['sis_success', 'sir_success', 'sis_collision', 'sir_collision',
                                'sis_dist', 'sir_dist', 'sis_time', 'sir_time']}
    
    print("\nRunning benchmark (SIS + SIR)...")
    t_total = time.time()
    
    for i, problem in enumerate(problems):
        theta = Params(goal=problem['q_goal'], env=problem['env'],
                       q_min=Q_MIN, q_max=Q_MAX, robot=robot)
        
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
        sis_any_col, sis_col_frac, sis_dist = evaluate_path(path_sis, theta)
        sir_any_col, sir_col_frac, sir_dist = evaluate_path(path_sir, theta)
        
        results['sis_success'].append(bool((not sis_any_col) and (sis_dist < 0.5)))
        results['sir_success'].append(bool((not sir_any_col) and (sir_dist < 0.5)))
        results['sis_collision'].append(float(sis_col_frac))
        results['sir_collision'].append(float(sir_col_frac))
        results['sis_dist'].append(float(sis_dist))
        results['sir_dist'].append(float(sir_dist))
        results['sis_time'].append(sis_time)
        results['sir_time'].append(sir_time)
        
        if (i + 1) % 10 == 0 or i == n - 1:
            print(f"  [{i+1}/{n}] SIS: {np.mean(results['sis_success'])*100:.1f}%, SIR: {np.mean(results['sir_success'])*100:.1f}%")
    
    t_total = time.time() - t_total
    
    print(f"\n{'='*60}")
    print(f"RESULTS ({n} problems, {t_total:.1f}s total)")
    print(f"{'='*60}")
    print(f"  {'Method':<8} {'Success':>10} {'Collision':>12} {'Dist':>10} {'Time (ms)':>12}")
    print(f"  {'-'*8} {'-'*10} {'-'*12} {'-'*10} {'-'*12}")
    for m in ['sis', 'sir']:
        print(f"  {m.upper():<8} {np.mean(results[f'{m}_success'])*100:>9.1f}% "
              f"{np.mean(results[f'{m}_collision'])*100:>11.1f}% "
              f"{np.mean(results[f'{m}_dist']):>10.3f} {np.mean(results[f'{m}_time'])*1000:>11.1f}")
    
    return results


def add_environment_to_viser(server, env, offset=np.array([0, 0, 0]), prefix="/env"):
    """Add environment obstacles to viser scene."""
    import trimesh
    
    handles = []
    
    for i, (c, r) in enumerate(zip(np.array(env.sphere_centers), np.array(env.sphere_radii))):
        handles.append(server.scene.add_icosphere(
            f"{prefix}/sphere_{i}", radius=float(r), position=tuple(c + offset), color=(128, 128, 128)))
    
    for i, (c, h, rot) in enumerate(zip(np.array(env.box_centers), np.array(env.box_half_extents), np.array(env.box_rotations))):
        quat_xyzw = Rotation.from_matrix(rot).as_quat()
        quat_wxyz = (quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2])
        handles.append(server.scene.add_box(
            f"{prefix}/box_{i}", dimensions=tuple(2 * h), position=tuple(c + offset),
            wxyz=quat_wxyz, color=(128, 128, 128)))
    
    for i, (c, r, hh, rot) in enumerate(zip(np.array(env.cylinder_centers), np.array(env.cylinder_radii),
                                            np.array(env.cylinder_half_heights), np.array(env.cylinder_rotations))):
        quat_xyzw = Rotation.from_matrix(rot).as_quat()
        quat_wxyz = (quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2])
        mesh = trimesh.creation.cylinder(radius=float(r), height=float(2*hh))
        handles.append(server.scene.add_mesh_trimesh(
            f"{prefix}/cylinder_{i}", mesh=mesh, position=tuple(c + offset), wxyz=quat_wxyz))
    
    return handles


def run_benchmark_viser(dataset_dir, urdf_path='models/panda_spherized.urdf', n_problems=None, seed=0):
    """Benchmark with viser visualization - tests both SIS and SIR side by side."""
    import viser
    from pfmp import PFMP
    from pfmp.interpolate import hermite_spline
    
    robot = load_robot(urdf_path)
    problems = load_mbm_dataset(dataset_dir, max_problems=n_problems)
    n = len(problems)
    print(f"Loaded {n} problems from {dataset_dir}")
    
    if not problems:
        return None
    
    pfmp = PFMP(dynamics, cost_fn, terminal_cost, init, proposal,
                horizon=1.0, num_knots=5, particles=512, ess_resample_thresh=0.2)
    
    @jax.jit
    def run_sis(key, q_start, theta):
        traj, scores = pfmp.run_sis(key, q_start, theta)
        return traj[jnp.argmin(jnp.sum(scores, axis=-1))]
    
    @jax.jit
    def run_sir(key, q_start, theta):
        traj, scores = pfmp.run_sir(key, q_start, theta)
        return traj[jnp.argmin(jnp.sum(scores, axis=-1))]
    
    @jax.jit
    def extract_path(traj_knots, q_start, theta):
        tq = jnp.linspace(0, pfmp.horizon, int(pfmp.horizon / pfmp.dt))
        us = hermite_spline(tq, pfmp.tk(), traj_knots)
        _, path = jax.lax.scan(lambda x, u: (dynamics(x, u, theta, 0.0)[0], x), q_start, us)
        return path
    
    get_spheres_batch = jax.jit(jax.vmap(lambda q: get_sphere_positions(q, robot)))
    
    server = viser.ViserServer()
    print("Open http://localhost:8080")
    
    offset_sis, offset_sir = np.array([-0.8, 0, 0]), np.array([0.8, 0, 0])
    server.scene.add_grid("/ground_sis", width=2, height=2, position=(offset_sis[0], 0, 0), plane="xy")
    server.scene.add_grid("/ground_sir", width=2, height=2, position=(offset_sir[0], 0, 0), plane="xy")
    
    label_sis = server.scene.add_label("/label_sis", "SIS: 0%", position=tuple(offset_sis + [0, 0, 1.2]))
    label_sir = server.scene.add_label("/label_sir", "SIR: 0%", position=tuple(offset_sir + [0, 0, 1.2]))
    
    with server.gui.add_folder("Stats"):
        problem_text = server.gui.add_text("Problem", initial_value="0/0", disabled=True)
        sis_rate_text = server.gui.add_text("SIS Success", initial_value="0%", disabled=True)
        sir_rate_text = server.gui.add_text("SIR Success", initial_value="0%", disabled=True)
    
    results = {k: [] for k in ['sis_success', 'sir_success', 'sis_collision', 'sir_collision', 'sis_time', 'sir_time']}
    key = jax.random.key(seed)
    
    env_handles, sis_spheres, sir_spheres, ghost_handles = [], [], [], []
    
    def clear_dynamic():
        for h in env_handles + sis_spheres + sir_spheres + ghost_handles:
            h.remove()
        env_handles.clear(); sis_spheres.clear(); sir_spheres.clear(); ghost_handles.clear()
    
    sphere_radii = np.array(robot['sphere_radii'])
    
    for prob_idx, problem in enumerate(problems):
        print(f"\n=== Problem {prob_idx + 1}/{n}: {problem['name']} ===")
        
        env, q_start, q_goal = problem['env'], problem['q_start'], problem['q_goal']
        theta = Params(goal=q_goal, env=env, q_min=Q_MIN, q_max=Q_MAX, robot=robot)
        
        key, run_key = jax.random.split(key)
        
        # Run both methods
        t0 = time.time()
        path_sis = extract_path(run_sis(run_key, q_start, theta), q_start, theta)
        sis_time = time.time() - t0
        
        t0 = time.time()
        path_sir = extract_path(run_sir(run_key, q_start, theta), q_start, theta)
        sir_time = time.time() - t0
        
        # Evaluate
        col_sis = np.array(jax.vmap(lambda q: robot_sdf(q, env, robot) < 0)(path_sis))
        col_sir = np.array(jax.vmap(lambda q: robot_sdf(q, env, robot) < 0)(path_sir))
        
        sis_dist = float(jnp.linalg.norm(path_sis[-1] - q_goal))
        sir_dist = float(jnp.linalg.norm(path_sir[-1] - q_goal))
        
        sis_success = not col_sis.any() and sis_dist < 0.5
        sir_success = not col_sir.any() and sir_dist < 0.5
        
        results['sis_success'].append(sis_success)
        results['sir_success'].append(sir_success)
        results['sis_collision'].append(col_sis.mean())
        results['sir_collision'].append(col_sir.mean())
        results['sis_time'].append(sis_time)
        results['sir_time'].append(sir_time)
        
        print(f"  SIS: col={col_sis.mean():.0%}, dist={sis_dist:.2f}, success={sis_success}")
        print(f"  SIR: col={col_sir.mean():.0%}, dist={sir_dist:.2f}, success={sir_success}")
        
        # Visualization
        clear_dynamic()
        
        env_handles.extend(add_environment_to_viser(server, env, offset_sis, "/env_sis"))
        env_handles.extend(add_environment_to_viser(server, env, offset_sir, "/env_sir"))
        
        for i, r in enumerate(sphere_radii):
            sis_spheres.append(server.scene.add_icosphere(f"/sis/s{i}", radius=float(r), position=(0,0,0), color=(76, 175, 80)))
            sir_spheres.append(server.scene.add_icosphere(f"/sir/s{i}", radius=float(r), position=(0,0,0), color=(76, 175, 80)))
        
        # Start/goal ghosts
        for prefix, q, color in [("start", q_start, (200,200,200)), ("goal", q_goal, (255,193,7))]:
            pos = np.array(get_sphere_positions(q, robot))
            for i, (p, r) in enumerate(zip(pos, sphere_radii)):
                ghost_handles.append(server.scene.add_icosphere(f"/{prefix}_sis/{i}", float(r), color=color))
                ghost_handles.append(server.scene.add_icosphere(f"/{prefix}_sir/{i}", float(r), color=color))
        
        # Update stats
        sis_rate = np.mean(results['sis_success']) * 100
        sir_rate = np.mean(results['sir_success']) * 100
        label_sis.text = f"SIS: {sis_rate:.0f}%"
        label_sir.text = f"SIR: {sir_rate:.0f}%"
        problem_text.value = f"{prob_idx + 1}/{n}"
        sis_rate_text.value = f"{sis_rate:.0f}%"
        sir_rate_text.value = f"{sir_rate:.0f}%"
        
        # Animate
        geom_sis = np.array(get_spheres_batch(path_sis))
        geom_sir = np.array(get_spheres_batch(path_sir))
        
        for _ in range(2):
            for frame in range(len(path_sis)):
                c_sis = (244, 67, 54) if col_sis[frame] else (76, 175, 80)
                c_sir = (244, 67, 54) if col_sir[frame] else (76, 175, 80)
                
                for s, p in zip(sis_spheres, geom_sis[frame]):
                    s.position = tuple(p + offset_sis); s.color = c_sis
                for s, p in zip(sir_spheres, geom_sir[frame]):
                    s.position = tuple(p + offset_sir); s.color = c_sir
                
                time.sleep(0.02)
    
    print(f"\n{'='*60}")
    print(f"FINAL RESULTS")
    print(f"{'='*60}")
    print(f"  SIS: {np.mean(results['sis_success'])*100:.1f}% success, {np.mean(results['sis_collision'])*100:.1f}% collision")
    print(f"  SIR: {np.mean(results['sir_success'])*100:.1f}% success, {np.mean(results['sir_collision'])*100:.1f}% collision")
    
    print("\nVisualization running. Press Ctrl+C to exit.")
    while True:
        time.sleep(1)


run_benchmark = run_benchmark_fast


# ============================================================================
# Test / Demo
# ============================================================================

def test_robot(urdf_path='models/panda_spherized.urdf'):
    """Test robot loading and FK."""
    print("Testing robot model...")
    
    robot = load_robot(urdf_path)
    
    q_zero = jnp.zeros(7)
    transforms = forward_kinematics(q_zero, robot)
    print(f"  Zero config EE pos: {transforms[-1, :3, 3]}")
    
    spheres = get_sphere_positions(q_zero, robot)
    print(f"  Sphere positions shape: {spheres.shape}")
    print(f"  First sphere: {spheres[0]}")
    print(f"  Last sphere:  {spheres[-1]}")
    
    # Timing
    fk_jit = jax.jit(lambda q: get_sphere_positions(q, robot))
    _ = fk_jit(q_zero)
    
    t0 = time.time()
    for _ in range(1000):
        _ = fk_jit(q_zero)
    jax.block_until_ready(_)
    print(f"  FK time: {(time.time() - t0)}ms for 1000 calls")


def test_collision(urdf_path='models/panda_spherized.urdf'):
    """Test collision checking."""
    print("\nTesting collision...")
    
    robot = load_robot(urdf_path)
    
    env = Environment(
        sphere_centers=jnp.zeros((0, 3)), sphere_radii=jnp.zeros((0,)),
        box_centers=jnp.array([[0.5, 0.0, 0.5]]),
        box_half_extents=jnp.array([[0.1, 0.3, 0.1]]),
        box_rotations=jnp.eye(3)[None],
        cylinder_centers=jnp.zeros((0, 3)), cylinder_radii=jnp.zeros((0,)),
        cylinder_half_heights=jnp.zeros((0,)), cylinder_rotations=jnp.zeros((0, 3, 3)),
    )
    
    q_zero = jnp.zeros(7)
    sdf = robot_sdf(q_zero, env, robot)
    print(f"  SDF at zero config: {sdf:.3f}")
    
    sdf_jit = jax.jit(lambda q: robot_sdf(q, env, robot))
    _ = sdf_jit(q_zero)
    
    t0 = time.time()
    for _ in range(1000):
        _ = sdf_jit(q_zero)
    jax.block_until_ready(_)
    print(f"  SDF time: {(time.time() - t0)}ms for 1000 calls")


if __name__ == "__main__":
    import sys
    
    args = sys.argv[1:]
    
    if not args or args[0] == "test":
        urdf = args[1] if len(args) > 1 else 'models/panda_spherized.urdf'
        test_robot(urdf)
        test_collision(urdf)
    elif args[0] == "viser":
        dataset = args[1] if len(args) > 1 else "./models/problems/table_pick_panda"
        urdf = args[2] if len(args) > 2 else 'models/panda_spherized.urdf'
        n = int(args[3]) if len(args) > 3 else None
        run_benchmark_viser(dataset, urdf, n)
    elif args[0] == "fast":
        dataset = args[1] if len(args) > 1 else "./models/problems/table_pick_panda"
        urdf = args[2] if len(args) > 2 else 'models/panda_spherized.urdf'
        n = int(args[3]) if len(args) > 3 else None
        run_benchmark_fast(dataset, urdf, n)
    else:
        dataset = args[0]
        urdf = args[1] if len(args) > 1 else 'models/panda_spherized.urdf'
        n = int(args[2]) if len(args) > 2 else None
        run_benchmark_fast(dataset, urdf, n)
    
    print("\nUsage:")
    print("  python benchmark_mbm_simple.py test [urdf]")
    print("  python benchmark_mbm_simple.py fast <dataset> [urdf] [n]")
    print("  python benchmark_mbm_simple.py viser <dataset> [urdf] [n]")
    print("  python benchmark_mbm_simple.py <dataset> [urdf] [n]")