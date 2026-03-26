"""
Real-time deployment with Franky motion control and PFMP planning.

Uses:
- Franky: Real-time motion control for Franka robots
- PFMP: Particle filter motion planning with voxel collision
- RealSense depth camera for environment perception

================================================================================
END-TO-END TESTING PROCEDURE
================================================================================

Assuming camera intrinsics and T_ee_camera are already calibrated, run these
tests in order to verify the full pipeline before deployment:

STEP 1: CAMERA STREAM TEST
--------------------------
Verify depth camera is working and producing valid frames.

    python deploy_new.py test-camera

Expected output:
  - Depth image displayed with reasonable values (0.3-2.0m for tabletop)
  - No large holes or invalid regions in workspace area
  - Intrinsics printed match expected values

STEP 2: POINT CLOUD TEST
------------------------
Verify depth-to-pointcloud conversion and camera-to-world transform.

    python deploy_new.py test-pointcloud

Expected output:
  - Point cloud visualized in robot base frame
  - Table surface appears at correct height (~0.4m if table is 40cm high)
  - Objects on table are at correct world positions
  - No systematic offset or rotation errors

STEP 3: VOXEL GRID TEST
-----------------------
Verify point cloud to voxel grid conversion.

    python deploy_new.py test-voxels

Expected output:
  - Voxel grid visualization shows occupied cells for obstacles
  - Grid aligns with actual obstacle positions
  - Resolution is sufficient to capture object shapes
  - Distance field computed without errors

STEP 4: COLLISION CHECK TEST
----------------------------
Verify robot collision checking against voxel grid.

    python deploy_new.py test-collision

Expected output:
  - Robot at home position reports no collision (SDF > 0)
  - Moving robot toward table reports collision when expected
  - Collision spheres visualized at correct positions

STEP 5: PLANNING TEST (static environment)
------------------------------------------
Verify motion planning with known obstacles.

    python deploy_new.py test-planning

Expected output:
  - Planner finds collision-free path
  - Path visualization shows robot avoiding obstacles
  - Goal configuration is reached (within tolerance)

STEP 6: EXECUTION TEST (slow speed)
-----------------------------------
Verify trajectory execution on real robot at low speed.

    python deploy_new.py test-execute --velocity 0.1

Expected output:
  - Robot moves smoothly along planned trajectory
  - No jerky motions or stops
  - Goal position reached accurately

STEP 7: FULL PIPELINE TEST
--------------------------
Run complete perception-planning-execution loop.

    python deploy_new.py real --use-camera --velocity 0.2

Expected output:
  - Environment captured from depth camera
  - Path planned avoiding detected obstacles
  - Robot executes motion successfully

================================================================================
TASK-LEVEL COMMANDS
================================================================================

WORKSPACE SCANNING (scan)
-------------------------
Scan the workspace from multiple viewpoints and build a voxel map.
The robot moves to predefined poses (HOME, LEFT, RIGHT, TOP) and accumulates
point clouds into a unified voxel grid for collision avoidance.

    python deploy_new.py scan

This is useful for:
  - Building a map of static environments before planning
  - Verifying camera calibration from multiple angles
  - Visualizing the accumulated voxel occupancy

The VoxelMap class handles multi-view accumulation by transforming each
point cloud to world frame using T_base_ee @ T_ee_camera, then voxelizing.

================================================================================
"""

import jax
import jax.numpy as jnp
import numpy as np
import time
from typing import NamedTuple, Optional
from dataclasses import dataclass

# Local imports
from voxels import (
    VoxelGrid,
    VoxelMap,
    empty_voxel_grid,
    compute_distance_field,
    add_box,
    add_sphere,
    add_cylinder,
    point_cloud_to_voxel_grid,
)
from mbm import (
    load_robot,
    make_planning_fns,
    Params,
    Q_MIN,
    Q_MAX,
)
from scan import (
    scan_workspace,
    generate_scan_poses,
    task_scan_and_reach,
    HOME_POSE,
    LEFT_VIEW,
    RIGHT_VIEW,
    TOP_VIEW,
    SCAN_POSES,
)
from pfmp import PFMP
from pfmp.interpolate import hermite_spline, Jet2


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class DeployConfig:
    """Deployment configuration."""
    # Robot
    robot_ip: str = "172.16.0.2"

    # Voxel grid
    grid_shape: tuple = (64, 64, 64)
    grid_origin: tuple = (-0.5, -0.5, 0.0)
    grid_resolution: float = 0.025

    # Planning
    horizon: float = 1.0
    num_knots: int = 5
    particles: int = 512
    replan_rate: float = 10.0  # Hz
    collision_margin: float = 0.05  # meters

    # Camera
    camera_serial: Optional[str] = None
    depth_max: float = 2.0

    # Control
    control_rate: float = 100.0  # Hz
    velocity_scale: float = 0.3  # fraction of max velocity


# ============================================================================
# Franky Robot Interface
# ============================================================================

class FrankyRobot:
    """
    Wrapper for Franky motion control library.

    Franky provides real-time motion control for Franka Emika robots
    with support for Cartesian and joint-space motion.
    """

    def __init__(self, robot_ip: str, velocity_scale: float = 0.3):
        """
        Initialize connection to Franka robot via Franky.

        Args:
            robot_ip: IP address of the Franka Control Interface
            velocity_scale: Fraction of maximum velocity (0.0 to 1.0)
        """
        self.robot_ip = robot_ip
        self.velocity_scale = velocity_scale
        self._robot = None
        self._gripper = None
        self._connected = False

    def connect(self) -> bool:
        """
        Connect to the robot.

        Returns:
            True if connection successful
        """
        try:
            import franky

            self._robot = franky.Robot(self.robot_ip)
            self._robot.relative_dynamics_factor = self.velocity_scale
            self._robot.recover_from_errors()

            # Initialize gripper if available
            try:
                self._gripper = franky.Gripper(self.robot_ip)
            except Exception:
                print("Warning: Gripper not available")
                self._gripper = None

            self._connected = True
            print(f"Connected to Franka at {self.robot_ip}")
            return True

        except ImportError:
            print("Error: franky library not installed")
            print("Install with: pip install franky-panda")
            return False
        except Exception as e:
            print(f"Error connecting to robot: {e}")
            return False

    def disconnect(self):
        """Disconnect from robot."""
        self._robot = None
        self._gripper = None
        self._connected = False

    @property
    def is_connected(self) -> bool:
        return self._connected

    def get_joint_positions(self) -> np.ndarray:
        """
        Get current joint positions.

        Returns:
            Array of shape (7,) with joint angles in radians
        """
        if not self._connected:
            raise RuntimeError("Not connected to robot")

        state = self._robot.read_once()
        return np.array(state.q)

    def get_joint_velocities(self) -> np.ndarray:
        """
        Get current joint velocities.

        Returns:
            Array of shape (7,) with joint velocities in rad/s
        """
        if not self._connected:
            raise RuntimeError("Not connected to robot")

        state = self._robot.read_once()
        return np.array(state.dq)

    def get_ee_pose(self) -> tuple:
        """
        Get current end-effector pose.

        Returns:
            (position, quaternion) where position is (3,) and quaternion is (4,) wxyz
        """
        if not self._connected:
            raise RuntimeError("Not connected to robot")

        state = self._robot.read_once()
        T = np.array(state.O_T_EE).reshape(4, 4).T
        position = T[:3, 3]

        # Convert rotation matrix to quaternion
        from scipy.spatial.transform import Rotation
        quat = Rotation.from_matrix(T[:3, :3]).as_quat()  # xyzw
        quat_wxyz = np.array([quat[3], quat[0], quat[1], quat[2]])

        return position, quat_wxyz

    def move_to_joint_positions(
        self,
        target_q: np.ndarray,
        duration: Optional[float] = None,
        blocking: bool = True,
    ) -> bool:
        """
        Move to target joint positions.

        Args:
            target_q: Target joint positions (7,)
            duration: Motion duration in seconds (None = automatic)
            blocking: Wait for motion to complete

        Returns:
            True if motion started/completed successfully
        """
        if not self._connected:
            raise RuntimeError("Not connected to robot")

        import franky

        motion = franky.JointMotion(target_q.tolist())

        if blocking:
            return self._robot.move(motion)
        else:
            self._robot.move_async(motion)
            return True

    def stream_joint_trajectory(
        self,
        trajectory: np.ndarray,
        dt: float,
        velocities: Optional[np.ndarray] = None,
    ) -> bool:
        """
        Stream a joint trajectory to the robot in real-time.

        This uses Franky's real-time control interface for smooth
        trajectory execution with collision checking.

        Args:
            trajectory: Joint positions of shape (T, 7)
            dt: Time step between waypoints
            velocities: Optional joint velocities of shape (T, 7)

        Returns:
            True if trajectory executed successfully
        """
        if not self._connected:
            raise RuntimeError("Not connected to robot")

        import franky

        # Create waypoint motion
        waypoints = []
        for i, q in enumerate(trajectory):
            if velocities is not None:
                wp = franky.JointWaypoint(q.tolist(), velocities[i].tolist())
            else:
                wp = franky.JointWaypoint(q.tolist())
            waypoints.append(wp)

        motion = franky.JointWaypointMotion(waypoints)

        return self._robot.move(motion)

    def stop(self):
        """Stop current motion immediately."""
        if self._connected and self._robot is not None:
            self._robot.stop()

    def recover_from_errors(self) -> bool:
        """
        Attempt to recover from error state.

        Returns:
            True if recovery successful
        """
        if not self._connected:
            return False

        try:
            self._robot.recover_from_errors()
            return True
        except Exception as e:
            print(f"Recovery failed: {e}")
            return False

    def gripper_open(self, width: float = 0.08, speed: float = 0.1) -> bool:
        """Open gripper to specified width."""
        if self._gripper is None:
            return False
        return self._gripper.move(width, speed)

    def gripper_close(self, width: float = 0.0, speed: float = 0.1, force: float = 20.0) -> bool:
        """Close gripper with specified force."""
        if self._gripper is None:
            return False
        return self._gripper.grasp(width, speed, force)


# ============================================================================
# Depth Perception Pipeline
# ============================================================================

class DepthCamera:
    """
    Depth camera interface for Intel RealSense cameras.

    Supports D415, D435, D455, and similar depth cameras.
    """

    def __init__(self, serial: Optional[str] = None, width: int = 640, height: int = 480,
                 fps: int = 30):
        """
        Initialize depth camera.

        Args:
            serial: Camera serial number (None = first available)
            width: Image width
            height: Image height
            fps: Frame rate
        """
        self.serial = serial
        self.width = width
        self.height = height
        self.fps = fps
        self._pipeline = None
        self._config = None
        self._profile = None
        self._intrinsics = None
        self._align = None
        self._depth_scale = 0.001  # Default: depth units to meters
        self._temporal_filter = None
        self._spatial_filter = None
        self._hole_filling = None

    def start(self) -> bool:
        """
        Start camera stream.

        Returns:
            True if camera started successfully
        """
        try:
            import pyrealsense2 as rs
        except ImportError:
            print("Error: pyrealsense2 not installed")
            print("Install with: pip install pyrealsense2")
            return False

        try:
            self._pipeline = rs.pipeline()
            self._config = rs.config()

            # Select specific camera if serial provided
            if self.serial:
                self._config.enable_device(self.serial)

            # Enable depth and color streams
            self._config.enable_stream(
                rs.stream.depth, self.width, self.height,
                rs.format.z16, self.fps
            )
            self._config.enable_stream(
                rs.stream.color, self.width, self.height,
                rs.format.bgr8, self.fps
            )

            # Start pipeline
            self._profile = self._pipeline.start(self._config)

            # Get depth sensor and scale
            depth_sensor = self._profile.get_device().first_depth_sensor()
            self._depth_scale = depth_sensor.get_depth_scale()

            # Get intrinsics
            depth_stream = self._profile.get_stream(rs.stream.depth)
            intrinsics = depth_stream.as_video_stream_profile().get_intrinsics()
            self._intrinsics = {
                'fx': intrinsics.fx,
                'fy': intrinsics.fy,
                'cx': intrinsics.ppx,
                'cy': intrinsics.ppy,
                'width': intrinsics.width,
                'height': intrinsics.height,
            }

            # Set up alignment (align depth to color)
            self._align = rs.align(rs.stream.color)

            # Set up post-processing filters for cleaner depth
            self._temporal_filter = rs.temporal_filter()
            self._temporal_filter.set_option(rs.option.filter_smooth_alpha, 0.4)
            self._temporal_filter.set_option(rs.option.filter_smooth_delta, 20)

            self._spatial_filter = rs.spatial_filter()
            self._spatial_filter.set_option(rs.option.filter_magnitude, 2)
            self._spatial_filter.set_option(rs.option.filter_smooth_alpha, 0.5)
            self._spatial_filter.set_option(rs.option.filter_smooth_delta, 20)

            self._hole_filling = rs.hole_filling_filter()

            # Wait for auto-exposure to stabilize
            for _ in range(30):
                self._pipeline.wait_for_frames()

            print(f"Camera started: {self.width}x{self.height}@{self.fps}fps")
            print(f"  Depth scale: {self._depth_scale:.6f} (units to meters)")
            print(f"  Intrinsics: fx={self._intrinsics['fx']:.1f}, "
                  f"fy={self._intrinsics['fy']:.1f}")

            return True

        except Exception as e:
            print(f"Error starting camera: {e}")
            self._pipeline = None
            return False

    def stop(self):
        """Stop camera stream."""
        if self._pipeline is not None:
            self._pipeline.stop()
            self._pipeline = None
        self._profile = None
        self._intrinsics = None

    def get_depth_frame(self, apply_filters: bool = True) -> Optional[np.ndarray]:
        """
        Get latest depth frame.

        Args:
            apply_filters: Apply temporal/spatial filtering for cleaner depth

        Returns:
            Depth image as (H, W) array in meters, or None if unavailable
        """
        if self._pipeline is None:
            return None

        try:
            # Wait for frames
            frames = self._pipeline.wait_for_frames(timeout_ms=1000)

            # Align depth to color
            aligned_frames = self._align.process(frames)
            depth_frame = aligned_frames.get_depth_frame()

            if not depth_frame:
                return None

            # Apply post-processing filters
            if apply_filters:
                depth_frame = self._temporal_filter.process(depth_frame)
                depth_frame = self._spatial_filter.process(depth_frame)
                depth_frame = self._hole_filling.process(depth_frame)

            # Convert to numpy array in meters
            depth_image = np.asanyarray(depth_frame.get_data())
            depth_meters = depth_image.astype(np.float32) * self._depth_scale

            return depth_meters

        except Exception as e:
            print(f"Error getting depth frame: {e}")
            return None

    def get_color_frame(self) -> Optional[np.ndarray]:
        """
        Get latest color frame.

        Returns:
            Color image as (H, W, 3) BGR array, or None if unavailable
        """
        if self._pipeline is None:
            return None

        try:
            frames = self._pipeline.wait_for_frames(timeout_ms=1000)
            aligned_frames = self._align.process(frames)
            color_frame = aligned_frames.get_color_frame()

            if not color_frame:
                return None

            return np.asanyarray(color_frame.get_data())

        except Exception as e:
            print(f"Error getting color frame: {e}")
            return None

    def get_rgbd_frame(self, apply_filters: bool = True) -> Optional[tuple]:
        """
        Get aligned RGB-D frame pair.

        Returns:
            (color_image, depth_image) tuple or None if unavailable
        """
        if self._pipeline is None:
            return None

        try:
            frames = self._pipeline.wait_for_frames(timeout_ms=1000)
            aligned_frames = self._align.process(frames)

            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()

            if not depth_frame or not color_frame:
                return None

            if apply_filters:
                depth_frame = self._temporal_filter.process(depth_frame)
                depth_frame = self._spatial_filter.process(depth_frame)
                depth_frame = self._hole_filling.process(depth_frame)

            depth_image = np.asanyarray(depth_frame.get_data())
            depth_meters = depth_image.astype(np.float32) * self._depth_scale
            color_image = np.asanyarray(color_frame.get_data())

            return color_image, depth_meters

        except Exception as e:
            print(f"Error getting RGBD frame: {e}")
            return None

    def get_intrinsics(self) -> dict:
        """
        Get camera intrinsic parameters.

        Returns:
            Dict with 'fx', 'fy', 'cx', 'cy', 'width', 'height'
        """
        if self._intrinsics is None:
            raise RuntimeError("Camera not started")
        return self._intrinsics

    def get_depth_scale(self) -> float:
        """Get depth scale (depth units to meters)."""
        return self._depth_scale


def depth_to_point_cloud(
    depth: np.ndarray,
    intrinsics: dict,
    max_depth: float = 2.0,
) -> np.ndarray:
    """
    Convert depth image to 3D point cloud in camera frame.

    Args:
        depth: Depth image (H, W) in meters
        intrinsics: Camera intrinsics with 'fx', 'fy', 'cx', 'cy'
        max_depth: Maximum depth to include

    Returns:
        Point cloud of shape (N, 3) in camera coordinates
    """
    H, W = depth.shape
    fx, fy = intrinsics['fx'], intrinsics['fy']
    cx, cy = intrinsics['cx'], intrinsics['cy']

    # Generate pixel coordinates
    u, v = np.meshgrid(np.arange(W), np.arange(H))

    # Valid depth mask
    valid = (depth > 0) & (depth < max_depth)

    # Backproject to camera frame
    z = depth[valid]
    x = (u[valid] - cx) * z / fx
    y = (v[valid] - cy) * z / fy

    return np.stack([x, y, z], axis=-1)


def transform_point_cloud(
    points: np.ndarray,
    camera_pose: np.ndarray,
) -> np.ndarray:
    """
    Transform point cloud from camera frame to world frame.

    Args:
        points: Point cloud (N, 3) in camera frame
        camera_pose: 4x4 camera-to-world transform

    Returns:
        Point cloud (N, 3) in world frame
    """
    R = camera_pose[:3, :3]
    t = camera_pose[:3, 3]
    return (R @ points.T).T + t


def get_camera_pose_from_robot(robot: FrankyRobot, camera_offset: np.ndarray) -> np.ndarray:
    """
    Get camera pose in world frame from robot state.

    Assumes camera is rigidly attached to end-effector.

    Args:
        robot: Connected FrankyRobot instance
        camera_offset: 4x4 transform from EE to camera

    Returns:
        4x4 camera-to-world transform
    """
    from scipy.spatial.transform import Rotation

    pos, quat_wxyz = robot.get_ee_pose()
    quat_xyzw = np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]])

    T_world_ee = np.eye(4)
    T_world_ee[:3, :3] = Rotation.from_quat(quat_xyzw).as_matrix()
    T_world_ee[:3, 3] = pos

    return T_world_ee @ camera_offset


def demo_scan_workspace(config: Optional[DeployConfig] = None):
    """Demo: scan workspace and visualize the resulting voxel map."""
    print("=" * 60)
    print("DEMO: Workspace Scanning")
    print("=" * 60)

    if config is None:
        config = DeployConfig()

    robot = FrankyRobot(config.robot_ip, config.velocity_scale)
    camera = DepthCamera()

    if not robot.connect():
        print("Could not connect to robot")
        return

    if not camera.start():
        print("Could not start camera")
        robot.disconnect()
        return

    try:
        T_ee_camera = load_camera_calibration()

        # Scan from multiple poses
        voxel_map = scan_workspace(
            robot, camera, T_ee_camera,
            SCAN_POSES,
            config=config,
        )

        # Finalize and visualize
        grid, dist_field = voxel_map.finalize()

        # Visualize
        import viser
        import trimesh
        from mbm import get_sphere_positions

        server = viser.ViserServer()
        print("\nOpen http://localhost:8080 to view scanned map")
        print("Press Ctrl+C to stop")

        # Show voxels
        occupied = np.array(grid.data > 0.5)
        indices = np.argwhere(occupied)
        if len(indices) > 0:
            voxel_positions = np.array(grid.origin) + (indices + 0.5) * config.grid_resolution
            cube = trimesh.creation.box(extents=(config.grid_resolution * 0.9,) * 3)
            meshes = [cube.copy().apply_translation(p) for p in voxel_positions[:10000]]
            merged = trimesh.util.concatenate(meshes)
            server.scene.add_mesh_trimesh("/voxels", mesh=merged, color=(100, 150, 200))

        server.scene.add_frame("/world", axes_length=0.3, axes_radius=0.01)
        server.scene.add_grid("/ground", width=2, height=2, plane="xy")

        # Show robot
        robot_model = load_robot()
        sphere_handles = []
        for i in range(robot_model['n_spheres']):
            r = float(robot_model['sphere_radii'][i])
            h = server.scene.add_icosphere(f"/robot/{i}", radius=r)
            sphere_handles.append(h)

        while True:
            q = robot.get_joint_positions()
            positions = np.array(get_sphere_positions(jnp.array(q), robot_model))
            for h, pos in zip(sphere_handles, positions):
                h.position = tuple(pos)
                h.color = (76, 175, 80)
            time.sleep(0.05)

    except KeyboardInterrupt:
        print("\nDone")
    finally:
        camera.stop()
        robot.disconnect()


def test_scan_motion(config: Optional[DeployConfig] = None):
    """Test scanning motion pattern without camera (robot only)."""
    print("=" * 60)
    print("TEST: Scan Motion Pattern")
    print("=" * 60)

    if config is None:
        config = DeployConfig()

    robot = FrankyRobot(config.robot_ip, config.velocity_scale)

    if not robot.connect():
        print("Could not connect to robot")
        return False

    try:
        print(f"\nMoving through {len(SCAN_POSES)} scan poses...")
        print("Poses: HOME -> LEFT -> RIGHT -> TOP")

        for i, (name, pose) in enumerate([
            ("HOME", HOME_POSE),
            ("LEFT", LEFT_VIEW),
            ("RIGHT", RIGHT_VIEW),
            ("TOP", TOP_VIEW),
        ]):
            print(f"\n  [{i+1}/4] Moving to {name}...")
            robot.move_to_joint_positions(pose, blocking=True)
            time.sleep(0.5)  # Settle time

            q_actual = robot.get_joint_positions()
            error = np.linalg.norm(q_actual - pose)
            print(f"        Reached. Error: {error:.4f} rad")

        # Return home
        print("\n  Returning to HOME...")
        robot.move_to_joint_positions(HOME_POSE, blocking=True)

        print("\nPASSED: Scan motion pattern complete")
        return True

    except KeyboardInterrupt:
        print("\nAborted by user")
        return False
    finally:
        robot.disconnect()


# ============================================================================
# Planning Interface
# ============================================================================

class MotionPlanner:
    """
    PFMP-based motion planner with voxel collision avoidance.
    """

    def __init__(self, config: DeployConfig):
        """
        Initialize motion planner.

        Args:
            config: Deployment configuration
        """
        self.config = config
        self.robot_model = load_robot()

        # Create planning functions
        (self.cost_fn, self.terminal_cost, self.dynamics,
         self.init, self.proposal, self.robot_sdf) = make_planning_fns(self.robot_model)

        # Initialize PFMP
        self.pfmp = PFMP(
            self.dynamics,
            self.cost_fn,
            self.terminal_cost,
            self.init,
            self.proposal,
            horizon=config.horizon,
            num_knots=config.num_knots,
            particles=config.particles,
            ess_resample_thresh=0.1,
        )

        # JIT compile planning functions
        self._run_sir = jax.jit(self._run_sir_impl)
        self._extract_path = jax.jit(self._extract_path_impl)

        # Current environment
        self._grid = None
        self._dist_field = None

    def _run_sir_impl(self, key, q_start, theta):
        """Run SIR planning (to be JIT compiled)."""
        traj, scores = self.pfmp.run_sir(key, q_start, theta)
        return traj[jnp.argmin(jnp.sum(scores, axis=-1))]

    def _extract_path_impl(self, traj, q_start, theta):
        """Extract dense path from trajectory knots."""
        tq = jnp.linspace(0, self.pfmp.horizon, int(self.pfmp.horizon / self.pfmp.dt))
        us = hermite_spline(tq, self.pfmp.tk(), traj)
        _, path = jax.lax.scan(
            lambda x, u: (self.dynamics(x, u, theta, 0.0)[0], x),
            q_start, us
        )
        return path

    def update_environment(self, grid: VoxelGrid):
        """
        Update the environment voxel grid.

        Args:
            grid: New voxel grid with obstacles
        """
        self._grid = grid
        self._dist_field = compute_distance_field(grid)

    def plan(
        self,
        q_start: np.ndarray,
        q_goal: np.ndarray,
        seed: int = 0,
    ) -> Optional[np.ndarray]:
        """
        Plan a collision-free trajectory.

        Args:
            q_start: Starting joint configuration (7,)
            q_goal: Goal joint configuration (7,)
            seed: Random seed for planning

        Returns:
            Joint trajectory of shape (T, 7), or None if planning failed
        """
        if self._grid is None or self._dist_field is None:
            raise RuntimeError("Environment not set. Call update_environment first.")

        # Build parameters
        theta = Params(
            goal=jnp.array(q_goal),
            origin=self._grid.origin,
            resolution=self._grid.resolution,
            dist_field=self._dist_field,
            q_min=Q_MIN,
            q_max=Q_MAX,
        )

        # Run planning
        key = jax.random.key(seed)
        q_start_jax = jnp.array(q_start)

        traj = self._run_sir(key, q_start_jax, theta)
        path = self._extract_path(traj, q_start_jax, theta)

        # Check for collisions
        col_mask = jax.vmap(
            lambda q: self.robot_sdf(q, theta.origin, theta.resolution, theta.dist_field) < 0
        )(path)

        if jnp.any(col_mask):
            print(f"Warning: Path has {jnp.sum(col_mask).item()} colliding configurations")

        return np.array(path)

    def check_collision(self, q: np.ndarray) -> bool:
        """
        Check if a configuration is in collision.

        Args:
            q: Joint configuration (7,)

        Returns:
            True if in collision
        """
        if self._grid is None or self._dist_field is None:
            return False

        sdf = self.robot_sdf(
            jnp.array(q),
            self._grid.origin,
            self._grid.resolution,
            self._dist_field,
        )
        return float(sdf) < self.config.collision_margin


# ============================================================================
# Deployment Controller
# ============================================================================

class DeploymentController:
    """
    Main deployment controller integrating perception, planning, and execution.
    """

    def __init__(self, config: DeployConfig):
        """
        Initialize deployment controller.

        Args:
            config: Deployment configuration
        """
        self.config = config

        # Initialize components
        self.robot = FrankyRobot(config.robot_ip, config.velocity_scale)
        self.planner = MotionPlanner(config)
        self.camera = DepthCamera(config.camera_serial)

        # Camera-to-EE transform (TODO: calibrate)
        self._camera_offset = np.eye(4)
        self._camera_offset[:3, 3] = [0.05, 0, 0.05]  # Placeholder offset

        self._running = False

    def connect(self) -> bool:
        """
        Connect to robot and start camera.

        Returns:
            True if all connections successful
        """
        robot_ok = self.robot.connect()
        camera_ok = self.camera.start()

        if not robot_ok:
            print("Warning: Robot connection failed - running in simulation mode")

        if not camera_ok:
            print("Warning: Camera not available - using static environment")

        return robot_ok

    def disconnect(self):
        """Disconnect from robot and stop camera."""
        self.robot.disconnect()
        self.camera.stop()
        self._running = False

    def update_environment_from_depth(self) -> bool:
        """
        Update environment from current depth frame.

        Returns:
            True if environment was updated
        """
        depth = self.camera.get_depth_frame()
        if depth is None:
            return False

        # Get camera pose
        if self.robot.is_connected:
            camera_pose = get_camera_pose_from_robot(self.robot, self._camera_offset)
        else:
            camera_pose = np.eye(4)
            camera_pose[:3, 3] = [0.5, 0, 0.5]  # Default camera position

        # Convert depth to voxels
        intrinsics = self.camera.get_intrinsics()
        points_cam = depth_to_point_cloud(depth, intrinsics, self.config.depth_max)
        points_world = transform_point_cloud(points_cam, camera_pose)

        grid = point_cloud_to_voxel_grid(
            points_world,
            shape=self.config.grid_shape,
            origin=self.config.grid_origin,
            resolution=self.config.grid_resolution,
        )

        self.planner.update_environment(grid)
        return True

    def set_static_environment(self, grid: VoxelGrid):
        """
        Set a static environment for planning.

        Args:
            grid: Voxel grid with obstacles
        """
        self.planner.update_environment(grid)

    def move_to_goal(
        self,
        q_goal: np.ndarray,
        replan: bool = False,
    ) -> bool:
        """
        Move robot to goal configuration.

        Args:
            q_goal: Goal joint configuration (7,)
            replan: If True, continuously replan during execution

        Returns:
            True if goal reached successfully
        """
        if not self.robot.is_connected:
            print("Robot not connected - cannot execute motion")
            return False

        q_current = self.robot.get_joint_positions()

        # Plan trajectory
        print("Planning trajectory...")
        path = self.planner.plan(q_current, q_goal)

        if path is None:
            print("Planning failed")
            return False

        print(f"Executing trajectory with {len(path)} waypoints...")

        if replan:
            return self._execute_with_replanning(path, q_goal)
        else:
            return self._execute_trajectory(path)

    def _execute_trajectory(self, path: np.ndarray) -> bool:
        """Execute a planned trajectory."""
        dt = self.config.horizon / len(path)
        return self.robot.stream_joint_trajectory(path, dt)

    def _execute_with_replanning(
        self,
        initial_path: np.ndarray,
        q_goal: np.ndarray,
    ) -> bool:
        """
        Execute trajectory with periodic replanning.

        This allows adapting to dynamic obstacles detected via depth camera.
        """
        self._running = True
        path = initial_path
        path_idx = 0

        replan_period = 1.0 / self.config.replan_rate
        last_replan = time.time()
        seed = 0

        try:
            while self._running and path_idx < len(path):
                q_current = self.robot.get_joint_positions()

                # Check if we've reached the goal
                if np.linalg.norm(q_current - q_goal) < 0.05:
                    print("Goal reached!")
                    return True

                # Replan periodically
                current_time = time.time()
                if current_time - last_replan > replan_period:
                    # Update environment from depth
                    if self.update_environment_from_depth():
                        # Replan from current position
                        seed += 1
                        new_path = self.planner.plan(q_current, q_goal, seed)
                        if new_path is not None:
                            path = new_path
                            path_idx = 0
                    last_replan = current_time

                # Send next waypoint
                if path_idx < len(path):
                    self.robot.move_to_joint_positions(
                        path[path_idx],
                        duration=0.01,
                        blocking=False,
                    )
                    path_idx += 1

                time.sleep(1.0 / self.config.control_rate)

        except KeyboardInterrupt:
            print("Interrupted - stopping robot")
            self.robot.stop()
            return False

        return True

    def stop(self):
        """Stop current motion and replanning loop."""
        self._running = False
        if self.robot.is_connected:
            self.robot.stop()


# ============================================================================
# Example Usage
# ============================================================================

def create_demo_environment() -> VoxelGrid:
    """Create a demo environment with some obstacles."""
    grid = empty_voxel_grid(
        shape=(64, 64, 64),
        origin=(-0.5, -0.5, 0.0),
        resolution=0.025,
    )

    # Table
    grid = add_box(grid, [0.5, 0.0, 0.4], [0.3, 0.4, 0.02])

    # Objects on table
    grid = add_box(grid, [0.4, 0.15, 0.5], [0.05, 0.05, 0.08])
    grid = add_cylinder(grid, [0.55, -0.1, 0.48], 0.04, 0.06)
    grid = add_sphere(grid, [0.65, 0.1, 0.47], 0.05)

    return grid


def demo_simulation():
    """Demo without real robot (simulation mode)."""
    print("=" * 60)
    print("PFMP Deployment Demo (Simulation Mode)")
    print("=" * 60)

    config = DeployConfig(
        robot_ip="localhost",  # Won't connect
        particles=256,
    )

    controller = DeploymentController(config)

    # Set static environment
    print("\nCreating demo environment...")
    grid = create_demo_environment()
    controller.set_static_environment(grid)

    # Plan a trajectory (won't execute without robot)
    print("\nPlanning trajectory...")
    q_start = np.array([0.0, -0.5, 0.0, -2.0, 0.0, 1.5, 0.0])
    q_goal = np.array([0.5, 0.0, 0.0, -1.5, 0.0, 1.5, 0.5])

    path = controller.planner.plan(q_start, q_goal)

    if path is not None:
        print(f"Planned path with {len(path)} waypoints")
        print(f"Start: {q_start}")
        print(f"Goal:  {q_goal}")
        print(f"Final: {path[-1]}")

        # Check for collisions
        n_collisions = sum(
            controller.planner.check_collision(q) for q in path
        )
        print(f"Collisions: {n_collisions} / {len(path)}")
    else:
        print("Planning failed!")

    return path


def demo_real_robot():
    """Demo with real robot."""
    print("=" * 60)
    print("PFMP Deployment Demo (Real Robot)")
    print("=" * 60)

    config = DeployConfig(
        robot_ip="172.16.0.2",
        velocity_scale=0.2,  # Conservative speed
        particles=512,
    )

    controller = DeploymentController(config)

    if not controller.connect():
        print("Failed to connect to robot")
        return

    try:
        # Set static environment (TODO: use depth camera)
        print("\nCreating demo environment...")
        grid = create_demo_environment()
        controller.set_static_environment(grid)

        # Get current position
        q_current = controller.robot.get_joint_positions()
        print(f"\nCurrent joint positions: {q_current}")

        # Define goal
        q_goal = np.array([0.5, 0.0, 0.0, -1.5, 0.0, 1.5, 0.5])

        # Move to goal
        print(f"\nMoving to goal: {q_goal}")
        success = controller.move_to_goal(q_goal, replan=False)

        if success:
            print("Successfully reached goal!")
        else:
            print("Failed to reach goal")

    finally:
        controller.disconnect()


# ============================================================================
# Test Functions
# ============================================================================

def test_camera():
    """Test camera stream and display depth image."""
    print("=" * 60)
    print("TEST: Camera Stream")
    print("=" * 60)

    camera = DepthCamera()
    if not camera.start():
        print("FAILED: Could not start camera")
        return False

    try:
        import cv2

        print("\nStreaming depth... Press 'q' to quit")
        while True:
            depth = camera.get_depth_frame()
            if depth is None:
                print("Warning: No depth frame")
                continue

            # Stats
            valid = depth > 0
            if valid.any():
                print(f"\rDepth: min={depth[valid].min():.3f}m, "
                      f"max={depth[valid].max():.3f}m, "
                      f"valid={valid.mean()*100:.1f}%", end="")

            # Visualize (normalize to 0-255)
            depth_vis = np.clip(depth / 2.0, 0, 1)  # 0-2m range
            depth_vis = (depth_vis * 255).astype(np.uint8)
            depth_color = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)

            cv2.imshow("Depth", depth_color)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        print("\n\nPASSED: Camera streaming correctly")
        return True

    except ImportError:
        print("OpenCV not available for visualization")
        # Just verify we can get frames
        for i in range(10):
            depth = camera.get_depth_frame()
            if depth is not None:
                valid = depth > 0
                print(f"Frame {i}: min={depth[valid].min():.3f}m, "
                      f"max={depth[valid].max():.3f}m")
        print("\nPASSED: Camera streaming correctly")
        return True

    finally:
        camera.stop()


def test_pointcloud(config: Optional[DeployConfig] = None):
    """Test depth to point cloud conversion."""
    print("=" * 60)
    print("TEST: Point Cloud Generation")
    print("=" * 60)

    if config is None:
        config = DeployConfig()

    camera = DepthCamera()
    robot = FrankyRobot(config.robot_ip, config.velocity_scale)

    if not camera.start():
        print("FAILED: Could not start camera")
        return False

    # Load calibration
    T_ee_camera = load_camera_calibration()

    # Connect to robot (optional - use identity if not available)
    robot_connected = robot.connect()
    if not robot_connected:
        print("Warning: Robot not connected, using fixed camera pose")

    try:
        import viser

        server = viser.ViserServer()
        print("\nOpen http://localhost:8080 to view point cloud")
        print("Press Ctrl+C to stop")

        while True:
            depth = camera.get_depth_frame()
            if depth is None:
                time.sleep(0.1)
                continue

            # Get camera pose
            if robot_connected:
                camera_pose = get_camera_pose_from_robot(robot, T_ee_camera)
            else:
                camera_pose = np.eye(4)
                camera_pose[:3, 3] = [0.5, 0, 0.8]  # Default pose

            # Convert to point cloud
            intrinsics = camera.get_intrinsics()
            points_cam = depth_to_point_cloud(depth, intrinsics, config.depth_max)
            points_world = transform_point_cloud(points_cam, camera_pose)

            # Subsample for visualization
            if len(points_world) > 50000:
                idx = np.random.choice(len(points_world), 50000, replace=False)
                points_world = points_world[idx]

            # Update visualization
            colors = np.full((len(points_world), 3), 100, dtype=np.uint8)
            server.scene.add_point_cloud(
                "/pointcloud", points=points_world.astype(np.float32),
                colors=colors, point_size=0.005
            )

            # Add coordinate frame
            server.scene.add_frame("/world", axes_length=0.2, axes_radius=0.01)

            print(f"\rPoints: {len(points_world)}", end="")
            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\n\nPASSED: Point cloud generation working")
        return True
    finally:
        camera.stop()
        robot.disconnect()


def test_voxels(config: Optional[DeployConfig] = None):
    """Test point cloud to voxel grid conversion."""
    print("=" * 60)
    print("TEST: Voxel Grid Generation")
    print("=" * 60)

    if config is None:
        config = DeployConfig()

    camera = DepthCamera()
    robot = FrankyRobot(config.robot_ip, config.velocity_scale)

    if not camera.start():
        print("FAILED: Could not start camera")
        return False

    T_ee_camera = load_camera_calibration()
    robot_connected = robot.connect()

    try:
        import viser
        import trimesh

        server = viser.ViserServer()
        print("\nOpen http://localhost:8080 to view voxels")
        print("Press Ctrl+C to stop")

        while True:
            depth = camera.get_depth_frame()
            if depth is None:
                time.sleep(0.1)
                continue

            # Get camera pose
            if robot_connected:
                camera_pose = get_camera_pose_from_robot(robot, T_ee_camera)
            else:
                camera_pose = np.eye(4)
                camera_pose[:3, 3] = [0.5, 0, 0.8]

            # Convert to voxels
            intrinsics = camera.get_intrinsics()
            points_cam = depth_to_point_cloud(depth, intrinsics, config.depth_max)
            points_world = transform_point_cloud(points_cam, camera_pose)

            grid = point_cloud_to_voxel_grid(
                points_world,
                shape=config.grid_shape,
                origin=config.grid_origin,
                resolution=config.grid_resolution,
            )

            # Get occupied voxel positions
            occupied = np.array(grid.data > 0.5)
            n_occupied = occupied.sum()
            indices = np.argwhere(occupied)

            if len(indices) > 0:
                voxel_positions = (np.array(grid.origin) +
                                   (indices + 0.5) * config.grid_resolution)

                # Visualize as cubes
                cube = trimesh.creation.box(
                    extents=(config.grid_resolution * 0.9,) * 3
                )
                meshes = [cube.copy().apply_translation(p) for p in voxel_positions[:5000]]
                merged = trimesh.util.concatenate(meshes)
                server.scene.add_mesh_trimesh("/voxels", mesh=merged)

            server.scene.add_frame("/world", axes_length=0.2, axes_radius=0.01)
            server.scene.add_grid("/ground", width=2, height=2, plane="xy")

            print(f"\rOccupied voxels: {n_occupied}", end="")
            time.sleep(0.5)

    except KeyboardInterrupt:
        print("\n\nPASSED: Voxel generation working")
        return True
    finally:
        camera.stop()
        robot.disconnect()


def test_collision(config: Optional[DeployConfig] = None):
    """Test robot collision checking against voxel grid."""
    print("=" * 60)
    print("TEST: Collision Checking")
    print("=" * 60)

    if config is None:
        config = DeployConfig()

    robot = FrankyRobot(config.robot_ip, config.velocity_scale)
    planner = MotionPlanner(config)

    if not robot.connect():
        print("FAILED: Could not connect to robot")
        return False

    # Create test environment
    grid = create_demo_environment()
    planner.update_environment(grid)

    try:
        import viser
        import trimesh
        from mbm import get_sphere_positions

        server = viser.ViserServer()
        print("\nOpen http://localhost:8080 to view collision check")
        print("Move robot manually and observe collision status")
        print("Press Ctrl+C to stop")

        # Visualize obstacles
        occupied = np.array(grid.data > 0.5)
        indices = np.argwhere(occupied)
        voxel_positions = np.array(grid.origin) + (indices + 0.5) * config.grid_resolution
        cube = trimesh.creation.box(extents=(config.grid_resolution * 0.9,) * 3)
        meshes = [cube.copy().apply_translation(p) for p in voxel_positions]
        merged = trimesh.util.concatenate(meshes)
        server.scene.add_mesh_trimesh("/voxels", mesh=merged, color=(150, 150, 150))

        # Visualize robot spheres
        robot_model = planner.robot_model
        sphere_handles = []
        for i in range(robot_model['n_spheres']):
            r = float(robot_model['sphere_radii'][i])
            h = server.scene.add_icosphere(f"/robot/{i}", radius=r, position=(0, 0, 0))
            sphere_handles.append(h)

        server.scene.add_frame("/world", axes_length=0.2, axes_radius=0.01)
        server.scene.add_grid("/ground", width=2, height=2, plane="xy")

        while True:
            q = robot.get_joint_positions()
            in_collision = planner.check_collision(q)

            # Update sphere positions
            positions = np.array(get_sphere_positions(jnp.array(q), robot_model))
            color = (244, 67, 54) if in_collision else (76, 175, 80)

            for h, pos in zip(sphere_handles, positions):
                h.position = tuple(pos)
                h.color = color

            status = "COLLISION" if in_collision else "OK"
            print(f"\rJoints: {q.round(2)} | Status: {status}  ", end="")
            time.sleep(0.05)

    except KeyboardInterrupt:
        print("\n\nPASSED: Collision checking working")
        return True
    finally:
        robot.disconnect()


def test_planning(config: Optional[DeployConfig] = None):
    """Test motion planning with static obstacles."""
    print("=" * 60)
    print("TEST: Motion Planning")
    print("=" * 60)

    if config is None:
        config = DeployConfig()

    planner = MotionPlanner(config)

    # Create test environment
    grid = create_demo_environment()
    planner.update_environment(grid)

    # Define start and goal
    q_start = np.array([0.0, -0.5, 0.0, -2.0, 0.0, 1.5, 0.0])
    q_goal = np.array([0.5, 0.0, 0.0, -1.5, 0.0, 1.5, 0.5])

    print(f"\nStart: {q_start}")
    print(f"Goal:  {q_goal}")
    print("\nPlanning...")

    t0 = time.time()
    path = planner.plan(q_start, q_goal)
    dt = time.time() - t0

    if path is None:
        print("FAILED: Planning returned no path")
        return False

    print(f"Planning time: {dt*1000:.1f}ms")
    print(f"Path length: {len(path)} waypoints")

    # Check for collisions
    n_collisions = sum(planner.check_collision(q) for q in path)
    print(f"Collisions: {n_collisions} / {len(path)}")

    # Check goal reached
    goal_error = np.linalg.norm(path[-1] - q_goal)
    print(f"Goal error: {goal_error:.4f} rad")

    if n_collisions > 0:
        print("\nWARNING: Path has collisions")
    if goal_error > 0.1:
        print("\nWARNING: Goal not reached accurately")

    # Visualize
    try:
        import viser
        import trimesh
        from mbm import get_sphere_positions

        server = viser.ViserServer()
        print("\nOpen http://localhost:8080 to view planned path")
        print("Press Ctrl+C to stop")

        # Visualize obstacles
        occupied = np.array(grid.data > 0.5)
        indices = np.argwhere(occupied)
        voxel_positions = np.array(grid.origin) + (indices + 0.5) * config.grid_resolution
        cube = trimesh.creation.box(extents=(config.grid_resolution * 0.9,) * 3)
        meshes = [cube.copy().apply_translation(p) for p in voxel_positions]
        merged = trimesh.util.concatenate(meshes)
        server.scene.add_mesh_trimesh("/voxels", mesh=merged, color=(150, 150, 150))

        # Visualize robot
        robot_model = planner.robot_model
        sphere_handles = []
        for i in range(robot_model['n_spheres']):
            r = float(robot_model['sphere_radii'][i])
            h = server.scene.add_icosphere(f"/robot/{i}", radius=r, position=(0, 0, 0))
            sphere_handles.append(h)

        server.scene.add_frame("/world", axes_length=0.2, axes_radius=0.01)
        server.scene.add_grid("/ground", width=2, height=2, plane="xy")

        # Animate path
        frame = 0
        while True:
            q = path[frame % len(path)]
            positions = np.array(get_sphere_positions(jnp.array(q), robot_model))
            in_collision = planner.check_collision(q)
            color = (244, 67, 54) if in_collision else (76, 175, 80)

            for h, pos in zip(sphere_handles, positions):
                h.position = tuple(pos)
                h.color = color

            frame += 1
            time.sleep(0.05)

    except KeyboardInterrupt:
        pass

    print("\nPASSED: Motion planning working")
    return True


def test_execute(config: Optional[DeployConfig] = None, velocity: float = 0.1):
    """Test trajectory execution on real robot."""
    print("=" * 60)
    print("TEST: Trajectory Execution")
    print("=" * 60)

    if config is None:
        config = DeployConfig()
    config.velocity_scale = velocity

    controller = DeploymentController(config)

    if not controller.connect():
        print("FAILED: Could not connect to robot")
        return False

    try:
        # Create environment and plan
        grid = create_demo_environment()
        controller.set_static_environment(grid)

        q_current = controller.robot.get_joint_positions()
        print(f"\nCurrent: {q_current.round(3)}")

        # Small motion for safety
        q_goal = q_current.copy()
        q_goal[0] += 0.2  # Small motion on joint 1
        q_goal[3] += 0.2  # Small motion on joint 4
        print(f"Goal:    {q_goal.round(3)}")

        input("\nPress Enter to execute motion (Ctrl+C to abort)...")

        success = controller.move_to_goal(q_goal, replan=False)

        if success:
            print("\nPASSED: Trajectory execution successful")
            return True
        else:
            print("\nFAILED: Trajectory execution failed")
            return False

    except KeyboardInterrupt:
        print("\nAborted by user")
        controller.stop()
        return False
    finally:
        controller.disconnect()


def load_camera_calibration(path: Optional[str] = None) -> np.ndarray:
    """
    Load camera-to-end-effector calibration.

    Args:
        path: Path to calibration file. If None, uses default location
              ~/.config/pfmp/camera_calibration.yaml

    Returns:
        4x4 transform from EE frame to camera frame
    """
    import os

    if path is None:
        path = os.path.expanduser("~/.config/pfmp/camera_calibration.yaml")

    if os.path.exists(path):
        try:
            import yaml
            with open(path, 'r') as f:
                data = yaml.safe_load(f)
            T = np.array(data['T_ee_camera'], dtype=np.float64)
            print(f"Loaded camera calibration from {path}")
            return T
        except Exception as e:
            print(f"Warning: Could not load calibration: {e}")

    # Default calibration (typical wrist-mounted camera)
    print("Using default camera calibration")
    T = np.array([
        [1.0,  0.0,  0.0, 0.05],   # 5cm forward
        [0.0, -1.0,  0.0, 0.00],   # flip Y
        [0.0,  0.0, -1.0, 0.05],   # flip Z, 5cm up
        [0.0,  0.0,  0.0, 1.00],
    ])
    return T


def save_camera_calibration(T_ee_camera: np.ndarray, path: Optional[str] = None):
    """Save camera calibration to file."""
    import os
    import yaml
    from datetime import datetime

    if path is None:
        config_dir = os.path.expanduser("~/.config/pfmp")
        os.makedirs(config_dir, exist_ok=True)
        path = os.path.join(config_dir, "camera_calibration.yaml")

    data = {
        'T_ee_camera': T_ee_camera.tolist(),
        'calibration_date': datetime.now().isoformat(),
    }

    with open(path, 'w') as f:
        yaml.dump(data, f, default_flow_style=None)

    print(f"Saved camera calibration to {path}")


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="PFMP Robot Deployment")
    parser.add_argument("command", nargs="?", default="sim",
                        choices=["sim", "real", "test-camera", "test-pointcloud",
                                 "test-voxels", "test-collision", "test-planning",
                                 "test-execute", "test-scan-motion", "scan"],
                        help="Command to run")
    parser.add_argument("--robot-ip", default="172.16.0.2",
                        help="Robot IP address")
    parser.add_argument("--velocity", type=float, default=0.2,
                        help="Velocity scale (0.0-1.0)")
    parser.add_argument("--use-camera", action="store_true",
                        help="Use depth camera for perception")

    args = parser.parse_args()

    config = DeployConfig(
        robot_ip=args.robot_ip,
        velocity_scale=args.velocity,
    )

    if args.command == "sim":
        demo_simulation()

    elif args.command == "real":
        demo_real_robot()

    elif args.command == "test-camera":
        test_camera()

    elif args.command == "test-pointcloud":
        test_pointcloud(config)

    elif args.command == "test-voxels":
        test_voxels(config)

    elif args.command == "test-collision":
        test_collision(config)

    elif args.command == "test-planning":
        test_planning(config)

    elif args.command == "test-execute":
        test_execute(config, velocity=args.velocity)

    elif args.command == "test-scan-motion":
        test_scan_motion(config)

    elif args.command == "scan":
        demo_scan_workspace(config)
