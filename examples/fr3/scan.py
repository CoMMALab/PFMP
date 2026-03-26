"""
Workspace scanning for multi-view voxel map construction.

This module provides functions for scanning a workspace from multiple
robot viewpoints to build a complete voxel map for collision avoidance.
"""

import numpy as np
import time
from typing import Optional, TYPE_CHECKING

from voxels import VoxelMap, VoxelGrid, compute_distance_field

if TYPE_CHECKING:
    from deploy_new import FrankyRobot, DepthCamera, DeployConfig


# ============================================================================
# Standard Joint Configurations
# ============================================================================

# Home position - safe starting pose
HOME_POSE = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])

# Viewpoints for workspace scanning
LEFT_VIEW = np.array([0.5, -0.5, 0.0, -2.0, 0.0, 1.5, 0.785])
RIGHT_VIEW = np.array([-0.5, -0.5, 0.0, -2.0, 0.0, 1.5, 0.785])
TOP_VIEW = np.array([0.0, -0.3, 0.0, -1.8, 0.0, 1.8, 0.785])

# Default scanning poses
SCAN_POSES = [HOME_POSE, LEFT_VIEW, RIGHT_VIEW, TOP_VIEW]


# ============================================================================
# Workspace Scanning
# ============================================================================

def scan_workspace(
    robot: 'FrankyRobot',
    camera: 'DepthCamera',
    T_ee_camera: np.ndarray,
    scan_poses: list,
    voxel_map: Optional[VoxelMap] = None,
    config: Optional['DeployConfig'] = None,
    settle_time: float = 0.5,
    n_frames_per_pose: int = 3,
    depth_to_points_fn=None,
    get_camera_pose_fn=None,
) -> VoxelMap:
    """
    Scan workspace from multiple robot poses to build a complete voxel map.

    Args:
        robot: Connected FrankyRobot
        camera: Started DepthCamera
        T_ee_camera: Camera-to-EE transform (4x4)
        scan_poses: List of joint configurations to scan from
        voxel_map: Existing map to add to (None = create new)
        config: Deployment config for grid parameters
        settle_time: Wait time after moving before capturing
        n_frames_per_pose: Number of frames to capture per pose
        depth_to_points_fn: Custom depth-to-points function
        get_camera_pose_fn: Custom function to get camera pose from robot

    Returns:
        VoxelMap with accumulated observations
    """
    # Get grid parameters from config or use defaults
    if config is not None:
        grid_shape = config.grid_shape
        grid_origin = config.grid_origin
        grid_resolution = config.grid_resolution
        depth_max = config.depth_max
    else:
        grid_shape = (64, 64, 64)
        grid_origin = (-0.5, -0.5, 0.0)
        grid_resolution = 0.025
        depth_max = 2.0

    if voxel_map is None:
        voxel_map = VoxelMap(
            shape=grid_shape,
            origin=grid_origin,
            resolution=grid_resolution,
        )

    intrinsics = camera.get_intrinsics()

    print(f"Scanning workspace from {len(scan_poses)} poses...")

    for i, q_target in enumerate(scan_poses):
        # Move to scan pose
        print(f"  Pose {i+1}/{len(scan_poses)}: moving...", end="", flush=True)
        robot.move_to_joint_positions(np.array(q_target), blocking=True)

        # Wait for robot to settle
        time.sleep(settle_time)

        # Capture multiple frames
        for _ in range(n_frames_per_pose):
            depth = camera.get_depth_frame()
            if depth is not None:
                # Get camera pose in world frame
                if get_camera_pose_fn is not None:
                    camera_pose = get_camera_pose_fn(robot, T_ee_camera)
                else:
                    camera_pose = _get_camera_pose_from_robot(robot, T_ee_camera)

                voxel_map.integrate(
                    depth, intrinsics, camera_pose, depth_max,
                    depth_to_points_fn=depth_to_points_fn,
                )
            time.sleep(0.1)

        print(f" captured ({voxel_map.n_occupied} voxels)")

    print(f"Scan complete: {voxel_map.n_frames} frames, {voxel_map.n_occupied} voxels")
    return voxel_map


def generate_scan_poses(
    center: np.ndarray,
    n_poses: int = 4,
    radius: float = 0.3,
) -> list:
    """
    Generate scan poses looking at a center point from different angles.

    This is a simple heuristic - in practice you'd want IK-based pose generation.

    Args:
        center: Joint configuration at center/home
        n_poses: Number of scan poses
        radius: How much to vary joint angles (radians)

    Returns:
        List of joint configurations
    """
    poses = [center.copy()]  # Always include center

    # Vary joints 1 (base rotation) and 5 (wrist) for different viewpoints
    for i in range(n_poses - 1):
        angle = 2 * np.pi * i / (n_poses - 1)
        pose = center.copy()
        pose[0] += radius * np.cos(angle)  # Base rotation
        pose[5] += radius * 0.5 * np.sin(angle)  # Wrist
        poses.append(pose)

    return poses


# ============================================================================
# Task: Scan and Reach
# ============================================================================

def task_scan_and_reach(
    controller,
    goal_pose: np.ndarray,
    scan_poses: list = None,
    T_ee_camera: np.ndarray = None,
):
    """
    Complete task: scan workspace, then plan and reach a goal.

    Args:
        controller: Connected DeploymentController
        goal_pose: Target joint configuration
        scan_poses: Poses to scan from (default: SCAN_POSES)
        T_ee_camera: Camera calibration (default: load from file)

    Returns:
        True if goal reached successfully
    """
    if scan_poses is None:
        scan_poses = SCAN_POSES

    print("=" * 60)
    print("TASK: Scan and Reach")
    print("=" * 60)

    # Load calibration if not provided
    if T_ee_camera is None:
        from deploy_new import load_camera_calibration
        T_ee_camera = load_camera_calibration()

    # Scan workspace
    voxel_map = scan_workspace(
        controller.robot,
        controller.camera,
        T_ee_camera,
        scan_poses,
        config=controller.config,
    )

    # Finalize map
    grid, dist_field = voxel_map.finalize()
    controller.planner.update_environment(grid)
    controller.planner._dist_field = dist_field

    # Plan and execute
    print(f"\nPlanning to goal: {goal_pose.round(2)}")
    success = controller.move_to_goal(goal_pose, replan=False)

    return success


# ============================================================================
# Internal Helpers
# ============================================================================

def _get_camera_pose_from_robot(robot: 'FrankyRobot', T_ee_camera: np.ndarray) -> np.ndarray:
    """
    Get camera pose in world frame from robot state.

    Args:
        robot: Connected FrankyRobot instance
        T_ee_camera: 4x4 transform from EE to camera

    Returns:
        4x4 camera-to-world transform
    """
    from scipy.spatial.transform import Rotation

    # Get EE pose from robot
    pos, quat_wxyz = robot.get_ee_pose()

    # Convert to transformation matrix
    quat_xyzw = [quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]]
    T_world_ee = np.eye(4)
    T_world_ee[:3, :3] = Rotation.from_quat(quat_xyzw).as_matrix()
    T_world_ee[:3, 3] = pos

    return T_world_ee @ T_ee_camera
