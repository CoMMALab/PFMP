"""
Voxel grid representation and manipulation for collision avoidance.

This module provides:
- VoxelGrid: Core voxel occupancy representation
- VoxelMap: Multi-view point cloud accumulation
- Grid construction primitives (add_box, add_sphere, add_cylinder)
- Point cloud to voxel conversion
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import NamedTuple, Optional


# ============================================================================
# Voxel Grid
# ============================================================================

class VoxelGrid(NamedTuple):
    """Voxel occupancy grid with origin and resolution."""
    data: jax.Array      # 3D occupancy array (1=occupied, 0=free)
    origin: jax.Array    # World position of grid corner (x, y, z)
    resolution: float    # Voxel size in meters


def empty_voxel_grid(
    shape: tuple = (64, 64, 64),
    origin: tuple = (-1.0, -1.0, 0.0),
    resolution: float = 0.05,
) -> VoxelGrid:
    """Create an empty voxel grid."""
    return VoxelGrid(
        data=jnp.zeros(shape, dtype=jnp.float32),
        origin=jnp.array(origin, dtype=jnp.float32),
        resolution=float(resolution),
    )


def compute_distance_field(grid: VoxelGrid) -> jax.Array:
    """
    Compute signed distance field from occupancy grid.

    Returns:
        SDF where negative = inside obstacle, positive = free space
    """
    from scipy.ndimage import distance_transform_edt
    occupied = np.array(grid.data > 0.5)
    dist_free = distance_transform_edt(~occupied) * grid.resolution
    dist_occ = distance_transform_edt(occupied) * grid.resolution
    return jnp.array(np.where(occupied, -dist_occ, dist_free), dtype=jnp.float32)


# ============================================================================
# Grid Construction Primitives
# ============================================================================

def add_box(
    grid: VoxelGrid,
    center: tuple,
    half_extents: tuple,
    rotation: Optional[np.ndarray] = None,
) -> VoxelGrid:
    """
    Add axis-aligned box to voxel grid.

    Args:
        grid: Voxel grid to modify
        center: Box center (x, y, z)
        half_extents: Half-sizes (hx, hy, hz)
        rotation: Ignored (axis-aligned only)

    Returns:
        Updated VoxelGrid
    """
    data = np.array(grid.data)
    center = np.array(center)
    half_extents = np.array(half_extents)
    origin = np.array(grid.origin)
    res = grid.resolution
    shape = data.shape

    min_idx = np.maximum(((center - half_extents - origin) / res).astype(int), 0)
    max_idx = np.minimum(((center + half_extents - origin) / res).astype(int) + 1, shape)

    data[min_idx[0]:max_idx[0], min_idx[1]:max_idx[1], min_idx[2]:max_idx[2]] = 1.0

    return grid._replace(data=jnp.array(data))


def add_sphere(grid: VoxelGrid, center: tuple, radius: float) -> VoxelGrid:
    """
    Add sphere to voxel grid.

    Args:
        grid: Voxel grid to modify
        center: Sphere center (x, y, z)
        radius: Sphere radius

    Returns:
        Updated VoxelGrid
    """
    data = np.array(grid.data)
    center = np.array(center)
    origin = np.array(grid.origin)
    res = grid.resolution
    shape = np.array(data.shape)

    min_idx = np.maximum(((center - radius - origin) / res).astype(int), 0)
    max_idx = np.minimum(((center + radius - origin) / res).astype(int) + 1, shape)

    # Vectorized sphere test
    i = np.arange(min_idx[0], max_idx[0])
    j = np.arange(min_idx[1], max_idx[1])
    k = np.arange(min_idx[2], max_idx[2])

    if len(i) == 0 or len(j) == 0 or len(k) == 0:
        return grid

    ii, jj, kk = np.meshgrid(i, j, k, indexing='ij')

    # Voxel centers
    px = origin[0] + (ii + 0.5) * res
    py = origin[1] + (jj + 0.5) * res
    pz = origin[2] + (kk + 0.5) * res

    # Distance from center
    dist = np.sqrt((px - center[0])**2 + (py - center[1])**2 + (pz - center[2])**2)
    inside = dist <= radius

    data[min_idx[0]:max_idx[0], min_idx[1]:max_idx[1], min_idx[2]:max_idx[2]][inside] = 1.0

    return grid._replace(data=jnp.array(data))


def add_cylinder(
    grid: VoxelGrid,
    center: tuple,
    radius: float,
    half_height: float,
    rotation: Optional[np.ndarray] = None,
) -> VoxelGrid:
    """
    Add axis-aligned cylinder (Z-axis) to voxel grid.

    Args:
        grid: Voxel grid to modify
        center: Cylinder center (x, y, z)
        radius: Cylinder radius
        half_height: Half-height along Z
        rotation: Ignored (axis-aligned only)

    Returns:
        Updated VoxelGrid
    """
    data = np.array(grid.data)
    center = np.array(center)
    origin = np.array(grid.origin)
    res = grid.resolution
    shape = data.shape

    # Bounding box indices
    extent = max(radius, half_height)
    min_idx = np.maximum(((center - extent - origin) / res).astype(int) - 1, 0)
    max_idx = np.minimum(((center + extent - origin) / res).astype(int) + 2, shape)

    # Create grid of voxel centers in the bounding box
    i = np.arange(min_idx[0], max_idx[0])
    j = np.arange(min_idx[1], max_idx[1])
    k = np.arange(min_idx[2], max_idx[2])

    if len(i) == 0 or len(j) == 0 or len(k) == 0:
        return grid

    ii, jj, kk = np.meshgrid(i, j, k, indexing='ij')

    # Voxel centers
    px = origin[0] + (ii + 0.5) * res
    py = origin[1] + (jj + 0.5) * res
    pz = origin[2] + (kk + 0.5) * res

    # Cylinder test (axis-aligned along Z)
    radial_dist = np.sqrt((px - center[0])**2 + (py - center[1])**2)
    axial_dist = np.abs(pz - center[2])

    inside = (radial_dist <= radius) & (axial_dist <= half_height)

    data[min_idx[0]:max_idx[0], min_idx[1]:max_idx[1], min_idx[2]:max_idx[2]][inside] = 1.0

    return grid._replace(data=jnp.array(data))


# ============================================================================
# Point Cloud to Voxels
# ============================================================================

def point_cloud_to_voxel_grid(
    points: np.ndarray,
    grid: Optional[VoxelGrid] = None,
    shape: tuple = (64, 64, 64),
    origin: tuple = (-0.5, -0.5, 0.0),
    resolution: float = 0.025,
) -> VoxelGrid:
    """
    Convert point cloud to voxel occupancy grid.

    Args:
        points: Point cloud (N, 3) in world frame
        grid: Existing grid to update (None = create new)
        shape: Grid dimensions if creating new
        origin: Grid origin if creating new
        resolution: Voxel size if creating new

    Returns:
        Updated VoxelGrid
    """
    if grid is None:
        grid = empty_voxel_grid(shape=shape, origin=origin, resolution=resolution)

    data = np.zeros(shape, dtype=np.float32)
    origin_arr = np.array(grid.origin)
    res = grid.resolution

    # Convert points to voxel indices
    voxel_indices = ((points - origin_arr) / res).astype(int)

    # Filter in-bounds points
    in_bounds = np.all((voxel_indices >= 0) & (voxel_indices < np.array(shape)), axis=1)
    voxel_indices = voxel_indices[in_bounds]

    # Mark occupied voxels
    if len(voxel_indices) > 0:
        data[voxel_indices[:, 0], voxel_indices[:, 1], voxel_indices[:, 2]] = 1.0

    return grid._replace(data=jnp.array(data))


# ============================================================================
# VoxelMap (Multi-View Accumulation)
# ============================================================================

class VoxelMap:
    """
    Accumulates point clouds from multiple viewpoints into a single voxel grid.

    Simple union-based fusion - marks voxels as occupied if any view sees them.
    Good for static scenes with a wrist-mounted camera.
    """

    def __init__(
        self,
        shape: tuple = (64, 64, 64),
        origin: tuple = (-0.5, -0.5, 0.0),
        resolution: float = 0.025,
        min_hits: int = 1,
    ):
        """
        Initialize voxel map.

        Args:
            shape: Grid dimensions (X, Y, Z)
            origin: World position of grid corner
            resolution: Voxel size in meters
            min_hits: Minimum observations to mark voxel as occupied
        """
        self.shape = shape
        self.origin = np.array(origin)
        self.resolution = resolution
        self.min_hits = min_hits

        # Hit counts per voxel
        self._hits = np.zeros(shape, dtype=np.int32)
        self._n_frames = 0

    def integrate(
        self,
        depth: np.ndarray,
        intrinsics: dict,
        camera_pose: np.ndarray,
        max_depth: float = 2.0,
        depth_to_points_fn=None,
    ):
        """
        Integrate a depth frame into the map.

        Args:
            depth: Depth image (H, W) in meters
            intrinsics: Camera intrinsics dict with fx, fy, cx, cy
            camera_pose: 4x4 camera-to-world transform
            max_depth: Maximum depth to include
            depth_to_points_fn: Function to convert depth to points (optional)
        """
        if depth_to_points_fn is None:
            # Default implementation
            points_cam = _depth_to_point_cloud(depth, intrinsics, max_depth)
        else:
            points_cam = depth_to_points_fn(depth, intrinsics, max_depth)

        points_world = _transform_point_cloud(points_cam, camera_pose)
        self.integrate_pointcloud(points_world)

    def integrate_pointcloud(self, points_world: np.ndarray):
        """
        Integrate a point cloud directly (already in world frame).

        Args:
            points_world: Point cloud (N, 3) in world coordinates
        """
        voxel_indices = ((points_world - self.origin) / self.resolution).astype(int)

        in_bounds = np.all(
            (voxel_indices >= 0) & (voxel_indices < np.array(self.shape)),
            axis=1
        )
        voxel_indices = voxel_indices[in_bounds]

        if len(voxel_indices) > 0:
            unique_voxels = np.unique(voxel_indices, axis=0)
            self._hits[unique_voxels[:, 0], unique_voxels[:, 1], unique_voxels[:, 2]] += 1

        self._n_frames += 1

    def clear(self):
        """Clear all accumulated data."""
        self._hits.fill(0)
        self._n_frames = 0

    def get_occupancy(self) -> np.ndarray:
        """Get binary occupancy grid."""
        return (self._hits >= self.min_hits).astype(np.float32)

    def get_hit_counts(self) -> np.ndarray:
        """Get raw hit counts per voxel."""
        return self._hits.copy()

    def finalize(self) -> tuple:
        """
        Finalize map and compute distance field.

        Returns:
            (grid, dist_field) tuple ready for planning
        """
        occupancy = self.get_occupancy()
        grid = VoxelGrid(
            data=jnp.array(occupancy),
            origin=jnp.array(self.origin),
            resolution=self.resolution,
        )
        dist_field = compute_distance_field(grid)

        n_occupied = int(occupancy.sum())
        print(f"VoxelMap: {self._n_frames} frames, {n_occupied} occupied voxels")

        return grid, dist_field

    @property
    def n_frames(self) -> int:
        return self._n_frames

    @property
    def n_occupied(self) -> int:
        return int((self._hits >= self.min_hits).sum())


# ============================================================================
# Internal Helpers
# ============================================================================

def _depth_to_point_cloud(
    depth: np.ndarray,
    intrinsics: dict,
    max_depth: float = 2.0,
) -> np.ndarray:
    """
    Convert depth image to 3D point cloud in camera frame.

    Args:
        depth: Depth image (H, W) in meters
        intrinsics: Dict with 'fx', 'fy', 'cx', 'cy'
        max_depth: Maximum depth to include

    Returns:
        Point cloud (N, 3) in camera frame
    """
    fx, fy = intrinsics['fx'], intrinsics['fy']
    cx, cy = intrinsics['cx'], intrinsics['cy']

    h, w = depth.shape

    # Create pixel coordinate grids
    u = np.arange(w)
    v = np.arange(h)
    u, v = np.meshgrid(u, v)

    # Filter valid depth
    valid = (depth > 0) & (depth < max_depth)

    # Unproject to 3D
    z = depth[valid]
    x = (u[valid] - cx) * z / fx
    y = (v[valid] - cy) * z / fy

    return np.stack([x, y, z], axis=1)


def _transform_point_cloud(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """
    Transform point cloud by 4x4 matrix.

    Args:
        points: Point cloud (N, 3)
        transform: 4x4 transformation matrix

    Returns:
        Transformed points (N, 3)
    """
    # Homogeneous coordinates
    ones = np.ones((len(points), 1))
    points_h = np.hstack([points, ones])

    # Transform
    transformed = (transform @ points_h.T).T

    return transformed[:, :3]
