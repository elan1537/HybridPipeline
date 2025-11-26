"""
Project Loader for COLMAP Data
Loads cameras.json and gaussian.ply files from a project directory structure.
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import open3d as o3d

try:
    from plyfile import PlyData
    PLYFILE_AVAILABLE = True
except ImportError:
    print("[Warning] plyfile library not found. Install with: pip install plyfile")
    print("          Falling back to basic Open3D PLY loader (may not show correct colors)")
    PLYFILE_AVAILABLE = False


class GaussianPLYLoader:
    """Load and process Gaussian Splatting PLY files with SH coefficient support."""

    @staticmethod
    def sh_to_rgb(sh_coeffs: np.ndarray) -> np.ndarray:
        """
        Convert SH DC coefficients to RGB colors.

        Gaussian Splatting stores colors as Spherical Harmonics (SH) DC coefficients
        (f_dc_0, f_dc_1, f_dc_2). This function converts them to RGB values in [0, 1] range.

        Formula (from official graphdeco-inria/gaussian-splatting):
            RGB = C0 * sh_coeffs + 0.5
            where C0 = 0.28209479177387814 (zeroth-order SH coefficient)

        Args:
            sh_coeffs: SH DC coefficients (N, 3) array for (R, G, B)

        Returns:
            RGB colors (N, 3) array in [0, 1] range
        """
        # Zeroth-order spherical harmonics coefficient: 1 / sqrt(4 * pi)
        C0 = 0.28209479177387814

        # Official formula: RGB = C0 * sh + 0.5
        rgb = sh_coeffs * C0 + 0.5

        # Clamp to valid range [0, 1]
        rgb = np.clip(rgb, 0.0, 1.0)

        return rgb

    @staticmethod
    def load_ply_with_sh(ply_path: str) -> Optional[o3d.geometry.PointCloud]:
        """
        Load Gaussian Splatting PLY file and extract RGB from SH coefficients.

        Args:
            ply_path: Path to PLY file

        Returns:
            Open3D PointCloud with RGB colors, or None if failed
        """
        if not PLYFILE_AVAILABLE:
            return None

        try:
            # Read PLY file with plyfile library
            plydata = PlyData.read(ply_path)
            vertex = plydata['vertex']

            # Extract positions
            positions = np.stack([
                vertex['x'],
                vertex['y'],
                vertex['z']
            ], axis=1)

            # Try to extract SH DC coefficients (Gaussian Splatting format)
            sh_dc = None
            if 'f_dc_0' in vertex.data.dtype.names:
                sh_dc = np.stack([
                    vertex['f_dc_0'],
                    vertex['f_dc_1'],
                    vertex['f_dc_2']
                ], axis=1)

            # Try to extract standard RGB colors
            rgb = None
            if 'red' in vertex.data.dtype.names:
                rgb = np.stack([
                    vertex['red'],
                    vertex['green'],
                    vertex['blue']
                ], axis=1).astype(np.float32)

                # Normalize if values are in [0, 255] range
                if rgb.max() > 1.0:
                    rgb = rgb / 255.0

            # Create Open3D point cloud
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(positions)

            # Set colors (prefer SH coefficients over standard RGB)
            if sh_dc is not None:
                rgb_colors = GaussianPLYLoader.sh_to_rgb(sh_dc)
                pcd.colors = o3d.utility.Vector3dVector(rgb_colors)
                print(f"  [PLY] Colors: Converted from SH coefficients ({len(positions)} points)")
            elif rgb is not None:
                pcd.colors = o3d.utility.Vector3dVector(rgb)
                print(f"  [PLY] Colors: Loaded from RGB attributes ({len(positions)} points)")
            else:
                # No color information, use default
                print(f"  [PLY] Colors: None found, using default white ({len(positions)} points)")
                pcd.paint_uniform_color([1.0, 1.0, 1.0])

            return pcd

        except Exception as e:
            print(f"  [PLY] Warning: Failed to load with SH extraction: {e}")
            return None


def load_project(project_dir: str, frame_idx: int = 0,
                 checkpoints_subdir: str = "checkpoints",
                 return_available_frames: bool = False) -> Tuple:
    """
    Load project data including cameras and point cloud.

    Args:
        project_dir: Path to project directory containing cameras.json and checkpoints/
        frame_idx: Which frame to load (default: 0)
        checkpoints_subdir: Checkpoints subdirectory name (default: "checkpoints")
        return_available_frames: If True, return list of available frame IDs (default: False)

    Returns:
        If return_available_frames is False:
            cameras: Dictionary of camera data from cameras.json
            point_cloud: Open3D PointCloud object
            ply_path: Path to the loaded PLY file

        If return_available_frames is True:
            cameras: Dictionary of camera data from cameras.json
            point_cloud: Open3D PointCloud object
            ply_path: Path to the loaded PLY file
            available_frames: List of available frame indices
    """
    project_path = Path(project_dir)

    if not project_path.exists():
        raise FileNotFoundError(f"Project directory not found: {project_dir}")

    # Load cameras.json
    cameras_path = project_path / "cameras.json"
    if not cameras_path.exists():
        raise FileNotFoundError(f"cameras.json not found in {project_dir}")

    cameras = parse_cameras_json(cameras_path)
    print(f"[ProjectLoader] Loaded {len(cameras)} cameras from cameras.json")

    # Get checkpoints directory
    checkpoints_dir = project_path / checkpoints_subdir

    # Find available frames
    available_frames = []
    if checkpoints_dir.exists():
        available_frames = find_available_frames(checkpoints_dir)
        if available_frames:
            print(f"[ProjectLoader] Found {len(available_frames)} frames: "
                  f"{min(available_frames)}-{max(available_frames)}")

    # Find and load gaussian.ply
    ply_path = find_gaussian_ply(project_path, frame_idx)
    if ply_path is None:
        raise FileNotFoundError(f"No gaussian.ply found for frame {frame_idx}")

    point_cloud = load_gaussian_ply(ply_path)
    print(f"[ProjectLoader] Loaded point cloud with {len(point_cloud.points)} points")

    if return_available_frames:
        return cameras, point_cloud, ply_path, available_frames
    else:
        return cameras, point_cloud, ply_path


def parse_cameras_json(filepath: Path) -> Dict:
    """
    Parse cameras.json file.

    Args:
        filepath: Path to cameras.json

    Returns:
        Dictionary with camera data
    """
    with open(filepath, 'r') as f:
        cameras_data = json.load(f)

    return cameras_data


def find_available_frames(checkpoints_dir: Path) -> List[int]:
    """
    Find all available frame indices in checkpoints directory.

    Args:
        checkpoints_dir: Path to checkpoints directory

    Returns:
        List of frame indices (sorted)
    """
    if not checkpoints_dir.exists():
        return []

    frame_dirs = sorted(checkpoints_dir.glob("frame*"))
    frame_indices = []

    for frame_dir in frame_dirs:
        # Extract frame number from directory name (e.g., "frame000123" -> 123)
        try:
            frame_num = int(frame_dir.name.replace("frame", "").replace("_", ""))
            frame_indices.append(frame_num)
        except ValueError:
            continue

    return sorted(frame_indices)


def validate_weight_pattern(weight_pattern: str, available_frames: List[int],
                           max_check: int = 5) -> Tuple[bool, List[int], List[int]]:
    """
    Validate weight pattern path against available frames.

    Args:
        weight_pattern: Path pattern with {frame_id:06d} placeholder
            Example: '/path/checkpoints/frame_{frame_id:06d}/gaussian.ply'
        available_frames: List of available frame indices
        max_check: Maximum number of frames to check (default: 5)

    Returns:
        Tuple of (all_valid, valid_frames, missing_frames)
            all_valid: True if all checked frames exist
            valid_frames: List of frame IDs that exist
            missing_frames: List of frame IDs that don't exist
    """
    if not weight_pattern or not available_frames:
        return False, [], []

    # Check if pattern contains frame_id placeholder
    if '{frame_id' not in weight_pattern:
        print(f"[Warning] Weight pattern missing {{frame_id}} placeholder: {weight_pattern}")
        return False, [], []

    valid_frames = []
    missing_frames = []

    # Check a subset of frames
    frames_to_check = available_frames[:max_check]

    for frame_id in frames_to_check:
        # Format the path with frame_id
        try:
            path_str = weight_pattern.format(frame_id=frame_id)
            path = Path(path_str)

            if path.exists():
                valid_frames.append(frame_id)
            else:
                missing_frames.append(frame_id)
        except (KeyError, ValueError) as e:
            print(f"[Warning] Failed to format weight pattern: {e}")
            return False, [], []

    all_valid = len(missing_frames) == 0

    if valid_frames:
        print(f"[WeightPattern] Validated: {len(valid_frames)}/{len(frames_to_check)} frames exist")
        if missing_frames:
            print(f"[WeightPattern] Missing frames: {missing_frames}")

    return all_valid, valid_frames, missing_frames


def find_gaussian_ply(project_path: Path, frame_idx: int = 0) -> Optional[Path]:
    """
    Find gaussian.ply for specified frame.

    Args:
        project_path: Path to project directory
        frame_idx: Frame index to load

    Returns:
        Path to gaussian.ply or None if not found
    """
    checkpoints_dir = project_path / "checkpoints"

    if not checkpoints_dir.exists():
        print(f"[ProjectLoader] Warning: checkpoints directory not found")
        return None

    # Construct frame directory name (e.g., "frame000000")
    frame_dir_name = f"frame_{frame_idx:06d}"
    frame_dir = checkpoints_dir / frame_dir_name

    print(frame_dir)

    if not frame_dir.exists():
        # Try to find any available frame
        available_frames = find_available_frames(checkpoints_dir)
        if available_frames:
            print(f"[ProjectLoader] Frame {frame_idx} not found, using frame {available_frames[0]}")
            frame_dir_name = f"frame_{available_frames[0]:06d}"
            frame_dir = checkpoints_dir / frame_dir_name
        else:
            print(f"[ProjectLoader] No frames found in {checkpoints_dir}")
            return None

    ply_path = frame_dir / "gaussian.ply"

    if not ply_path.exists():
        print(f"[ProjectLoader] gaussian.ply not found in {frame_dir}")
        return None

    return ply_path


def load_gaussian_ply(ply_path: Path) -> o3d.geometry.PointCloud:
    """
    Load gaussian.ply file as Open3D PointCloud with SH coefficient support.

    Attempts to load using GaussianPLYLoader (with SH extraction) first,
    then falls back to basic Open3D loader if that fails.

    Args:
        ply_path: Path to gaussian.ply file

    Returns:
        Open3D PointCloud object
    """
    print(f"[ProjectLoader] Loading PLY: {ply_path.name}")

    # Try loading with SH coefficient extraction
    if PLYFILE_AVAILABLE:
        pcd = GaussianPLYLoader.load_ply_with_sh(str(ply_path))
        if pcd is not None and pcd.has_points():
            return pcd

    # Fallback to basic Open3D loader
    print(f"  [PLY] Falling back to basic Open3D loader")
    pcd = o3d.io.read_point_cloud(str(ply_path))

    if not pcd.has_points():
        raise ValueError(f"Loaded PLY file has no points: {ply_path}")

    print(f"  [PLY] Loaded {len(pcd.points)} points with basic loader")
    return pcd


class CameraVisualizer:
    """Visualize cameras from COLMAP cameras.json with transform_to_local_frame support."""

    @staticmethod
    def transform_to_local_frame(cameras: list, point_cloud: o3d.geometry.PointCloud,
                                 reference_cam_id: int = 0) -> Tuple[list, o3d.geometry.PointCloud, np.ndarray]:
        """
        Transform all cameras and point cloud to use reference camera as origin.

        This creates a local coordinate system where the reference camera is at the origin (0,0,0)
        and all other cameras/objects are expressed relative to it.

        Args:
            cameras: List of camera dictionaries with 'position' and 'rotation'
            point_cloud: Open3D PointCloud to transform
            reference_cam_id: ID of the reference camera (default: 0)

        Returns:
            Tuple of (transformed_cameras, transformed_pcd, world_to_local_transform)
        """
        if not cameras:
            raise ValueError("No cameras provided")

        # Find reference camera
        ref_cam = None
        for cam in cameras:
            if cam.get('id', -1) == reference_cam_id:
                ref_cam = cam
                break

        if ref_cam is None:
            print(f"\n[Warning] Reference camera {reference_cam_id} not found, using first camera")
            ref_cam = cameras[0]

        # Get reference camera pose (camera-to-world)
        ref_position = np.array(ref_cam['position'])
        ref_rotation = np.array(ref_cam['rotation'])  # 3x3 rotation matrix

        print(f"\n[Coordinate Transform] Using camera {ref_cam.get('id', 0)} as local origin")
        print(f"  Reference position: [{ref_position[0]:.3f}, {ref_position[1]:.3f}, {ref_position[2]:.3f}]")

        # Build camera-to-world transform for reference camera
        T_c2w = np.eye(4)
        T_c2w[:3, :3] = ref_rotation
        T_c2w[:3, 3] = ref_position

        # Compute world-to-camera transform (inverse)
        T_w2c = np.eye(4)
        T_w2c[:3, :3] = ref_rotation.T
        T_w2c[:3, 3] = -ref_rotation.T @ ref_position

        # Transform all cameras to local frame
        transformed_cameras = []

        for cam in cameras:
            cam_id = cam.get('id', -1)
            position = np.array(cam['position'])
            rotation = np.array(cam['rotation'])

            # Build camera-to-world transform
            T_cam_c2w = np.eye(4)
            T_cam_c2w[:3, :3] = rotation
            T_cam_c2w[:3, 3] = position

            # Transform to local frame: T_local = T_w2c @ T_cam_c2w
            T_local = T_w2c @ T_cam_c2w

            # Extract transformed position and rotation
            local_position = T_local[:3, 3]
            local_rotation = T_local[:3, :3]

            transformed_cam = {
                'id': cam_id,
                'position': local_position.tolist(),
                'rotation': local_rotation.tolist(),
                'original_position': position.tolist(),
                'original_rotation': rotation.tolist()
            }

            transformed_cameras.append(transformed_cam)

            # Print first few cameras
            if len(transformed_cameras) <= 3:
                print(f"  Camera {cam_id}:")
                print(f"    World position:  [{position[0]:7.3f}, {position[1]:7.3f}, {position[2]:7.3f}]")
                print(f"    Local position:  [{local_position[0]:7.3f}, {local_position[1]:7.3f}, {local_position[2]:7.3f}]")

        # Verify reference camera is at origin
        ref_cam_pos = np.array(transformed_cameras[reference_cam_id]['position'])
        if np.linalg.norm(ref_cam_pos) < 1e-6:
            print(f"\n  ✓ Camera {reference_cam_id} successfully placed at origin")
        else:
            print(f"\n  ⚠ Warning: Camera {reference_cam_id} not at origin (distance: {np.linalg.norm(ref_cam_pos):.6f})")

        # Transform point cloud
        transformed_pcd = o3d.geometry.PointCloud(point_cloud)
        transformed_pcd.transform(T_w2c)
        print(f"  ✓ Point cloud transformed to local frame")

        return transformed_cameras, transformed_pcd, T_w2c

    @staticmethod
    def create_camera_geometry(position: np.ndarray, rotation: np.ndarray,
                               size: float = 0.3, color: Tuple[float, float, float] = (1.0, 0.0, 0.0)) -> List:
        """
        Create camera visualization geometry (COLMAP convention).

        COLMAP Convention:
        - X: right
        - Y: down
        - Z: forward (camera looks along +Z)

        Args:
            position: Camera position (x, y, z)
            rotation: 3x3 camera-to-world rotation matrix
            size: Camera visualization size
            color: RGB color tuple (0-1 range)

        Returns:
            List of Open3D geometries representing the camera
        """
        geometries = []

        # Camera coordinate frame (COLMAP convention)
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=size, origin=[0, 0, 0]
        )

        # Build camera-to-world transform
        T = np.eye(4)
        T[:3, :3] = rotation
        T[:3, 3] = position

        # Transform coordinate frame to camera pose
        coord_frame.transform(T)
        geometries.append(coord_frame)

        # Create a small sphere at camera position
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=size * 0.3)
        sphere.translate(position)
        sphere.paint_uniform_color(color)
        sphere.compute_vertex_normals()
        geometries.append(sphere)

        # Create camera frustum
        frustum = CameraVisualizer.create_frustum(size * 1.5, color=[0.7, 0.7, 0.7])
        frustum.transform(T)
        geometries.append(frustum)

        return geometries

    @staticmethod
    def create_frustum(size: float, color: List[float] = None) -> o3d.geometry.LineSet:
        """
        Create camera frustum as LineSet (COLMAP convention: looks along +Z).

        Args:
            size: Frustum size
            color: RGB color for the lines (default: gray [0.7, 0.7, 0.7])

        Returns:
            Open3D LineSet with colors
        """
        if color is None:
            color = [0.7, 0.7, 0.7]

        # Frustum points (at camera origin, looking along +Z)
        far = size * 1.0
        aspect = 1.0

        # Points: camera center + far plane corners
        points = [
            [0, 0, 0],  # Camera center
            # Far plane corners (COLMAP: Y down, Z forward)
            [-far * aspect, -far, far],  # Top-left (Y down!)
            [far * aspect, -far, far],   # Top-right
            [far * aspect, far, far],    # Bottom-right
            [-far * aspect, far, far],   # Bottom-left
        ]

        # Lines connecting camera center to frustum corners
        lines = [
            [0, 1], [0, 2], [0, 3], [0, 4],  # Center to corners
            [1, 2], [2, 3], [3, 4], [4, 1],  # Far plane rectangle
        ]

        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(points)
        line_set.lines = o3d.utility.Vector2iVector(lines)

        # Add colors to all lines
        line_colors = [color for _ in range(len(lines))]
        line_set.colors = o3d.utility.Vector3dVector(line_colors)

        return line_set

    @staticmethod
    def visualize_cameras(cameras: list, camera_size: float = 0.3) -> List:
        """
        Create visualization geometries for all cameras.

        Args:
            cameras: List of camera dictionaries from cameras.json
            camera_size: Size of camera visualization

        Returns:
            List of Open3D geometries
        """
        print(f"\n[Visualizing Cameras] Creating geometries for {len(cameras)} cameras...")

        geometries = []

        for i, cam in enumerate(cameras):
            cam_id = cam.get('id', i)
            position = np.array(cam['position'])
            rotation = np.array(cam['rotation'])

            # Create camera geometry
            cam_geoms = CameraVisualizer.create_camera_geometry(
                position, rotation, size=camera_size,
                color=(1.0, 0.3, 0.3)  # Red for cameras
            )

            geometries.extend(cam_geoms)

            # Print camera info
            if i < 5 or i % 10 == 0:  # Print first 5 and every 10th
                print(f"  Camera {cam_id}: Position = [{position[0]:.3f}, {position[1]:.3f}, {position[2]:.3f}]")

        if len(cameras) > 5:
            print(f"  ... and {len(cameras) - 5} more cameras")

        print(f"  Created {len(geometries)} geometries")

        return geometries


def create_camera_frustums(cameras: Dict, frustum_size: float = 0.1) -> List[o3d.geometry.LineSet]:
    """
    Create camera frustum visualizations from cameras.json data.

    Args:
        cameras: Dictionary of camera data
        frustum_size: Size of the frustum visualization

    Returns:
        List of Open3D LineSet objects representing camera frustums
    """
    frustums = []

    for _cam_id, cam_data in cameras.items():
        # Extract camera pose
        # Assuming cameras.json has 'position' and 'rotation' fields
        # You may need to adjust this based on actual format

        if 'position' in cam_data and 'rotation' in cam_data:
            position = np.array(cam_data['position'])
            rotation = np.array(cam_data['rotation'])  # Rotation matrix or quaternion

            # Create frustum geometry
            frustum = create_frustum_lineset(position, rotation, frustum_size)
            frustums.append(frustum)

    return frustums


def create_frustum_lineset(position: np.ndarray, rotation: np.ndarray,
                           size: float = 0.1) -> o3d.geometry.LineSet:
    """
    Create a camera frustum as a LineSet.

    Args:
        position: Camera position [x, y, z]
        rotation: Camera rotation matrix (3x3)
        size: Frustum size

    Returns:
        Open3D LineSet representing the frustum
    """
    # Define frustum points in camera space
    points = np.array([
        [0, 0, 0],  # Camera center
        [-size, -size, size],  # Bottom-left
        [size, -size, size],   # Bottom-right
        [size, size, size],    # Top-right
        [-size, size, size],   # Top-left
    ])

    # Transform to world space
    if rotation.shape == (3, 3):
        points = (rotation @ points.T).T + position
    else:
        points = points + position

    # Define lines connecting points
    lines = [
        [0, 1], [0, 2], [0, 3], [0, 4],  # Camera center to corners
        [1, 2], [2, 3], [3, 4], [4, 1],  # Rectangle
    ]

    # Create LineSet
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)

    # Color the frustum
    colors = [[0.0, 0.7, 1.0] for _ in range(len(lines))]  # Cyan
    line_set.colors = o3d.utility.Vector3dVector(colors)

    return line_set


if __name__ == "__main__":
    # Test the loader
    import sys

    if len(sys.argv) < 2:
        print("Usage: python project_loader.py <project_dir>")
        sys.exit(1)

    project_dir = sys.argv[1]

    try:
        cameras, pcd, ply_path = load_project(project_dir)
        print(f"\nSuccessfully loaded project:")
        print(f"  - Cameras: {len(cameras)}")
        print(f"  - Points: {len(pcd.points)}")
        print(f"  - PLY: {ply_path}")

        # List available frames
        checkpoints_dir = Path(project_dir) / "checkpoints"
        frames = find_available_frames(checkpoints_dir)
        print(f"  - Available frames: {frames}")

    except Exception as e:
        print(f"Error loading project: {e}")
        sys.exit(1)
