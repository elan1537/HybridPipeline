"""
Transform Math Utilities
Matrix composition, decomposition, and transformation computation.
"""

import numpy as np
import json
from pathlib import Path
from typing import Tuple, Dict
from scipy.spatial.transform import Rotation as R


def decompose_transform(matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Decompose a 4x4 transformation matrix into translation, rotation, and scale.

    Args:
        matrix: 4x4 homogeneous transformation matrix

    Returns:
        translation: (3,) translation vector
        rotation: (3, 3) rotation matrix
        scale: (3,) scale vector for each axis
    """
    if matrix.shape != (4, 4):
        raise ValueError(f"Expected 4x4 matrix, got shape {matrix.shape}")

    # Extract translation
    translation = matrix[:3, 3].copy()

    # Extract upper-left 3x3 matrix
    M = matrix[:3, :3].copy()

    # Extract scale (magnitude of each column vector)
    scale = np.array([
        np.linalg.norm(M[:, 0]),
        np.linalg.norm(M[:, 1]),
        np.linalg.norm(M[:, 2])
    ])

    # Extract rotation (normalize columns)
    rotation = M.copy()
    for i in range(3):
        if scale[i] > 1e-10:  # Avoid division by zero
            rotation[:, i] /= scale[i]

    # Ensure it's a proper rotation matrix (det = 1)
    if np.linalg.det(rotation) < 0:
        rotation[:, 0] *= -1
        scale[0] *= -1

    return translation, rotation, scale


def compose_transform(translation: np.ndarray, rotation: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """
    Compose a 4x4 transformation matrix from translation, rotation, and scale.

    Args:
        translation: (3,) translation vector
        rotation: (3, 3) rotation matrix
        scale: (3,) scale vector for each axis

    Returns:
        4x4 homogeneous transformation matrix
    """
    # Create scale matrix
    S = np.diag([scale[0], scale[1], scale[2]])

    # Compose: T * R * S
    M = rotation @ S

    # Build 4x4 matrix
    transform = np.eye(4)
    transform[:3, :3] = M
    transform[:3, 3] = translation

    return transform


def rotation_matrix_to_euler(rotation_matrix: np.ndarray, degrees: bool = True) -> np.ndarray:
    """
    Convert rotation matrix to Euler angles (XYZ convention).

    Args:
        rotation_matrix: (3, 3) rotation matrix
        degrees: Return angles in degrees (default: True)

    Returns:
        (3,) array of Euler angles [rx, ry, rz]
    """
    r = R.from_matrix(rotation_matrix)
    euler = r.as_euler('xyz', degrees=degrees)
    return euler


def euler_to_rotation_matrix(euler_angles: np.ndarray, degrees: bool = True) -> np.ndarray:
    """
    Convert Euler angles to rotation matrix (XYZ convention).

    Args:
        euler_angles: (3,) array of Euler angles [rx, ry, rz]
        degrees: Input angles are in degrees (default: True)

    Returns:
        (3, 3) rotation matrix
    """
    r = R.from_euler('xyz', euler_angles, degrees=degrees)
    return r.as_matrix()


def rotation_matrix_to_quaternion(rotation_matrix: np.ndarray) -> np.ndarray:
    """
    Convert rotation matrix to quaternion.

    Args:
        rotation_matrix: (3, 3) rotation matrix

    Returns:
        (4,) quaternion [w, x, y, z]
    """
    r = R.from_matrix(rotation_matrix)
    quat = r.as_quat()  # Returns [x, y, z, w]
    return np.array([quat[3], quat[0], quat[1], quat[2]])  # Reorder to [w, x, y, z]


def quaternion_to_rotation_matrix(quaternion: np.ndarray) -> np.ndarray:
    """
    Convert quaternion to rotation matrix.

    Args:
        quaternion: (4,) quaternion [w, x, y, z]

    Returns:
        (3, 3) rotation matrix
    """
    # Reorder from [w, x, y, z] to [x, y, z, w] for scipy
    quat_scipy = [quaternion[1], quaternion[2], quaternion[3], quaternion[0]]
    r = R.from_quat(quat_scipy)
    return r.as_matrix()


def rotation_matrix_from_axis_angle(axis: np.ndarray, angle: float, degrees: bool = False) -> np.ndarray:
    """
    Create rotation matrix from axis and angle.

    Args:
        axis: (3,) unit vector representing rotation axis
        angle: Rotation angle
        degrees: Angle is in degrees (default: False)

    Returns:
        (3, 3) rotation matrix
    """
    axis = axis / np.linalg.norm(axis)  # Normalize

    if degrees:
        angle = np.deg2rad(angle)

    r = R.from_rotvec(angle * axis)
    return r.as_matrix()


def compute_colmap_to_cube_transform(cube_transform: np.ndarray) -> np.ndarray:
    """
    Compute the transformation to align COLMAP coordinate system to the cube.

    This returns the transformation that should be applied to COLMAP points/cameras
    to align them with the cube's coordinate system.

    Args:
        cube_transform: 4x4 transformation matrix of the cube

    Returns:
        4x4 transformation matrix to apply to COLMAP data
    """
    # The cube transform represents the desired coordinate system
    # We want to transform COLMAP data so that the cube is at the origin with identity transform
    # This means we need the inverse of the cube transform

    transform = np.linalg.inv(cube_transform)

    return transform


def export_transform_matrix(matrix: np.ndarray, filepath: str) -> None:
    """
    Export transformation matrix to a text file.

    Args:
        matrix: 4x4 transformation matrix
        filepath: Output file path
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    # Save as space-separated values
    np.savetxt(filepath, matrix, fmt='%.10f', delimiter=' ',
               header='4x4 Transformation Matrix (row-major order)')

    print(f"[Export] Saved transformation matrix to {filepath}")


def export_transform_params(translation: np.ndarray, rotation: np.ndarray, scale: np.ndarray,
                            filepath: str) -> None:
    """
    Export transformation parameters to a JSON file.

    Args:
        translation: (3,) translation vector
        rotation: (3, 3) rotation matrix
        scale: (3,) scale vector
        filepath: Output file path
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    # Convert to serializable formats
    euler_deg = rotation_matrix_to_euler(rotation, degrees=True)
    quaternion = rotation_matrix_to_quaternion(rotation)

    # Compose full matrix
    full_matrix = compose_transform(translation, rotation, scale)

    params = {
        "translation": translation.tolist(),
        "rotation_matrix": rotation.tolist(),
        "rotation_euler_xyz_deg": euler_deg.tolist(),
        "rotation_quaternion_wxyz": quaternion.tolist(),
        "scale": scale.tolist(),
        "transform_matrix_4x4": full_matrix.tolist(),
        "note": "Transform to apply to COLMAP data to align with cube coordinate system"
    }

    with open(filepath, 'w') as f:
        json.dump(params, f, indent=2)

    print(f"[Export] Saved transformation parameters to {filepath}")


def load_transform_params(filepath: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load transformation parameters from JSON file.

    Args:
        filepath: Input file path

    Returns:
        translation: (3,) translation vector
        rotation: (3, 3) rotation matrix
        scale: (3,) scale vector
    """
    with open(filepath, 'r') as f:
        params = json.load(f)

    translation = np.array(params['translation'])
    rotation = np.array(params['rotation_matrix'])
    scale = np.array(params['scale'])

    return translation, rotation, scale


def apply_transform_to_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """
    Apply transformation matrix to a set of 3D points.

    Args:
        points: (N, 3) array of 3D points
        transform: 4x4 transformation matrix

    Returns:
        (N, 3) array of transformed points
    """
    # Convert to homogeneous coordinates
    points_homogeneous = np.hstack([points, np.ones((points.shape[0], 1))])

    # Apply transformation
    points_transformed = (transform @ points_homogeneous.T).T

    # Convert back to 3D
    return points_transformed[:, :3]


def interpolate_transforms(transform1: np.ndarray, transform2: np.ndarray, t: float) -> np.ndarray:
    """
    Interpolate between two transformation matrices.

    Args:
        transform1: First 4x4 transformation matrix
        transform2: Second 4x4 transformation matrix
        t: Interpolation parameter (0 = transform1, 1 = transform2)

    Returns:
        Interpolated 4x4 transformation matrix
    """
    # Decompose both transforms
    t1, r1, s1 = decompose_transform(transform1)
    t2, r2, s2 = decompose_transform(transform2)

    # Interpolate translation and scale linearly
    t_interp = (1 - t) * t1 + t * t2
    s_interp = (1 - t) * s1 + t * s2

    # Interpolate rotation using SLERP (via quaternions)
    q1 = rotation_matrix_to_quaternion(r1)
    q2 = rotation_matrix_to_quaternion(r2)

    # Convert back to scipy format [x, y, z, w]
    q1_scipy = [q1[1], q1[2], q1[3], q1[0]]
    q2_scipy = [q2[1], q2[2], q2[3], q2[0]]

    r1_scipy = R.from_quat(q1_scipy)
    r2_scipy = R.from_quat(q2_scipy)

    # SLERP
    key_times = [0, 1]
    key_rots = R.from_quat([q1_scipy, q2_scipy])
    slerp = R.from_quat(key_rots.as_quat())  # Simplified
    r_interp_scipy = R.from_rotvec((1 - t) * r1_scipy.as_rotvec() + t * r2_scipy.as_rotvec())
    r_interp = r_interp_scipy.as_matrix()

    # Compose interpolated transform
    return compose_transform(t_interp, r_interp, s_interp)


if __name__ == "__main__":
    # Test transformation operations
    print("Testing transformation math...\n")

    # Create a test transform
    translation = np.array([1.0, 2.0, 3.0])
    euler_angles = np.array([30, 45, 60])  # degrees
    scale = np.array([1.5, 2.0, 2.5])

    rotation = euler_to_rotation_matrix(euler_angles, degrees=True)
    transform = compose_transform(translation, rotation, scale)

    print("Original parameters:")
    print(f"  Translation: {translation}")
    print(f"  Euler (deg): {euler_angles}")
    print(f"  Scale: {scale}")
    print(f"\nComposed 4x4 matrix:\n{transform}\n")

    # Decompose
    t_dec, r_dec, s_dec = decompose_transform(transform)
    euler_dec = rotation_matrix_to_euler(r_dec, degrees=True)

    print("Decomposed parameters:")
    print(f"  Translation: {t_dec}")
    print(f"  Euler (deg): {euler_dec}")
    print(f"  Scale: {s_dec}")

    # Check accuracy
    print(f"\nTranslation error: {np.linalg.norm(translation - t_dec)}")
    print(f"Rotation error: {np.linalg.norm(rotation - r_dec)}")
    print(f"Scale error: {np.linalg.norm(scale - s_dec)}")

    # Test export
    print("\nTesting export...")
    export_transform_matrix(transform, "test_transform_matrix.txt")
    export_transform_params(translation, rotation, scale, "test_transform_params.json")
    print("Export test complete!")
