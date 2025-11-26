from skimage.transform import SimilarityTransform
import numpy as np


def decompose_transform(M):
    """
    Decompose a 4x4 transformation matrix into translation, rotation, and scale.

    Args:
        M: 4x4 transformation matrix (numpy array)
           M = [s*R | t]
               [0   | 1]

    Returns:
        translation: (3,) translation vector
        rotation: (3, 3) rotation matrix (without scale)
        scale: (3,) scale factors [sx, sy, sz]

    Note:
        For isotropic scaling (uniform in all directions), all scale components are equal.
        Local→Recon transform has scale ≈ 5.7 (1 meter Local = 5.7 units Recon)
    """
    # Extract upper-left 3x3 matrix (contains scale*rotation)
    upper_3x3 = M[:3, :3]

    # Extract translation (last column, first 3 rows)
    translation = M[:3, 3]

    # Compute scale factors (length of each column)
    scale = np.linalg.norm(upper_3x3, axis=0)

    # Extract rotation by removing scale
    rotation = upper_3x3 / scale

    return translation, rotation, scale


def get_scale_factor(M):
    """
    Extract the isotropic scale factor from a 4x4 transformation matrix.

    Args:
        M: 4x4 transformation matrix

    Returns:
        scale_factor: float, scale factor for Local→Recon transform
                     (approximately 5.7 for current setup)

    Note:
        Assumes isotropic scaling (same scale in all directions).
        If scaling is not isotropic, returns the average scale.
    """
    _, _, scale = decompose_transform(M)

    # Check if scaling is isotropic (within 1% tolerance)
    if not np.allclose(scale, scale[0], rtol=0.01):
        print(f"Warning: Non-isotropic scaling detected: {scale}")
        print(f"Using average scale: {np.mean(scale):.4f}")

    return float(np.mean(scale))

# 1. Source (Local World) - 4x3 행렬
src_points = np.array([
    [ 0.7, -0.05,  0.25], # v_local_1
    [-0.7, -0.05,  0.25], # v_local_2
    [-0.7, -0.05, -0.25], # v_local_3
    [ 0.7, -0.05, -0.25]  # v_local_4
])

# 2. Target (Recon World) - 4x3 행렬 (순서 중요!)
dst_points = np.array([
    [ 6.646002, 7.225370,  8.947520], # p_recon_v8 (v1과 대응)
    [-5.471739, 6.820036,  8.749257], # p_recon_v7 (v2와 대응)
    [-5.622741, 7.122706, 13.082649], # p_recon_v3 (v3와 대응)
    [ 6.495000, 7.528070, 13.280912]  # p_recon_v4 (v4와 대응)
])

# 3. 변환 계산
transform = SimilarityTransform()
transform.estimate(src_points, dst_points)

# 4. 결과 (4x4 행렬 M)
M = transform.params
print("4x4 변환 행렬 M:\n", M)

# 5. 결과 (스케일, 회전, 이동 분리)
print("Scale (s):", transform.scale)
print("Rotation (R):\n", transform.rotation)
print("Translation (t):", transform.translation)

# 6. Scale factor 추출 (유틸리티 함수 검증)
scale_factor = get_scale_factor(M)
print(f"\nScale factor (Local→Recon): {scale_factor:.4f}")
print(f"This means: 1 meter in Local = {scale_factor:.4f} units in Recon")
print(f"Or: 1 unit in Recon = {1/scale_factor:.4f} meters")