"""Camera coordinate transformation and synchronization manager."""

import copy
from typing import Optional
import numpy as np
import open3d as o3d
import torch

from renderer.data_types import CameraFrame


class CameraManager:
    """Manages camera transformations between Local and Reconstructed worlds."""

    def __init__(
        self,
        local_to_recon_transform: np.ndarray,
        near_plane: float = 0.03,
        far_plane: float = 10.0
    ):
        """
        Initialize camera manager.

        Args:
            local_to_recon_transform: 4x4 transformation matrix (Local → Reconstructed)
            near_plane: Near clipping plane
            far_plane: Far clipping plane
        """
        self.local_to_recon_transform = local_to_recon_transform
        self.recon_to_local_transform = np.linalg.inv(local_to_recon_transform)

        # Extract pure rotation matrices (scale removed)
        self.R_local_to_recon = self._extract_rotation_matrix(local_to_recon_transform)
        self.R_recon_to_local = self._extract_rotation_matrix(self.recon_to_local_transform)

        self.near_plane = near_plane
        self.far_plane = far_plane

        # Synchronization state
        self.last_local_camera: Optional[o3d.camera.PinholeCameraParameters] = None
        self.last_recon_camera: Optional[o3d.camera.PinholeCameraParameters] = None
        self.sync_lock = False

    def build_extrinsic_from_vectors(
        self,
        position: np.ndarray,
        view_direction: np.ndarray,
        up_vector: np.ndarray
    ) -> np.ndarray:
        """
        Build camera extrinsic matrix from position, view direction, and up vector.

        Args:
            position: Camera position in world coordinates
            view_direction: Camera view direction (normalized)
            up_vector: Camera up vector (normalized)

        Returns:
            4x4 extrinsic matrix [R|t; 0|1]
        """
        view_dir = view_direction / np.linalg.norm(view_direction)
        up = up_vector / np.linalg.norm(up_vector)

        z_axis = -view_dir
        x_axis = np.cross(up, z_axis)
        x_axis = x_axis / np.linalg.norm(x_axis)
        y_axis = np.cross(z_axis, x_axis)

        R = np.column_stack([x_axis, y_axis, z_axis])
        t = -R @ position

        extrinsic = np.eye(4)
        extrinsic[:3, :3] = R
        extrinsic[:3, 3] = t

        return extrinsic

    def _extract_rotation_matrix(self, transform_matrix: np.ndarray) -> np.ndarray:
        """Extract pure rotation matrix from transformation matrix (removes scale)."""
        SR = transform_matrix[:3, :3]

        scale_x = np.linalg.norm(SR[:, 0])
        scale_y = np.linalg.norm(SR[:, 1])
        scale_z = np.linalg.norm(SR[:, 2])

        R = SR.copy()
        R[:, 0] /= scale_x
        R[:, 1] /= scale_y
        R[:, 2] /= scale_z

        return R

    def camera_changed(
        self,
        cam1: o3d.camera.PinholeCameraParameters,
        cam2: Optional[o3d.camera.PinholeCameraParameters],
        threshold: float = 1e-5
    ) -> bool:
        """Check if camera parameters changed significantly."""
        if cam2 is None:
            return True

        R1 = cam1.extrinsic[:3, :3]
        t1 = cam1.extrinsic[:3, 3]
        pos1 = -R1.T @ t1
        view1 = R1.T @ np.array([0.0, 0.0, -1.0])
        up1 = R1.T @ np.array([0.0, 1.0, 0.0])

        R2 = cam2.extrinsic[:3, :3]
        t2 = cam2.extrinsic[:3, 3]
        pos2 = -R2.T @ t2
        view2 = R2.T @ np.array([0.0, 0.0, -1.0])
        up2 = R2.T @ np.array([0.0, 1.0, 0.0])

        position_diff = np.linalg.norm(pos1 - pos2)
        view_diff = np.linalg.norm(view1 - view2)
        up_diff = np.linalg.norm(up1 - up2)

        return position_diff > threshold or view_diff > threshold or up_diff > threshold

    def transform_camera_local_to_recon(
        self,
        local_cam: o3d.camera.PinholeCameraParameters
    ) -> o3d.camera.PinholeCameraParameters:
        """Transform Local World camera to Reconstructed World coordinates."""
        recon_cam = copy.deepcopy(local_cam)

        R_local = local_cam.extrinsic[:3, :3]
        t_local = local_cam.extrinsic[:3, 3]

        local_pos = -R_local.T @ t_local
        local_view_dir = R_local.T @ np.array([0.0, 0.0, -1.0])
        local_up = R_local.T @ np.array([0.0, 1.0, 0.0])

        recon_pos = (self.local_to_recon_transform @ np.append(local_pos, 1.0))[:3]
        recon_view_dir = self.R_local_to_recon @ local_view_dir
        recon_up = self.R_local_to_recon @ local_up

        recon_cam.extrinsic = self.build_extrinsic_from_vectors(
            recon_pos, recon_view_dir, recon_up
        )

        return recon_cam

    def apply_camera(self, view_control, extrinsic: np.ndarray, intrinsic_matrix=None):
        """Apply camera using complete PinholeCameraParameters."""
        camera_params = o3d.camera.PinholeCameraParameters()
        camera_params.extrinsic = extrinsic

        if intrinsic_matrix is None:
            current_params = view_control.convert_to_pinhole_camera_parameters()
            camera_params.intrinsic = current_params.intrinsic
        else:
            camera_params.intrinsic = intrinsic_matrix

        view_control.convert_from_pinhole_camera_parameters(camera_params)

        # Re-apply constant clipping planes after camera update
        view_control.set_constant_z_near(self.near_plane)
        view_control.set_constant_z_far(self.far_plane)

    def sync_cameras(self, local_vis, recon_vis):
        """Synchronize cameras between Local and Reconstructed worlds."""
        if self.sync_lock:
            return

        local_cam = local_vis.get_view_control().convert_to_pinhole_camera_parameters()
        recon_cam = recon_vis.get_view_control().convert_to_pinhole_camera_parameters()

        local_changed = self.camera_changed(local_cam, self.last_local_camera)
        recon_changed = self.camera_changed(recon_cam, self.last_recon_camera)

        if local_changed and not recon_changed:
            self.sync_lock = True
            transformed_extrinsic = local_cam.extrinsic @ self.recon_to_local_transform
            self.apply_camera(recon_vis.get_view_control(), transformed_extrinsic)
            self.last_local_camera = copy.deepcopy(local_cam)
            self.last_recon_camera = recon_vis.get_view_control().convert_to_pinhole_camera_parameters()
            self.sync_lock = False
        elif local_changed and recon_changed:
            self.last_local_camera = copy.deepcopy(local_cam)
            self.last_recon_camera = copy.deepcopy(recon_cam)

    def get_cam0_view_local(
        self,
        cameras_list: list,
        reference_cam_id: int,
        width: int,
        height: int,
        camera_width: int,
        camera_height: int,
        fx: float,
        fy: float
    ) -> Optional[o3d.camera.PinholeCameraParameters]:
        """Get cam0 camera parameters in Local World coordinates."""
        cam0 = None
        for cam in cameras_list:
            if cam.get('id', -1) == reference_cam_id:
                cam0 = cam
                break

        if cam0 is None:
            return None

        position = np.array(cam0['position'])
        rotation = np.array(cam0['rotation'])

        scale_x = width / camera_width
        scale_y = height / camera_height
        fx_scaled = fx * scale_x
        fy_scaled = fy * scale_y
        cx = width / 2.0
        cy = height / 2.0

        t_recon = -rotation @ position
        extrinsic_recon = np.eye(4)
        extrinsic_recon[:3, :3] = rotation
        extrinsic_recon[:3, 3] = t_recon

        extrinsic_local = extrinsic_recon @ self.local_to_recon_transform

        intrinsic_matrix = np.array([
            [fx_scaled, 0, cx],
            [0, fy_scaled, cy],
            [0, 0, 1]
        ])

        camera_params = o3d.camera.PinholeCameraParameters()
        camera_params.extrinsic = extrinsic_local
        camera_params.intrinsic = o3d.camera.PinholeCameraIntrinsic(
            width, height, intrinsic_matrix
        )

        return camera_params

    def setup_cam0_camera(
        self,
        recon_vis,
        cameras_list: list,
        reference_cam_id: int,
        width: int,
        height: int,
        camera_width: int,
        camera_height: int,
        fx: float,
        fy: float
    ):
        """Set Reconstructed world camera to cam0 view."""
        cam0 = None
        for cam in cameras_list:
            if cam.get('id', -1) == reference_cam_id:
                cam0 = cam
                break

        if cam0 is None:
            cam0 = cameras_list[0]

        position = np.array(cam0['position'])
        rotation = np.array(cam0['rotation'])

        scale_x = width / camera_width
        scale_y = height / camera_height
        fx_scaled = fx * scale_x
        fy_scaled = fy * scale_y
        cx = width / 2.0
        cy = height / 2.0

        t = -rotation @ position
        extrinsic = np.eye(4)
        extrinsic[:3, :3] = rotation
        extrinsic[:3, 3] = t

        R = extrinsic[:3, :3]
        t_vec = extrinsic[:3, 3]
        camera_position = -R.T @ t_vec
        view_direction = R.T @ np.array([0.0, 0.0, -1.0])
        up_vector = np.array([0, -1, 0])

        view_control = recon_vis.get_view_control()
        view_control.set_lookat([0.0, 0.0, 0.0])
        view_control.set_front(view_direction)
        view_control.set_up(up_vector)

    def extract_camera_position(self, extrinsic: np.ndarray) -> np.ndarray:
        """Extract camera position from extrinsic matrix."""
        R = extrinsic[:3, :3]
        t = extrinsic[:3, 3]
        return -R.T @ t

    def to_camera_frame(
        self,
        o3d_camera: o3d.camera.PinholeCameraParameters,
        current_frame_id: int,
        available_frame_ids: list
    ) -> CameraFrame:
        """Convert Open3D camera parameters to CameraFrame."""
        extrinsic = o3d_camera.extrinsic
        view_matrix = extrinsic.astype(np.float32)

        intrinsic = o3d_camera.intrinsic
        fx = intrinsic.intrinsic_matrix[0, 0]
        fy = intrinsic.intrinsic_matrix[1, 1]
        cx = intrinsic.intrinsic_matrix[0, 2]
        cy = intrinsic.intrinsic_matrix[1, 2]

        intrinsics_matrix = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ], dtype=np.float32)

        view_matrix_torch = torch.from_numpy(view_matrix).cuda()
        intrinsics_torch = torch.from_numpy(intrinsics_matrix).cuda()

        time_index = 0.0
        if available_frame_ids and len(available_frame_ids) > 1:
            frame_idx = available_frame_ids.index(current_frame_id)
            time_index = frame_idx / (len(available_frame_ids) - 1)

        camera_frame = CameraFrame(
            view_matrix=view_matrix_torch,
            intrinsics=intrinsics_torch,
            width=intrinsic.width,
            height=intrinsic.height,
            near=self.near_plane,
            far=self.far_plane,
            time_index=time_index,
            frame_id=current_frame_id
        )

        return camera_frame
