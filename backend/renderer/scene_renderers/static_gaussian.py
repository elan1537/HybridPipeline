"""
Static Gaussian Splatting Renderer using gsplat library.

This renderer loads a PLY file containing 3D Gaussians and renders them
using the gsplat library's rasterization function.
"""

from typing import Dict, Any, Optional
import torch
import numpy as np
from plyfile import PlyData

from renderer.data_types import CameraFrame, RenderOutput
from renderer.scene_renderers.base import BaseSceneRenderer

try:
    from gsplat import rasterization
except ImportError:
    print("Warning: gsplat not installed. Install with: pip install gsplat")
    rasterization = None


class GsplatRenderer(BaseSceneRenderer):
    """
    3D Gaussian Splatting renderer using gsplat library.

    This renderer loads Gaussian parameters from a PLY file and renders
    them using gsplat's CUDA rasterization kernel.
    """

    def __init__(self):
        """Initialize the renderer."""
        self.means_gpu: Optional[torch.Tensor] = None
        self.quats_gpu: Optional[torch.Tensor] = None
        self.scales_gpu: Optional[torch.Tensor] = None
        self.ops_gpu: Optional[torch.Tensor] = None
        self.shs_gpu: Optional[torch.Tensor] = None
        self.device: str = "cuda"
        self.sh_degree: int = 3
        self.initialized: bool = False

    async def on_init(self, config: Dict[str, Any]) -> bool:
        """
        Initialize the renderer and load the scene.

        Args:
            config: Configuration containing:
                - model_path: Path to PLY file (required)
                - device: Device ('cuda' or 'cuda:0', default: 'cuda')
                - sh_degree: Spherical harmonics degree (default: 3)
                - gaussian_scale: Scale factor for Gaussians (default: 1.0)

        Returns:
            True if successful
        """
        self._validate_config(config, ['model_path'])

        model_path = config['model_path']
        self.device = config.get('device', 'cuda')
        self.sh_degree = config.get('sh_degree', 3)
        gaussian_scale = config.get('gaussian_scale', 1.0)

        print(f"Loading Gaussian scene from: {model_path}")

        # Load PLY file
        try:
            ply_data = PlyData.read(model_path)
            vertex_data = ply_data['vertex']
        except Exception as e:
            print(f"Error loading PLY file: {e}")
            return False

        # Extract Gaussian parameters
        means, quats, scales, opacities, shs = self._load_gaussian_params(vertex_data)

        # Apply scale factor if specified
        if gaussian_scale != 1.0:
            print(f"Applying Gaussian scale factor: {gaussian_scale}")
            scales = scales * gaussian_scale

        # Upload to GPU
        print(f"Uploading {means.shape[0]} Gaussians to GPU...")
        self.means_gpu = torch.from_numpy(means).to(self.device)
        self.quats_gpu = torch.from_numpy(quats).to(self.device)
        self.scales_gpu = torch.exp(torch.from_numpy(scales)).to(self.device)
        # Reshape opacities from (N, 1) to (N,) as required by gsplat
        self.ops_gpu = torch.sigmoid(torch.from_numpy(opacities).reshape(-1)).to(self.device)
        self.shs_gpu = torch.from_numpy(shs).to(self.device)

        self.initialized = True
        print("Gaussian scene loaded successfully")
        return True

    def _load_gaussian_params(self, vertex_data) -> tuple:
        """
        Extract Gaussian parameters from PLY vertex data.

        Args:
            vertex_data: PLY vertex element

        Returns:
            Tuple of (means, quats, scales, opacities, shs)
        """
        # Positions (xyz)
        means = np.vstack([
            vertex_data['x'],
            vertex_data['y'],
            vertex_data['z']
        ]).T.astype(np.float32)

        # Rotations (quaternions)
        quats = np.vstack([
            vertex_data['rot_0'],
            vertex_data['rot_1'],
            vertex_data['rot_2'],
            vertex_data['rot_3']
        ]).T.astype(np.float32)

        # Scales (log space)
        scales = np.vstack([
            vertex_data['scale_0'],
            vertex_data['scale_1'],
            vertex_data['scale_2']
        ]).T.astype(np.float32)

        # Opacities (logit space)
        opacities = np.array([vertex_data['opacity']]).T.astype(np.float32)

        # Spherical harmonics coefficients
        # DC component (0-th degree)
        shs_dc = np.vstack([
            vertex_data['f_dc_0'],
            vertex_data['f_dc_1'],
            vertex_data['f_dc_2']
        ]).T.astype(np.float32)
        shs = np.expand_dims(shs_dc, axis=1)  # (N, 1, 3)

        # Higher-order SH coefficients (1st, 2nd, 3rd degree)
        for i in range(15):
            sh_r = vertex_data[f'f_rest_{i}']
            sh_g = vertex_data[f'f_rest_{15 + i}']
            sh_b = vertex_data[f'f_rest_{30 + i}']
            sh_coeff = np.vstack([sh_r, sh_g, sh_b]).T
            sh_coeff = np.expand_dims(sh_coeff, axis=1)  # (N, 1, 3)
            shs = np.concatenate([shs, sh_coeff], axis=1)

        return means, quats, scales, opacities, shs

    async def render(self, camera: CameraFrame) -> RenderOutput:
        """
        Render the scene from the given camera.

        Args:
            camera: Camera parameters

        Returns:
            RenderOutput with color, depth, and alpha

        Raises:
            RuntimeError: If renderer not initialized
        """
        if not self.initialized:
            raise RuntimeError("Renderer not initialized. Call on_init() first.")

        if rasterization is None:
            raise RuntimeError("gsplat library not available")

        # Move camera matrices to GPU
        viewmats = camera.view_matrix.to(self.device)
        if viewmats.ndim == 2:
            viewmats = viewmats.unsqueeze(0)  # (4, 4) -> (1, 4, 4)

        Ks = camera.intrinsics.to(self.device)
        if Ks.ndim == 2:
            Ks = Ks.unsqueeze(0)  # (3, 3) -> (1, 3, 3)

        # Render using gsplat
        render_colors, render_alphas, _ = rasterization(
            means=self.means_gpu,
            quats=self.quats_gpu,
            scales=self.scales_gpu,
            opacities=self.ops_gpu,
            colors=self.shs_gpu,
            viewmats=viewmats,
            Ks=Ks,
            width=camera.width,
            height=camera.height,
            near_plane=camera.near,
            far_plane=camera.far,
            packed=False,
            radius_clip=0.1,
            sh_degree=self.sh_degree,
            eps2d=0.3,
            render_mode="RGB+D",
            rasterize_mode="antialiased",
            camera_model="pinhole"
        )

        # Extract outputs (batch index 0)
        color = render_colors[0, :, :, :3]  # (H, W, 3)
        depth = render_colors[0, :, :, -1]  # (H, W)
        alpha = render_alphas[0, :, :, 0]   # (H, W)

        output = RenderOutput(
            color=color,
            depth=depth,
            alpha=alpha,
            metadata={'sh_degree': self.sh_degree}
        )
        output.validate()

        return output

    async def on_cleanup(self) -> None:
        """Clean up GPU resources."""
        if self.initialized:
            del self.means_gpu, self.quats_gpu, self.scales_gpu, self.ops_gpu, self.shs_gpu
            torch.cuda.empty_cache()
            self.initialized = False
            print("Renderer cleanup complete")
