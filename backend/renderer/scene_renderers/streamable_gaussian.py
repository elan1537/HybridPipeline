"""
Streamable Gaussian Splatting Renderer (3DGStream)
"""

import sys
import os
from typing import Dict, Any, List, Optional
import torch
import numpy as np
from tqdm import tqdm
from random import randint

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from renderer.data_types import CameraFrame, RenderOutput
from renderer.scene_renderers.base import BaseSceneRenderer
from renderer.scene_renderers.gaussian_state import FastGaussianState
from renderer.scene_renderers.optimization_params import SimplePipelineParams, OptimizationParams

# Add 3DGStream module path (Docker: /workspace/research/3DGStream)
stream_path = "/workspace/research/3DGStream"
sys.path.insert(0, stream_path)

# Import 3DGStream modules
from stream3d import TemporalGaussianModel, NeuralTransformationCache
from gaussian_renderer import render
from utils.loss_utils import l1_loss, ssim
from utils.image_utils import psnr
import tinycudann as tcnn
import commentjson as ctjs


class StreamableGaussian(BaseSceneRenderer):
    """
    3DGStream Renderer using Neural Transformation Cache (NTC).
    """

    def __init__(self):
        self.gaussians_state = FastGaussianState()
        self.gaussians: Optional[TemporalGaussianModel] = None
        self.ntc: Optional[NeuralTransformationCache] = None
        self.ntc_optimizer = None
        self.opt: Optional[OptimizationParams] = None
        self.pipe: Optional[SimplePipelineParams] = None
        self.background = None
        self.dataset_path: str = ""
        self.current_frame_idx: int = 0
        self.iterations_s1: int = 50
        self.iterations_s2: int = 50
        self.initialized: bool = False

    async def on_init(self, config: Dict[str, Any]) -> bool:
        """
        Initialize the renderer.

        Args:
            config: Configuration dictionary with keys:
                - model_path: Initial Gaussian PLY file (required)
                - ntc_path: NTC weight path (required)
                - ntc_config: NTC config JSON (required)
                - dataset_path: COLMAP dataset path (required)
                - weight_path: Pre-trained weights directory (optional, for inference-only)
                - iterations_s1: Stage 1 iterations (optional, default: 50)
                - iterations_s2: Stage 2 iterations (optional, default: 50)

        Returns:
            True if initialization succeeded
        """
        self._validate_config(config, ['model_path', 'ntc_path', 'ntc_config', 'dataset_path'])

        model_path = config['model_path']
        ntc_path = config['ntc_path']
        ntc_config_path = config['ntc_config']
        self.dataset_path = config['dataset_path']
        weight_path = config.get('weight_path', None)
        self.iterations_s1 = config.get('iterations_s1', 50)
        self.iterations_s2 = config.get('iterations_s2', 50)

        print("[INIT] StreamableGaussian Renderer Initialization")
        print(f"[INIT] Model: {model_path}")
        print(f"[INIT] NTC: {ntc_path}")
        print(f"[INIT] Dataset: {self.dataset_path}")
        if weight_path:
            print(f"[INIT] Weight Path: {weight_path} (inference-only mode)")
        print(f"[INIT] Iterations: S1={self.iterations_s1}, S2={self.iterations_s2}")

        # Setup optimization parameters
        self.opt = OptimizationParams()
        self.opt.ntc_path = ntc_path
        self.opt.ntc_conf_path = ntc_config_path
        self.opt.batch_size = 1
        self.opt.densification_interval = 100
        self.opt.densify_from_iter = 5

        # Setup pipeline parameters
        self.pipe = SimplePipelineParams()
        self.background = torch.tensor([0, 0, 0], dtype=torch.float32).cuda()

        # Load NTC configuration
        print("[INIT] Loading NTC configuration...")
        try:
            with open(ntc_config_path) as f:
                ntc_conf = ctjs.load(f)
        except Exception as e:
            print(f"[ERROR] Failed to load NTC config: {e}")
            return False

        # Load initial Gaussian model
        print(f"[INIT] Loading initial Gaussian model...")
        try:
            self.gaussians = TemporalGaussianModel(sh_degree=1, rotate_sh=False)
            self.gaussians.load_ply(model_path)
            self.gaussians_state.save_state(0, self.gaussians)
            print(f"[INIT] Loaded {self.gaussians.get_xyz.shape[0]} Gaussians (frame 0)")
        except Exception as e:
            print(f"[ERROR] Failed to load Gaussian model: {e}")
            return False

        # Load pre-trained weights if provided (inference-only mode)
        if weight_path:
            loaded_frames = self._load_weights_from_directory(weight_path)
            if loaded_frames > 0:
                print(f"[INIT] Loaded {loaded_frames} pre-trained frames from {weight_path}")
            else:
                print(f"[WARNING] No PLY files found in {weight_path}")

        # Setup NTC model
        try:
            model = tcnn.NetworkWithInputEncoding(
                n_input_dims=3,
                n_output_dims=8,
                encoding_config=ntc_conf["encoding"],
                network_config=ntc_conf["network"]
            ).to(torch.device("cuda"))

            xyz_bound_min, xyz_bound_max = self.gaussians.get_xyz_bound(86.6)
            self.ntc = NeuralTransformationCache(model, xyz_bound_min, xyz_bound_max)
            self.ntc.load_state_dict(torch.load(ntc_path))

            ntc_lr = ntc_conf.get("optimizer", {}).get("learning_rate", 0.002)
            self.ntc_optimizer = torch.optim.Adam(self.ntc.parameters(), lr=ntc_lr)

            print("[INIT] NTC model loaded successfully")
        except Exception as e:
            print(f"[ERROR] Failed to setup NTC: {e}")
            return False

        self.initialized = True
        print("[INIT] Initialization completed successfully")
        return True

    def _load_weights_from_directory(self, weight_path: str) -> int:
        """
        Load pre-trained PLY files from a directory.

        Args:
            weight_path: Directory containing PLY files (e.g., temp_states/)

        Returns:
            Number of frames loaded
        """
        import glob

        if not os.path.exists(weight_path):
            print(f"[WARNING] Weight path does not exist: {weight_path}")
            return 0

        # Find all PLY files matching pattern: frame_XXXXXX.ply
        ply_files = sorted(glob.glob(os.path.join(weight_path, "frame_*.ply")))

        if not ply_files:
            print(f"[WARNING] No PLY files found in {weight_path}")
            return 0

        print(f"[INIT] Loading {len(ply_files)} pre-trained frames...")
        loaded_count = 0

        for ply_path in ply_files:
            # Extract frame_id from filename: frame_000001.ply -> 1
            basename = os.path.basename(ply_path)
            try:
                frame_id = int(basename.replace("frame_", "").replace(".ply", ""))
            except ValueError:
                print(f"[WARNING] Invalid PLY filename format: {basename}")
                continue

            # Register PLY path in gaussian_state
            self.gaussians_state.ply_states[frame_id] = ply_path
            loaded_count += 1
            print(f"[INIT] Registered frame {frame_id}: {ply_path}")

        return loaded_count

    async def train(self, cameras: List, frame_id: int, **kwargs) -> Dict[str, Any]:
        """
        Train one frame using 3DGStream feed-forward pipeline.

        This method performs 2-stage training:
        - Stage 1: Neural Transformation Cache (NTC) training
        - Stage 2: Gaussian refinement with densification

        Multi-view training uses all provided cameras.
        Training results are automatically saved to memory and PLY files.

        Args:
            cameras: List of 3DGStream Camera objects (from setup_frame_data)
            frame_id: Frame index for state management
            **kwargs:
                - iterations_s1: Stage 1 iterations (default: self.iterations_s1)
                - iterations_s2: Stage 2 iterations (default: self.iterations_s2)

        Returns:
            Dictionary containing:
                - psnr: PSNR metric
                - losses_s1: Stage 1 loss history
                - losses_s2: Stage 2 loss history
                - num_gaussians: Number of Gaussians after training
        """
        if not self.initialized:
            raise RuntimeError("Renderer not initialized. Call on_init() first.")

        if frame_id is None or frame_id < 1:
            raise ValueError("frame_id must be a positive integer")

        if not cameras or len(cameras) == 0:
            raise ValueError("cameras list cannot be empty")

        iterations_s1 = kwargs.get('iterations_s1', self.iterations_s1)
        iterations_s2 = kwargs.get('iterations_s2', self.iterations_s2)

        print(f"\n[TRAIN] Training frame {frame_id:06d} with {len(cameras)} cameras")

        # Load previous frame state (for frame > 1)
        if frame_id > 1:
            fresh_gaussians = self.gaussians_state.create_fresh_gaussian_with_ply(
                frame_id - 1,
                sh_degree=1,
                rotate_sh=False,
                gaussian_class=TemporalGaussianModel
            )
            if fresh_gaussians is not None:
                self.gaussians = fresh_gaussians
                print(f"[TRAIN] Using fresh Gaussian from frame {frame_id-1}")

        # Reset NTC bounds and optimizer
        xyz_bound_min, xyz_bound_max = self.gaussians.get_xyz_bound(86.6)
        self.ntc.xyz_bound_min = xyz_bound_min
        self.ntc.xyz_bound_max = xyz_bound_max

        # Create fresh NTC optimizer
        ntc_lr = 0.002
        self.ntc_optimizer = torch.optim.Adam(self.ntc.parameters(), lr=ntc_lr)

        # Stage 1: NTC Training
        s1_losses = await self._train_stage1(cameras, frame_id, iterations_s1)

        # Stage 2: Gaussian Refinement
        s2_losses, psnr_val = await self._train_stage2(cameras, frame_id, iterations_s2)

        # Save current frame state
        self.gaussians_state.save_state(frame_id, self.gaussians)

        # Memory cleanup (keep PLY files for all frames)
        if frame_id % 3 == 0:
            self.gaussians_state.clear_old_states(keep_last_n=2, keep_ply=True)
            torch.cuda.empty_cache()

        print(f"[TRAIN] Frame {frame_id:06d} completed: {self.gaussians.get_xyz.shape[0]} gaussians, PSNR={psnr_val:.2f}")

        return {
            'psnr': psnr_val,
            'losses_s1': s1_losses,
            'losses_s2': s2_losses,
            'num_gaussians': self.gaussians.get_xyz.shape[0]
        }

    async def render(self, camera: CameraFrame) -> RenderOutput:
        """
        Pure inference rendering using current or saved Gaussian state.

        This method performs rendering only (no training). If the current frame's
        Gaussian state is not in memory, it will be loaded from PLY files.

        Args:
            camera: Camera parameters for rendering

        Returns:
            RenderOutput with color, depth, and alpha

        Raises:
            RuntimeError: If renderer not initialized or state unavailable
        """
        if not self.initialized:
            raise RuntimeError("Renderer not initialized. Call on_init() first.")

        frame_idx = camera.frame_id
        if frame_idx is None:
            frame_idx = 1

        # Ensure Gaussian state is loaded
        if frame_idx not in self.gaussians_state.states and frame_idx not in self.gaussians_state.ply_states:
            raise RuntimeError(
                f"No Gaussian state found for frame {frame_idx}. "
                f"Call train() first or ensure PLY file exists."
            )

        # Load from PLY if not in memory
        if self.gaussians is None or (hasattr(self, 'current_frame_idx') and self.current_frame_idx != frame_idx):
            fresh_gaussians = self.gaussians_state.create_fresh_gaussian_with_ply(
                frame_idx,
                sh_degree=1,
                rotate_sh=False,
                gaussian_class=TemporalGaussianModel
            )
            if fresh_gaussians is None:
                raise RuntimeError(f"Failed to load Gaussian state for frame {frame_idx}")
            self.gaussians = fresh_gaussians
            self.current_frame_idx = frame_idx
            print(f"[RENDER] Loaded Gaussian state for frame {frame_idx}")

        # Convert CameraFrame to 3DGStream Camera
        stream_camera = self._convert_to_stream_camera(camera)

        # Pure inference
        print(f"[RENDER] Rendering frame {frame_idx:06d}...")
        with torch.no_grad():
            render_pkg = render(stream_camera, self.gaussians, self.pipe, self.background)

        # Extract outputs
        image = render_pkg["render"]
        depth = render_pkg.get("depth")
        alpha = render_pkg.get("alpha", torch.ones_like(depth))

        # Convert to (H, W, C) format
        color = image.permute(1, 2, 0)
        depth = depth.squeeze(0)
        alpha = alpha.squeeze(0)

        output = RenderOutput(
            color=color,
            depth=depth,
            alpha=alpha,
            metadata={'renderer': '3dgstream', 'frame_id': frame_idx}
        )
        output.validate()

        return output

    def _convert_to_stream_camera(self, camera: CameraFrame):
        """
        Convert CameraFrame to 3DGStream Camera object.

        Args:
            camera: CameraFrame with view_matrix and intrinsics

        Returns:
            A camera object compatible with 3DGStream render()
        """
        # Extract intrinsic parameters
        K = camera.intrinsics.cpu().numpy()
        fx = K[0, 0]
        fy = K[1, 1]

        # Convert to FoV (radians)
        FoVx = 2.0 * np.arctan(camera.width / (2.0 * fx))
        FoVy = 2.0 * np.arctan(camera.height / (2.0 * fy))

        # Get view matrix
        view_matrix = camera.view_matrix.cuda()

        # Compute projection matrix (same as 3DGStream)
        from utils.graphics_utils import getProjectionMatrix
        projection_matrix = getProjectionMatrix(
            znear=camera.near,
            zfar=camera.far,
            fovX=FoVx,
            fovY=FoVy
        ).transpose(0, 1).cuda()

        # Compute full projection transform
        full_proj_transform = (
            view_matrix.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))
        ).squeeze(0)

        # Compute camera center (critical for correct rendering!)
        view_inv = torch.inverse(view_matrix)
        camera_center = view_inv[3, :3]

        # Create camera object matching 3DGStream Camera class
        class SimpleCameraWrapper:
            def __init__(self, world_view_transform, full_proj_transform, camera_center,
                        width, height, FoVx, FoVy, znear, zfar):
                self.world_view_transform = world_view_transform
                self.full_proj_transform = full_proj_transform
                self.camera_center = camera_center
                self.image_width = width
                self.image_height = height
                self.FoVx = FoVx
                self.FoVy = FoVy
                self.znear = znear
                self.zfar = zfar
                self.original_image = None  # For inference, no GT needed

        return SimpleCameraWrapper(
            world_view_transform=view_matrix,
            full_proj_transform=full_proj_transform,
            camera_center=camera_center,
            width=camera.width,
            height=camera.height,
            FoVx=FoVx,
            FoVy=FoVy,
            znear=camera.near,
            zfar=camera.far
        )

    async def _train_stage1(
        self,
        train_cameras: List,
        frame_idx: int,
        iterations_s1: int
    ) -> List[float]:
        """Stage 1: NTC Training."""
        print(f"[TRAIN] Stage 1: NTC Training ({iterations_s1} iterations)")

        # Setup NTC and Gaussian training
        xyz_bound_min, xyz_bound_max = self.gaussians.get_xyz_bound(86.6)
        self.ntc.xyz_bound_min = xyz_bound_min
        self.ntc.xyz_bound_max = xyz_bound_max

        self.gaussians.ntc = self.ntc
        self.gaussians.ntc_optimizer = self.ntc_optimizer

        self.opt.iterations = iterations_s1
        self.gaussians.training_one_frame_setup(self.opt)

        print(f"[TRAIN] Ready: {self.gaussians.get_xyz.shape[0]} gaussians")

        lambda_dssim = 0.2
        s1_losses = []
        progress_bar = tqdm(range(iterations_s1), desc="Stage 1")

        for iteration in range(iterations_s1):
            self.gaussians.query_ntc()
            total_loss = torch.tensor(0.).cuda()

            for batch_iter in range(self.opt.batch_size):
                camera = train_cameras[randint(0, len(train_cameras) - 1)]
                render_pkg = render(camera, self.gaussians, self.pipe, self.background)
                image = render_pkg["render"]
                last_visibility_filter = render_pkg["visibility_filter"]
                last_radii = render_pkg["radii"]
                last_viewspace_points = render_pkg["viewspace_points"]

                gt_image = camera.original_image
                Ll1 = l1_loss(image, gt_image)
                loss = (1.0 - lambda_dssim) * Ll1 + lambda_dssim * (1.0 - ssim(image, gt_image))
                total_loss += loss

            total_loss /= self.opt.batch_size
            total_loss.backward()

            with torch.no_grad():
                if iteration % 10 == 0:
                    progress_bar.set_postfix({"Loss": f"{total_loss.item():.6f}"})

                if iteration > self.opt.densify_from_iter:
                    self.gaussians.max_radii2D[last_visibility_filter] = torch.max(
                        self.gaussians.max_radii2D[last_visibility_filter],
                        last_radii[last_visibility_filter]
                    )
                    self.gaussians.add_densification_stats(last_viewspace_points, last_visibility_filter)

                if iteration < iterations_s1:
                    self.gaussians.ntc_optimizer.step()
                    self.gaussians.ntc_optimizer.zero_grad(set_to_none=True)

            s1_losses.append(total_loss.item())
            progress_bar.update(1)

        progress_bar.close()

        # Apply NTC transformations permanently
        print("[TRAIN] Applying NTC transformations permanently...")
        self.gaussians.update_by_ntc()

        print("[TRAIN] Setting up Stage 2 with added points...")
        self.gaussians.training_one_frame_s2_setup(self.opt)

        return s1_losses

    async def _train_stage2(
        self,
        train_cameras: List,
        frame_idx: int,
        iterations_s2: int
    ) -> tuple:
        """Stage 2: Gaussian Refinement. Returns (losses, psnr)."""
        print(f"[TRAIN] Stage 2: Added Points Training ({iterations_s2} iterations)")
        added_count = self.gaussians._added_xyz.shape[0] if hasattr(self.gaussians, '_added_xyz') else 0
        print(f"[TRAIN] Added points: {added_count}")

        lambda_dssim = 0.2
        s2_losses = []
        progress_bar = tqdm(total=self.opt.iterations + iterations_s2, desc="Stage 2")
        ema_loss_for_log = 0.0

        for iteration in range(self.opt.iterations + 1, self.opt.iterations + iterations_s2 + 1):
            total_loss = torch.tensor(0.).cuda()

            for batch_iter in range(self.opt.batch_size):
                camera = train_cameras[randint(0, len(train_cameras) - 1)]
                render_pkg = render(camera, self.gaussians, self.pipe, self.background)
                image = render_pkg["render"]

                last_visibility_filter = render_pkg["visibility_filter"]
                last_radii = render_pkg["radii"]
                last_viewspace_points = render_pkg["viewspace_points"]

                gt_image = camera.original_image
                Ll1 = l1_loss(image, gt_image)
                loss = (1.0 - lambda_dssim) * Ll1 + lambda_dssim * (1.0 - ssim(image, gt_image))
                total_loss += loss

            total_loss /= self.opt.batch_size
            total_loss.backward()

            with torch.no_grad():
                ema_loss_for_log = 0.4 * total_loss.item() + 0.6 * ema_loss_for_log

                if (iteration - self.opt.iterations) % 10 == 0:
                    progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.6f}"})
                    progress_bar.update(10)

                if (iteration - self.opt.iterations) % self.opt.densification_interval == 0:
                    self.gaussians.adding_and_prune(self.opt, 0.08)

                if iteration < self.opt.iterations + iterations_s2:
                    self.gaussians.optimizer.step()
                    self.gaussians.optimizer.zero_grad(set_to_none=True)

            s2_losses.append(total_loss.item())
            progress_bar.update(1)

        progress_bar.close()

        # Final evaluation
        print("[TRAIN] Final evaluation...")
        final_camera = train_cameras[0]
        with torch.no_grad():
            render_pkg = render(final_camera, self.gaussians, self.pipe, self.background)
            image = render_pkg["render"]
            gt_image = final_camera.original_image
            psnr_val = psnr(image, gt_image).mean().item()

        return s2_losses, psnr_val

    async def on_cleanup(self) -> None:
        """Clean up GPU resources."""
        if self.initialized:
            print("[CLEANUP] Cleaning up renderer resources...")

            if self.gaussians is not None:
                del self.gaussians

            if self.ntc is not None:
                del self.ntc

            if self.ntc_optimizer is not None:
                del self.ntc_optimizer

            torch.cuda.empty_cache()

            if os.path.exists("temp_states"):
                import shutil
                shutil.rmtree("temp_states")
                print("[CLEANUP] Removed temp_states directory")

            self.initialized = False
            print("[CLEANUP] Cleanup complete")