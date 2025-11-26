"""Neural rendering and depth fusion coordinator."""

import asyncio
from pathlib import Path
from typing import Optional
import numpy as np
import cv2
import open3d as o3d
import json
from tqdm import tqdm

from renderer.scene_renderers.streamable_gaussian import StreamableGaussian
from renderer.data_types import RenderOutput
from collision_detector import DepthCollisionDetector
from transform_to_matrix import get_scale_factor


class RenderingCoordinator:
    """Coordinates neural rendering, depth fusion, and frame saving."""

    def __init__(
        self,
        renderer: StreamableGaussian,
        output_dir: Path,
        near_plane: float = 0.03,
        far_plane: float = 10.0,
        enable_debug: bool = False,
        local_to_recon_transform: Optional[np.ndarray] = None,
        ground_y_local: float = 0.0
    ):
        """
        Initialize rendering coordinator.

        Args:
            renderer: StreamableGaussian renderer instance
            output_dir: Output directory for saved frames
            near_plane: Near clipping plane
            far_plane: Far clipping plane
            enable_debug: Enable alpha/depth debug visualization
            local_to_recon_transform: Transformation matrix (Local → Recon)
                                     If provided, will calculate ground_y in Recon space
            ground_y_local: Ground plane Y coordinate in Local space (default: -1.0m)
                           Will be transformed to Recon space if transform is provided
        """
        self.renderer = renderer
        self.near_plane = near_plane
        self.far_plane = far_plane
        self.enable_debug = enable_debug

        # Calculate ground_y in Recon space
        if local_to_recon_transform is not None:
            scale_factor = get_scale_factor(local_to_recon_transform)
            # Transform ground plane from Local to Recon space
            # Simple scaling for Y coordinate (assumes Y-axis aligned)
            ground_y_recon = ground_y_local * scale_factor
            print(f"[RENDERING_COORDINATOR] Ground plane: {ground_y_local:.2f}m Local → {ground_y_recon:.2f} Recon units (scale={scale_factor:.2f})")
        else:
            # Fallback: assume no scaling
            ground_y_recon = ground_y_local
            print(f"[RENDERING_COORDINATOR] No transform provided, using ground_y={ground_y_recon:.2f} (assumed same scale)")

        # Output directories
        self.local_scene_output_dir = output_dir / "local_scene_view"
        self.neural_scene_output_dir = output_dir / "neural_scene_view"
        self.recon_scene_output_dir = output_dir / "recon_scene_view"
        self.debug_output_dir = Path("debug_alpha_output")

        self.local_scene_output_dir.mkdir(parents=True, exist_ok=True)
        self.neural_scene_output_dir.mkdir(parents=True, exist_ok=True)
        self.recon_scene_output_dir.mkdir(parents=True, exist_ok=True)
        if self.enable_debug:
            self.debug_output_dir.mkdir(parents=True, exist_ok=True)

        # Frame buffers for fusion (Recon Scene View + Neural Scene View)
        self.latest_recon_scene_rgb: Optional[np.ndarray] = None
        self.latest_recon_scene_depth: Optional[np.ndarray] = None
        self.latest_neural_scene_rgb: Optional[np.ndarray] = None
        self.latest_neural_scene_depth: Optional[np.ndarray] = None
        self.latest_neural_scene_alpha: Optional[np.ndarray] = None  # Alpha map (opacity)
        self.latest_fused_depth: Optional[np.ndarray] = None  # Fused depth map

        # Fusion settings
        self.fusion_enabled = False
        self.fusion_depth_threshold = 0.01  # Tighter depth fusion (was 0.02)
        self.alpha_threshold = 0.6  # Stricter opacity threshold for valid depth (was 0.3)
        self.fusion_debug = False  # Enable fusion debug statistics

        # Collision detection
        self.collision_detector = DepthCollisionDetector(
            near=near_plane,
            far=far_plane,
            epsilon=0.005,  # Very tight collision tolerance (was 0.01)
            ground_y=ground_y_recon  # Ground plane in Recon space (scaled from Local)
        )

    async def render_neural(self, camera_frame) -> Optional[np.ndarray]:
        """
        Render using neural renderer.

        Args:
            camera_frame: CameraFrame for rendering

        Returns:
            Rendered RGB image (H, W, 3) uint8
        """
        render_output: RenderOutput = await self.renderer.render(camera_frame)

        # Save camera parameters to JSON
        cam_params = {
            "view_matrix": camera_frame.view_matrix.tolist(),
            "intrinsics": camera_frame.intrinsics.tolist(),
            "width": int(camera_frame.width),
            "height": int(camera_frame.height),
            "near": float(camera_frame.near),
            "far": float(camera_frame.far),
            "frame_id": int(camera_frame.frame_id)
        }

        # with open(f"{self.debug_output_dir}/camera_params.json", "w") as f:
        #     json.dump(cam_params, f, indent=2)

        color_np = render_output.color.cpu().numpy()
        color_bgr = cv2.cvtColor(color_np, cv2.COLOR_RGB2BGR)
        color_uint8 = (color_bgr * 255).astype(np.uint8)

        # cv2.imwrite("debug_alpha_output/neural_color.png", color_uint8)

        # Store Neural Scene View for fusion
        self.latest_neural_scene_rgb = color_uint8.copy()
        if render_output.depth is not None:
            depth_np = render_output.depth.cpu().numpy().astype(np.float32)

            # np.save(f"{self.debug_output_dir}/neural_depth", depth_np)

            # depth_normalized = (depth_np - depth_np.min()) / (depth_np.max() - depth_np.min())
            # depth_uint8 = (depth_normalized * 255).astype(np.uint8)
            # cv2.imwrite(f"{self.debug_output_dir}/neural_depth.png", depth_uint8)

            # Apply alpha masking to filter out invalid depth regions
            # (background, transparent areas, etc.)
            if render_output.alpha is not None:
                # Note: render_output.alpha contains transmittance (T)
                # Convert to opacity: alpha = 1 - T
                transmittance_np = render_output.alpha.cpu().numpy().astype(np.float32)
                alpha_np = 1.0 - transmittance_np

                # Debug visualization (if enabled)
                if self.enable_debug:
                    self._debug_render_result(depth_np, transmittance_np, alpha_np)

                # Invalidate depth where alpha (opacity) is low (< 0.5)
                # This prevents false collisions with background/transparent pixels
                invalid_mask = alpha_np < 0.5
                depth_np[invalid_mask] = np.nan

                # Store alpha map for depth fusion
                self.latest_neural_scene_alpha = alpha_np
            else:
                # No alpha available, clear alpha map
                self.latest_neural_scene_alpha = None

            self.latest_neural_scene_depth = depth_np

            # Build projection matrix from intrinsics
            proj_matrix = self._build_projection_matrix(
                camera_frame.intrinsics,
                camera_frame.width,
                camera_frame.height,
                camera_frame.near,
                camera_frame.far
            )

            # Update collision detector with neural depth
            self.collision_detector.update_depths(self.latest_neural_scene_depth)
            self.collision_detector.update_camera(
                camera_frame.view_matrix,
                proj_matrix,
                camera_frame.width,
                camera_frame.height
            )

        return color_uint8

    def update_recon_scene_frame(self, recon_scene_vis):
        """
        Update Recon Scene View frame for fusion (no file I/O).

        Args:
            recon_scene_vis: Recon scene visualizer
        """
        # Capture RGB (do_render=False, already rendered by update_renderer())
        rgb_buffer = recon_scene_vis.capture_screen_float_buffer(do_render=False)
        rgb_array = np.asarray(rgb_buffer)
        rgb_uint8 = (rgb_array * 255).astype(np.uint8)
        self.latest_recon_scene_rgb = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2BGR)

        # Capture depth
        depth_buffer = recon_scene_vis.capture_depth_float_buffer(do_render=False)
        self.latest_recon_scene_depth = np.asarray(depth_buffer, dtype=np.float32)

    def fuse_depth_based(self) -> Optional[np.ndarray]:
        """
        Depth-based fusion with alpha-aware validity checking.

        Uses alpha map to improve depth validity detection:
        - Neural depth is only valid if alpha (opacity) > threshold
        - Recon depth validity is checked by finite values
        - Depth comparison uses configurable threshold τ

        Returns:
            Fused RGB image or None if fusion is not possible
        """
        if (self.latest_recon_scene_rgb is None or self.latest_recon_scene_depth is None or
            self.latest_neural_scene_rgb is None or self.latest_neural_scene_depth is None):
            return None

        if (self.latest_recon_scene_rgb.shape != self.latest_neural_scene_rgb.shape or
            self.latest_recon_scene_depth.shape != self.latest_neural_scene_depth.shape):
            print("[FUSION] Shape mismatch")
            return None

        d_recon  = self.latest_recon_scene_depth
        d_neural = self.latest_neural_scene_depth
        alpha    = self.latest_neural_scene_alpha

        # === Validity Check ===

        # Recon: 0, NaN, inf → invalid
        recon_valid = np.isfinite(d_recon) & (d_recon > 0)

        # Neural: Use alpha-aware validity
        if alpha is not None:
            # Alpha-based validity: depth must be finite AND opacity > threshold
            neural_valid = (
                np.isfinite(d_neural) &
                (d_neural > 0) &
                (alpha >= self.alpha_threshold)  # Opacity threshold
            )
        else:
            # Fallback: traditional validity check (no alpha available)
            neural_valid = np.isfinite(d_neural) & (d_neural > 0)

        # === Depth Comparison ===

        both_valid  = recon_valid & neural_valid
        recon_only  = recon_valid & ~neural_valid
        neural_only = ~recon_valid & neural_valid

        # Depth comparison: choose closer one (smaller depth = closer)
        # Recon is preferred if it's τ closer than neural
        τ = self.fusion_depth_threshold
        recon_closer = both_valid & (d_recon + τ < d_neural)
        neural_closer = both_valid & (d_neural + τ < d_recon)
        similar_depth = both_valid & ~recon_closer & ~neural_closer

        # === Fusion Strategy ===

        # Use recon if:
        # 1. Recon is closer by threshold τ
        # 2. Only recon is valid
        # 3. Similar depth (within τ) → prefer neural (higher quality)
        use_recon = recon_closer | recon_only

        # Use neural if:
        # 1. Neural is closer by threshold τ
        # 2. Only neural is valid
        # 3. Similar depth → use neural (default)
        use_neural = neural_closer | neural_only | similar_depth

        # === RGB Blending ===

        # Create 3-channel mask for RGB
        fusion_mask = np.repeat(use_recon[:, :, np.newaxis], 3, axis=2)
        fused_rgb = np.where(fusion_mask, self.latest_recon_scene_rgb, self.latest_neural_scene_rgb)

        # === Depth Fusion ===

        # Create fused depth map
        fused_depth = np.full(d_recon.shape, np.nan, dtype=np.float32)
        fused_depth[use_recon] = d_recon[use_recon]
        fused_depth[use_neural] = d_neural[use_neural]

        # Store the latest fused depth
        self.latest_fused_depth = fused_depth.copy()

        # Update collision detector with fused depth (not just neural!)
        self.collision_detector.update_depths(fused_depth)

        # Debug visualization (if enabled)
        if self.enable_debug:
            self._debug_fusion_result(d_recon, d_neural, fused_depth,
                                     recon_valid, neural_valid, use_recon, use_neural)

        # === Debug Statistics (optional) ===
        if self.fusion_debug:
            total_pixels = recon_valid.size
            print(f"[FUSION] Validity:")
            print(f"  Recon valid:  {recon_valid.sum():6d} / {total_pixels} ({recon_valid.sum()/total_pixels:.1%})")
            print(f"  Neural valid: {neural_valid.sum():6d} / {total_pixels} ({neural_valid.sum()/total_pixels:.1%})")
            if alpha is not None:
                alpha_filtered = (alpha >= self.alpha_threshold).sum()
                print(f"  Alpha > {self.alpha_threshold}: {alpha_filtered:6d} / {total_pixels} ({alpha_filtered/total_pixels:.1%})")
            print(f"[FUSION] Decision:")
            print(f"  Use Recon:    {use_recon.sum():6d} ({use_recon.sum()/total_pixels:.1%})")
            print(f"  Use Neural:   {use_neural.sum():6d} ({use_neural.sum()/total_pixels:.1%})")

        return fused_rgb

    def toggle_fusion(self):
        """Toggle depth-based fusion on/off."""
        self.fusion_enabled = not self.fusion_enabled
        return self.fusion_enabled

    def toggle_fusion_debug(self):
        """Toggle fusion debug statistics on/off."""
        self.fusion_debug = not self.fusion_debug
        status = "enabled" if self.fusion_debug else "disabled"
        print(f"[FUSION] Debug statistics {status}")
        return self.fusion_debug

    def set_alpha_threshold(self, threshold: float):
        """
        Set alpha threshold for valid depth determination.

        Args:
            threshold: Alpha (opacity) threshold [0.0, 1.0]
                      - 0.0: All pixels valid (no filtering)
                      - 0.5: Medium filtering (default, recommended)
                      - 0.9: Aggressive filtering (only very opaque pixels)

        Returns:
            Updated threshold value
        """
        self.alpha_threshold = np.clip(threshold, 0.0, 1.0)
        print(f"[FUSION] Alpha threshold set to {self.alpha_threshold:.2f}")
        return self.alpha_threshold

    def set_fusion_depth_threshold(self, threshold: float):
        """
        Set depth threshold (τ) for fusion decision.

        Args:
            threshold: Depth difference threshold in meters
                      - Smaller values: More sensitive to depth differences
                      - Larger values: More tolerant to depth differences

        Returns:
            Updated threshold value
        """
        self.fusion_depth_threshold = max(0.0, threshold)
        print(f"[FUSION] Depth threshold set to {self.fusion_depth_threshold:.3f}m")
        return self.fusion_depth_threshold

    def _build_projection_matrix(self, intrinsics, width, height, near, far):
        """
        Build OpenGL projection matrix from camera intrinsics.

        Args:
            intrinsics: Camera intrinsics matrix (3x3) - can be torch.Tensor or np.ndarray
            width: Image width
            height: Image height
            near: Near clipping plane
            far: Far clipping plane

        Returns:
            Projection matrix (4x4) as numpy array
        """
        # Convert to numpy if torch tensor
        if hasattr(intrinsics, 'cpu'):
            intrinsics = intrinsics.cpu().numpy()

        fx = intrinsics[0, 0]
        fy = intrinsics[1, 1]
        cx = intrinsics[0, 2]
        cy = intrinsics[1, 2]

        # OpenGL projection matrix
        proj = np.zeros((4, 4), dtype=np.float32)
        proj[0, 0] = 2.0 * fx / width
        proj[1, 1] = 2.0 * fy / height
        proj[0, 2] = (width - 2.0 * cx) / width
        proj[1, 2] = (2.0 * cy - height) / height
        proj[2, 2] = -(far + near) / (far - near)
        proj[2, 3] = -2.0 * far * near / (far - near)
        proj[3, 2] = -1.0

        return proj

    def _debug_render_result(self, depth_np: np.ndarray,
                             transmittance_np: np.ndarray, alpha_np: np.ndarray):
        """
        Debug visualization for render output (alpha, depth, correlation).

        Args:
            depth_np: Depth array (H, W)
            transmittance_np: Transmittance array (H, W)
            alpha_np: Opacity array (H, W)
        """
        debug_dir = self.debug_output_dir

        # === Basic visualizations ===

        # Save neural depth (normalized for visualization)
        depth_valid = depth_np[np.isfinite(depth_np)]
        if len(depth_valid) > 0:
            depth_min, depth_max = depth_valid.min(), depth_valid.max()
            depth_normalized = np.clip((depth_np - depth_min) / (depth_max - depth_min + 1e-8), 0, 1)
            depth_vis = (depth_normalized * 255).astype(np.uint8)
            cv2.imwrite(str(debug_dir / "neural_depth.png"), depth_vis)

            # === 10-Level Depth Visualization (Blue=Near, Red=Far) ===
            # Discretize depth into 10 levels
            depth_levels = np.full(depth_np.shape, -1, dtype=np.int32)  # -1 for invalid
            valid_mask = np.isfinite(depth_np)
            depth_levels[valid_mask] = np.digitize(
                depth_normalized[valid_mask],
                bins=np.linspace(0, 1, 11)
            ) - 1
            depth_levels = np.clip(depth_levels, -1, 9)

            # Create color map: Blue (near) -> Red (far)
            # Level 0 (nearest): Blue (255, 0, 0) in BGR
            # Level 9 (farthest): Red (0, 0, 255) in BGR
            depth_colored = np.zeros((*depth_np.shape, 3), dtype=np.uint8)
            for level in range(10):
                mask = (depth_levels == level)
                # Interpolate from blue to red
                # BGR: (255, 0, 0) -> (0, 0, 255)
                t = level / 9.0  # Normalized level [0, 1]
                b = int(255 * (1 - t))  # Blue channel: 255 -> 0
                g = 0                    # Green channel: always 0
                r = int(255 * t)         # Red channel: 0 -> 255
                depth_colored[mask] = [b, g, r]

            # Invalid depth pixels remain black
            cv2.imwrite(str(debug_dir / "depth_10levels_colored.png"), depth_colored)

            # Save depth level statistics
            depth_level_counts = np.bincount(depth_levels[depth_levels >= 0].flatten(), minlength=10)
            depth_level_percentages = depth_level_counts / max(depth_level_counts.sum(), 1) * 100
        else:
            depth_min, depth_max = 0, 1
            depth_level_counts = np.zeros(10, dtype=np.int64)
            depth_level_percentages = np.zeros(10)

        # Save transmittance (T) - raw from CUDA
        transmittance_vis = (transmittance_np * 255).astype(np.uint8)
        cv2.imwrite(str(debug_dir / "transmittance_raw.png"), transmittance_vis)

        # Save opacity (alpha = 1 - T) - grayscale
        alpha_vis = (alpha_np * 255).astype(np.uint8)
        cv2.imwrite(str(debug_dir / "opacity_alpha.png"), alpha_vis)

        # Save mask (alpha < 0.5)
        invalid_mask = alpha_np < 0.5
        mask_vis = ((~invalid_mask) * 255).astype(np.uint8)  # White = valid, Black = invalid
        cv2.imwrite(str(debug_dir / "valid_mask.png"), mask_vis)

        # === 10-Level Alpha Visualization (White=Transparent, Red=Opaque) ===

        # Discretize alpha into 10 levels: [0.0-0.1), [0.1-0.2), ..., [0.9-1.0]
        alpha_levels = np.digitize(alpha_np, bins=np.linspace(0, 1, 11)) - 1
        alpha_levels = np.clip(alpha_levels, 0, 9)  # Ensure range [0, 9]

        # Create color map: White (255,255,255) -> Red (0,0,255) in BGR
        # Level 0 (0.0-0.1): White (most transparent)
        # Level 9 (0.9-1.0): Red (most opaque)
        alpha_colored = np.zeros((*alpha_np.shape, 3), dtype=np.uint8)
        for level in range(10):
            mask = (alpha_levels == level)
            # Interpolate from white to red
            # BGR: (255, 255, 255) -> (0, 0, 255)
            t = level / 9.0  # Normalized level [0, 1]
            b = int(255 * (1 - t))  # Blue channel: 255 -> 0
            g = int(255 * (1 - t))  # Green channel: 255 -> 0
            r = 255                  # Red channel: always 255
            alpha_colored[mask] = [b, g, r]

        cv2.imwrite(str(debug_dir / "alpha_10levels_colored.png"), alpha_colored)

        # === Depth-Alpha Correlation Analysis ===

        # Check if high alpha corresponds to valid depth
        depth_finite = np.isfinite(depth_np)
        depth_zero = (depth_np == 0.0) if depth_finite.any() else np.zeros_like(depth_finite)

        # Analyze alpha distribution by depth validity
        high_alpha_mask = (alpha_np >= 0.9)
        high_alpha_valid_depth = high_alpha_mask & depth_finite & ~depth_zero
        high_alpha_invalid_depth = high_alpha_mask & (~depth_finite | depth_zero)

        # Create diagnostic image: show alpha levels only where depth is valid
        alpha_colored_depth_filtered = alpha_colored.copy()
        invalid_depth_mask = ~depth_finite | depth_zero
        alpha_colored_depth_filtered[invalid_depth_mask] = [0, 0, 0]  # Black for invalid depth
        cv2.imwrite(str(debug_dir / "alpha_10levels_depth_filtered.png"), alpha_colored_depth_filtered)

        # === Statistics ===

        level_counts = np.bincount(alpha_levels.flatten(), minlength=10)
        level_percentages = level_counts / alpha_levels.size * 100

        with open(str(debug_dir / "alpha_stats.txt"), "w") as f:
            f.write(f"Transmittance (T):\n")
            f.write(f"  min={transmittance_np.min():.4f}\n")
            f.write(f"  max={transmittance_np.max():.4f}\n")
            f.write(f"  mean={transmittance_np.mean():.4f}\n\n")

            f.write(f"Opacity (Alpha = 1-T):\n")
            f.write(f"  min={alpha_np.min():.4f}\n")
            f.write(f"  max={alpha_np.max():.4f}\n")
            f.write(f"  mean={alpha_np.mean():.4f}\n\n")

            f.write(f"Invalid pixels (alpha < 0.5):\n")
            f.write(f"  count={invalid_mask.sum()}\n")
            f.write(f"  ratio={invalid_mask.sum() / invalid_mask.size:.2%}\n\n")

            f.write(f"10-Level Alpha Distribution:\n")
            for level in range(10):
                alpha_range = f"[{level*0.1:.1f}, {(level+1)*0.1:.1f})"
                f.write(f"  Level {level} {alpha_range}: {level_counts[level]:6d} pixels ({level_percentages[level]:5.2f}%)\n")

            f.write(f"\nDepth-Alpha Correlation:\n")
            f.write(f"  High alpha (>0.9) pixels: {high_alpha_mask.sum():6d} ({high_alpha_mask.sum()/alpha_np.size:.2%})\n")
            f.write(f"    with valid depth:       {high_alpha_valid_depth.sum():6d} ({high_alpha_valid_depth.sum()/max(high_alpha_mask.sum(), 1):.2%})\n")
            f.write(f"    with invalid depth:     {high_alpha_invalid_depth.sum():6d} ({high_alpha_invalid_depth.sum()/max(high_alpha_mask.sum(), 1):.2%})\n")

            if len(depth_valid) > 0:
                f.write(f"\nDepth statistics (valid pixels only):\n")
                f.write(f"  min={depth_min:.4f}\n")
                f.write(f"  max={depth_max:.4f}\n")
                f.write(f"  mean={depth_valid.mean():.4f}\n")
                f.write(f"  valid_pixels={len(depth_valid)} ({len(depth_valid)/depth_np.size:.2%})\n")

                f.write(f"\n10-Level Depth Distribution:\n")
                for level in range(10):
                    depth_range_min = depth_min + (depth_max - depth_min) * level / 10.0
                    depth_range_max = depth_min + (depth_max - depth_min) * (level + 1) / 10.0
                    f.write(f"  Level {level} [{depth_range_min:.4f}, {depth_range_max:.4f}): "
                           f"{depth_level_counts[level]:6d} pixels ({depth_level_percentages[level]:5.2f}%)\n")

            f.write(f"\nColor Maps:\n")
            f.write(f"  Alpha: White (transparent) -> Red (opaque)\n")
            f.write(f"  Depth: Blue (near) -> Red (far)\n")

    def _debug_fusion_result(self, recon_depth: np.ndarray, neural_depth: np.ndarray,
                            fused_depth: np.ndarray, recon_valid: np.ndarray,
                            neural_valid: np.ndarray, use_recon: np.ndarray,
                            use_neural: np.ndarray):
        """
        Debug visualization for depth fusion result.

        Args:
            recon_depth: Recon scene depth (H, W)
            neural_depth: Neural scene depth (H, W)
            fused_depth: Fused depth (H, W)
            recon_valid: Recon depth validity mask (H, W)
            neural_valid: Neural depth validity mask (H, W)
            use_recon: Pixels using recon depth (H, W)
            use_neural: Pixels using neural depth (H, W)
        """
        debug_dir = self.debug_output_dir / "fusion"
        debug_dir.mkdir(exist_ok=True)

        def save_depth_10levels(depth_np: np.ndarray, filename: str):
            """Helper function to save depth as 10-level colored image."""
            depth_valid = depth_np[np.isfinite(depth_np)]
            if len(depth_valid) == 0:
                # All invalid - save black image
                depth_colored = np.zeros((*depth_np.shape, 3), dtype=np.uint8)
                cv2.imwrite(str(debug_dir / filename), depth_colored)
                return None, None, None

            depth_min, depth_max = depth_valid.min(), depth_valid.max()
            depth_normalized = np.clip((depth_np - depth_min) / (depth_max - depth_min + 1e-8), 0, 1)

            # Discretize depth into 10 levels
            depth_levels = np.full(depth_np.shape, -1, dtype=np.int32)
            valid_mask = np.isfinite(depth_np)
            depth_levels[valid_mask] = np.digitize(
                depth_normalized[valid_mask],
                bins=np.linspace(0, 1, 11)
            ) - 1
            depth_levels = np.clip(depth_levels, -1, 9)

            # Create color map: Blue (near) -> Red (far)
            depth_colored = np.zeros((*depth_np.shape, 3), dtype=np.uint8)
            for level in range(10):
                mask = (depth_levels == level)
                t = level / 9.0
                b = int(255 * (1 - t))
                g = 0
                r = int(255 * t)
                depth_colored[mask] = [b, g, r]

            cv2.imwrite(str(debug_dir / filename), depth_colored)

            # Calculate statistics
            level_counts = np.bincount(depth_levels[depth_levels >= 0].flatten(), minlength=10)
            level_percentages = level_counts / max(level_counts.sum(), 1) * 100
            return depth_min, depth_max, list(zip(level_counts, level_percentages))

        # Save all three depth maps
        recon_min, recon_max, recon_stats = save_depth_10levels(recon_depth, "recon_depth_10levels.png")
        neural_min, neural_max, neural_stats = save_depth_10levels(neural_depth, "neural_depth_10levels.png")
        fused_min, fused_max, fused_stats = save_depth_10levels(fused_depth, "fused_depth_10levels.png")

        # Create decision visualization (which source was used)
        # Green = Recon, Blue = Neural, Black = Invalid
        decision_vis = np.zeros((*fused_depth.shape, 3), dtype=np.uint8)
        decision_vis[use_recon] = [0, 255, 0]    # Green for recon
        decision_vis[use_neural] = [255, 0, 0]   # Blue for neural
        cv2.imwrite(str(debug_dir / "fusion_decision.png"), decision_vis)

        # Save statistics
        with open(str(debug_dir / "fusion_stats.txt"), "w") as f:
            f.write("Depth Fusion Statistics\n")
            f.write("=" * 50 + "\n\n")

            # Validity statistics
            total_pixels = recon_valid.size
            f.write(f"Validity:\n")
            f.write(f"  Recon valid:  {recon_valid.sum():6d} / {total_pixels} ({recon_valid.sum()/total_pixels:.2%})\n")
            f.write(f"  Neural valid: {neural_valid.sum():6d} / {total_pixels} ({neural_valid.sum()/total_pixels:.2%})\n\n")

            # Decision statistics
            f.write(f"Fusion Decision:\n")
            f.write(f"  Use Recon:    {use_recon.sum():6d} ({use_recon.sum()/total_pixels:.2%})\n")
            f.write(f"  Use Neural:   {use_neural.sum():6d} ({use_neural.sum()/total_pixels:.2%})\n\n")

            # Depth range statistics
            if recon_min is not None:
                f.write(f"Recon Depth Range: [{recon_min:.4f}, {recon_max:.4f}]\n")
                f.write(f"  10-Level Distribution:\n")
                for level, (count, percentage) in enumerate(recon_stats):
                    depth_range_min = recon_min + (recon_max - recon_min) * level / 10.0
                    depth_range_max = recon_min + (recon_max - recon_min) * (level + 1) / 10.0
                    f.write(f"    Level {level} [{depth_range_min:.4f}, {depth_range_max:.4f}): "
                           f"{count:6d} pixels ({percentage:5.2f}%)\n")
                f.write("\n")

            if neural_min is not None:
                f.write(f"Neural Depth Range: [{neural_min:.4f}, {neural_max:.4f}]\n")
                f.write(f"  10-Level Distribution:\n")
                for level, (count, percentage) in enumerate(neural_stats):
                    depth_range_min = neural_min + (neural_max - neural_min) * level / 10.0
                    depth_range_max = neural_min + (neural_max - neural_min) * (level + 1) / 10.0
                    f.write(f"    Level {level} [{depth_range_min:.4f}, {depth_range_max:.4f}): "
                           f"{count:6d} pixels ({percentage:5.2f}%)\n")
                f.write("\n")

            if fused_min is not None:
                f.write(f"Fused Depth Range: [{fused_min:.4f}, {fused_max:.4f}]\n")
                f.write(f"  10-Level Distribution:\n")
                for level, (count, percentage) in enumerate(fused_stats):
                    depth_range_min = fused_min + (fused_max - fused_min) * level / 10.0
                    depth_range_max = fused_min + (fused_max - fused_min) * (level + 1) / 10.0
                    f.write(f"    Level {level} [{depth_range_min:.4f}, {depth_range_max:.4f}): "
                           f"{count:6d} pixels ({percentage:5.2f}%)\n")

            f.write(f"\nColor Maps:\n")
            f.write(f"  Depth: Blue (near) -> Red (far)\n")

    def create_fusion_depth_grayscale(self) -> Optional[np.ndarray]:
        """
        Create a grayscale visualization of the fused depth map.

        Returns:
            Grayscale depth image (uint8) or None if no fused depth available
        """
        if self.latest_fused_depth is None:
            return None

        depth = self.latest_fused_depth.copy()

        # Handle NaN and invalid values
        valid_mask = np.isfinite(depth) & (depth > 0) & (depth < self.far_plane)
        if not np.any(valid_mask):
            return np.zeros(depth.shape, dtype=np.uint8)

        # Get valid depth range
        valid_depth = depth[valid_mask]
        depth_min = np.min(valid_depth)
        depth_max = np.max(valid_depth)

        # Normalize depth to 0-255 range (inverted: near=dark, far=bright)
        depth_normalized = np.zeros_like(depth)
        if depth_max > depth_min:
            depth_normalized[valid_mask] = (depth[valid_mask] - depth_min) / (depth_max - depth_min)
        else:
            depth_normalized[valid_mask] = 0.5

        # Convert to grayscale (0-255)
        depth_gray = (depth_normalized * 255).astype(np.uint8)

        # Mark invalid areas as middle gray
        depth_gray[~valid_mask] = 128

        return depth_gray

    def create_depth_contours(self, depth_gray: Optional[np.ndarray] = None,
                             num_levels: int = 10) -> Optional[np.ndarray]:
        """
        Create contour lines overlay for depth visualization.

        Args:
            depth_gray: Grayscale depth image. If None, will generate from fused depth.
            num_levels: Number of contour levels (default: 10)

        Returns:
            RGB image with depth grayscale and contour overlay, or None if no data
        """
        # Get or create grayscale depth
        if depth_gray is None:
            depth_gray = self.create_fusion_depth_grayscale()
            if depth_gray is None:
                return None

        # Convert to 3-channel for colored contours
        depth_colored = cv2.cvtColor(depth_gray, cv2.COLOR_GRAY2BGR)

        # Create contour levels
        levels = np.linspace(0, 255, num_levels + 1).astype(np.uint8)

        # Draw contours for each level
        for i in range(len(levels) - 1):
            # Create binary mask for this level
            mask = ((depth_gray >= levels[i]) & (depth_gray < levels[i + 1])).astype(np.uint8) * 255

            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Choose color based on level (blue to red gradient)
            color_ratio = i / (num_levels - 1) if num_levels > 1 else 0
            color = (
                int(255 * (1 - color_ratio)),  # Blue component (decreases)
                int(128 * (1 - abs(color_ratio - 0.5) * 2)),  # Green (peaks at middle)
                int(255 * color_ratio)  # Red component (increases)
            )

            # Draw contours
            cv2.drawContours(depth_colored, contours, -1, color, 1)

        return depth_colored

    def get_fusion_depth_visualization(self, with_contours: bool = True,
                                      num_contour_levels: int = 10) -> Optional[np.ndarray]:
        """
        Get the fusion depth visualization with optional contours.

        Args:
            with_contours: Whether to include contour lines
            num_contour_levels: Number of contour levels

        Returns:
            Visualization image (BGR format) or None if no data
        """
        if with_contours:
            return self.create_depth_contours(num_levels=num_contour_levels)
        else:
            # Just return grayscale converted to BGR
            gray = self.create_fusion_depth_grayscale()
            if gray is None:
                return None
            return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    def create_collision_mask(self) -> Optional[np.ndarray]:
        """
        Generate per-pixel collision boundary mask using depth gradient analysis.

        Returns:
            BGR image (H, W, 3) showing collision boundaries:
            - Black [0, 0, 0]: Invalid depth (no data)
            - Green [0, 255, 0]: Smooth surface (low depth gradient)
            - Yellow [0, 255, 255]: Object boundaries (high depth gradient/edges)
            - Red [0, 0, 255]: Very close objects (< 2× near_plane)

        Algorithm:
            1. Compute Sobel depth gradient (dx, dy) to find depth discontinuities
            2. Calculate gradient magnitude to identify edges/boundaries
            3. Classify pixels based on gradient strength and proximity

        Note:
            Uses latest_fused_depth from current frame only (no temporal filtering).
        """
        if self.latest_fused_depth is None:
            return None

        fused_depth = self.latest_fused_depth
        H, W = fused_depth.shape

        # Create output mask (BGR format for OpenCV)
        mask = np.zeros((H, W, 3), dtype=np.uint8)

        # Get thresholds from collision detector
        near = self.collision_detector.near
        far = self.collision_detector.far

        # Create valid depth mask
        valid_mask = (
            np.isfinite(fused_depth) &
            (fused_depth > 0.0) &
            (fused_depth < far * 0.99)
        )

        # Compute depth gradient using Sobel operator
        # ksize=5 for better edge detection on noisy depth maps
        grad_x = cv2.Sobel(fused_depth, cv2.CV_64F, 1, 0, ksize=5)
        grad_y = cv2.Sobel(fused_depth, cv2.CV_64F, 0, 1, ksize=5)

        # Gradient magnitude (strength of depth discontinuity)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)

        # Normalize gradient by local depth for scale-invariant edge detection
        # Avoid division by zero with safe minimum
        normalized_gradient = gradient_magnitude / np.maximum(fused_depth, 0.01)

        # Define thresholds
        # Gradient threshold: edges where depth changes significantly
        # Very tight setting - highly sensitive to depth discontinuities
        edge_threshold = 0.03  # Normalized gradient threshold (was 0.1, now 3x more sensitive)

        # Proximity threshold: objects very close to camera
        proximity_threshold = 2.0 * near

        # Classify each pixel
        for y in range(H):
            for x in range(W):
                if not valid_mask[y, x]:
                    # Invalid depth → Black
                    mask[y, x] = [0, 0, 0]
                    continue

                depth = fused_depth[y, x]
                gradient = normalized_gradient[y, x]

                if depth < proximity_threshold:
                    # Very close to camera → Red (danger zone)
                    mask[y, x] = [0, 0, 255]
                elif gradient > edge_threshold:
                    # High gradient → Yellow (object boundary)
                    mask[y, x] = [0, 255, 255]
                else:
                    # Low gradient → Green (smooth surface)
                    mask[y, x] = [0, 255, 0]

        return mask

    def create_cross_section_view(self, sphere_center: np.ndarray, sphere_radius: float) -> Optional[np.ndarray]:
        """
        Generate 3D cross-section visualization of collision boundaries.

        Creates three orthogonal plane cross-sections (XY, XZ, YZ) passing through
        the sphere center, showing collision points and sphere outline in each plane.

        Args:
            sphere_center: (3,) position in RECON space (sphere center)
            sphere_radius: Sphere radius in RECON space units

        Returns:
            BGR image showing 1×3 grid [XY plane | XZ plane | YZ plane]
            - Red points: Collision detected
            - Green points: No collision (surface)
            - Yellow circle: Sphere outline
            - Black: Invalid depth
        """
        if self.latest_fused_depth is None:
            return None

        fused_depth = self.latest_fused_depth
        H, W = fused_depth.shape

        # Get camera matrices
        if self.collision_detector.view_matrix is None or self.collision_detector.proj_matrix is None:
            return None

        view_matrix = self.collision_detector.view_matrix
        proj_matrix = self.collision_detector.proj_matrix
        near = self.collision_detector.near
        far = self.collision_detector.far

        # Compute inverse matrices for unprojection
        try:
            inv_view = np.linalg.inv(view_matrix)
            inv_proj = np.linalg.inv(proj_matrix)
        except np.linalg.LinAlgError:
            return None

        # Unproject all valid pixels to 3D world space
        world_points = []
        collision_flags = []

        # Progress bar for pixel processing
        total_pixels = H * W
        with tqdm(total=total_pixels, desc="Unprojecting pixels", unit="px") as pbar:
            for y in range(H):
                for x in range(W):
                    depth = fused_depth[y, x]

                    # Check valid depth
                    if not np.isfinite(depth) or depth <= 0 or depth >= far * 0.99:
                        pbar.update(1)
                        continue

                    # Screen to NDC
                    ndc_x = (x / W) * 2.0 - 1.0
                    ndc_y = 1.0 - (y / H) * 2.0  # Flip Y
                    ndc_z = (depth - near) / (far - near) * 2.0 - 1.0  # Depth to NDC

                    # NDC to clip space (reverse perspective divide)
                    clip_pos = np.array([ndc_x * depth, ndc_y * depth, ndc_z * depth, depth])

                    # Clip to view space
                    view_pos = inv_proj @ clip_pos

                    # View to world space
                    world_pos_hom = inv_view @ view_pos
                    world_pos = world_pos_hom[:3] / world_pos_hom[3]

                    # Check collision at this point
                    collision = self.collision_detector.check_object_collision(world_pos, 0.0)  # Point collision
                    world_points.append(world_pos)
                    collision_flags.append(collision["collision"])

                    # Update progress bar
                    pbar.update(1)

        if len(world_points) == 0:
            print("[CROSS_SECTION] ERROR: No valid world points unprojected!")
            return None

        world_points = np.array(world_points)  # (N, 3)
        collision_flags = np.array(collision_flags)  # (N,)

        print(f"[CROSS_SECTION] Total unprojected points: {len(world_points)}")

        # Sphere center coordinates
        cx, cy, cz = sphere_center

        # Slice thickness (how close to plane)
        # Increase to 50% of radius for better coverage (was 10%)
        slice_thickness = sphere_radius * 0.5
        print(f"[CROSS_SECTION] Sphere center: ({cx:.2f}, {cy:.2f}, {cz:.2f})")
        print(f"[CROSS_SECTION] Sphere radius: {sphere_radius:.2f}")
        print(f"[CROSS_SECTION] Slice thickness: {slice_thickness:.2f}")

        # Print world points range for debugging
        print(f"[CROSS_SECTION] World points X range: [{world_points[:, 0].min():.2f}, {world_points[:, 0].max():.2f}]")
        print(f"[CROSS_SECTION] World points Y range: [{world_points[:, 1].min():.2f}, {world_points[:, 1].max():.2f}]")
        print(f"[CROSS_SECTION] World points Z range: [{world_points[:, 2].min():.2f}, {world_points[:, 2].max():.2f}]")

        # Create three cross-section views
        views = []

        # XY plane (constant Z = cz)
        xy_mask = np.abs(world_points[:, 2] - cz) < slice_thickness
        xy_points = world_points[xy_mask][:, :2]  # (x, y)
        xy_collisions = collision_flags[xy_mask]
        print(f"[CROSS_SECTION] XY plane: {len(xy_points)} points in slice (Z={cz:.2f} ± {slice_thickness:.2f})")
        xy_view = self._render_2d_cross_section(
            xy_points, xy_collisions, sphere_center[:2], sphere_radius,
            xlabel="X", ylabel="Y", title="XY Plane (Z={:.2f})".format(cz)
        )
        views.append(xy_view)

        # XZ plane (constant Y = cy)
        xz_mask = np.abs(world_points[:, 1] - cy) < slice_thickness
        xz_points = world_points[xz_mask][:, [0, 2]]  # (x, z)
        xz_collisions = collision_flags[xz_mask]
        print(f"[CROSS_SECTION] XZ plane: {len(xz_points)} points in slice (Y={cy:.2f} ± {slice_thickness:.2f})")
        xz_view = self._render_2d_cross_section(
            xz_points, xz_collisions, sphere_center[[0, 2]], sphere_radius,
            xlabel="X", ylabel="Z", title="XZ Plane (Y={:.2f})".format(cy)
        )
        views.append(xz_view)

        # YZ plane (constant X = cx)
        yz_mask = np.abs(world_points[:, 0] - cx) < slice_thickness
        yz_points = world_points[yz_mask][:, [1, 2]]  # (y, z)
        yz_collisions = collision_flags[yz_mask]
        print(f"[CROSS_SECTION] YZ plane: {len(yz_points)} points in slice (X={cx:.2f} ± {slice_thickness:.2f})")
        yz_view = self._render_2d_cross_section(
            yz_points, yz_collisions, sphere_center[[1, 2]], sphere_radius,
            xlabel="Y", ylabel="Z", title="YZ Plane (X={:.2f})".format(cx)
        )
        views.append(yz_view)

        # Combine into 1×3 grid
        combined = np.hstack(views)
        return combined

    def _render_2d_cross_section(self, points_2d, collision_flags, sphere_center_2d, sphere_radius,
                                  xlabel="X", ylabel="Y", title="Cross Section", img_size=512):
        """
        Render 2D cross-section view with collision points and sphere outline.

        Args:
            points_2d: (N, 2) array of 2D projected points
            collision_flags: (N,) boolean array of collision status
            sphere_center_2d: (2,) sphere center in 2D plane
            sphere_radius: Sphere radius
            xlabel, ylabel: Axis labels
            title: Plot title
            img_size: Output image size (square)

        Returns:
            BGR image (img_size, img_size, 3)
        """
        img = np.zeros((img_size, img_size, 3), dtype=np.uint8)

        if len(points_2d) == 0:
            # No points in this slice, just draw empty view with labels
            cv2.putText(img, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(img, "No data in slice", (img_size // 4, img_size // 2),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (128, 128, 128), 2)
            return img

        # Compute bounds (with margin)
        margin = sphere_radius * 3  # Show 3× sphere radius around center
        x_min, x_max = sphere_center_2d[0] - margin, sphere_center_2d[0] + margin
        y_min, y_max = sphere_center_2d[1] - margin, sphere_center_2d[1] + margin

        # World to image coordinate transform
        def world_to_img(pt):
            x_norm = (pt[0] - x_min) / (x_max - x_min)
            y_norm = (pt[1] - y_min) / (y_max - y_min)
            px = int(x_norm * (img_size - 1))
            py = int((1.0 - y_norm) * (img_size - 1))  # Flip Y for image coords
            return (px, py)

        # Draw collision points
        for pt, is_collision in zip(points_2d, collision_flags):
            img_pt = world_to_img(pt)
            if 0 <= img_pt[0] < img_size and 0 <= img_pt[1] < img_size:
                color = (0, 0, 255) if is_collision else (0, 255, 0)  # Red or Green
                cv2.circle(img, img_pt, 2, color, -1)

        # Draw sphere outline (circle in 2D cross-section)
        sphere_center_img = world_to_img(sphere_center_2d)
        radius_px = int(sphere_radius / (x_max - x_min) * img_size)
        cv2.circle(img, sphere_center_img, radius_px, (0, 255, 255), 2)  # Yellow outline

        # Draw crosshair at sphere center
        cv2.line(img, (sphere_center_img[0] - 10, sphere_center_img[1]),
                (sphere_center_img[0] + 10, sphere_center_img[1]), (255, 255, 255), 1)
        cv2.line(img, (sphere_center_img[0], sphere_center_img[1] - 10),
                (sphere_center_img[0], sphere_center_img[1] + 10), (255, 255, 255), 1)

        # Add labels
        cv2.putText(img, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(img, f"{xlabel} →", (img_size - 80, img_size - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(img, f"{ylabel} ↑", (10, img_size - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        return img

    def create_transform_overlay(self, local_mesh_points: Optional[np.ndarray] = None,
                                 transform_matrix: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        """
        Create transform validation overlay showing Local mesh transformed to Recon space.

        Args:
            local_mesh_points: (N, 3) array of points in Local space
            transform_matrix: (4, 4) Local→Recon transformation matrix

        Returns:
            BGR image with Neural scene + transformed Local mesh overlay
            - Green points: Correctly aligned mesh points
            - Neural rendering as background

        Note:
            If local_mesh_points or transform_matrix not provided, returns Neural scene only.
        """
        if self.latest_neural_scene_rgb is None:
            print("[TRANSFORM_OVERLAY] No neural scene available")
            return None

        # Start with Neural scene as base
        overlay = self.latest_neural_scene_rgb.copy()

        # If no mesh data provided, just return Neural scene
        if local_mesh_points is None or transform_matrix is None:
            print("[TRANSFORM_OVERLAY] No mesh data provided, returning Neural scene only")
            return overlay

        # Transform Local points to Recon space
        N = local_mesh_points.shape[0]
        local_points_hom = np.hstack([local_mesh_points, np.ones((N, 1))])  # (N, 4)
        recon_points_hom = (transform_matrix @ local_points_hom.T).T  # (N, 4)
        recon_points = recon_points_hom[:, :3]  # (N, 3)

        # Project Recon points to screen space
        view_matrix = self.collision_detector.view_matrix
        proj_matrix = self.collision_detector.proj_matrix

        if view_matrix is None or proj_matrix is None:
            print("[TRANSFORM_OVERLAY] No camera matrices available")
            return overlay

        # Project points
        for point_3d in recon_points:
            screen_coords = self.collision_detector.world_to_screen(point_3d)
            if screen_coords is not None:
                x, y = screen_coords
                # Draw green point on overlay
                cv2.circle(overlay, (x, y), 2, (0, 255, 0), -1)  # Green

        return overlay

    def create_combined_debug_view(self, local_mesh_points: Optional[np.ndarray] = None,
                                   transform_matrix: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        """
        Create combined debug view with 4 panels in 2×2 grid:
        [Neural Scene]      [Transform Overlay]
        [Fused Depth]       [Collision Mask]

        Args:
            local_mesh_points: (N, 3) array of points in Local space (for transform overlay)
            transform_matrix: (4, 4) Local→Recon transformation matrix

        Returns:
            Combined BGR image with 4 panels, or None if data unavailable
        """
        # Generate each panel
        neural_scene = self.latest_neural_scene_rgb
        transform_overlay = self.create_transform_overlay(local_mesh_points, transform_matrix)
        fused_depth_vis = self.get_fusion_depth_visualization(with_contours=True)
        collision_mask = self.create_collision_mask()

        # Check if all panels are available
        panels = [neural_scene, transform_overlay, fused_depth_vis, collision_mask]
        if any(panel is None for panel in panels):
            print("[COMBINED_DEBUG] Some panels unavailable, skipping")
            return None

        # Ensure all panels have the same size
        H, W = neural_scene.shape[:2]
        panels_resized = []
        for panel in panels:
            if panel.shape[:2] != (H, W):
                panel = cv2.resize(panel, (W, H))
            panels_resized.append(panel)

        # Add labels to panels
        font = cv2.FONT_HERSHEY_SIMPLEX
        labels = ["Neural Scene", "Transform Overlay", "Fused Depth", "Collision Mask"]
        for panel, label in zip(panels_resized, labels):
            cv2.putText(panel, label, (10, 30), font, 1.0, (255, 255, 255), 2)

        # Combine into 2×2 grid
        top_row = np.hstack([panels_resized[0], panels_resized[1]])
        bottom_row = np.hstack([panels_resized[2], panels_resized[3]])
        combined = np.vstack([top_row, bottom_row])

        return combined
