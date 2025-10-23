"""
Gaussian state management for temporal rendering.
"""

import os
import torch
from typing import Optional


class FastGaussianState:
    """
    Manages Gaussian state across frames using hybrid memory + PLY approach.
    """

    def __init__(self):
        self.states = {}
        self.ply_states = {}
        self.current_frame = 0

    def save_state(self, frame_id: int, gaussians):
        """Save Gaussian state for a frame (memory + PLY)."""
        state = {
            '_xyz': gaussians.get_xyz.clone().detach(),
            '_features_dc': gaussians._features_dc.clone().detach(),
            '_features_rest': gaussians._features_rest.clone().detach(),
            '_opacity': gaussians._opacity.clone().detach(),
            '_scaling': gaussians._scaling.clone().detach(),
            '_rotation': gaussians._rotation.clone().detach(),
            'max_radii2D': gaussians.max_radii2D.clone().detach(),
            'xyz_gradient_accum': gaussians.xyz_gradient_accum.clone().detach(),
            'denom': gaussians.denom.clone().detach(),
            'point_count': gaussians.get_xyz.shape[0],
        }
        self.states[frame_id] = state

        # PLY file saving for stability
        ply_path = f"temp_states/frame_{frame_id:06d}.ply"
        os.makedirs("temp_states", exist_ok=True)
        gaussians.save_ply(ply_path, 'all')
        self.ply_states[frame_id] = ply_path

        print(f"[STATE] Saved frame {frame_id} ({state['point_count']} points)")

    def create_fresh_gaussian_with_ply(self, frame_id: int, sh_degree: int = 1,
                                       rotate_sh: bool = False,
                                       gaussian_class=None) -> Optional[object]:
        """Create fresh Gaussian from PLY (stable loading)."""
        if frame_id not in self.ply_states:
            print(f"[WARNING] No PLY state found for frame {frame_id}")
            return None

        ply_path = self.ply_states[frame_id]
        if not os.path.exists(ply_path):
            print(f"[WARNING] PLY file not found: {ply_path}")
            return None

        print(f"[STATE] Loading Gaussian from PLY: {ply_path}")

        # Create fresh TemporalGaussianModel instance
        fresh_gaussians = gaussian_class(sh_degree=sh_degree, rotate_sh=rotate_sh)
        fresh_gaussians.load_ply(ply_path)

        # Restore training statistics from memory
        if frame_id in self.states:
            state = self.states[frame_id]
            with torch.no_grad():
                point_count = fresh_gaussians.get_xyz.shape[0]
                if point_count == state['point_count']:
                    fresh_gaussians.max_radii2D = state['max_radii2D'].clone().detach()
                    fresh_gaussians.xyz_gradient_accum = state['xyz_gradient_accum'].clone().detach()
                    fresh_gaussians.denom = state['denom'].clone().detach()
                    print(f"[STATE] Training stats restored from memory")
                else:
                    # Size mismatch, reinitialize
                    fresh_gaussians.max_radii2D = torch.zeros((point_count,), device="cuda")
                    fresh_gaussians.xyz_gradient_accum = torch.zeros((point_count, 1), device="cuda")
                    fresh_gaussians.denom = torch.zeros((point_count, 1), device="cuda")
                    print(f"[WARNING] Size mismatch, initialized new training stats")

        # Clear CUDA cache
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        print(f"[STATE] Loaded {fresh_gaussians.get_xyz.shape[0]} points for frame {frame_id}")
        return fresh_gaussians

    def clear_old_states(self, keep_last_n: int = 3, keep_ply: bool = False):
        """
        Clear old states to save memory.

        Args:
            keep_last_n: Number of recent frames to keep in memory
            keep_ply: If True, keep PLY files on disk (only clear memory states)
        """
        if len(self.states) > keep_last_n:
            frames_to_delete = sorted(self.states.keys())[:-keep_last_n]
            for frame_id in frames_to_delete:
                # Delete memory state
                if frame_id in self.states:
                    del self.states[frame_id]

                # Delete PLY file (only if keep_ply=False)
                if not keep_ply and frame_id in self.ply_states:
                    ply_path = self.ply_states[frame_id]
                    if os.path.exists(ply_path):
                        os.remove(ply_path)
                    del self.ply_states[frame_id]

            torch.cuda.empty_cache()
            if keep_ply:
                print(f"[STATE] Cleared old memory states (keeping PLY files), last {keep_last_n} frames in memory")
            else:
                print(f"[STATE] Cleared old states, keeping last {keep_last_n} frames")