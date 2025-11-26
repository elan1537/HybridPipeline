#!/usr/bin/env python3
"""
Triple Scene Viewer with Combined 2×2 View

Interactive viewer with three windows:
1. Local Scene View: Manipulatable spheres + ground plane (local coordinates)
2. Recon Scene View: Transformed local scene + optional point cloud (recon coordinates)
3. Combined View (2×2): Four views in one window
   ┌─────────────────┬─────────────────┐
   │ Neural RGB      │ Neural Scene    │
   │ (Pure rendering)│ (With Fusion)   │
   ├─────────────────┼─────────────────┤
   │ Fusion Depth    │ Local RGB       │
   │ (Depth viz)     │ (Local mesh)    │
   └─────────────────┴─────────────────┘

Controls (in Local Scene View window):
Object Manipulation:
- T/R: Switch mode (translate/rotate)
- X/Y/Z/A: Axis locking
- Arrow keys: Manipulate selected sphere
- Space: Reset selected sphere
- U: Reset all spheres
- E: Export transformation

Sphere Selection:
- 1-9: Select sphere by index (1-9)
- 0: Select 10th sphere
- Tab: Select next sphere
- Note: Shift+Tab not available in Open3D

Frame Control:
- N: Next frame
- B: Previous frame (Back)
- Home: Jump to first frame
- End: Jump to last frame
- A: Toggle auto-play mode

Physics Control:
- P: Toggle physics for selected sphere
- O: Toggle physics for all spheres

View Options:
- F: Toggle depth fusion
- D: Toggle fusion depth display (4th window)
- Q: Quit

Usage:
    python triple_world_renderer.py \
        --project-dir /path/to/project \
        --frame-idx 0 \
        --show-point-cloud  # Optional: show point cloud in Recon Scene View
"""

import argparse
import asyncio
import sys
import os
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
import cv2
from concurrent.futures import ThreadPoolExecutor
import time

import open3d as o3d

# Add backend to path
project_path = Path(__file__).absolute().parent.parent.parent
backend_path = project_path / "backend"
sys.path.insert(0, str(backend_path))

from renderer.scene_renderers.streamable_gaussian import StreamableGaussian
from project_loader import load_project
from transform_to_matrix import M as LOCAL_TO_RECON_TRANSFORM

# Import new modules
from camera_manager import CameraManager
from scene_builder import SceneBuilder
from object_controller import ObjectController
from rendering_coordinator import RenderingCoordinator
from physics_playback_manager import PhysicsPlaybackManager


class TripleSceneViewer:
    """Triple scene interactive viewer (Local Scene + Recon Scene + Neural Scene)."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize the triple scene viewer."""
        self.config = config

        # Project configuration
        self.project_dir = Path(config['project_dir'])
        self.frame_idx = config.get('frame_idx', 1)
        self.width = config.get('width', 640)
        self.height = config.get('height', 720)

        # Camera intrinsics
        self.camera_width = config.get('camera_width', 2686)
        self.camera_height = config.get('camera_height', 2012)
        self.fx = config.get('fx', 1453.889)
        self.fy = config.get('fy', 1460.737)

        # Scene configuration
        self.sphere_radius = config.get('sphere_radius', 0.2)
        self.ground_size = config.get('ground_size', 10.0)
        self.voxel_size = config.get('voxel_size', 0.02)
        self.show_point_cloud = config.get('show_point_cloud', False)

        # Transformation matrix
        self.local_to_recon_transform = LOCAL_TO_RECON_TRANSFORM.copy()

        # Near/far planes (configurable via CLI)
        self.near_plane = config.get('near_plane', 0.3)
        self.far_plane = config.get('far_plane', 1000.0)

        # Initialize components
        self.camera_manager = CameraManager(
            self.local_to_recon_transform,
            self.near_plane,
            self.far_plane
        )

        self.scene_builder = SceneBuilder(
            self.project_dir,
            self.frame_idx,
            self.local_to_recon_transform,
            self.sphere_radius,
            self.ground_size,
            self.voxel_size,
            self.show_point_cloud
        )

        self.object_controller = ObjectController(
            self.local_to_recon_transform,
            sphere_configs=self.scene_builder.sphere_configs
        )

        # StreamableGaussian renderer
        self.renderer: StreamableGaussian = None
        self.rendering_coordinator: RenderingCoordinator = None

        # Cameras from project
        self.cameras_list: List[Dict] = []
        self.reference_cam_id = 0

        # Frame tracking
        self.current_frame_id = 0
        self.available_frame_ids: List[int] = []

        # Frame control
        self.auto_play_enabled = False
        self.playback_fps = config.get('playback_fps', 30.0)  # Frames per second for auto-play (matches video_fps)
        self.last_frame_advance_time = 0.0

        # Scene View Visualizers
        self.local_scene_vis = None
        self.recon_scene_vis = None

        # Scene geometries
        self.local_scene = {}
        self.recon_scene = {}

        # State
        self.should_exit = False
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.show_fusion_depth = False  # Flag for showing fusion depth window

        # Video recording
        self.video_output_path = config.get('video_output', './combined_view.mp4')
        self.video_writer = None  # Will be initialized when V key is pressed
        self.video_fps = config.get('video_fps', 30.0)
        self.is_recording = False  # Recording state (toggle with V key)

        # Physics simulation
        self.last_physics_update = time.time()

    async def initialize(self) -> bool:
        """Initialize renderer and load project data."""
        # Initialize renderer
        self.renderer = StreamableGaussian()

        model_path = self.project_dir / f"checkpoints/frame_{self.frame_idx:06d}/gaussian.ply"
        weight_pattern = str(self.project_dir / "checkpoints/frame_{frame_id:06d}/gaussian.ply")
        checkpoints_dir = str(self.project_dir / "checkpoints")

        renderer_config = {
            'model_path': str(model_path),
            'weight_path_pattern': weight_pattern,
            'checkpoints_dir': checkpoints_dir,
        }

        success = await self.renderer.on_init(renderer_config)
        if not success:
            return False

        if self.renderer.available_frame_ids:
            self.available_frame_ids = self.renderer.available_frame_ids
            if self.available_frame_ids:
                self.current_frame_id = self.available_frame_ids[0]

        # Initialize rendering coordinator
        output_dir = self.project_dir / "output"
        self.rendering_coordinator = RenderingCoordinator(
            self.renderer,
            output_dir,
            self.near_plane,
            self.far_plane,
            enable_debug=False,  # Enable alpha/depth debug visualization
            local_to_recon_transform=self.local_to_recon_transform,
            ground_y_local=-0.5  # Ground plane at 0.0m in Local space
        )

        # Connect object_controller with rendering_coordinator for collision detection
        self.object_controller.set_rendering_coordinator(self.rendering_coordinator)

        # Physics playback manager
        self.physics_playback_manager = PhysicsPlaybackManager(
            self.object_controller,
            playback_fps=self.playback_fps,
            cache_size=100
        )

        # Load project data (cameras)
        result = load_project(
            str(self.project_dir),
            self.frame_idx,
            return_available_frames=True
        )
        cameras_raw, _, _, _ = result

        if isinstance(cameras_raw, dict):
            self.cameras_list = []
            for cam_id, cam_data in cameras_raw.items():
                cam_dict = {'id': int(cam_id) if isinstance(cam_id, str) else cam_id}
                cam_dict.update(cam_data)
                self.cameras_list.append(cam_dict)
        else:
            self.cameras_list = cameras_raw

        return True

    def setup_scenes(self):
        """Setup Local Scene and Recon Scene."""
        self.local_scene = self.scene_builder.create_local_scene()
        self.recon_scene = self.scene_builder.create_recon_scene_display()

    def _update_all_spheres(self):
        """Update all spheres in both scene views with collision feedback."""
        for i, sphere in enumerate(self.object_controller.spheres):
            sphere_transform = self.object_controller.get_transform_by_index(i)

            # Choose color based on collision state and selection
            is_selected = (i == self.object_controller.selected_sphere_idx)
            if sphere.is_colliding:
                color = [1.0, 0.2, 0.2]  # Red (collision)
            else:
                color = sphere.color  # Original color

            # Add selection highlight
            if is_selected:
                # Make selected sphere slightly brighter
                color = [min(1.0, c + 0.3) for c in color]

            # Update Local Scene View
            sphere_key = f'sphere_{i}'
            if sphere_key in self.local_scene:
                self.local_scene_vis.remove_geometry(self.local_scene[sphere_key], reset_bounding_box=False)
            self.local_scene[sphere_key] = self.scene_builder.update_sphere_by_index(i, sphere_transform, color, is_selected)
            self.local_scene_vis.add_geometry(self.local_scene[sphere_key], reset_bounding_box=False)

            # Update Recon Scene View
            if sphere_key in self.recon_scene:
                self.recon_scene_vis.remove_geometry(self.recon_scene[sphere_key], reset_bounding_box=False)
            self.recon_scene[sphere_key] = self.scene_builder.update_sphere_recon_by_index(i, sphere_transform, color, is_selected)
            self.recon_scene_vis.add_geometry(self.recon_scene[sphere_key], reset_bounding_box=False)

            # Update collision marker for this sphere
            marker_key = f'collision_marker_{i}'
            if marker_key in self.recon_scene:
                self.recon_scene_vis.remove_geometry(self.recon_scene[marker_key], reset_bounding_box=False)
                del self.recon_scene[marker_key]

            if sphere.collision_point is not None:
                marker = self.scene_builder.create_collision_marker(sphere.collision_point)
                self.recon_scene[marker_key] = marker
                self.recon_scene_vis.add_geometry(marker, reset_bounding_box=False)

    # ==================== Frame Navigation Methods ====================

    def _advance_frame(self):
        """Advance to the next frame."""
        if not self.available_frame_ids:
            print("[FRAME] No available frames")
            return

        current_idx = self.available_frame_ids.index(self.current_frame_id)
        next_idx = (current_idx + 1) % len(self.available_frame_ids)
        self.current_frame_id = self.available_frame_ids[next_idx]

        self._update_window_title()

    def _previous_frame(self):
        """Go to the previous frame."""
        if not self.available_frame_ids:
            print("[FRAME] No available frames")
            return

        current_idx = self.available_frame_ids.index(self.current_frame_id)
        prev_idx = (current_idx - 1) % len(self.available_frame_ids)
        self.current_frame_id = self.available_frame_ids[prev_idx]

        print(f"[FRAME] Rewound to frame {self.current_frame_id} ({prev_idx + 1}/{len(self.available_frame_ids)})")
        self._update_window_title()

    def _goto_first_frame(self):
        """Jump to the first frame."""
        if not self.available_frame_ids:
            print("[FRAME] No available frames")
            return

        self.current_frame_id = self.available_frame_ids[0]
        print(f"[FRAME] Jumped to first frame {self.current_frame_id}")
        self._update_window_title()

    def _goto_last_frame(self):
        """Jump to the last frame."""
        if not self.available_frame_ids:
            print("[FRAME] No available frames")
            return

        self.current_frame_id = self.available_frame_ids[-1]
        print(f"[FRAME] Jumped to last frame {self.current_frame_id}")
        self._update_window_title()

    def _toggle_auto_play(self):
        """Toggle auto-play mode."""
        self.auto_play_enabled = not self.auto_play_enabled
        status = "ENABLED" if self.auto_play_enabled else "DISABLED"
        print(f"[FRAME] Auto-play {status} (FPS: {self.playback_fps})")

        if self.auto_play_enabled:
            self.last_frame_advance_time = time.time()

        self._update_window_title()

    def _update_window_title(self):
        """Update Combined View window title with frame info."""
        if not self.available_frame_ids:
            title = "Combined View (2x2)"
        else:
            current_idx = self.available_frame_ids.index(self.current_frame_id)
            total_frames = len(self.available_frame_ids)
            auto_status = " [AUTO-PLAY]" if self.auto_play_enabled else ""
            title = f"Combined View (2x2) - Frame {self.current_frame_id} ({current_idx + 1}/{total_frames}){auto_status}"

        cv2.setWindowTitle("Combined View (2x2)", title)

    def _run_async_render(self, o3d_camera):
        """Run async render in a separate thread."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        camera_frame = self.camera_manager.to_camera_frame(
            o3d_camera,
            self.current_frame_id,
            self.available_frame_ids
        )

        render_output = loop.run_until_complete(
            self.rendering_coordinator.render_neural(camera_frame)
        )

        loop.close()
        return render_output

    def _animation_callback(self, vis):
        """Animation callback for Neural Scene View rendering."""
        if self.should_exit:
            return

        current_time = time.time()
        old_frame_id = self.current_frame_id

        # Auto-play frame advancement
        if self.auto_play_enabled and self.available_frame_ids:
            time_since_last_advance = current_time - self.last_frame_advance_time
            frame_interval = 1.0 / self.playback_fps

            if time_since_last_advance >= frame_interval:
                self._advance_frame()
                self.last_frame_advance_time = current_time

        # Neural rendering (updates depth map)
        o3d_camera = vis.get_view_control().convert_to_pinhole_camera_parameters()

        future = self.executor.submit(self._run_async_render, o3d_camera)
        render_output = future.result(timeout=1.0)

        # Frame-by-frame physics update (AFTER rendering, depth map is now available)
        if old_frame_id != self.current_frame_id:
            self.physics_playback_manager.on_frame_change(
                self.current_frame_id,
                old_frame_id,
                self.rendering_coordinator.collision_detector,
                self.object_controller.sphere_radius
            )
            self._update_all_spheres()

        # TEMPORARY: Direct physics update for testing
        # Update physics for all enabled spheres
        # Use fixed dt based on video_fps when recording, otherwise playback_fps
        if self.object_controller.physics_enabled_any():
            time_since_last_update = current_time - self.last_physics_update if hasattr(self, 'last_physics_update') else 0.0

            # Use video_fps when recording to match output video timing
            # Otherwise use playback_fps for interactive viewing
            target_fps = self.video_fps if self.is_recording else self.playback_fps
            frame_interval = 1.0 / target_fps  # Fixed time step

            # Accumulate time and update physics at fixed intervals
            if not hasattr(self, 'physics_time_accumulator'):
                self.physics_time_accumulator = 0.0

            self.physics_time_accumulator += time_since_last_update

            # Update physics with fixed time step
            if self.physics_time_accumulator >= frame_interval:
                # Use fixed dt for consistent physics simulation
                dt = frame_interval
                self.object_controller.update_physics(dt)
                self._update_all_spheres()
                self.physics_time_accumulator -= frame_interval

            self.last_physics_update = current_time

        if render_output is not None:
            # ==================== Create Combined 2×2 View ====================

            # 1. Local RGB View: Capture Local Scene
            local_rgb = self.local_scene_vis.capture_screen_float_buffer(do_render=False)
            local_rgb = (np.asarray(local_rgb) * 255).astype(np.uint8)
            local_rgb = cv2.cvtColor(local_rgb, cv2.COLOR_RGB2BGR)

            # 2. Neural RGB View: Pure RGB, no fusion
            neural_rgb = render_output.copy()

            # 3. Fusion RGB (Neural Scene): Fusion or RGB
            if self.rendering_coordinator.fusion_enabled:
                fused_output = self.rendering_coordinator.fuse_depth_based()
                fusion_rgb = fused_output if fused_output is not None else render_output
            else:
                fusion_rgb = render_output

            # 4. Fusion Depth View: Depth visualization
            fusion_depth = self.rendering_coordinator.get_fusion_depth_visualization(
                with_contours=True,
                num_contour_levels=10
            )
            if fusion_depth is None:
                # Create black placeholder if no depth
                fusion_depth = np.zeros_like(render_output)

            # Ensure all views have same size
            target_h, target_w = render_output.shape[:2]
            local_rgb = cv2.resize(local_rgb, (target_w, target_h))
            neural_rgb = cv2.resize(neural_rgb, (target_w, target_h))
            fusion_rgb = cv2.resize(fusion_rgb, (target_w, target_h))
            fusion_depth = cv2.resize(fusion_depth, (target_w, target_h))

            # Add labels to each view
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(local_rgb, "1. Local RGB", (10, 30), font, 1.0, (0, 255, 0), 2)
            cv2.putText(neural_rgb, "2. Neural RGB", (10, 30), font, 1.0, (0, 255, 0), 2)
            cv2.putText(fusion_rgb, "3. Fusion RGB" if self.rendering_coordinator.fusion_enabled else "3. Neural RGB",
                       (10, 30), font, 1.0, (0, 255, 0), 2)
            cv2.putText(fusion_depth, "4. Fusion Depth", (10, 30), font, 1.0, (255, 255, 255), 2)

            # Combine into 2×2 grid in order: 2-3-1-4
            # Top row: 1 (Local RGB) | 3 (Fusion RGB)
            # Bottom row: 2 (Neural RGB) | 4 (Fusion Depth)

            top_row = np.hstack([local_rgb, fusion_rgb])
            bottom_row = np.hstack([neural_rgb, fusion_depth])
            combined_view = np.vstack([top_row, bottom_row])

            # Display combined view
            cv2.imshow("Combined View (2x2)", combined_view)

            # Save to video (only when recording)
            if self.is_recording and self.video_writer is not None:
                self.video_writer.write(combined_view)

    # ==================== Keyboard Callbacks ====================

    def _key_quit(self, _vis):
        self.should_exit = True
        return False

    def _key_mode_translate(self, _vis):
        self.object_controller.set_mode("translate")
        return False

    def _key_mode_rotate(self, _vis):
        self.object_controller.set_mode("rotate")
        return False

    def _key_lock_x(self, _vis):
        self.object_controller.lock_axis(0)
        return False

    def _key_lock_y(self, _vis):
        self.object_controller.lock_axis(1)
        return False

    def _key_lock_z(self, _vis):
        self.object_controller.lock_axis(2)
        return False

    def _key_unlock_axis(self, _vis):
        self.object_controller.lock_axis(None)
        return False

    def _key_arrow_right(self, _vis):
        print("[KEY] Arrow RIGHT pressed")
        self.object_controller.manipulate(0, 1)
        self._update_all_spheres()
        if self.object_controller.physics_enabled:
            self.physics_playback_manager.on_manual_movement(self.current_frame_id)
        return False

    def _key_arrow_left(self, _vis):
        print("[KEY] Arrow LEFT pressed")
        self.object_controller.manipulate(0, -1)
        self._update_all_spheres()
        if self.object_controller.physics_enabled:
            self.physics_playback_manager.on_manual_movement(self.current_frame_id)
        return False

    def _key_arrow_up(self, _vis):
        print("[KEY] Arrow UP pressed")
        self.object_controller.manipulate(1, 1)
        self._update_all_spheres()
        if self.object_controller.physics_enabled:
            self.physics_playback_manager.on_manual_movement(self.current_frame_id)
        return False

    def _key_arrow_down(self, _vis):
        print("[KEY] Arrow DOWN pressed")
        self.object_controller.manipulate(1, -1)
        self._update_all_spheres()
        if self.object_controller.physics_enabled:
            self.physics_playback_manager.on_manual_movement(self.current_frame_id)
        return False

    def _key_page_up(self, _vis):
        print("[KEY] Page UP pressed")
        self.object_controller.manipulate(2, 1)
        self._update_all_spheres()
        if self.object_controller.physics_enabled:
            self.physics_playback_manager.on_manual_movement(self.current_frame_id)
        return False

    def _key_page_down(self, _vis):
        print("[KEY] Page DOWN pressed")
        self.object_controller.manipulate(2, -1)
        self._update_all_spheres()
        if self.object_controller.physics_enabled:
            self.physics_playback_manager.on_manual_movement(self.current_frame_id)
        return False

    def _key_reset_sphere(self, _vis):
        self.object_controller.reset()
        self._update_all_spheres()
        return False

    def _key_reset_all_spheres(self, _vis):
        """Reset all spheres to initial positions (Shift+R)."""
        self.object_controller.reset_all()
        self._update_all_spheres()
        print("[KEY] Shift+R pressed - All spheres reset to initial positions")
        return False

    def _key_select_next(self, _vis):
        """Select next sphere (Tab)."""
        self.object_controller.select_next_sphere()
        self._update_all_spheres()
        print(f"[KEY] Tab pressed - Selected sphere {self.object_controller.selected_sphere_idx}")
        return False

    def _key_select_previous(self, _vis):
        """Select previous sphere (Shift+Tab)."""
        self.object_controller.select_previous_sphere()
        self._update_all_spheres()
        print(f"[KEY] Shift+Tab pressed - Selected sphere {self.object_controller.selected_sphere_idx}")
        return False

    def _key_toggle_physics_all(self, _vis):
        """Toggle physics for all spheres (Shift+P)."""
        if self.object_controller.physics_enabled_any():
            self.object_controller.disable_physics_all()
            self.physics_playback_manager.on_physics_disabled()
            print("[KEY] Shift+P pressed - Physics disabled for all spheres")
        else:
            self.object_controller.enable_physics_all()
            self.physics_playback_manager.on_physics_enabled(self.current_frame_id)
            print("[KEY] Shift+P pressed - Physics enabled for all spheres")
        return False

    def _key_export(self, _vis):
        output_dir = self.project_dir / "sphere_export"
        self.object_controller.export_transformation(output_dir)
        return False

    def _key_toggle_fusion(self, _vis):
        self.rendering_coordinator.toggle_fusion()
        return False

    def _key_toggle_depth_display(self, _vis):
        """Toggle fusion depth display window (D key)."""
        self.show_fusion_depth = not self.show_fusion_depth
        if self.show_fusion_depth:
            print("[KEY] D pressed - Fusion Depth Display ENABLED")
            # Create fusion depth window if it doesn't exist
            cv2.namedWindow("Fusion Depth View", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Fusion Depth View", self.width, self.height)
            # Position in 2x2 grid: bottom-right corner
            cv2.moveWindow("Fusion Depth View", self.width + 70, self.height + 130)
        else:
            print("[KEY] D pressed - Fusion Depth Display DISABLED")
            # Destroy the fusion depth window
            cv2.destroyWindow("Fusion Depth View")
        return False

    def _key_save_collision_mask(self, _vis):
        """Save collision boundary mask (C key)."""
        if self.rendering_coordinator:
            mask = self.rendering_coordinator.create_collision_mask()
            if mask is not None:
                os.makedirs("./debug_output", exist_ok=True)
                output_path = f"./debug_output/collision_mask_frame_{self.current_frame_id:06d}.png"
                cv2.imwrite(str(output_path), mask)
                print(f"[KEY] C pressed - Collision mask saved to {output_path}")
            else:
                print("[KEY] C pressed - Failed to generate collision mask (no depth data)")
        return False

    def _key_toggle_recording(self, _vis):
        """Toggle video recording (V key) + Auto-start physics and playback."""
        if not self.is_recording:
            # Start recording
            # Get current frame size from Combined View window
            # We need to wait until we have a valid combined_view
            # For now, use default resolution (will be updated on first frame)
            h, w = self.height * 2, self.width * 2  # 2x2 grid

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')

            # Add timestamp to filename to avoid overwriting
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"./combined_view_{timestamp}.mp4"

            self.video_writer = cv2.VideoWriter(
                output_path,
                fourcc,
                self.video_fps,
                (w, h)
            )
            self.video_output_path = output_path
            self.is_recording = True
            print(f"[VIDEO] Recording STARTED: {output_path} at {self.video_fps} FPS")
            print(f"[VIDEO] Resolution: {w}x{h}")

            # Auto-trigger G key (physics reset + start)
            print("[VIDEO] Auto-triggering physics reset (G key)")
            self._key_start_physics(_vis)

            # Auto-trigger A key (auto-play)
            if not self.auto_play_enabled:
                print("[VIDEO] Auto-triggering auto-play (A key)")
                self._toggle_auto_play()
        else:
            # Stop recording
            if self.video_writer is not None:
                self.video_writer.release()
                print(f"[VIDEO] Recording STOPPED and saved to: {self.video_output_path}")
                self.video_writer = None
            self.is_recording = False

            # Auto-stop auto-play
            if self.auto_play_enabled:
                print("[VIDEO] Auto-stopping auto-play")
                self._toggle_auto_play()
        return False

    def _key_save_transform_overlay(self, _vis):
        """Save transform validation overlay (T key)."""
        if self.rendering_coordinator:
            # Extract Local mesh points from scene builder
            local_mesh_points = None
            transform_matrix = self.local_to_recon_transform

            # Try to get mesh points from scene_builder
            if hasattr(self, 'scene_builder') and hasattr(self.scene_builder, 'local_mesh'):
                local_mesh = self.scene_builder.local_mesh
                if local_mesh is not None and hasattr(local_mesh, 'vertices'):
                    local_mesh_points = np.asarray(local_mesh.vertices)

            overlay = self.rendering_coordinator.create_transform_overlay(
                local_mesh_points, transform_matrix
            )
            if overlay is not None:
                os.makedirs("./debug_output", exist_ok=True)
                output_path = f"./debug_output/transform_overlay_frame_{self.current_frame_id:06d}.png"
                cv2.imwrite(str(output_path), overlay)
                print(f"[KEY] T pressed - Transform overlay saved to {output_path}")
            else:
                print("[KEY] T pressed - Failed to generate transform overlay")
        return False

    def _key_save_combined_debug(self, _vis):
        """Save combined debug view with 4 panels (M key)."""
        if self.rendering_coordinator:
            # Extract Local mesh points
            local_mesh_points = None
            transform_matrix = self.local_to_recon_transform

            if hasattr(self, 'scene_builder') and hasattr(self.scene_builder, 'local_mesh'):
                local_mesh = self.scene_builder.local_mesh
                if local_mesh is not None and hasattr(local_mesh, 'vertices'):
                    local_mesh_points = np.asarray(local_mesh.vertices)

            combined = self.rendering_coordinator.create_combined_debug_view(
                local_mesh_points, transform_matrix
            )
            if combined is not None:
                os.makedirs("./debug_output", exist_ok=True)
                output_path = f"./debug_output/debug_combined_frame_{self.current_frame_id:06d}.png"
                cv2.imwrite(str(output_path), combined)
                print(f"[KEY] M pressed - Combined debug view saved to {output_path}")
            else:
                print("[KEY] M pressed - Failed to generate combined debug view")
        return False

    def _key_save_cross_section(self, _vis):
        """Save 3D cross-section view (S key)."""
        if self.rendering_coordinator and self.object_controller:
            # Get current selected sphere
            selected_sphere = self.object_controller.get_selected_sphere()
            if selected_sphere is None:
                print("[KEY] S pressed - No sphere selected for cross-section view")
                return False

            # Get sphere position in Recon space using ObjectController
            selected_idx = self.object_controller.selected_sphere_idx
            sphere_center = self.object_controller.get_sphere_center_recon_space_by_index(selected_idx)
            sphere_radius = selected_sphere.radius * self.object_controller.scale_factor

            # Generate cross-section view
            cross_section = self.rendering_coordinator.create_cross_section_view(
                sphere_center, sphere_radius
            )

            if cross_section is not None:
                os.makedirs("./debug_output", exist_ok=True)
                output_path = f"./debug_output/cross_section_frame_{self.current_frame_id:06d}.png"
                cv2.imwrite(str(output_path), cross_section)
                print(f"[KEY] S pressed - Cross-section view saved to {output_path}")
                print(f"[KEY] Sphere center (Recon): [{sphere_center[0]:.2f}, {sphere_center[1]:.2f}, {sphere_center[2]:.2f}]")
            else:
                print("[KEY] S pressed - Failed to generate cross-section view (no depth data)")
        return False

    def _key_toggle_physics(self, _vis):
        """Toggle physics simulation (P key)."""
        if self.object_controller.physics_enabled:
            self.object_controller.disable_physics()
            self.physics_playback_manager.on_physics_disabled()
        else:
            self.object_controller.enable_physics()
            self.physics_playback_manager.on_physics_enabled(self.current_frame_id)
        return False

    def _key_start_physics(self, _vis):
        """Start physics simulation (G key - Go!)."""
        # Reset all spheres to initial positions
        self.object_controller.reset_all()
        print("[KEY] G pressed - All spheres reset to initial positions")

        # Disable physics first (to ensure clean state)
        if self.object_controller.physics_enabled:
            self.object_controller.disable_physics()
            self.physics_playback_manager.on_physics_disabled()

        # Enable physics for free-fall
        self.object_controller.enable_physics()
        self.physics_playback_manager.on_physics_enabled(self.current_frame_id)
        print("[KEY] G pressed - Physics STARTED (free-fall)")

        return False

    def _key_next_frame(self, _vis):
        """Advance to next frame (N key)."""
        print("[KEY] N pressed - Next frame")
        self._advance_frame()
        return False

    def _key_previous_frame(self, _vis):
        """Go to previous frame (B key)."""
        print("[KEY] B pressed - Previous frame")
        self._previous_frame()
        return False

    def _key_first_frame(self, _vis):
        """Jump to first frame (Home key)."""
        print("[KEY] Home pressed - First frame")
        self._goto_first_frame()
        return False

    def _key_last_frame(self, _vis):
        """Jump to last frame (End key)."""
        print("[KEY] End pressed - Last frame")
        self._goto_last_frame()
        return False

    def _key_toggle_auto_play(self, _vis):
        """Toggle auto-play mode (A key)."""
        print("[KEY] A pressed - Toggle auto-play")
        self._toggle_auto_play()
        return False

    def run(self):
        """Run the triple-world interactive viewer."""
        # Setup scenes
        self.setup_scenes()

        # ==================== Local Scene View Window ====================
        # 2x2 Grid Layout: Top-left
        self.local_scene_vis = o3d.visualization.VisualizerWithKeyCallback()
        self.local_scene_vis.create_window(
            window_name="Local Scene View (Manipulatable)",
            width=self.width,
            height=self.height,
            left=50,
            top=100
        )

        for geom in self.local_scene.values():
            self.local_scene_vis.add_geometry(geom)

        opt_local = self.local_scene_vis.get_render_option()
        opt_local.background_color = np.array([0.9, 0.9, 0.9])
        opt_local.line_width = 2.0

        # Setup camera
        view_local = self.local_scene_vis.get_view_control()
        cam0_local_params = self.camera_manager.get_cam0_view_local(
            self.cameras_list,
            self.reference_cam_id,
            self.width,
            self.height,
            self.camera_width,
            self.camera_height,
            self.fx,
            self.fy
        )

        if cam0_local_params:
            view_local.set_lookat([0.0, 0.0, 0.0])
            view_local.set_front((cam0_local_params.extrinsic[:3, :3].T @ np.array([0, 0, -1])).tolist())
            view_local.set_up([0, 1, 0])
        else:
            view_local.set_lookat([0.0, 0.0, 0.0])
            view_local.set_front([0.577, -0.408, -0.707])
            view_local.set_up([0.0, 1.0, 0.0])
            view_local.set_zoom(0.7)

        # Set constant clipping planes
        view_local.set_constant_z_near(self.near_plane)
        view_local.set_constant_z_far(self.far_plane)
        print(f"[Local ViewControl] Set z_near={self.near_plane}, z_far={self.far_plane}")

        # Save initial camera state
        self.camera_manager.last_local_camera = view_local.convert_to_pinhole_camera_parameters()

        # Register keyboard callbacks
        self.local_scene_vis.register_key_callback(ord("Q"), self._key_quit)
        self.local_scene_vis.register_key_callback(ord("T"), self._key_mode_translate)
        self.local_scene_vis.register_key_callback(ord("R"), self._key_mode_rotate)
        self.local_scene_vis.register_key_callback(ord("X"), self._key_lock_x)
        self.local_scene_vis.register_key_callback(ord("Y"), self._key_lock_y)
        self.local_scene_vis.register_key_callback(ord("Z"), self._key_lock_z)
        self.local_scene_vis.register_key_callback(ord("A"), self._key_unlock_axis)
        self.local_scene_vis.register_key_callback(262, self._key_arrow_right)
        self.local_scene_vis.register_key_callback(263, self._key_arrow_left)
        self.local_scene_vis.register_key_callback(265, self._key_arrow_up)
        self.local_scene_vis.register_key_callback(264, self._key_arrow_down)
        self.local_scene_vis.register_key_callback(266, self._key_page_up)
        self.local_scene_vis.register_key_callback(267, self._key_page_down)
        self.local_scene_vis.register_key_callback(32, self._key_reset_sphere)
        self.local_scene_vis.register_key_callback(ord("E"), self._key_export)
        self.local_scene_vis.register_key_callback(ord("F"), self._key_toggle_fusion)
        self.local_scene_vis.register_key_callback(ord("D"), self._key_toggle_depth_display)
        self.local_scene_vis.register_key_callback(ord("P"), self._key_toggle_physics)
        self.local_scene_vis.register_key_callback(ord("G"), self._key_start_physics)  # Start physics

        # Video recording callback
        self.local_scene_vis.register_key_callback(ord("V"), self._key_toggle_recording)

        # Debug visualization callbacks
        self.local_scene_vis.register_key_callback(ord("C"), self._key_save_collision_mask)
        self.local_scene_vis.register_key_callback(ord("M"), self._key_save_combined_debug)
        self.local_scene_vis.register_key_callback(ord("S"), self._key_save_cross_section)

        # Frame control callbacks
        self.local_scene_vis.register_key_callback(ord("N"), self._key_next_frame)
        self.local_scene_vis.register_key_callback(ord("B"), self._key_previous_frame)
        self.local_scene_vis.register_key_callback(268, self._key_first_frame)  # Home key
        self.local_scene_vis.register_key_callback(269, self._key_last_frame)   # End key
        self.local_scene_vis.register_key_callback(ord("A"), self._key_toggle_auto_play)

        # Multi-sphere control callbacks
        self.local_scene_vis.register_key_callback(9, self._key_select_next)  # Tab
        # Note: Shift+Tab and other modifier keys might need special handling
        self.local_scene_vis.register_key_callback(ord("O"), self._key_toggle_physics_all)  # O for all physics
        self.local_scene_vis.register_key_callback(ord("U"), self._key_reset_all_spheres)  # U for reset all

        # Number keys for sphere selection
        for i in range(10):
            key_code = ord('0') + i if i > 0 else ord('0')  # 1-9, 0 for 10th sphere
            def make_select_callback(idx):
                def callback(_vis):
                    actual_idx = idx if idx > 0 else 9  # Map 0 key to index 9
                    if self.object_controller.select_sphere(actual_idx):
                        self._update_all_spheres()
                        print(f"[KEY] {idx} pressed - Selected sphere {actual_idx}")
                    return False
                return callback
            self.local_scene_vis.register_key_callback(key_code, make_select_callback(i))

        # ==================== Recon Scene View Window ====================
        # 2x2 Grid Layout: Top-right
        self.recon_scene_vis = o3d.visualization.VisualizerWithKeyCallback()
        self.recon_scene_vis.create_window(
            window_name="Recon Scene View (Transformed)",
            width=self.width,
            height=self.height,
            left=self.width + 70,
            top=100
        )

        for geom in self.recon_scene.values():
            self.recon_scene_vis.add_geometry(geom)

        opt_recon = self.recon_scene_vis.get_render_option()
        opt_recon.background_color = np.array([0.1, 0.1, 0.1])
        opt_recon.point_size = 2.0
        opt_recon.line_width = 2.0

        # Setup camera
        view_recon = self.recon_scene_vis.get_view_control()
        if self.scene_builder.point_cloud is not None:
            self.recon_scene_vis.reset_view_point(True)
        else:
            view_recon.set_lookat([0.0, 0.0, 0.0])
            view_recon.set_front([0.5, -0.3, -0.8])
            view_recon.set_up([0.0, -1.0, 0.0])
            view_recon.set_zoom(0.3)

        # Set constant clipping planes
        view_recon.set_constant_z_near(self.near_plane)
        view_recon.set_constant_z_far(self.far_plane)
        print(f"[Recon ViewControl] Set z_near={self.near_plane}, z_far={self.far_plane}")

        self.recon_scene_vis.register_key_callback(ord("Q"), self._key_quit)

        # Setup cam0
        self.camera_manager.setup_cam0_camera(
            self.recon_scene_vis,
            self.cameras_list,
            self.reference_cam_id,
            self.width,
            self.height,
            self.camera_width,
            self.camera_height,
            self.fx,
            self.fy
        )

        self.camera_manager.last_recon_camera = self.recon_scene_vis.get_view_control().convert_to_pinhole_camera_parameters()

        # Register animation callback
        self.recon_scene_vis.register_animation_callback(self._animation_callback)

        # ==================== Combined View Window (2×2 Grid) ====================
        # Shows 4 views in one window:
        # Top-left: Neural RGB | Top-right: Neural Scene (Fusion)
        # Bottom-left: Fusion Depth | Bottom-right: Local RGB
        cv2.namedWindow("Combined View (2x2)", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Combined View (2x2)", self.width * 2, self.height * 2)
        cv2.moveWindow("Combined View (2x2)", 50, self.height + 130)

        # Initialize window title with frame info
        self._update_window_title()

        # ==================== Enable physics for all spheres initially ====================
        print("[INIT] Enabling physics for all 10 spheres (free-fall mode)...")
        # Skip collision check to enable all spheres regardless of initial position
        self.object_controller.enable_physics_all(skip_collision_check=True)
        if hasattr(self, 'physics_playback_manager'):
            self.physics_playback_manager.on_physics_enabled(self.current_frame_id)
        print("[INIT] All spheres now have gravity enabled - Free fall!")

        # ==================== Run all scene views ====================
        while not self.should_exit:
            # Update Local Scene View
            if not self.local_scene_vis.poll_events():
                break
            self.local_scene_vis.update_renderer()

            # Update Recon Scene View
            if not self.recon_scene_vis.poll_events():
                break

            # Camera synchronization (BEFORE rendering Recon Scene)
            self.camera_manager.sync_cameras(self.local_scene_vis, self.recon_scene_vis)

            # Render Recon Scene (with synchronized camera)
            self.recon_scene_vis.update_renderer()

            # Update Recon Scene frame for fusion (AFTER rendering)
            self.rendering_coordinator.update_recon_scene_frame(self.recon_scene_vis)

            # Neural rendering (uses updated Recon Scene depth for fusion)
            self._animation_callback(self.recon_scene_vis)

            # Handle cv2 events
            key = cv2.waitKey(1)
            if key == ord('q') or key == ord('Q') or key == 27:
                self.should_exit = True

        # Cleanup video writer
        if self.video_writer is not None:
            self.video_writer.release()
            print(f"[VIDEO] Recording saved to {self.video_output_path}")
            self.video_writer = None

        self.local_scene_vis.destroy_window()
        self.recon_scene_vis.destroy_window()
        self.executor.shutdown(wait=True)
        cv2.destroyAllWindows()


async def main_async():
    """Main async entry point."""
    parser = argparse.ArgumentParser(
        description="Triple Scene Viewer (Local Scene + Recon Scene + Neural Scene)"
    )

    parser.add_argument('--project-dir', type=str, required=True,
                       help='Project directory (cameras.json + checkpoints/)')
    parser.add_argument('--frame-idx', type=int, default=1,
                       help='Frame index to load (default: 1)')
    parser.add_argument('--width', type=int, default=1280,
                       help='Window width per viewport (default: 1280)')
    parser.add_argument('--height', type=int, default=720,
                       help='Window height per viewport (default: 720)')
    parser.add_argument('--sphere-radius', type=float, default=0.2,
                       help='Sphere radius in meters (default: 0.2)')
    parser.add_argument('--ground-size', type=float, default=10.0,
                       help='Ground plane size in meters (default: 10.0)')
    parser.add_argument('--voxel-size', type=float, default=0.02,
                       help='Point cloud voxel downsampling size (default: 0.02)')
    parser.add_argument('--camera-width', type=int, default=2686,
                       help='Original camera width (default: 2686)')
    parser.add_argument('--camera-height', type=int, default=2012,
                       help='Original camera height (default: 2012)')
    parser.add_argument('--fx', type=float, default=1453.889,
                       help='Focal length x (default: 1453.889)')
    parser.add_argument('--fy', type=float, default=1460.737,
                       help='Focal length y (default: 1460.737)')
    parser.add_argument('--near-plane', type=float, default=0.03,
                       help='Near clipping plane distance (default: 0.03)')
    parser.add_argument('--far-plane', type=float, default=100.0,
                       help='Far clipping plane distance (default: 100.0)')
    parser.add_argument('--show-point-cloud', action='store_true', default=False,
                       help='Show point cloud in Recon Scene View (default: False)')
    parser.add_argument('--playback-fps', type=float, default=30.0,
                       help='Auto-play frame rate in FPS (default: 30.0)')
    parser.add_argument('--video-output', type=str, default='./combined_view.mp4',
                       help='Output video file path (default: ./combined_view.mp4)')
    parser.add_argument('--video-fps', type=float, default=30.0,
                       help='Video recording FPS (default: 30.0)')

    args = parser.parse_args()

    config = {
        'project_dir': args.project_dir,
        'frame_idx': args.frame_idx,
        'width': args.width,
        'height': args.height,
        'sphere_radius': args.sphere_radius,
        'ground_size': args.ground_size,
        'voxel_size': args.voxel_size,
        'camera_width': args.camera_width,
        'camera_height': args.camera_height,
        'fx': args.fx,
        'fy': args.fy,
        'near_plane': args.near_plane,
        'far_plane': args.far_plane,
        'show_point_cloud': args.show_point_cloud,
        'playback_fps': args.playback_fps,
        'video_output': args.video_output,
        'video_fps': args.video_fps,
    }

    viewer = TripleSceneViewer(config)
    success = await viewer.initialize()

    if not success:
        return 1

    viewer.run()
    return 0


def main():
    """Main entry point."""
    exit_code = asyncio.run(main_async())
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
