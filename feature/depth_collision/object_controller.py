"""Object manipulation controller."""

from pathlib import Path
from typing import Optional
import numpy as np
import time

from transform_math import (
    compose_transform,
    decompose_transform,
    rotation_matrix_from_axis_angle,
    export_transform_matrix,
    export_transform_params
)
from transform_to_matrix import get_scale_factor
from physics_engine import PhysicsEngine


class SphereObject:
    """Individual sphere object with its own physics state."""

    def __init__(self, position: np.ndarray, radius: float, color: list, idx: int):
        """
        Initialize a sphere object.

        Args:
            position: Initial position
            radius: Sphere radius
            color: RGB color values [0-1]
            idx: Sphere index
        """
        self.idx = idx
        self.initial_position = position.copy()
        self.translation = position.copy()
        self.rotation = np.eye(3)
        self.radius = radius
        self.color = color

        # Physics state
        # NOTE: Gravity sign depends on Y-axis direction in transform
        # Gravity reduced to 0.56 m/s² for slow-motion effect (3× slower fall)
        # Scale factor will be set by ObjectController after initialization
        self.physics_engine = PhysicsEngine(
            gravity_local=0.56,  # 0.56 m/s² for slow-motion fall (1/9 of 5.0, gives 3× longer fall time)
            damping=0.95,
            friction=0.3,
            scale_factor=1.0  # Will be updated by ObjectController
        )
        self.physics_enabled = False
        self.is_colliding = False
        self.collision_point = None
        self.last_update_time = None

    def get_transform(self) -> np.ndarray:
        """Get 4x4 transformation matrix."""
        return compose_transform(self.translation, self.rotation, np.array([1.0, 1.0, 1.0]))

    def reset(self):
        """Reset to initial position."""
        self.translation = self.initial_position.copy()
        self.rotation = np.eye(3)
        self.physics_engine.velocity = np.array([0.0, 0.0, 0.0])
        self.physics_engine.is_grounded = False
        self.is_colliding = False
        self.collision_point = None


class ObjectController:
    """Manages multiple sphere objects."""

    def __init__(
        self,
        local_to_recon_transform: np.ndarray,
        sphere_configs: list,  # List of (position, radius, color) tuples
    ):
        """
        Initialize object controller with multiple spheres.

        Args:
            local_to_recon_transform: Transformation matrix (Local → Reconstructed)
            sphere_configs: List of (position, radius, color) tuples for each sphere
        """
        self.local_to_recon_transform = local_to_recon_transform

        # Extract scale factor from transform matrix
        self.scale_factor = get_scale_factor(local_to_recon_transform)
        print(f"[OBJECT_CONTROLLER] Scale factor (Local→Recon): {self.scale_factor:.4f}")
        print(f"[OBJECT_CONTROLLER] 1 meter Local = {self.scale_factor:.4f} Recon units")

        # Create sphere objects
        self.spheres = []
        for i, (position, radius, color) in enumerate(sphere_configs):
            sphere = SphereObject(position, radius, color, i)
            # Configure physics engine with proper scale factor
            sphere.physics_engine.scale_factor = self.scale_factor
            sphere.physics_engine.gravity = sphere.physics_engine.gravity_local * self.scale_factor
            self.spheres.append(sphere)

        # Manipulation state
        self.selected_sphere_idx = 0 if self.spheres else None
        self.mode = "translate"  # "translate" or "rotate"
        self.locked_axis: Optional[int] = None  # None, 0 (X), 1 (Y), or 2 (Z)
        self.step_size = 0.1  # Translation step or rotation angle (degrees)

        # Collision detection
        self.rendering_coordinator = None

        # Backward compatibility properties
        if self.spheres:
            # These are now references to the selected sphere
            self.sphere_radius = self.spheres[0].radius
            self.physics_enabled = False
            self.is_colliding = False
            self.collision_point = None
            self.physics_engine = None
            # Update all backward compatibility properties
            self._update_backward_compat_properties()
        else:
            # Initialize with default values if no spheres
            self.translation = np.array([0.0, 0.0, 0.0])
            self.rotation = np.eye(3)
            self.sphere_radius = 0.15
            self.physics_enabled = False
            self.is_colliding = False
            self.collision_point = None
            self.physics_engine = None

    def set_rendering_coordinator(self, rendering_coordinator):
        """Set rendering coordinator for collision detection."""
        self.rendering_coordinator = rendering_coordinator

    def _update_backward_compat_properties(self):
        """Update backward compatibility properties from selected sphere."""
        if self.selected_sphere_idx is not None and self.selected_sphere_idx < len(self.spheres):
            sphere = self.spheres[self.selected_sphere_idx]
            self.translation = sphere.translation
            self.rotation = sphere.rotation
            self.sphere_radius = sphere.radius
            self.is_colliding = sphere.is_colliding
            self.collision_point = sphere.collision_point
            self.physics_engine = sphere.physics_engine
            self.physics_enabled = sphere.physics_enabled
        else:
            # Default values if no sphere selected
            self.translation = np.array([0.0, 0.0, 0.0])
            self.rotation = np.eye(3)
            self.physics_engine = None
            self.physics_enabled = False

    def select_sphere(self, idx: int) -> bool:
        """
        Select a sphere by index.

        Args:
            idx: Sphere index to select

        Returns:
            True if selection successful, False if index out of range
        """
        if 0 <= idx < len(self.spheres):
            self.selected_sphere_idx = idx
            # Update backward compatibility properties
            self._update_backward_compat_properties()
            return True
        return False

    def select_next_sphere(self):
        """Select the next sphere in the list."""
        if self.spheres and self.selected_sphere_idx is not None:
            self.selected_sphere_idx = (self.selected_sphere_idx + 1) % len(self.spheres)
            self.select_sphere(self.selected_sphere_idx)

    def select_previous_sphere(self):
        """Select the previous sphere in the list."""
        if self.spheres and self.selected_sphere_idx is not None:
            self.selected_sphere_idx = (self.selected_sphere_idx - 1) % len(self.spheres)
            self.select_sphere(self.selected_sphere_idx)

    def get_selected_sphere(self):
        """Get currently selected sphere object.

        Returns:
            SphereObject or None if no sphere is selected
        """
        if self.selected_sphere_idx is not None and self.selected_sphere_idx < len(self.spheres):
            return self.spheres[self.selected_sphere_idx]
        return None

    def get_transform(self) -> np.ndarray:
        """Get current selected sphere transformation matrix (4x4)."""
        if self.selected_sphere_idx is not None and self.selected_sphere_idx < len(self.spheres):
            return self.spheres[self.selected_sphere_idx].get_transform()
        return np.eye(4)

    def get_transform_by_index(self, idx: int) -> np.ndarray:
        """Get transformation matrix for a specific sphere."""
        if 0 <= idx < len(self.spheres):
            return self.spheres[idx].get_transform()
        return np.eye(4)

    def get_all_transforms(self) -> list:
        """Get transformation matrices for all spheres."""
        return [sphere.get_transform() for sphere in self.spheres]

    def get_sphere_center_recon_space(self) -> np.ndarray:
        """Get selected sphere center position in Recon space."""
        full_transform = self.local_to_recon_transform @ self.get_transform()
        return full_transform[:3, 3]

    def get_sphere_center_recon_space_by_index(self, idx: int) -> np.ndarray:
        """Get sphere center position in Recon space by index."""
        transform = self.get_transform_by_index(idx)
        full_transform = self.local_to_recon_transform @ transform
        return full_transform[:3, 3]

    def manipulate(self, axis_idx: int, direction: int):
        """
        Manipulate selected sphere (translate or rotate).

        Args:
            axis_idx: Axis index (0=X, 1=Y, 2=Z)
            direction: Direction (+1 or -1)
        """
        if self.selected_sphere_idx is None:
            return

        sphere = self.spheres[self.selected_sphere_idx]
        print(f"[MANIPULATE] Sphere {self.selected_sphere_idx}: axis={axis_idx}, direction={direction}, mode={self.mode}")

        if self.locked_axis is not None:
            axis_idx = self.locked_axis

        if self.mode == "translate":
            self._translate(axis_idx, direction)
        elif self.mode == "rotate":
            self._rotate(axis_idx, direction)

    def _translate(self, axis_idx: int, direction: int):
        """Translate selected sphere along a specific axis with binary search ray-marching."""
        if self.selected_sphere_idx is None:
            return

        sphere = self.spheres[self.selected_sphere_idx]
        print(f"[TRANSLATE] Sphere {self.selected_sphere_idx}: axis={axis_idx}, direction={direction}")
        print(f"[TRANSLATE] rendering_coordinator = {self.rendering_coordinator is not None}")

        axis_dir = sphere.rotation[:, axis_idx]
        full_delta = axis_dir * self.step_size * direction

        # Reset collision state
        sphere.is_colliding = False
        sphere.collision_point = None

        # No collision detection available, just move
        if self.rendering_coordinator is None:
            sphere.translation += full_delta
            # Update backward compatibility
            self._update_backward_compat_properties()
            return

        # First, check target position
        target_translation = sphere.translation + full_delta
        old_translation = sphere.translation.copy()
        sphere.translation = target_translation
        target_center = self.get_sphere_center_recon_space_by_index(self.selected_sphere_idx)
        sphere.translation = old_translation

        target_collision = self.rendering_coordinator.collision_detector.check_object_collision(
            target_center, sphere.radius
        )

        # If target is safe, just move
        if not target_collision["collision"]:
            print(f"[TRANSLATE] Target is safe, moving fully")
            sphere.translation = target_translation
            sphere.is_colliding = False
            sphere.collision_point = None
            # Update backward compatibility
            self._update_backward_compat_properties()
            return

        # Target has collision, use binary search
        print(f"[TRANSLATE] Target has collision, using binary search")
        min_ratio = 0.0  # Current position (assumed safe)
        max_ratio = 1.0  # Target position (has collision)
        safe_ratio = 0.0

        # 10 iterations = 1/1024 precision
        for iteration in range(10):
            test_ratio = (min_ratio + max_ratio) / 2.0
            test_translation = sphere.translation + full_delta * test_ratio

            # Get test position in recon space
            sphere.translation = test_translation
            test_center = self.get_sphere_center_recon_space_by_index(self.selected_sphere_idx)
            sphere.translation = old_translation

            # Check collision
            collision = self.rendering_coordinator.collision_detector.check_object_collision(
                test_center, sphere.radius
            )

            if collision["collision"]:
                # Collision detected, reduce range
                max_ratio = test_ratio
                sphere.is_colliding = True
                sphere.collision_point = collision.get("contact_point")
                print(f"[BINARY_SEARCH] iter={iteration}, ratio={test_ratio:.4f} -> COLLISION")
            else:
                # Safe, increase range
                safe_ratio = test_ratio
                min_ratio = test_ratio
                print(f"[BINARY_SEARCH] iter={iteration}, ratio={test_ratio:.4f} -> SAFE")

        # Apply safe movement
        final_delta = full_delta * safe_ratio
        sphere.translation += final_delta

        # Update backward compatibility
        self._update_backward_compat_properties()

        print(f"[TRANSLATE] Final safe_ratio={safe_ratio:.4f}, is_colliding={sphere.is_colliding}")

        if sphere.is_colliding:
            print(f"[TRANSLATE] Blocked! Moved {safe_ratio*100:.1f}% of requested distance")

    def _rotate(self, axis_idx: int, direction: int):
        """Rotate selected sphere around a specific axis."""
        if self.selected_sphere_idx is None:
            return

        sphere = self.spheres[self.selected_sphere_idx]
        axis_dir = sphere.rotation[:, axis_idx]
        angle_delta = np.deg2rad(self.step_size) * direction
        rot_delta = rotation_matrix_from_axis_angle(axis_dir, angle_delta, degrees=False)
        sphere.rotation = rot_delta @ sphere.rotation

        # Update backward compatibility
        self._update_backward_compat_properties()

    def reset(self):
        """Reset selected sphere to initial position."""
        if self.selected_sphere_idx is not None:
            self.spheres[self.selected_sphere_idx].reset()
            self._update_backward_compat_properties()

    def reset_all(self):
        """Reset all spheres to initial positions."""
        for sphere in self.spheres:
            sphere.reset()

    def set_mode(self, mode: str):
        """Set manipulation mode ('translate' or 'rotate')."""
        self.mode = mode

    def lock_axis(self, axis_idx: Optional[int]):
        """Lock to a specific axis (0=X, 1=Y, 2=Z, None=unlock)."""
        self.locked_axis = axis_idx

    def export_transformation(self, output_dir: Path):
        """Export transformation matrices and parameters to files."""
        output_dir.mkdir(parents=True, exist_ok=True)

        object_transform = self.get_transform()
        full_transform = self.local_to_recon_transform @ object_transform

        t, r, s = decompose_transform(full_transform)

        matrix_path = output_dir / "object_transform_matrix.txt"
        export_transform_matrix(full_transform, str(matrix_path))

        params_path = output_dir / "object_transform_params.json"
        export_transform_params(t, r, s, str(params_path))

    def update_physics(self, dt, ground_y_local=None):
        """
        Update physics simulation for all enabled spheres.

        Args:
            dt: Time delta in seconds
            ground_y_local: Ground plane Y coordinate in Local space
                           If None, will be calculated from RenderingCoordinator's ground_y

        Note:
            Physics runs in Recon space, but we convert deltas back to Local space
            accounting for the scale factor. Ground plane is automatically synced
            from RenderingCoordinator's collision_detector.
        """
        if self.rendering_coordinator is None:
            return

        # Get ground_y from RenderingCoordinator if not provided
        if ground_y_local is None:
            # Convert from Recon space back to Local space
            ground_y_recon = self.rendering_coordinator.collision_detector.ground_y
            ground_y_local = ground_y_recon / self.scale_factor

        # Update physics for each sphere that has physics enabled
        for i, sphere in enumerate(self.spheres):
            if not sphere.physics_enabled:
                continue

            # Get current position in Recon space
            current_center = self.get_sphere_center_recon_space_by_index(i)

            # Physics update (runs in Recon space)
            new_center = sphere.physics_engine.update(
                dt,
                current_center,
                self.rendering_coordinator.collision_detector,
                sphere.radius * self.scale_factor  # Convert radius to Recon space
            )

            # Calculate delta in Recon space
            delta_recon = new_center - current_center

            # Transform delta back to Local space
            # Need to extract rotation AND divide by scale factor
            # delta_local = R^-1 @ delta_recon / scale
            # But since M[:3,:3] = scale * R, we have:
            # R^-1 @ delta_recon / scale = (scale * R)^-1 @ delta_recon
            R_scaled_inv = np.linalg.inv(self.local_to_recon_transform[:3, :3])
            delta_local = R_scaled_inv @ delta_recon

            # Apply delta
            proposed_translation = sphere.translation + delta_local

            # Check ground collision in Local space
            sphere_bottom_y_local = proposed_translation[1] - sphere.radius

            if sphere_bottom_y_local <= ground_y_local:
                # Ground collision in Local space!
                penetration = ground_y_local - sphere_bottom_y_local

                # Place sphere exactly on ground surface (Local space)
                proposed_translation[1] = ground_y_local + sphere.radius

                # Stop velocity
                sphere.physics_engine.velocity[1] = 0.0
                sphere.physics_engine.is_grounded = True
            else:
                sphere.physics_engine.is_grounded = False

            sphere.translation = proposed_translation

            # Update collision state
            sphere.is_colliding = sphere.physics_engine.is_grounded

        # Update backward compatibility for selected sphere after all physics updates
        if self.selected_sphere_idx is not None:
            self._update_backward_compat_properties()

    def enable_physics(self):
        """Enable physics for selected sphere."""
        if self.selected_sphere_idx is None:
            return

        sphere = self.spheres[self.selected_sphere_idx]
        current_pos_recon = self.get_sphere_center_recon_space_by_index(self.selected_sphere_idx)

        # Safety check: don't enable if already in collision
        if self.rendering_coordinator:
            collision = self.rendering_coordinator.collision_detector.check_object_collision_with_gravity(
                current_pos_recon, sphere.radius, gravity_dir=[0, -1, 0]
            )
            if collision["collision"]:
                return  # Don't enable physics

        sphere.physics_enabled = True
        sphere.physics_engine.reset()
        sphere.last_update_time = time.time()
        # Update backward compatibility
        self._update_backward_compat_properties()

    def disable_physics(self):
        """Disable physics for selected sphere."""
        if self.selected_sphere_idx is None:
            return

        sphere = self.spheres[self.selected_sphere_idx]
        sphere.physics_enabled = False
        # Update backward compatibility
        self._update_backward_compat_properties()

    def enable_physics_all(self, skip_collision_check=False):
        """
        Enable physics for all spheres.

        Args:
            skip_collision_check: If True, skip collision check and enable all spheres (for free-fall testing)
        """
        enabled_count = 0
        for i, sphere in enumerate(self.spheres):
            current_pos_recon = self.get_sphere_center_recon_space_by_index(i)

            # Safety check: don't enable if already in collision (unless skipping)
            if not skip_collision_check and self.rendering_coordinator:
                collision = self.rendering_coordinator.collision_detector.check_object_collision_with_gravity(
                    current_pos_recon, sphere.radius, gravity_dir=[0, -1, 0]
                )
                if collision["collision"]:
                    continue

            sphere.physics_enabled = True
            sphere.physics_engine.reset()
            sphere.last_update_time = time.time()
            enabled_count += 1

    def disable_physics_all(self):
        """Disable physics for all spheres."""
        for sphere in self.spheres:
            sphere.physics_enabled = False

    def physics_enabled_any(self) -> bool:
        """Check if any sphere has physics enabled."""
        return any(sphere.physics_enabled for sphere in self.spheres)
