"""Simple physics engine for gravity and collision-based sliding."""

import numpy as np


class PhysicsEngine:
    """
    Simple physics engine with gravity, damping, and friction.

    Coordinate System:
        - All physics calculations run in RECON (Fusion) SPACE
        - Gravity is specified in Local meters/s² and auto-scaled to Recon units/s²
        - Scale factor typically ~5.7 (1 meter Local = 5.7 units Recon)

    Note:
        Physics runs in Recon space because depth collision detection requires
        fusion of Mesh + Neural depth maps (which are in Recon/camera coordinates).
    """

    def __init__(self, gravity_local=-9.8, damping=0.95, friction=0.3, scale_factor=1.0):
        """
        Initialize physics engine.

        Args:
            gravity_local: Gravity acceleration in LOCAL space (m/s²)
                          - Negative = downward in Local Y-axis
                          - Default: -9.8 m/s² (Earth's gravity)
                          - Will be scaled to Recon units automatically
            damping: Velocity damping coefficient (0-1, energy loss per frame)
            friction: Surface friction coefficient (0-1, tangent velocity reduction)
            scale_factor: Scale factor for Local→Recon transform
                         - Typically ~5.7 (1 meter Local = 5.7 units Recon)
                         - If 1.0, assumes no scaling (units match)
        """
        # Store both Local and Recon gravity for clarity
        self.gravity_local = gravity_local  # m/s² in Local space
        self.scale_factor = scale_factor
        self.gravity = gravity_local * scale_factor  # Recon units/s² (actual value used in physics)

        self.damping = damping
        self.friction = friction

        # Physics state (all in Recon space)
        self.velocity = np.array([0.0, 0.0, 0.0])  # Recon units/s
        self.is_grounded = False

    def update(self, dt, current_pos, collision_detector, sphere_radius):
        """
        Update physics simulation for one timestep.

        Args:
            dt: Time delta in seconds
            current_pos: Current position (3,) in RECON space
            collision_detector: DepthCollisionDetector instance
            sphere_radius: Sphere radius in RECON space units

        Returns:
            New position (3,) in RECON space

        Note:
            All positions and velocities are in RECON (Fusion) coordinate system.
            Gravity is automatically scaled from Local (m/s²) to Recon (units/s²).
        """
        # Safety check: detect NaN or inf in input
        if not np.all(np.isfinite(current_pos)):
            print(f"[PHYSICS] WARNING: NaN/inf detected in current_pos: {current_pos}")
            print(f"[PHYSICS] Resetting velocity and returning last valid position")
            self.velocity = np.array([0.0, 0.0, 0.0])
            return current_pos

        # Safety check: detect NaN or inf in velocity
        if not np.all(np.isfinite(self.velocity)):
            print(f"[PHYSICS] WARNING: NaN/inf detected in velocity: {self.velocity}")
            print(f"[PHYSICS] Resetting velocity to zero")
            self.velocity = np.array([0.0, 0.0, 0.0])

        # Screen-space collision detection (depth-based) is now ENABLED
        # Previously disabled due to inverted depth logic bug (now fixed)
        DISABLE_SCREEN_COLLISION = False  # Re-enabled after fixing depth comparison logic

        # Apply gravity
        gravity_accel = self.gravity * dt
        self.velocity[1] += gravity_accel

        # Apply damping
        self.velocity *= self.damping

        # Calculate proposed position
        proposed_pos = current_pos + self.velocity * dt

        if DISABLE_SCREEN_COLLISION:
            # Only check ground collision, skip screen-space depth collision
            ground_collision = collision_detector.check_ground_plane_collision(
                proposed_pos, sphere_radius
            )

            if ground_collision["collision"]:
                # Hit ground! Stop at ground level
                ground_y = collision_detector.ground_y

                # Place sphere exactly on ground surface
                final_pos = proposed_pos.copy()
                final_pos[1] = ground_y + sphere_radius  # Center = ground + radius

                # Stop vertical velocity (elastic collision with restitution=0)
                self.velocity[1] = 0.0
                self.is_grounded = True

                return final_pos
            else:
                # Free fall continues
                self.is_grounded = False
                return proposed_pos

        # Original collision detection code
        # Emergency escape: check if already deeply penetrating
        gravity_dir = np.array([0.0, self.gravity / abs(self.gravity), 0.0])  # Normalized gravity direction
        current_collision = collision_detector.check_object_collision_with_gravity(
            current_pos, sphere_radius, gravity_dir=gravity_dir
        )

        if current_collision["collision"]:
            penetration = current_collision.get("penetration_depth", 0)

            # Deep penetration threshold: half the sphere radius
            if penetration > sphere_radius * 0.5:
                # Emergency escape upward (anti-gravity push)
                escape_normal = np.array([0.0, 1.0, 0.0])  # Up direction
                escape_distance = penetration + sphere_radius * 0.1  # Extra margin
                escaped_pos = current_pos + escape_normal * escape_distance

                # Reset velocity on escape
                self.velocity = np.array([0.0, 0.0, 0.0])

                return escaped_pos

        # Apply gravity
        self.velocity[1] += self.gravity * dt

        # Calculate proposed position
        proposed_pos = current_pos + self.velocity * dt

        # Check collision (using gravity-aware detection)
        collision = collision_detector.check_object_collision_with_gravity(
            proposed_pos, sphere_radius, gravity_dir=gravity_dir
        )

        if collision["collision"]:
            # Get surface normal
            normal = self._get_surface_normal(collision_detector, proposed_pos, sphere_radius)

            # Resolve collision (find safe position)
            final_pos = self._resolve_collision(
                current_pos, proposed_pos, normal,
                collision_detector, sphere_radius
            )

            # Slide velocity (project onto tangent plane)
            v_normal = np.dot(self.velocity, normal) * normal
            v_tangent = self.velocity - v_normal
            self.velocity = v_tangent * (1 - self.friction)
            self.velocity *= self.damping

            self.is_grounded = True
        else:
            # No collision, free fall
            final_pos = proposed_pos
            self.is_grounded = False

        return final_pos

    def _get_surface_normal(self, collision_detector, pos, radius):
        """
        Estimate surface normal using depth gradient.

        Args:
            collision_detector: DepthCollisionDetector instance
            pos: Position in world space
            radius: Sphere radius (unused, for interface compatibility)

        Returns:
            Normal vector (3,) in world space
        """
        screen_coords = collision_detector.world_to_screen(pos)
        if screen_coords is None:
            return np.array([0.0, 1.0, 0.0])  # Default: upward

        x, y = screen_coords
        depth_map = collision_detector.neural_depth
        h, w = depth_map.shape

        # Depth gradient (Sobel-like)
        dx = depth_map[y, min(x + 1, w - 1)] - depth_map[y, max(x - 1, 0)]
        dy = depth_map[min(y + 1, h - 1), x] - depth_map[max(y - 1, 0), x]

        # Normal in camera space (gradient → tangent → normal)
        normal_cam = np.array([-dx, -dy, 1.0])
        normal_cam /= np.linalg.norm(normal_cam)

        # Transform to world space
        inv_view = np.linalg.inv(collision_detector.view_matrix)
        normal_world = inv_view[:3, :3] @ normal_cam
        normal_world /= np.linalg.norm(normal_world)

        return normal_world

    def _resolve_collision(self, current_pos, proposed_pos, normal, collision_detector, radius):
        """
        Resolve collision using binary search to find safe position.

        Args:
            current_pos: Current safe position
            proposed_pos: Proposed position (has collision)
            normal: Surface normal (unused, for future extension)
            collision_detector: DepthCollisionDetector instance
            radius: Sphere radius

        Returns:
            Safe position as close to proposed as possible
        """
        min_ratio = 0.0  # Current position (safe)
        max_ratio = 1.0  # Proposed position (collision)
        safe_ratio = 0.0

        delta = proposed_pos - current_pos

        # Binary search (8 iterations = 1/256 precision)
        for _ in range(8):
            test_ratio = (min_ratio + max_ratio) / 2.0
            test_pos = current_pos + delta * test_ratio

            collision = collision_detector.check_object_collision(test_pos, radius)
            if collision["collision"]:
                max_ratio = test_ratio
            else:
                safe_ratio = test_ratio
                min_ratio = test_ratio

        return current_pos + delta * safe_ratio

    def reset(self):
        """Reset physics state."""
        self.velocity = np.array([0.0, 0.0, 0.0])
        self.is_grounded = False
