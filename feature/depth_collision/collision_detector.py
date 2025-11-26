import numpy as np
from math import sqrt, cos, sin, pi


class DepthCollisionDetector:
    """
    Depth-based collision detection for triple_world_renderer.

    Coordinate System:
        - All positions are in RECON (Fusion) SPACE
        - This is required because depth fusion combines Mesh + Neural depth maps
        - Ground plane Y coordinate should be provided in Recon space units

    Note:
        Physics and collision run in Recon space to handle occlusion correctly
        in the fused depth buffer from Mesh + Neural rendering.
    """

    def __init__(self, near=0.03, far=10.0, epsilon=0.005, ground_y=0.0):
        """
        Initialize depth collision detector.

        Args:
            near: Near clipping plane distance
            far: Far clipping plane distance
            epsilon: Collision tolerance threshold in RECON space units (default 0.005, very tight)
                     Should be scaled from Local space if needed (e.g., 0.005m → 0.029 Recon units)
                     Accounts for:
                     - Gaussian Splatting depth noise (±0.5mm in Local, very precise)
                     - Coordinate system conversion errors
                     - Floating point precision
                     WARNING: Very tight setting, may be sensitive to depth noise
            ground_y: Ground plane Y coordinate in RECON space (NOT Local space!)
                     - Calculate from Local: ground_y_recon ≈ ground_y_local × scale_factor
                     - Example: -1.0m Local × 5.7 ≈ -5.7 Recon units
        """
        self.near = near
        self.far = far
        self.epsilon = epsilon
        self.ground_y = ground_y

        # Depth buffer (updated by RenderingCoordinator)
        self.neural_depth = None  # (H, W) numpy array

        # Camera parameters (for projection)
        self.view_matrix = None  # (4, 4)
        self.proj_matrix = None  # (4, 4)
        self.width = 640
        self.height = 720

    def update_depths(self, neural_depth):
        """Update depth buffer from rendering coordinator."""
        self.neural_depth = neural_depth

    def update_camera(self, view_matrix, proj_matrix, width, height):
        """Update camera parameters."""
        # Convert to numpy if torch tensor
        if hasattr(view_matrix, 'cpu'):
            view_matrix = view_matrix.cpu().numpy()
        if hasattr(proj_matrix, 'cpu'):
            proj_matrix = proj_matrix.cpu().numpy()

        self.view_matrix = view_matrix
        self.proj_matrix = proj_matrix
        self.width = width
        self.height = height

    def world_to_screen(self, world_pos):
        """
        Project world position to screen pixel coordinates.

        Args:
            world_pos: (3,) position in RECON space

        Returns:
            (x, y) pixel coordinates or None if out of bounds
        """
        # Check for NaN or inf in input
        if not np.all(np.isfinite(world_pos)):
            return None

        # Apply view and projection transforms
        pos_hom = np.append(world_pos, 1.0)
        view_pos = self.view_matrix @ pos_hom
        clip_pos = self.proj_matrix @ view_pos

        # Check for NaN or inf in clip space
        if not np.all(np.isfinite(clip_pos)):
            return None

        # Perspective divide
        if abs(clip_pos[3]) < 1e-6:
            return None
        ndc = clip_pos[:3] / clip_pos[3]

        # Check for NaN or inf in NDC
        if not np.all(np.isfinite(ndc)):
            return None

        # NDC to screen
        screen_x = int((ndc[0] + 1.0) * 0.5 * self.width)
        screen_y = int((1.0 - ndc[1]) * 0.5 * self.height)

        # Check bounds
        if screen_x < 0 or screen_x >= self.width or screen_y < 0 or screen_y >= self.height:
            return None

        return (screen_x, screen_y)

    def generate_sphere_samples(self, center, radius, n_samples=24):
        """
        Generate sample points on sphere surface using Fibonacci sphere.

        Args:
            center: (3,) sphere center in RECON space
            radius: Sphere radius in RECON space units
            n_samples: Number of sample points to generate (default 24 for tight boundaries)

        Returns:
            List of (3,) points on sphere surface in RECON space
        """
        points = []
        phi = (1 + sqrt(5)) / 2  # golden ratio

        for i in range(n_samples):
            y = 1 - (i / (n_samples - 1)) * 2  # -1 to 1
            r = sqrt(1 - y*y)
            theta = 2 * pi * i / phi

            x = cos(theta) * r
            z = sin(theta) * r

            point = center + radius * np.array([x, y, z])
            points.append(point)

        return points

    def check_point_collision(self, world_pos):
        """
        Check collision for a single point in world space.

        Args:
            world_pos: (3,) position in RECON space to check

        Returns:
            dict with keys:
                - collision (bool): True if point is inside surface
                - depth_value (float): Depth from camera to surface (from depth buffer)
                - object_depth (float): Depth from camera to test point
        """
        if self.neural_depth is None or self.view_matrix is None or self.proj_matrix is None:
            return {"collision": False}

        # Project to screen
        screen_coords = self.world_to_screen(world_pos)
        if screen_coords is None:
            return {"collision": False}

        screen_x, screen_y = screen_coords

        # Sample depth
        gaussian_depth = self.neural_depth[screen_y, screen_x]

        # Skip invalid depth
        if (not np.isfinite(gaussian_depth) or
            gaussian_depth <= 0.0 or
            gaussian_depth >= self.far * 0.99):
            return {"collision": False}

        # Calculate object depth
        camera_pos = np.linalg.inv(self.view_matrix)[:3, 3]
        object_depth = np.linalg.norm(world_pos - camera_pos)

        # Calculate penetration (positive = object behind surface = collision)
        # If object_depth > gaussian_depth: object is behind/inside surface → Collision!
        # If object_depth < gaussian_depth: object is in front of surface → No collision
        penetration = object_depth - gaussian_depth  # Original formula (REVERTED)

        # Check collision (positive penetration = object behind/inside surface)
        if penetration >= self.epsilon:
            return {
                "collision": True,
                "depth_value": gaussian_depth,
                "object_depth": object_depth,
                "penetration": penetration
            }

        return {"collision": False}

    def check_object_collision(self, center, radius):
        """
        Check collision using bounding sphere.

        Args:
            center: (3,) world position of sphere center
            radius: sphere radius

        Returns:
            dict with keys: collision, max_penetration, contact_point
        """
        if self.neural_depth is None or self.view_matrix is None or self.proj_matrix is None:
            return {"collision": False}

        # Generate sample points on sphere surface
        sphere_samples = self.generate_sphere_samples(center, radius, n_samples=12)

        # Extract camera position from view matrix
        camera_pos = np.linalg.inv(self.view_matrix)[:3, 3]

        max_penetration = -float('inf')
        contact_point = None
        valid_samples = 0
        collision_samples = 0

        # Check each sample point
        for sample_point in sphere_samples:
            # Project to screen
            screen_coords = self.world_to_screen(sample_point)
            if screen_coords is None:
                continue

            valid_samples += 1
            screen_x, screen_y = screen_coords

            # Sample Gaussian depth
            gaussian_depth = self.neural_depth[screen_y, screen_x]

            # Skip invalid depth samples (NaN, inf, zero, far-plane background)
            # This prevents false collisions with background/invalid pixels
            if (not np.isfinite(gaussian_depth) or
                gaussian_depth <= 0.0 or
                gaussian_depth >= self.far * 0.99):
                continue  # Skip this sample

            # Calculate object depth (distance from camera)
            object_depth = np.linalg.norm(sample_point - camera_pos)

            # Calculate penetration (positive = object behind surface = collision)
            # If object_depth > gaussian_depth: object is behind/inside surface → Collision!
            # If object_depth < gaussian_depth: object is in front of surface → No collision
            penetration = object_depth - gaussian_depth  # Original formula (REVERTED)

            # Track max penetration
            if penetration > max_penetration:
                max_penetration = penetration
                contact_point = sample_point.copy()

            # Early exit if collision detected (positive penetration = object behind/inside surface)
            if penetration >= self.epsilon:
                collision_samples += 1
                return {
                    "collision": True,
                    "max_penetration": penetration,
                    "contact_point": contact_point
                }

        return {
            "collision": False,
            "max_penetration": max_penetration,
            "contact_point": contact_point
        }

    def check_ground_plane_collision(self, center, radius, ground_y=None):
        """
        Check collision with horizontal ground plane at Y=ground_y.

        This provides a fallback collision surface when depth data is unavailable.

        Args:
            center: (3,) sphere center in RECON space
            radius: Sphere radius in RECON space units
            ground_y: Ground plane Y coordinate in RECON space
                     (default: None, uses self.ground_y)
                     NOTE: Should be in Recon units, not Local meters!

        Returns:
            dict with keys:
                - collision (bool): True if sphere penetrates ground
                - max_penetration (float): Penetration depth in Recon units
                - contact_point (3,): Contact point in Recon space
                - detection_type (str): "ground_plane"
        """
        # Use instance ground_y if not specified
        if ground_y is None:
            ground_y = self.ground_y

        # Calculate sphere bottom Y coordinate
        sphere_bottom_y = center[1] - radius

        # Check if sphere penetrates ground plane
        if sphere_bottom_y <= ground_y:
            penetration = ground_y - sphere_bottom_y
            return {
                "collision": True,
                "max_penetration": penetration,
                "contact_point": np.array([center[0], ground_y, center[2]]),
                "detection_type": "ground_plane"
            }

        return {"collision": False}

    def check_object_collision_with_gravity(self, center, radius, gravity_dir=None):
        """
        Check collision using bounding sphere + gravity-direction raycast.

        This method combines:
        1. Screen-space depth collision (existing method)
        2. Gravity-direction raycast (for out-of-view surfaces)

        Args:
            center: (3,) world position of sphere center
            radius: sphere radius
            gravity_dir: (3,) gravity direction vector (default: [0, -1, 0])

        Returns:
            dict with keys: collision, max_penetration, contact_point
        """
        if gravity_dir is None:
            gravity_dir = np.array([0.0, -1.0, 0.0])
        else:
            gravity_dir = np.array(gravity_dir, dtype=np.float32)
            # Normalize gravity direction
            gravity_dir = gravity_dir / np.linalg.norm(gravity_dir)

        # 1. Try screen-space collision check (existing method)
        screen_result = self.check_object_collision(center, radius)
        if screen_result["collision"]:
            return screen_result

        # 2. Gravity direction raycast (5 steps, 5cm each = 25cm total)
        step_size = 0.05  # 5cm per step
        num_steps = 5
        max_distance = step_size * num_steps

        for step in range(1, num_steps + 1):
            # Sample point below center in gravity direction
            test_point = center + gravity_dir * (step_size * step)

            # Check this point for collision
            point_result = self.check_point_collision(test_point)

            if point_result["collision"]:
                penetration_depth = step_size * step
                return {
                    "collision": True,
                    "max_penetration": point_result.get("penetration", penetration_depth),
                    "contact_point": test_point,
                    "detection_type": "gravity_raycast"
                }

        # 3. Final fallback: ground plane collision (uses self.ground_y)
        ground_result = self.check_ground_plane_collision(center, radius)
        if ground_result["collision"]:
            return ground_result

        return {
            "collision": False,
            "max_penetration": screen_result.get("max_penetration", -float('inf')),
            "contact_point": screen_result.get("contact_point", None)
        }
