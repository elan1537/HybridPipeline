"""Scene geometry builder with 2-Scene architecture (Display vs Render)."""

import copy
from pathlib import Path
from typing import Dict, Optional, Tuple
import numpy as np
import open3d as o3d

from project_loader import load_project


class SceneBuilder:
    """Builds scene geometries with separate Display and Render scenes."""

    def __init__(
        self,
        project_dir: Path,
        frame_idx: int,
        local_to_recon_transform: np.ndarray,
        sphere_radius: float = 0.1,
        ground_size: float = 10.0,
        voxel_size: float = 0.02,
        show_point_cloud: bool = False
    ):
        """
        Initialize scene builder.

        Args:
            project_dir: Project directory path
            frame_idx: Frame index to load
            local_to_recon_transform: Transformation matrix (Local → Reconstructed)
            sphere_radius: Sphere radius in meters
            ground_size: Ground plane size in meters
            voxel_size: Point cloud voxel downsampling size
            show_point_cloud: Whether to include point cloud in Recon Scene View
        """
        self.project_dir = project_dir
        self.frame_idx = frame_idx
        self.local_to_recon_transform = local_to_recon_transform
        self.sphere_radius = sphere_radius
        self.ground_size = ground_size
        self.voxel_size = voxel_size
        self.show_point_cloud = show_point_cloud

        # Sphere initial position (in Local coordinates)
        self.sphere_initial_position = np.array([0.0, 0.8, 1.5])

        # Multiple spheres configuration
        self.sphere_configs = []  # List of (position, radius, color) tuples
        self.generate_random_spheres(num_spheres=30)

        self.point_cloud: Optional[o3d.geometry.PointCloud] = None

    def generate_random_spheres(self, num_spheres: int = 10,
                               radius_range: Tuple[float, float] = (0.03, 0.08),
                               position_bounds: Tuple[float, float, float, float, float, float] = None):
        """
        Generate random sphere configurations.

        Args:
            num_spheres: Number of spheres to generate
            radius_range: Min and max radius for spheres
            position_bounds: (x_min, x_max, y_min, y_max, z_min, z_max) for positions
                           If None, uses src_points from transform_to_matrix.py for XZ, Y in [0, 1]
        """
        import random

        # Clear existing configs
        self.sphere_configs = []

        # Define a diverse color palette
        colors = [
            [1.0, 0.2, 0.2],  # Red
            [0.2, 1.0, 0.2],  # Green
            [0.2, 0.2, 1.0],  # Blue
            [1.0, 1.0, 0.2],  # Yellow
            [1.0, 0.2, 1.0],  # Magenta
            [0.2, 1.0, 1.0],  # Cyan
            [1.0, 0.5, 0.2],  # Orange
            [0.5, 0.2, 1.0],  # Purple
            [0.2, 1.0, 0.5],  # Light green
            [1.0, 0.2, 0.5],  # Pink
        ]

        # Use src_points from transform_to_matrix.py if bounds not specified
        if position_bounds is None:
            # Import src_points from transform_to_matrix
            from transform_to_matrix import src_points

            # Get XZ bounds from src_points
            x_min = -0.5  # -1.0
            x_max = 0.7
            z_min = -0.1
            z_max = 0.55

            # Y range as requested (0 to 1)
            y_min = 0.9
            y_max = 1.15
        else:
            x_min, x_max, y_min, y_max, z_min, z_max = position_bounds

        for i in range(num_spheres):
            # Random position
            x = random.uniform(x_min, x_max)
            y = random.uniform(y_min, y_max)
            z = random.uniform(z_min, z_max)
            position = np.array([x, y, z])

            # Random radius
            radius = random.uniform(radius_range[0], radius_range[1])

            # Cycle through colors
            color = colors[i % len(colors)]

            self.sphere_configs.append((position, radius, color))

    def create_local_scene(self) -> Dict[str, o3d.geometry.Geometry]:
        """
        Create Local World scene with multiple spheres.

        Returns:
            Dictionary of geometries: {'sphere_0', 'sphere_1', ..., 'ground', 'coord'}
        """
        geometries = {}

        # Create multiple spheres from configs
        for i, (position, radius, color) in enumerate(self.sphere_configs):
            sphere = o3d.geometry.TriangleMesh.create_sphere(
                radius=radius, resolution=40
            )
            sphere.translate(position)
            sphere.paint_uniform_color(color)
            sphere.compute_vertex_normals()
            geometries[f'sphere_{i}'] = sphere

        # # Additional spheres (different positions and colors)
        # sphere1 = o3d.geometry.TriangleMesh.create_sphere(radius=0.15, resolution=20)
        # sphere1.translate([0.8, 0.15, 0.5])
        # sphere1.paint_uniform_color([0.0, 1.0, 0.0])  # Green
        # sphere1.compute_vertex_normals()
        # geometries['sphere1'] = sphere1

        # sphere2 = o3d.geometry.TriangleMesh.create_sphere(radius=0.18, resolution=20)
        # sphere2.translate([-0.6, 0.18, -0.4])
        # sphere2.paint_uniform_color([0.0, 0.5, 1.0])  # Blue
        # sphere2.compute_vertex_normals()
        # geometries['sphere2'] = sphere2

        # sphere3 = o3d.geometry.TriangleMesh.create_sphere(radius=0.12, resolution=20)
        # sphere3.translate([0.3, 0.12, -0.7])
        # sphere3.paint_uniform_color([1.0, 1.0, 0.0])  # Yellow
        # sphere3.compute_vertex_normals()
        # geometries['sphere3'] = sphere3

        # # Cubes
        # cube1 = o3d.geometry.TriangleMesh.create_box(width=0.3, height=0.3, depth=0.3)
        # cube1.translate([-0.8, 0.0, 0.6])
        # cube1.paint_uniform_color([1.0, 0.0, 1.0])  # Magenta
        # cube1.compute_vertex_normals()
        # geometries['cube1'] = cube1

        # cube2 = o3d.geometry.TriangleMesh.create_box(width=0.25, height=0.4, depth=0.25)
        # cube2.translate([0.5, 0.0, -0.3])
        # cube2.paint_uniform_color([1.0, 0.5, 0.0])  # Orange
        # cube2.compute_vertex_normals()
        # geometries['cube2'] = cube2

        # # Cylinder
        # cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=0.12, height=0.5, resolution=20)
        # cylinder.translate([0.0, 0.25, 0.8])
        # cylinder.paint_uniform_color([0.5, 0.0, 0.5])  # Purple
        # cylinder.compute_vertex_normals()
        # geometries['cylinder'] = cylinder

        # # Cone
        # cone = o3d.geometry.TriangleMesh.create_cone(radius=0.15, height=0.4, resolution=20)
        # cone.translate([-0.4, 0.0, 0.2])
        # cone.paint_uniform_color([0.0, 1.0, 1.0])  # Cyan
        # cone.compute_vertex_normals()
        # geometries['cone'] = cone

        # # Torus
        # torus = o3d.geometry.TriangleMesh.create_torus(torus_radius=0.2, tube_radius=0.05, radial_resolution=20, tubular_resolution=20)
        # torus.translate([0.6, 0.2, 0.0])
        # torus.paint_uniform_color([0.8, 0.8, 0.0])  # Gold
        # torus.compute_vertex_normals()
        # geometries['torus'] = torus

        # Ground and coordinate frame
        geometries['ground'] = self._create_ground_plane(size=self.ground_size, divisions=20)
        geometries['coord'] = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.3, origin=[0, 0, 0]
        )

        return geometries

    def create_recon_scene_display(self) -> Dict[str, o3d.geometry.Geometry]:
        """
        Create Recon Scene View (optionally includes point cloud).

        Returns:
            Dictionary of geometries: All local scene objects transformed to reconstructed coordinates
        """
        geometries = {}

        # Load point cloud only if enabled
        if self.show_point_cloud:
            _, point_cloud_raw, _ = load_project(
                str(self.project_dir),
                self.frame_idx
            )
            self.point_cloud = point_cloud_raw.voxel_down_sample(voxel_size=self.voxel_size)
            geometries['point_cloud'] = self.point_cloud

        # Create multiple spheres with initial positions
        for i, (position, radius, color) in enumerate(self.sphere_configs):
            sphere = o3d.geometry.TriangleMesh.create_sphere(
                radius=radius, resolution=40
            )
            sphere.paint_uniform_color(color)
            sphere.compute_vertex_normals()

            # Create transform matrix for initial position
            initial_transform = np.eye(4)
            initial_transform[:3, 3] = position

            # Apply local-to-recon transformation
            full_transform = self.local_to_recon_transform @ initial_transform
            sphere.transform(full_transform)

            geometries[f'sphere_{i}'] = sphere

        # Ground and coordinate frame (transformed)
        geometries['ground'] = self._create_transformed_ground()
        geometries['coord'] = self._create_transformed_coord()

        return geometries

    def create_recon_scene_render(self, sphere_transform: np.ndarray) -> Dict[str, o3d.geometry.Geometry]:
        """
        Create Reconstructed World scene for Rendering (UI Helper excluded).

        Args:
            sphere_transform: Current sphere transformation matrix

        Returns:
            Dictionary of geometries: All objects except ground and coord (no UI helpers)
        """
        geometries = {}

        # Create local scene
        local_scene = self.create_local_scene()

        # Transform all objects except ground and coord (UI helpers)
        for name, geom in local_scene.items():
            if name in ['ground', 'coord']:
                continue  # Skip UI helpers
            elif name == 'sphere':
                # Main sphere is manipulatable
                geometries['sphere'] = self._create_transformed_sphere(sphere_transform)
            else:
                # Static objects: transform directly
                transformed_geom = copy.deepcopy(geom)
                transformed_geom.transform(self.local_to_recon_transform)
                geometries[name] = transformed_geom

        return geometries

    def update_sphere_local(self, sphere_transform: np.ndarray) -> o3d.geometry.TriangleMesh:
        """Create updated sphere in Local World with transformation."""
        sphere = o3d.geometry.TriangleMesh.create_sphere(
            radius=self.sphere_radius, resolution=20
        )
        sphere.paint_uniform_color([1.0, 0.0, 0.0])
        sphere.compute_vertex_normals()
        sphere.transform(sphere_transform)
        return sphere

    def update_sphere_recon(self, sphere_transform: np.ndarray) -> o3d.geometry.TriangleMesh:
        """Create updated sphere in Reconstructed World with transformation."""
        return self._create_transformed_sphere(sphere_transform)

    def _create_ground_plane(self, size: float, divisions: int) -> o3d.geometry.LineSet:
        """Create a grid-based ground plane at y=0."""
        points = []
        lines = []
        colors = []

        half_size = size / 2.0
        step = size / divisions

        for i in range(divisions + 1):
            z = -half_size + i * step
            points.append([-half_size, 0, z])
            points.append([half_size, 0, z])
            line_idx = len(points) - 2
            lines.append([line_idx, line_idx + 1])
            colors.append([0.7, 0.7, 0.7] if abs(z) < 1e-6 else [0.4, 0.4, 0.4])

        for i in range(divisions + 1):
            x = -half_size + i * step
            points.append([x, 0, -half_size])
            points.append([x, 0, half_size])
            line_idx = len(points) - 2
            lines.append([line_idx, line_idx + 1])
            colors.append([0.7, 0.7, 0.7] if abs(x) < 1e-6 else [0.4, 0.4, 0.4])

        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(points)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.colors = o3d.utility.Vector3dVector(colors)

        return line_set

    def _create_transformed_sphere(self, sphere_transform: np.ndarray) -> o3d.geometry.TriangleMesh:
        """Create sphere transformed to reconstructed coordinates."""
        sphere = o3d.geometry.TriangleMesh.create_sphere(
            radius=self.sphere_radius, resolution=20
        )
        sphere.paint_uniform_color([1.0, 0.0, 0.0])
        sphere.compute_vertex_normals()

        full_transform = self.local_to_recon_transform @ sphere_transform
        sphere.transform(full_transform)

        return sphere

    def _create_transformed_ground(self) -> o3d.geometry.LineSet:
        """Create ground plane transformed to reconstructed coordinates."""
        ground = self._create_ground_plane(size=self.ground_size, divisions=20)
        ground.transform(self.local_to_recon_transform)
        return ground

    def _create_transformed_coord(self) -> o3d.geometry.TriangleMesh:
        """Create coordinate frame transformed to reconstructed coordinates."""
        coord = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.3, origin=[0, 0, 0]
        )
        coord.transform(self.local_to_recon_transform)
        return coord

    def create_collision_marker(self, position: np.ndarray, radius: float = 0.02) -> o3d.geometry.TriangleMesh:
        """Create collision marker (red sphere) at given position."""
        marker = o3d.geometry.TriangleMesh.create_sphere(radius=radius, resolution=10)
        marker.translate(position)
        marker.paint_uniform_color([1.0, 0.0, 0.0])  # Red
        marker.compute_vertex_normals()
        return marker

    def update_sphere_local_with_color(self, sphere_transform: np.ndarray, color: list) -> o3d.geometry.TriangleMesh:
        """Create updated sphere in Local World with custom color."""
        sphere = o3d.geometry.TriangleMesh.create_sphere(
            radius=self.sphere_radius, resolution=20
        )
        sphere.paint_uniform_color(color)
        sphere.compute_vertex_normals()
        sphere.transform(sphere_transform)
        return sphere

    def update_sphere_recon_with_color(self, sphere_transform: np.ndarray, color: list) -> o3d.geometry.TriangleMesh:
        """Create updated sphere in Reconstructed World with custom color."""
        sphere = o3d.geometry.TriangleMesh.create_sphere(
            radius=self.sphere_radius, resolution=20
        )
        sphere.paint_uniform_color(color)
        sphere.compute_vertex_normals()

        full_transform = self.local_to_recon_transform @ sphere_transform
        sphere.transform(full_transform)

        return sphere

    def update_sphere_by_index(self, idx: int, transform: np.ndarray, color: list,
                               is_selected: bool = False) -> o3d.geometry.TriangleMesh:
        """
        Create updated sphere by index in Local World with custom color.

        Args:
            idx: Sphere index
            transform: 4x4 transformation matrix
            color: RGB color values [0-1]
            is_selected: Whether this sphere is currently selected

        Returns:
            Updated sphere mesh
        """
        if idx >= len(self.sphere_configs):
            return None

        _, radius, _ = self.sphere_configs[idx]

        sphere = o3d.geometry.TriangleMesh.create_sphere(
            radius=radius, resolution=40
        )
        sphere.transform(transform)
        sphere.paint_uniform_color(color)

        # Add selection indicator (yellow outline)
        if is_selected:
            # Create a slightly larger sphere as outline
            outline = o3d.geometry.TriangleMesh.create_sphere(
                radius=radius * 1.1, resolution=40
            )
            outline.transform(transform)
            outline.paint_uniform_color([1.0, 1.0, 0.0])  # Yellow
            # Note: In practice, we'd need to handle this differently
            # for proper outline rendering

        sphere.compute_vertex_normals()
        return sphere

    def update_sphere_recon_by_index(self, idx: int, transform: np.ndarray, color: list,
                                    is_selected: bool = False) -> o3d.geometry.TriangleMesh:
        """
        Create updated sphere by index in Recon World with custom color.

        Args:
            idx: Sphere index
            transform: 4x4 transformation matrix in local coordinates
            color: RGB color values [0-1]
            is_selected: Whether this sphere is currently selected

        Returns:
            Updated sphere mesh in recon coordinates
        """
        if idx >= len(self.sphere_configs):
            return None

        _, radius, _ = self.sphere_configs[idx]

        sphere = o3d.geometry.TriangleMesh.create_sphere(
            radius=radius, resolution=40
        )

        # Apply local transform, then local-to-recon transform
        full_transform = self.local_to_recon_transform @ transform
        sphere.transform(full_transform)
        sphere.paint_uniform_color(color)
        sphere.compute_vertex_normals()

        return sphere
