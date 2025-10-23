"""
Base class for all scene renderers.

Defines the common interface that all renderer implementations must follow.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any

from renderer.data_types import CameraFrame, RenderOutput


class BaseSceneRenderer(ABC):
    """
    Abstract base class for scene renderers.

    All renderer implementations (3DGS, 4DGS, NeRF, etc.) must inherit from
    this class and implement the required methods.
    """

    @abstractmethod
    async def on_init(self, config: Dict[str, Any]) -> bool:
        """
        Initialize the renderer.

        This method is called once at startup to load the scene, allocate
        GPU resources, and perform any necessary preprocessing.

        Args:
            config: Configuration dictionary containing:
                - model_path: Path to scene file (PLY, etc.)
                - device: Device to use ('cuda', 'cuda:0', etc.)
                - Additional renderer-specific parameters

        Returns:
            True if initialization succeeded, False otherwise

        Example:
            config = {
                'model_path': '/path/to/scene.ply',
                'device': 'cuda',
                'sh_degree': 3,
                'gaussian_scale': 1.0
            }
        """
        pass

    @abstractmethod
    async def render(self, camera: CameraFrame) -> RenderOutput:
        """
        Render the scene from the given camera viewpoint.

        This is the core rendering method called for each frame.

        Args:
            camera: Camera parameters (view matrix, intrinsics, etc.)

        Returns:
            RenderOutput containing color, depth, and alpha channels

        Raises:
            RuntimeError: If renderer is not initialized
        """
        pass

    @abstractmethod
    async def on_cleanup(self) -> None:
        """
        Clean up resources.

        This method is called on shutdown to free GPU memory and other
        resources. Should be safe to call multiple times.
        """
        pass

    def _validate_config(self, config: Dict[str, Any], required_keys: list) -> None:
        """
        Helper method to validate configuration.

        Args:
            config: Configuration dictionary
            required_keys: List of required configuration keys

        Raises:
            ValueError: If required keys are missing
        """
        missing = [key for key in required_keys if key not in config]
        if missing:
            raise ValueError(f"Missing required configuration keys: {missing}")
