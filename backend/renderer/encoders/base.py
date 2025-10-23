"""Base encoder interface."""

from abc import ABC, abstractmethod
from renderer.data_types import RenderOutput, RenderPayload


class BaseEncoder(ABC):
    """Abstract base class for output encoders."""

    @abstractmethod
    def get_format_type(self) -> str:
        """Return format type identifier (e.g., 'jpeg+depth', 'h264')."""
        pass

    @abstractmethod
    async def encode(self, output: RenderOutput, frame_id: int) -> RenderPayload:
        """
        Encode RenderOutput to RenderPayload.

        Args:
            output: Rendering result
            frame_id: Frame identifier

        Returns:
            RenderPayload with encoded data and metadata
        """
        pass

    async def on_init(self) -> bool:
        """Optional initialization hook. Returns True if successful."""
        return True

    async def on_shutdown(self):
        """Optional cleanup hook."""
        pass
