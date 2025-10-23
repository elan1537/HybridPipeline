"""
Base adapter interfaces for Transport Service.

Provides protocol-agnostic abstractions for:
- Frontend communication (WebSocket, WebRTC, UDP, etc.)
- Backend communication (Unix Socket, Shared Memory, TCP)
"""

from abc import ABC, abstractmethod
from typing import Optional
import sys
from pathlib import Path

# Flexible imports for renderer data types
try:
    from renderer.data_types import CameraFrame, RenderPayload
except ImportError:
    try:
        sys.path.insert(0, str(Path(__file__).parent.parent.parent))
        from renderer.data_types import CameraFrame, RenderPayload
    except ImportError:
        # For type hints only, will fail at runtime if not available
        CameraFrame = None
        RenderPayload = None


class BaseFrontendAdapter(ABC):
    """
    Abstract interface for Frontend communication adapters.

    Implementations: WebSocket, WebRTC, UDP, etc.

    Responsibilities:
    - Receive camera data from Frontend
    - Send video data to Frontend
    - Handle protocol-specific connection management
    """

    @abstractmethod
    async def start(self):
        """
        Start the adapter (e.g., start listening for connections).

        Raises:
            RuntimeError: If adapter fails to start
        """
        pass

    @abstractmethod
    async def recv_camera(self) -> Optional[CameraFrame]:
        """
        Receive camera data from Frontend.

        Returns:
            CameraFrame if received, None if connection closed

        Raises:
            ConnectionError: If connection is lost
        """
        pass

    @abstractmethod
    async def send_video(self, payload: RenderPayload):
        """
        Send video data to Frontend.

        Args:
            payload: Encoded video payload from Renderer

        Raises:
            ConnectionError: If send fails
        """
        pass

    @abstractmethod
    async def is_connected(self) -> bool:
        """
        Check if Frontend is connected.

        Returns:
            True if connected, False otherwise
        """
        pass

    @abstractmethod
    async def close(self):
        """Close the adapter and cleanup resources."""
        pass


class BaseBackendAdapter(ABC):
    """
    Abstract interface for Backend (Renderer) communication adapters.

    Implementations: Unix Socket, Shared Memory, TCP, etc.

    Responsibilities:
    - Send camera data to Renderer
    - Receive video data from Renderer
    - Handle connection lifecycle
    """

    @abstractmethod
    async def connect(self, timeout: float = 10.0) -> bool:
        """
        Connect to Backend Renderer.

        Args:
            timeout: Connection timeout in seconds

        Returns:
            True if connection succeeded, False otherwise
        """
        pass

    @abstractmethod
    async def send_camera(self, camera: CameraFrame):
        """
        Send camera data to Renderer.

        Args:
            camera: Camera frame to send

        Raises:
            ConnectionError: If send fails
        """
        pass

    @abstractmethod
    async def recv_video(self) -> Optional[RenderPayload]:
        """
        Receive video data from Renderer.

        Returns:
            RenderPayload if received, None if connection closed

        Raises:
            ConnectionError: If connection is lost
        """
        pass

    @abstractmethod
    async def is_connected(self) -> bool:
        """
        Check if Backend is connected.

        Returns:
            True if connected, False otherwise
        """
        pass

    @abstractmethod
    async def close(self):
        """Close the adapter and cleanup resources."""
        pass
