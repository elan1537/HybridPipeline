"""
Simple frame buffering strategies for camera data.

Provides two basic patterns:
- FIFOBuffer: Sequential processing (Queue)
- LatestFrameBuffer: Always process newest frame (Mailbox)
"""

import asyncio
from abc import ABC, abstractmethod
from typing import Optional, TypeVar, Generic

T = TypeVar('T')


class FrameBuffer(ABC, Generic[T]):
    """Abstract interface for frame buffering."""

    @abstractmethod
    async def put(self, frame: T):
        """Add a frame to the buffer."""
        pass

    @abstractmethod
    async def get(self) -> T:
        """Get a frame from the buffer."""
        pass

    @abstractmethod
    def clear(self):
        """Clear all buffered frames."""
        pass


class FIFOBuffer(FrameBuffer[T]):
    """
    FIFO Queue-based buffer.

    Characteristics:
    - Sequential processing (first-in, first-out)
    - Higher latency (processes old frames)
    - Configurable size

    Args:
        maxsize: Maximum queue size (default: 2)
    """

    def __init__(self, maxsize: int = 2):
        self.queue = asyncio.Queue(maxsize=maxsize)
        self.maxsize = maxsize

    async def put(self, frame: T):
        """Add frame to queue (blocks if full)."""
        await self.queue.put(frame)

    async def get(self) -> T:
        """Get frame from queue (FIFO order)."""
        return await self.queue.get()

    def clear(self):
        """Clear all frames from queue."""
        while not self.queue.empty():
            try:
                self.queue.get_nowait()
            except asyncio.QueueEmpty:
                break


class LatestFrameBuffer(FrameBuffer[T]):
    """
    Latest-frame-only buffer (Mailbox pattern).

    Characteristics:
    - Always process newest frame
    - Lowest latency
    - Aggressive frame dropping

    Best for real-time rendering and network streaming.
    """

    def __init__(self):
        self.frame: Optional[T] = None
        self.lock = asyncio.Lock()
        self.available = asyncio.Event()

    async def put(self, frame: T):
        """Update to latest frame (overwrites previous)."""
        async with self.lock:
            self.frame = frame
            self.available.set()

    async def get(self) -> T:
        """Get latest frame (waits until a valid frame is available)."""
        while True:
            await self.available.wait()

            async with self.lock:
                frame = self.frame
                # Only clear and return if we have a valid frame
                # This prevents returning None after buffer clear
                if frame is not None:
                    self.available.clear()
                    return frame

                # Frame is None (just cleared), wait for next frame
                self.available.clear()

    def clear(self):
        """Clear buffered frame."""
        # For LatestFrameBuffer, we just clear the available flag
        # The next get() will wait for a new frame
        self.available.clear()
        self.frame = None
