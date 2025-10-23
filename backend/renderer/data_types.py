"""
Data types for the renderer service.

Defines standard input/output formats for scene renderers and encoders.

Note: torch is optional - Transport Service can use numpy arrays only.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Union, TYPE_CHECKING
import numpy as np

# Import torch for type checking only
if TYPE_CHECKING:
    import torch

# Optional torch import (not required for Transport Service)
try:
    import torch as _torch
    TORCH_AVAILABLE = True
except ImportError:
    _torch = None
    TORCH_AVAILABLE = False


@dataclass
class CameraFrame:
    """
    Camera parameters for rendering.

    Attributes:
        view_matrix: Camera view matrix (4x4) or batch (N, 4, 4)
                     Can be torch.Tensor (Renderer) or np.ndarray (Transport)
        intrinsics: Camera intrinsics matrix (3x3) or batch (N, 3, 3)
                    Can be torch.Tensor (Renderer) or np.ndarray (Transport)
        width: Image width in pixels
        height: Image height in pixels
        near: Near clipping plane distance
        far: Far clipping plane distance
        time_index: Temporal index for dynamic scenes (4DGS, etc.)
        frame_id: Frame identifier for tracking
        client_timestamp: Client send timestamp (ms)
        server_timestamp: Server receive timestamp (ms)
    """
    view_matrix: Union['torch.Tensor', np.ndarray]
    intrinsics: Union['torch.Tensor', np.ndarray]
    width: int
    height: int
    near: float
    far: float
    time_index: Optional[float] = None
    frame_id: Optional[int] = None
    client_timestamp: Optional[float] = None
    server_timestamp: Optional[float] = None


@dataclass
class RenderOutput:
    """
    Raw rendering output before encoding.

    All tensors are in float32 format on GPU.
    Color values are in [0, 1] range.

    Attributes:
        color: RGB color image (H, W, 3) float32 [0-1]
        depth: Depth map (H, W) float32 in world units
        alpha: Alpha/opacity map (H, W) float32 [0-1]
        metadata: Optional metadata (render time, etc.)

    Note: This class requires torch (only used in Renderer Service).
    """
    color: 'torch.Tensor'
    depth: 'torch.Tensor'
    alpha: 'torch.Tensor'
    metadata: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> bool:
        """Validate tensor shapes and dtypes."""
        assert self.color.ndim == 3 and self.color.shape[2] == 3, \
            f"Color must be (H, W, 3), got {self.color.shape}"
        assert self.depth.ndim == 2, \
            f"Depth must be (H, W), got {self.depth.shape}"
        assert self.alpha.ndim == 2, \
            f"Alpha must be (H, W), got {self.alpha.shape}"
        assert self.color.shape[:2] == self.depth.shape == self.alpha.shape, \
            "Color, depth, alpha must have same spatial dimensions"
        return True


@dataclass
class RenderPayload:
    """
    Encoded rendering output ready for transmission.

    Attributes:
        data: Encoded binary data (JPEG, H.264, etc.)
        metadata: Metadata (format, size, timestamps, etc.)
    """
    data: bytes
    metadata: Dict[str, Any] = field(default_factory=dict)
