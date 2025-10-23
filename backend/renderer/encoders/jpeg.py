"""JPEG encoder with automatic backend selection."""

import torch
from renderer.encoders.base import BaseEncoder
from renderer.data_types import RenderOutput, RenderPayload
from renderer.utils.depth_utils import depth_to_log_float16


# Try nvimgcodec first (GPU accelerated)
try:
    from nvidia import nvimgcodec as nvc
    NVC_AVAILABLE = True
except ImportError:
    NVC_AVAILABLE = False

# Fallback to OpenCV
try:
    import cv2
    import numpy as np
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


class JpegEncoder(BaseEncoder):
    """
    JPEG + Log-normalized Float16 Depth encoder.

    Automatically selects backend:
    - nvimgcodec (GPU) if available
    - OpenCV (CPU) as fallback

    From backend-feedforward-20251021/server.py render_loop_jpeg().
    """

    def __init__(self, quality: int = 90):
        """
        Args:
            quality: JPEG quality (0-100)
        """
        self.quality = quality
        self.backend = None
        self.encoder = None

    async def on_init(self) -> bool:
        """Initialize encoder backend."""
        if NVC_AVAILABLE:
            try:
                self.encoder = nvc.Encoder()
                self.backend = "nvimgcodec"
                print(f"[JpegEncoder] Using nvimgcodec (GPU), quality={self.quality}")
                return True
            except Exception as e:
                print(f"[JpegEncoder] nvimgcodec init failed: {e}, falling back to OpenCV")

        if CV2_AVAILABLE:
            self.backend = "opencv"
            print(f"[JpegEncoder] Using OpenCV (CPU), quality={self.quality}")
            return True

        print(f"[JpegEncoder] No backend available")
        return False

    def get_format_type(self) -> str:
        return "jpeg+depth"

    async def encode(self, output: RenderOutput, frame_id: int) -> RenderPayload:
        """
        Encode color as JPEG and depth as log-normalized float16.

        From backend-feedforward-20251021/server.py:344-358
        """
        import time

        output.validate()
        H, W = output.color.shape[:2]

        # Color encoding
        if self.backend == "nvimgcodec":
            jpeg_bytes = self._encode_nvc(output.color)
        else:
            jpeg_bytes = self._encode_cv2(output.color)

        # Depth encoding: log-normalized [0,1] float16
        near = output.metadata.get("near", 1.0)
        far = output.metadata.get("far", 30.0)

        depth_f16 = depth_to_log_float16(output.depth, output.alpha, near, far)
        depth_bytes = depth_f16.cpu().numpy().tobytes()

        # Measure encode end time
        encode_end_timestamp = time.time() * 1000.0  # ms

        # Extract timestamps from RenderOutput metadata
        client_timestamp = output.metadata.get("client_timestamp", 0.0)
        server_timestamp = output.metadata.get("server_timestamp", 0.0)
        render_start_timestamp = output.metadata.get("render_start_timestamp", 0.0)

        return RenderPayload(
            data=jpeg_bytes + depth_bytes,
            metadata={
                "format_type": 0,  # 0=JPEG, 1=H264, 2=Raw
                "frame_id": frame_id,
                "color_len": len(jpeg_bytes),
                "depth_len": len(depth_bytes),
                "width": W,
                "height": H,
                # Timestamps for protocol
                "client_timestamp": client_timestamp,
                "server_timestamp": server_timestamp,
                "render_start_timestamp": render_start_timestamp,
                "encode_end_timestamp": encode_end_timestamp,
                # Extra metadata (not in wire protocol)
                "depth_encoding": "log01",
                "jpeg_quality": self.quality,
                "backend": self.backend
            }
        )

    def _encode_nvc(self, color: torch.Tensor) -> bytes:
        """Encode with nvimgcodec (GPU)."""
        rgb_uint8 = (color * 255).clamp(0, 255).to(torch.uint8).contiguous()
        img_nv = nvc.as_image(rgb_uint8)

        # Try new API first (quality_value)
        try:
            params = nvc.EncodeParams(quality_value=self.quality)
        except TypeError:
            # Fallback to old API (quality)
            params = nvc.EncodeParams(quality=self.quality)

        jpeg_bytes = self.encoder.encode(img_nv, "jpeg", params=params)
        return jpeg_bytes

    def _encode_cv2(self, color: torch.Tensor) -> bytes:
        """Encode with OpenCV (CPU)."""
        rgb_uint8 = (color * 255).clamp(0, 255).to(torch.uint8)
        rgb_np = rgb_uint8.cpu().numpy()
        bgr_np = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2BGR)

        encode_params = [cv2.IMWRITE_JPEG_QUALITY, self.quality]
        success, jpeg_arr = cv2.imencode('.jpg', bgr_np, encode_params)

        if not success:
            raise RuntimeError("JPEG encoding failed")

        return jpeg_arr.tobytes()

    async def on_shutdown(self):
        """Cleanup encoder."""
        if self.encoder:
            del self.encoder
            self.encoder = None
