"""H.264 encoder using NVENC."""

import time
import torch
from renderer.encoders.base import BaseEncoder
from renderer.data_types import RenderOutput, RenderPayload
from renderer.utils.depth_utils import depth_to_rgb
from renderer.utils.color_utils import rgb_to_nv12


class H264Encoder(BaseEncoder):
    """
    H.264 encoder with vertically stacked color + depth.

    From backend-feedforward-20251021/server.py render_loop() and session.py.
    """

    def __init__(self, width: int = None, height: int = None,
                 bitrate: int = 20_000_000,
                 fps: int = 60,
                 preset: str = "P3"):
        """
        Args:
            width: Initial frame width (optional, will be set from first frame)
            height: Initial frame height (optional, will be set from first frame)
            bitrate: Target bitrate (bps)
            fps: Target frame rate
            preset: NVENC preset (P1-P7, P3=low latency)
        """
        self.width = width
        self.height = height
        self.bitrate = bitrate
        self.fps = fps
        self.preset = preset
        self.encoder = None
        self.initialized = False

    async def on_init(self) -> bool:
        """
        Initialize NVENC encoder.

        If width/height are not set, initialization is deferred until first frame.
        """
        if self.width is None or self.height is None:
            print(f"[H264Encoder] Deferred initialization (waiting for first frame)")
            return True

        return await self._initialize_encoder(self.width, self.height)

    async def _initialize_encoder(self, width: int, height: int) -> bool:
        """
        Internal method to initialize encoder with specific resolution.

        Args:
            width: Frame width
            height: Frame height (single frame, not stacked)
        """
        try:
            import PyNvVideoCodec as nvvc

            # Validate even dimensions
            if width % 2 != 0 or height % 2 != 0:
                print(f"[H264Encoder] Warning: Adjusting odd resolution {width}x{height} to even")
                width = width if width % 2 == 0 else width - 1
                height = height if height % 2 == 0 else height - 1

            # Constant QP mode: bitrate MUST be 0 to avoid rate control buffering
            encoder_params = {
                "codec": "h264",
                "preset": self.preset,
                "tuning_info": "ultra_low_latency",
                "repeatspspps": 1,
                "bitrate": 0,       # Disable rate control (required for constqp)
                "constqp": 28,      # Fixed QP for zero latency
                "gop": 1,           # All I-frames
                "fps": self.fps,
                "delay": 0,         # No reordering delay
                "numB": 0,          # No B-frames
            }

            combined_height = height * 2

            self.encoder = nvvc.CreateEncoder(
                width=width,
                height=combined_height,
                fmt="NV12",
                usecpuinputbuffer=False,
                **encoder_params
            )

            self.width = width
            self.height = height
            self.initialized = True

            print(f"[H264Encoder] Initialized {width}x{combined_height} @ {self.fps}fps, {self.bitrate/1e6:.1f}Mbps")
            return True

        except ImportError:
            print(f"[H264Encoder] PyNvVideoCodec not found")
            return False
        except Exception as e:
            print(f"[H264Encoder] Init failed: {e}")
            return False

    def get_format_type(self) -> str:
        return "h264"

    async def encode(self, output: RenderOutput, frame_id: int) -> RenderPayload:
        """
        Encode color + depth to H.264 bitstream.

        From backend-feedforward-20251021/server.py:247-264
        """
        output.validate()

        # Get frame dimensions
        frame_height, frame_width = output.color.shape[:2]

        # Lazy initialization or resolution change detection
        if not self.initialized or self.width != frame_width or self.height != frame_height:
            if self.initialized:
                # Resolution changed, reinitialize encoder
                print(f"[H264Encoder] Resolution changed from {self.width}x{self.height} to {frame_width}x{frame_height}, reinitializing...")
                await self.on_shutdown()

            # Initialize encoder with frame resolution
            if not await self._initialize_encoder(frame_width, frame_height):
                raise RuntimeError("Failed to initialize H264 encoder")

        if self.encoder is None:
            raise RuntimeError("Encoder not initialized")

        # Color: (H, W, 3) uint8
        color_uint8 = (output.color * 255).clamp(0, 255).to(torch.uint8).contiguous()

        # Depth: log-normalized RGB uint8
        near = output.metadata.get("near", 1.0)
        far = output.metadata.get("far", 30.0)
        depth_rgb = depth_to_rgb(output.depth, output.alpha, near, far)

        # Vertical stack
        combined = torch.cat([color_uint8, depth_rgb], dim=0)

        # RGB to NV12
        nv12 = rgb_to_nv12(combined)

        # Encode with immediate output (constqp + zeroReorderDelay)
        h264_bytes = bytes(self.encoder.Encode(nv12))

        # Measure encode end time
        encode_end_timestamp = time.time() * 1000.0  # ms

        # Extract timestamps from RenderOutput metadata
        client_timestamp = output.metadata.get("client_timestamp", 0.0)
        server_timestamp = output.metadata.get("server_timestamp", 0.0)
        render_start_timestamp = output.metadata.get("render_start_timestamp", 0.0)

        return RenderPayload(
            data=h264_bytes,
            metadata={
                "format_type": 1,  # 0=JPEG, 1=H264, 2=Raw
                "frame_id": frame_id,
                "color_len": len(h264_bytes),  # For H.264, color_len = video_len
                "depth_len": 0,                 # No separate depth for H.264
                "width": self.width,
                "height": self.height * 2,      # Stacked height
                # Timestamps for protocol
                "client_timestamp": client_timestamp,
                "server_timestamp": server_timestamp,
                "render_start_timestamp": render_start_timestamp,
                "encode_end_timestamp": encode_end_timestamp,
                # Extra metadata (not in wire protocol)
                "codec": "h264",
                "bitrate": self.bitrate,
                "fps": self.fps,
                "preset": self.preset
            }
        )

    async def on_shutdown(self):
        """Destroy encoder."""
        if self.encoder:
            del self.encoder
            self.encoder = None
        self.initialized = False
