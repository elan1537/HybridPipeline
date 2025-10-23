"""Raw tensor encoder for debugging."""

import torch
import io
from renderer.encoders.base import BaseEncoder
from renderer.data_types import RenderOutput, RenderPayload


class RawEncoder(BaseEncoder):
    """
    Raw tensor encoder using torch.save.

    Warning: Very large output size. Use only for debugging.
    """

    def get_format_type(self) -> str:
        return "raw"

    async def encode(self, output: RenderOutput, frame_id: int) -> RenderPayload:
        """Serialize tensors using torch.save."""
        import time

        output.validate()

        data = {
            "color": output.color,
            "depth": output.depth,
            "alpha": output.alpha,
            "metadata": output.metadata
        }

        buffer = io.BytesIO()
        torch.save(data, buffer)
        serialized = buffer.getvalue()

        # Measure encode end time
        encode_end_timestamp = time.time() * 1000.0  # ms

        # Extract timestamps from RenderOutput metadata
        client_timestamp = output.metadata.get("client_timestamp", 0.0)
        server_timestamp = output.metadata.get("server_timestamp", 0.0)
        render_start_timestamp = output.metadata.get("render_start_timestamp", 0.0)

        return RenderPayload(
            data=serialized,
            metadata={
                "format_type": 2,  # 0=JPEG, 1=H264, 2=Raw
                "frame_id": frame_id,
                "color_len": len(serialized),  # For raw, color_len = total_len
                "depth_len": 0,
                "width": output.color.shape[1],
                "height": output.color.shape[0],
                # Timestamps for protocol
                "client_timestamp": client_timestamp,
                "server_timestamp": server_timestamp,
                "render_start_timestamp": render_start_timestamp,
                "encode_end_timestamp": encode_end_timestamp,
                # Extra metadata (not in wire protocol)
                "data_size": len(serialized)
            }
        )
