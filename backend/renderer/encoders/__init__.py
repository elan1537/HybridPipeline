"""Encoder factory."""

from renderer.encoders.base import BaseEncoder
from renderer.encoders.jpeg import JpegEncoder
from renderer.encoders.h264 import H264Encoder
from renderer.encoders.raw import RawEncoder


def create_encoder(format_type: str, **kwargs) -> BaseEncoder:
    """
    Create encoder instance.

    Args:
        format_type: "jpeg", "h264", or "raw"
        **kwargs: Encoder-specific parameters

    Returns:
        Encoder instance

    Example:
        encoder = create_encoder("jpeg", quality=95)
        encoder = create_encoder("h264", width=1280, height=720, bitrate=30_000_000)
    """
    if format_type in ("jpeg", "jpeg+depth"):
        return JpegEncoder(**kwargs)
    elif format_type == "h264":
        return H264Encoder(**kwargs)
    elif format_type == "raw":
        return RawEncoder(**kwargs)
    else:
        raise ValueError(f"Unknown encoder type: {format_type}")


__all__ = [
    "BaseEncoder",
    "JpegEncoder",
    "H264Encoder",
    "RawEncoder",
    "create_encoder"
]
