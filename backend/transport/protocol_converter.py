"""
Protocol Converter - Translates between Frontend and Renderer binary protocols.

Frontend Protocol (160 bytes camera, 44-48 bytes video header):
  - Camera: eye/target + intrinsics + metadata
  - Video: format-specific header + data

Renderer Protocol (168 bytes camera, 56 bytes video header):
  - Camera: view_matrix + intrinsics + metadata
  - Video: unified binary header + data

Note: Uses numpy for lightweight Transport Service (no torch dependency).
"""

import struct
import numpy as np
from typing import Tuple

# Try relative import first (when used as module), then absolute
try:
    from ..renderer.data_types import CameraFrame
except ImportError:
    try:
        from renderer.data_types import CameraFrame
    except ImportError:
        # For standalone testing
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from renderer.data_types import CameraFrame


# =============================================================================
# Camera Protocol Conversion (Frontend → Renderer)
# =============================================================================

def parse_frontend_camera(
    raw: bytes,
    width: int,
    height: int,
    near: float = 0.1,
    far: float = 100.0,
    device: str = "cpu"  # Ignored for numpy implementation
) -> CameraFrame:
    """
    Parse Frontend WebSocket camera data (160 bytes) to CameraFrame.

    Frontend format (160 bytes):
      [0:128]   - Camera data (32 floats):
                  - eye (3), target (3), intrinsics (9), unused (17)
      [128:132] - frame_id (uint32)
      [132:136] - padding (4 bytes)
      [136:144] - client_timestamp (float64)
      [144:148] - time_index (float32)
      [148:160] - padding (12 bytes)

    Args:
        raw: 160-byte camera data from Frontend
        width: Viewport width
        height: Viewport height
        near: Near clipping plane
        far: Far clipping plane
        device: Device (ignored, kept for API compatibility)

    Returns:
        CameraFrame object with numpy arrays

    Raises:
        ValueError: If data size is incorrect
    """
    if len(raw) != 160:
        raise ValueError(f"Invalid camera data size: {len(raw)} (expected 160)")

    # Parse camera data (128 bytes = 32 floats)
    vals = struct.unpack("<32f", raw[:128])

    # Extract eye, target (first 6 floats)
    # Note: Y-axis flip from Frontend coordinate system
    eye = np.array([vals[0], -vals[1], vals[2]], dtype=np.float32)
    target = np.array([vals[3], -vals[4], vals[5]], dtype=np.float32)
    up = np.array([0., 1., 0.], dtype=np.float32)

    # Compute view matrix (look-at transformation)
    # Forward axis (z)
    zaxis = target - eye
    zaxis = zaxis / np.linalg.norm(zaxis)

    # Right axis (x)
    xaxis = np.cross(up, zaxis)
    xaxis = xaxis / np.linalg.norm(xaxis)

    # Up axis (y)
    yaxis = np.cross(zaxis, xaxis)

    # Rotation matrix (world to camera)
    R_w2c = np.array([xaxis, yaxis, zaxis], dtype=np.float32)

    # Translation vector
    t = -R_w2c @ eye

    # Build 4x4 view matrix
    view_matrix = np.eye(4, dtype=np.float32)
    view_matrix[:3, :3] = R_w2c
    view_matrix[:3, 3] = t

    # Extract intrinsics (9 floats starting at index 6)
    intrinsics_vals = vals[6:15]
    intrinsics = np.array([
        [intrinsics_vals[0], intrinsics_vals[1], intrinsics_vals[2]],
        [intrinsics_vals[3], intrinsics_vals[4], intrinsics_vals[5]],
        [intrinsics_vals[6], intrinsics_vals[7], intrinsics_vals[8]]
    ], dtype=np.float32)

    # Parse metadata (32 bytes)
    frame_id = struct.unpack_from("<I", raw, 128)[0]
    client_timestamp = struct.unpack_from("<d", raw, 136)[0]
    time_index = struct.unpack_from("<f", raw, 144)[0]

    import time
    server_timestamp = time.time() * 1000.0  # ms

    return CameraFrame(
        view_matrix=view_matrix,  # (4, 4) numpy array
        intrinsics=intrinsics,     # (3, 3) numpy array
        width=width,
        height=height,
        near=near,
        far=far,
        time_index=time_index,
        frame_id=frame_id,
        client_timestamp=client_timestamp,
        server_timestamp=server_timestamp
    )


# =============================================================================
# Video Protocol Conversion (Renderer → Frontend)
# =============================================================================

def parse_renderer_video_header(header: bytes) -> dict:
    """
    Parse Renderer video header (56 bytes).

    Renderer format (56 bytes):
      0-3:   frame_id (uint32)
      4:     format_type (uint8: 0=JPEG, 1=H264, 2=Raw)
      5-7:   padding
      8-11:  color_len (uint32)
      12-15: depth_len (uint32)
      16-19: width (uint32)
      20-23: height (uint32)
      24-31: client_timestamp (float64)
      32-39: server_timestamp (float64)
      40-47: render_start_timestamp (float64)
      48-55: encode_end_timestamp (float64)

    Args:
        header: 56-byte header

    Returns:
        dict with parsed fields

    Raises:
        ValueError: If header size is incorrect
    """
    if len(header) != 56:
        raise ValueError(f"Invalid header size: {len(header)} (expected 56)")

    frame_id, format_type, color_len, depth_len, width, height, \
        client_ts, server_ts, render_start_ts, encode_end_ts = \
        struct.unpack("<IB3xIIIIdddd", header)

    return {
        'frame_id': frame_id,
        'format_type': format_type,
        'color_len': color_len,
        'depth_len': depth_len,
        'width': width,
        'height': height,
        'client_timestamp': client_ts,
        'server_timestamp': server_ts,
        'render_start_timestamp': render_start_ts,
        'encode_end_timestamp': encode_end_ts
    }


def create_frontend_video_header(
    renderer_metadata: dict,
    transport_send_timestamp: float
) -> bytes:
    """
    Create Frontend video header from Renderer metadata.

    JPEG format (44 bytes):
      0-3:   jpegLen (uint32)
      4-7:   depthLen (uint32)
      8-11:  frameId (uint32)
      12-19: clientSendTime (float64)
      20-27: serverReceiveTime (float64)
      28-35: serverProcessEndTime (float64)
      36-43: serverSendTime (float64)

    H.264/Raw format (40 bytes):
      0-3:   videoLen (uint32)
      4-7:   frameId (uint32)
      8-15:  clientSendTime (float64)
      16-23: serverReceiveTime (float64)
      24-31: serverProcessEndTime (float64)
      32-39: serverSendTime (float64)

    Args:
        renderer_metadata: Metadata from parse_renderer_video_header()
        transport_send_timestamp: Transport send timestamp (ms)

    Returns:
        bytes: Frontend video header

    Raises:
        ValueError: If format_type is invalid
    """
    format_type = renderer_metadata['format_type']
    frame_id = renderer_metadata['frame_id']
    color_len = renderer_metadata['color_len']
    depth_len = renderer_metadata['depth_len']

    client_ts = renderer_metadata['client_timestamp']
    server_ts = renderer_metadata['server_timestamp']
    encode_end_ts = renderer_metadata['encode_end_timestamp']

    if format_type == 0:  # JPEG
        # 44 bytes header
        return struct.pack("<IIIdddd",
            color_len,              # jpegLen
            depth_len,              # depthLen
            frame_id,               # frameId
            client_ts,              # clientSendTime
            server_ts,              # serverReceiveTime
            encode_end_ts,          # serverProcessEndTime
            transport_send_timestamp  # serverSendTime
        )

    elif format_type in (1, 2):  # H.264 or Raw
        # 40 bytes header
        video_len = color_len  # For H.264/Raw, color_len is total length
        return struct.pack("<IIdddd",
            video_len,              # videoLen
            frame_id,               # frameId
            client_ts,              # clientSendTime
            server_ts,              # serverReceiveTime
            encode_end_ts,          # serverProcessEndTime
            transport_send_timestamp  # serverSendTime
        )

    else:
        raise ValueError(f"Unknown format type: {format_type} (expected 0=JPEG, 1=H264, 2=Raw)")


# =============================================================================
# Validation Functions
# =============================================================================

def validate_frontend_camera_data(raw: bytes) -> Tuple[bool, str]:
    """
    Validate Frontend camera data.

    Args:
        raw: Camera data bytes

    Returns:
        Tuple[bool, str]: (is_valid, error_message)
    """
    import numpy as np

    # Check size
    if len(raw) != 160:
        return False, f"Invalid size: {len(raw)} (expected 160)"

    # Parse and check values
    try:
        vals = struct.unpack("<32f", raw[:128])

        # Check for NaN or Inf in critical fields
        for i, val in enumerate(vals[:15]):  # eye, target, intrinsics
            if not np.isfinite(val):
                return False, f"Invalid value at index {i}: {val}"

        # Check frame_id
        frame_id = struct.unpack_from("<I", raw, 128)[0]
        if frame_id > 1_000_000_000:  # Sanity check
            return False, f"Suspicious frame_id: {frame_id}"

        return True, ""

    except Exception as e:
        return False, f"Parse error: {e}"
