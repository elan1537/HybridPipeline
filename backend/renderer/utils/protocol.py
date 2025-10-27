"""
Protocol utilities for camera frame and render payload serialization.

Defines wire formats for communication between Transport and Renderer services.

Note: torch is optional - uses numpy for Transport Service compatibility.
"""

import struct
import numpy as np
from typing import Tuple
import asyncio

# Optional torch import
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    TORCH_AVAILABLE = False

from renderer.data_types import CameraFrame, RenderPayload


# =============================================================================
# Camera Frame Protocol (Frontend ↔ Transport ↔ Renderer)
# =============================================================================

CAMERA_FRAME_SIZE = 168  # bytes
CONTROL_MESSAGE_SIZE = 8  # bytes (legacy, for backward compatibility)
CONTROL_MESSAGE_EXT_SIZE = 12  # bytes (extended with parameter)

# Control message commands
CONTROL_CMD_RESET_ENCODER = 1
CONTROL_CMD_CHANGE_ENCODER = 2

"""
Control Message Wire Format:

Basic (8 bytes):
Offset  Size  Type     Field
------  ----  -------  ------------------
0       4     uint32   magic (0xDEADBEEF)
4       4     uint32   command
------
Total:  8 bytes

Extended (12 bytes):
Offset  Size  Type     Field
------  ----  -------  ------------------
0       4     uint32   magic (0xDEADBEEF)
4       4     uint32   command
8       4     uint32   param
------
Total:  12 bytes

Commands:
- 1: Reset encoder (for WebSocket reconnection) - 8 bytes
- 2: Change encoder type - 12 bytes
  * param: 0=JPEG, 1=H264, 2=Raw

Camera Frame Wire Format (168 bytes):

Offset  Size  Type     Field
------  ----  -------  ------------------
0       64    float32  view_matrix (4×4)
64      36    float32  intrinsics (3×3)
100     4     uint32   width
104     4     uint32   height
108     4     float32  near
112     4     float32  far
116     4     float32  time_index
120     4     uint32   frame_id
124     4     -        padding (alignment)
128     8     float64  client_timestamp
136     8     float64  server_timestamp
144     24    -        reserved (future use)
------
Total:  168 bytes
"""


def pack_camera_frame(camera: CameraFrame) -> bytes:
    """
    Serialize CameraFrame to wire format (168 bytes).

    Args:
        camera: CameraFrame object

    Returns:
        bytes: 168-byte packed camera data

    Raises:
        ValueError: If camera parameters are invalid
    """
    # Validate input
    if camera.view_matrix.shape != (4, 4):
        raise ValueError(f"view_matrix must be (4, 4), got {camera.view_matrix.shape}")
    if camera.intrinsics.shape != (3, 3):
        raise ValueError(f"intrinsics must be (3, 3), got {camera.intrinsics.shape}")

    # Convert tensors to numpy if needed
    if TORCH_AVAILABLE and isinstance(camera.view_matrix, torch.Tensor):
        view_matrix = camera.view_matrix.cpu().numpy()
        intrinsics = camera.intrinsics.cpu().numpy()
    else:
        view_matrix = camera.view_matrix
        intrinsics = camera.intrinsics

    # Pack view matrix (64 bytes)
    view_bytes = view_matrix.astype(np.float32).tobytes()

    # Pack intrinsics (36 bytes)
    intrinsics_bytes = intrinsics.astype(np.float32).tobytes()

    # Pack metadata
    metadata_bytes = struct.pack("<IIfffIIdd",
        camera.width,                                    # uint32
        camera.height,                                   # uint32
        camera.near,                                     # float32
        camera.far,                                      # float32
        camera.time_index if camera.time_index else 0.0,  # float32
        camera.frame_id if camera.frame_id else 0,       # uint32
        0,                                               # padding (4 bytes)
        camera.client_timestamp if camera.client_timestamp else 0.0,  # float64
        camera.server_timestamp if camera.server_timestamp else 0.0   # float64
    )

    # Reserved space (24 bytes)
    reserved = b'\x00' * 24

    return view_bytes + intrinsics_bytes + metadata_bytes + reserved


def parse_camera_frame(data: bytes) -> CameraFrame:
    """
    Parse wire format to CameraFrame (168 bytes).

    Args:
        data: 168-byte camera data

    Returns:
        CameraFrame object

    Raises:
        ValueError: If data size is incorrect
    """
    if len(data) != CAMERA_FRAME_SIZE:
        raise ValueError(f"Invalid camera frame size: {len(data)} (expected {CAMERA_FRAME_SIZE})")

    # Parse view matrix (64 bytes)
    view_floats = struct.unpack("<16f", data[0:64])
    view_matrix = np.array(view_floats, dtype=np.float32).reshape(4, 4)

    # Parse intrinsics (36 bytes)
    intrinsics_floats = struct.unpack("<9f", data[64:100])
    intrinsics = np.array(intrinsics_floats, dtype=np.float32).reshape(3, 3)

    # Parse metadata (44 bytes)
    width, height, near, far, time_index, frame_id, _, client_ts, server_ts = \
        struct.unpack("<IIfffIIdd", data[100:144])

    # Reserved space (24 bytes) - ignored

    return CameraFrame(
        view_matrix=view_matrix,
        intrinsics=intrinsics,
        width=width,
        height=height,
        near=near,
        far=far,
        time_index=time_index,
        frame_id=frame_id,
        client_timestamp=client_ts,
        server_timestamp=server_ts
    )


def pack_control_message(command: int, param: int = 0) -> bytes:
    """
    Create a control message for Renderer.

    Args:
        command: Control command (e.g., CONTROL_CMD_RESET_ENCODER)
        param: Optional parameter (for CONTROL_CMD_CHANGE_ENCODER: 0=JPEG, 1=H264, 2=Raw)

    Returns:
        bytes: 8 or 12-byte control message
    """
    magic = 0xDEADBEEF

    # If command requires parameter, use extended format (12 bytes)
    if command == CONTROL_CMD_CHANGE_ENCODER:
        return struct.pack("<III", magic, command, param)
    else:
        # Legacy format (8 bytes)
        return struct.pack("<II", magic, command)


def is_control_message(data: bytes) -> bool:
    """
    Check if received data is a control message.

    Args:
        data: Received bytes

    Returns:
        bool: True if control message
    """
    # Support both 8-byte and 12-byte control messages
    if len(data) != CONTROL_MESSAGE_SIZE and len(data) != CONTROL_MESSAGE_EXT_SIZE:
        return False
    magic = struct.unpack_from("<I", data, 0)[0]
    return magic == 0xDEADBEEF


def parse_control_message(data: bytes) -> Tuple[int, int]:
    """
    Parse control message.

    Args:
        data: 8 or 12-byte control message

    Returns:
        Tuple[int, int]: (command, param)
            - command: Command code
            - param: Parameter (0 if not present)

    Raises:
        ValueError: If invalid control message
    """
    if len(data) not in [CONTROL_MESSAGE_SIZE, CONTROL_MESSAGE_EXT_SIZE]:
        raise ValueError(f"Control message must be {CONTROL_MESSAGE_SIZE} or {CONTROL_MESSAGE_EXT_SIZE} bytes, got {len(data)}")

    magic = struct.unpack_from("<I", data, 0)[0]

    if magic != 0xDEADBEEF:
        raise ValueError(f"Invalid control message magic: 0x{magic:08X}")

    if len(data) == CONTROL_MESSAGE_EXT_SIZE:
        # Extended format with parameter
        _, command, param = struct.unpack("<III", data)
        return (command, param)
    else:
        # Legacy format without parameter
        _, command = struct.unpack("<II", data)
        return (command, 0)


# =============================================================================
# Render Payload Protocol (Renderer → Transport → Frontend)
# =============================================================================

"""
Render Payload Wire Format (Binary, 56 bytes fixed header):

Offset  Size  Type     Field
------  ----  -------  ------------------
0       4     uint32   frame_id
4       1     uint8    format_type (0=JPEG, 1=H264, 2=Raw)
5       3     -        padding
8       4     uint32   color_len (or video_len for H264/Raw)
12      4     uint32   depth_len (or 0 for H264/Raw)
16      4     uint32   width
20      4     uint32   height
24      8     float64  client_timestamp
32      8     float64  server_timestamp
40      8     float64  render_start_timestamp
48      8     float64  encode_end_timestamp
56      var   bytes    data (color + depth or video)
"""

# Format type enum
FORMAT_JPEG = 0
FORMAT_H264 = 1
FORMAT_RAW = 2

RENDER_PAYLOAD_HEADER_SIZE = 56


def pack_render_payload(payload: RenderPayload) -> bytes:
    """
    Serialize RenderPayload to wire format (56 bytes header + data).

    Args:
        payload: RenderPayload object

    Returns:
        bytes: Packed payload (56 bytes header + data)

    Wire format:
        - Fixed 56-byte binary header
        - Variable data (color + depth or video)

    Metadata keys expected:
        - frame_id (int)
        - format_type (int or str): 0=JPEG, 1=H264, 2=Raw (or 'jpeg', 'h264', 'raw')
        - color_len (int): Color data length (JPEG mode)
        - depth_len (int): Depth data length (JPEG mode)
        - width (int): Image width
        - height (int): Image height
        - client_timestamp (float): Client send time (ms)
        - server_timestamp (float): Server receive time (ms)
        - render_start_timestamp (float): Render start time (ms)
        - encode_end_timestamp (float): Encode end time (ms)
    """
    metadata = payload.metadata

    # Extract metadata fields
    frame_id = metadata.get('frame_id', 0)

    # Get format_type (support both int and str for backward compatibility)
    format_type_value = metadata.get('format_type', 0)

    if isinstance(format_type_value, int):
        # Already an int (0, 1, 2)
        format_type = format_type_value
    else:
        # String format, convert to int
        if format_type_value == 'jpeg' or format_type_value == 'jpeg+depth':
            format_type = FORMAT_JPEG
        elif format_type_value == 'h264':
            format_type = FORMAT_H264
        elif format_type_value == 'raw':
            format_type = FORMAT_RAW
        else:
            format_type = FORMAT_JPEG  # Default

    color_len = metadata.get('color_len', 0)
    depth_len = metadata.get('depth_len', 0)
    width = metadata.get('width', 0)
    height = metadata.get('height', 0)

    client_timestamp = metadata.get('client_timestamp', 0.0)
    server_timestamp = metadata.get('server_timestamp', 0.0)
    render_start_timestamp = metadata.get('render_start_timestamp', 0.0)
    encode_end_timestamp = metadata.get('encode_end_timestamp', 0.0)

    # Pack header (56 bytes)
    header = struct.pack("<IB3xIIIIdddd",
        frame_id,                   # uint32
        format_type,                # uint8
        # 3 bytes padding
        color_len,                  # uint32
        depth_len,                  # uint32
        width,                      # uint32
        height,                     # uint32
        client_timestamp,           # float64
        server_timestamp,           # float64
        render_start_timestamp,     # float64
        encode_end_timestamp        # float64
    )

    return header + payload.data


async def read_exact(reader: asyncio.StreamReader, n: int) -> bytes:
    """
    Read exactly n bytes from async stream.

    Args:
        reader: asyncio StreamReader
        n: Number of bytes to read

    Returns:
        bytes: Exactly n bytes

    Raises:
        EOFError: If stream ends before n bytes are read
    """
    data = b""
    while len(data) < n:
        chunk = await reader.read(n - len(data))
        if not chunk:
            raise EOFError(f"Incomplete read: expected {n}, got {len(data)}")
        data += chunk
    return data


async def parse_render_payload(reader: asyncio.StreamReader) -> RenderPayload:
    """
    Parse wire format to RenderPayload from async stream.

    Args:
        reader: asyncio StreamReader

    Returns:
        RenderPayload object

    Raises:
        EOFError: If stream ends unexpectedly
    """
    # Read header (56 bytes)
    header = await read_exact(reader, RENDER_PAYLOAD_HEADER_SIZE)

    # Unpack header
    frame_id, format_type, color_len, depth_len, width, height, \
        client_ts, server_ts, render_start_ts, encode_end_ts = \
        struct.unpack("<IB3xIIIIdddd", header)

    # Calculate data length
    if format_type == FORMAT_JPEG:
        data_len = color_len + depth_len
    else:
        data_len = color_len  # For H264/Raw, color_len is video_len

    # Read data
    data = await read_exact(reader, data_len)

    # Reconstruct metadata dict (keep format_type as int)
    metadata = {
        'frame_id': frame_id,
        'format_type': format_type,  # Keep as int (0=JPEG, 1=H264, 2=Raw)
        'color_len': color_len,
        'depth_len': depth_len,
        'width': width,
        'height': height,
        'client_timestamp': client_ts,
        'server_timestamp': server_ts,
        'render_start_timestamp': render_start_ts,
        'encode_end_timestamp': encode_end_ts
    }

    return RenderPayload(
        data=data,
        metadata=metadata
    )


# =============================================================================
# Utility Functions
# =============================================================================

def validate_camera_frame(camera: CameraFrame) -> Tuple[bool, str]:
    """
    Validate CameraFrame parameters.

    Args:
        camera: CameraFrame to validate

    Returns:
        Tuple[bool, str]: (is_valid, error_message)
    """
    # Check view_matrix
    if camera.view_matrix.shape != (4, 4):
        return False, f"Invalid view_matrix shape: {camera.view_matrix.shape}"

    # Check intrinsics
    if camera.intrinsics.shape != (3, 3):
        return False, f"Invalid intrinsics shape: {camera.intrinsics.shape}"

    # Check dimensions
    if camera.width <= 0 or camera.height <= 0:
        return False, f"Invalid dimensions: {camera.width}x{camera.height}"

    # Check clipping planes
    if camera.near <= 0 or camera.far <= camera.near:
        return False, f"Invalid clipping planes: near={camera.near}, far={camera.far}"

    # Check frame_id
    if camera.frame_id is not None and camera.frame_id < 0:
        return False, f"Invalid frame_id: {camera.frame_id}"

    return True, ""
