"""
Renderer Service - Main orchestration layer.

Connects Scene Renderer and Encoder with Transport Service via Unix Sockets.
"""

import asyncio
from typing import Optional, Literal
import numpy as np
import os
from pathlib import Path

from renderer.data_types import CameraFrame, RenderPayload

# Import torch for GPU operations
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    print("[ERROR] torch is required for Renderer Service")
    raise

from renderer.scene_renderers.base import BaseSceneRenderer
from renderer.encoders.base import BaseEncoder
from renderer.utils.protocol import (
    parse_camera_frame, pack_render_payload,
    is_control_message, parse_control_message, CONTROL_CMD_RESET_ENCODER
)
from renderer.utils.frame_buffer import FrameBuffer, FIFOBuffer, LatestFrameBuffer


def camera_to_torch(camera: CameraFrame, device: str = "cuda") -> CameraFrame:
    """
    Convert CameraFrame numpy arrays to torch tensors (if needed).

    Args:
        camera: CameraFrame with numpy or torch arrays
        device: Target device for torch tensors

    Returns:
        CameraFrame with torch tensors
    """
    # Check if conversion is needed
    if isinstance(camera.view_matrix, torch.Tensor):
        # Already torch tensors, just move to device if needed
        if camera.view_matrix.device != torch.device(device):
            return CameraFrame(
                view_matrix=camera.view_matrix.to(device),
                intrinsics=camera.intrinsics.to(device),
                width=camera.width,
                height=camera.height,
                near=camera.near,
                far=camera.far,
                time_index=camera.time_index,
                frame_id=camera.frame_id,
                client_timestamp=camera.client_timestamp,
                server_timestamp=camera.server_timestamp
            )
        return camera

    # Convert numpy to torch
    return CameraFrame(
        view_matrix=torch.from_numpy(camera.view_matrix).to(device),
        intrinsics=torch.from_numpy(camera.intrinsics).to(device),
        width=camera.width,
        height=camera.height,
        near=camera.near,
        far=camera.far,
        time_index=camera.time_index,
        frame_id=camera.frame_id,
        client_timestamp=camera.client_timestamp,
        server_timestamp=camera.server_timestamp
    )


class RendererService:
    """
    Main renderer service orchestrating rendering pipeline.

    Responsibilities:
    - Connect to Transport Service (Unix Sockets)
    - Receive camera frames
    - Render scenes using Scene Renderer
    - Encode output using Encoder
    - Send encoded payloads back to Transport

    Architecture:
        Transport → camera.sock → RendererService → video.sock → Transport
    """

    def __init__(self,
                 scene_renderer: BaseSceneRenderer,
                 encoder: BaseEncoder,
                 renderer_config: dict = None,
                 buffer_type: Literal['fifo', 'latest'] = 'latest',
                 camera_socket: str = "/run/ipc/camera.sock",
                 video_socket: str = "/run/ipc/video.sock",
                 save_debug_output: bool = False,
                 debug_output_dir: str = "backend/renderer/output"):
        """
        Initialize Renderer Service.

        Args:
            scene_renderer: Scene renderer instance (3DGS, 4DGS, etc.)
            encoder: Output encoder instance (JPEG, H.264, etc.)
            renderer_config: Configuration dict for scene renderer
            buffer_type: Frame buffering strategy ('fifo' or 'latest')
            camera_socket: Unix socket path for receiving camera data
            video_socket: Unix socket path for sending video data
            save_debug_output: Save rendered images for debugging
            debug_output_dir: Directory to save debug output
        """
        self.scene_renderer = scene_renderer
        self.encoder = encoder
        self.renderer_config = renderer_config or {}
        self.camera_socket = camera_socket
        self.video_socket = video_socket
        self.save_debug_output = save_debug_output
        self.debug_output_dir = debug_output_dir

        # Create debug output directory if needed
        if self.save_debug_output:
            os.makedirs(self.debug_output_dir, exist_ok=True)
            print(f"[DEBUG] Saving renderer output to: {self.debug_output_dir}")

        # Frame buffer
        if buffer_type == 'fifo':
            self.frame_buffer: FrameBuffer[CameraFrame] = FIFOBuffer(maxsize=2)
        elif buffer_type == 'latest':
            self.frame_buffer: FrameBuffer[CameraFrame] = LatestFrameBuffer()
        else:
            raise ValueError(f"Unknown buffer type: {buffer_type}")

        # Socket connections
        self.camera_reader: Optional[asyncio.StreamReader] = None
        self.camera_writer: Optional[asyncio.StreamWriter] = None
        self.video_reader: Optional[asyncio.StreamReader] = None
        self.video_writer: Optional[asyncio.StreamWriter] = None

        # State
        self.running = False

    async def initialize(self) -> bool:
        """
        Initialize the service.

        Returns:
            True if initialization succeeded
        """
        print("[INIT] Initializing Renderer Service...")

        # Initialize scene renderer
        print("[INIT] Initializing scene renderer...")
        if not await self.scene_renderer.on_init(self.renderer_config):
            print("[ERROR] Failed to initialize scene renderer")
            return False

        # Initialize encoder
        print("[INIT] Initializing encoder...")
        await self.encoder.on_init()

        # Connect to Transport Service
        if not await self.connect_to_transport():
            print("[ERROR] Failed to connect to Transport Service")
            return False

        self.running = True
        print("[INIT] Renderer Service initialized successfully")
        return True

    async def connect_to_transport(self, max_retries: int = 5) -> bool:
        """
        Connect to Transport Service via Unix Sockets.

        Args:
            max_retries: Maximum connection retry attempts

        Returns:
            True if connected successfully
        """
        for attempt in range(max_retries):
            try:
                # Connect to camera socket (receive camera data)
                self.camera_reader, self.camera_writer = \
                    await asyncio.open_unix_connection(self.camera_socket)

                # Connect to video socket (send video data)
                self.video_reader, self.video_writer = \
                    await asyncio.open_unix_connection(self.video_socket)

                print(f"[SOCKET] Connected to Transport Service")
                print(f"[SOCKET]   Camera: {self.camera_socket}")
                print(f"[SOCKET]   Video: {self.video_socket}")
                return True

            except FileNotFoundError:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff
                    print(f"[SOCKET] Transport not ready, retrying in {wait_time}s... "
                          f"(attempt {attempt + 1}/{max_retries})")
                    await asyncio.sleep(wait_time)
                else:
                    print(f"[SOCKET] Failed to connect after {max_retries} attempts")
                    return False

            except Exception as e:
                print(f"[SOCKET] Connection error: {e}")
                return False

    async def camera_receive_loop(self):
        """
        Receive camera frames and control messages from Transport Service.

        Handles:
        - 168-byte camera frames
        - 8-byte control messages (e.g., encoder reset)
        """
        print("[CAMERA] Starting camera receive loop...")

        frame_count = 0

        while self.running:
            try:
                # Peek first 8 bytes to check message type
                data = await self.camera_reader.read(8)

                if len(data) == 0:
                    print("[CAMERA] Socket closed by Transport")
                    break

                if len(data) < 8:
                    print(f"[CAMERA] Incomplete message: {len(data)} bytes")
                    continue

                # Check if control message
                if is_control_message(data):
                    # Process control message
                    try:
                        command = parse_control_message(data)
                        await self._handle_control_message(command)
                    except Exception as e:
                        print(f"[CAMERA] Control message error: {e}")
                    continue

                # Read remaining bytes for camera frame (168 - 8 = 160 bytes)
                remaining = await self.camera_reader.read(160)

                if len(remaining) < 160:
                    print(f"[CAMERA] Incomplete camera frame: {len(data) + len(remaining)} bytes")
                    continue

                # Parse camera frame (full 168 bytes)
                full_data = data + remaining
                camera = parse_camera_frame(full_data)

                # Log camera data (every 60 frames for performance)
                frame_count += 1
                if frame_count % 60 == 0 or frame_count <= 3:
                    # Extract camera position from view matrix (inverse translation)
                    view_matrix = camera.view_matrix
                    if isinstance(view_matrix, np.ndarray):
                        # numpy array
                        R = view_matrix[:3, :3]
                        t = view_matrix[:3, 3]
                        camera_pos = -R.T @ t
                        view_str = f"[{view_matrix[0, 0]:.2f}, {view_matrix[0, 1]:.2f}, {view_matrix[0, 2]:.2f}, {view_matrix[0, 3]:.2f}]"
                        intrinsics_str = f"[{camera.intrinsics[0, 0]:.1f}, {camera.intrinsics[0, 2]:.1f}, {camera.intrinsics[1, 1]:.1f}]"
                    else:
                        # torch tensor
                        R = view_matrix[:3, :3]
                        t = view_matrix[:3, 3]
                        camera_pos = -R.T @ t
                        view_str = f"[{view_matrix[0, 0].item():.2f}, {view_matrix[0, 1].item():.2f}, {view_matrix[0, 2].item():.2f}, {view_matrix[0, 3].item():.2f}]"
                        intrinsics_str = f"[{camera.intrinsics[0, 0].item():.1f}, {camera.intrinsics[0, 2].item():.1f}, {camera.intrinsics[1, 1].item():.1f}]"

                    print(f"[CAMERA] Received frame {camera.frame_id}:")
                    print(f"  Position: ({camera_pos[0]:.2f}, {camera_pos[1]:.2f}, {camera_pos[2]:.2f})")
                    print(f"  View[0]: {view_str}")
                    print(f"  Intrinsics (fx, cx, fy): {intrinsics_str}")
                    print(f"  Resolution: {camera.width}x{camera.height}")
                    if camera.client_timestamp is not None:
                        print(f"  Client TS: {camera.client_timestamp:.2f} ms")
                    if camera.server_timestamp is not None:
                        print(f"  Server TS: {camera.server_timestamp:.2f} ms")
                    if camera.time_index is not None:
                        print(f"  Time Index: {camera.time_index:.3f}")

                # Put in buffer
                await self.frame_buffer.put(camera)

            except asyncio.CancelledError:
                print("[CAMERA] Camera receive loop cancelled")
                break

            except Exception as e:
                print(f"[CAMERA] Error receiving camera: {e}")
                break

        print("[CAMERA] Camera receive loop stopped")

    async def render_and_send_loop(self):
        """
        Render frames and send encoded output to Transport Service.

        Pipeline:
        1. Get camera from buffer
        2. Convert to torch tensors (if needed)
        3. Render scene
        4. Encode output
        5. Send to Transport
        """
        print("[RENDER] Starting render and send loop...")

        while self.running:
            try:
                # Get camera from buffer
                camera = await self.frame_buffer.get()

                # Adjust resolution to even numbers for H264 compatibility (NV12 format)
                # This ensures encoder compatibility when frontend sends odd resolutions
                original_width = camera.width
                original_height = camera.height
                adjusted_width = camera.width if camera.width % 2 == 0 else camera.width - 1
                adjusted_height = camera.height if camera.height % 2 == 0 else camera.height - 1

                if adjusted_width != original_width or adjusted_height != original_height:
                    camera = CameraFrame(
                        view_matrix=camera.view_matrix,
                        intrinsics=camera.intrinsics,
                        width=adjusted_width,
                        height=adjusted_height,
                        near=camera.near,
                        far=camera.far,
                        time_index=camera.time_index,
                        frame_id=camera.frame_id,
                        client_timestamp=camera.client_timestamp,
                        server_timestamp=camera.server_timestamp
                    )

                # Convert numpy to torch (if needed)
                camera = camera_to_torch(camera, device="cuda")

                # Render scene
                try:
                    render_output = await self.scene_renderer.render(camera)
                except Exception as e:
                    print(f"[RENDER] Render error at frame {camera.frame_id}: {e}")
                    continue

                # Encode output
                try:
                    payload = await self.encoder.encode(render_output, camera.frame_id)
                except Exception as e:
                    print(f"[ENCODE] Encode error at frame {camera.frame_id}: {e}")
                    continue

                # Save debug output (if enabled)
                if self.save_debug_output:
                    try:
                        self._save_debug_frame(camera.frame_id, payload)
                    except Exception as e:
                        print(f"[DEBUG] Failed to save frame {camera.frame_id}: {e}")

                # Send to Transport
                try:
                    await self.send_payload(payload)
                except Exception as e:
                    print(f"[SEND] Send error at frame {camera.frame_id}: {e}")
                    continue

            except asyncio.CancelledError:
                print("[RENDER] Render and send loop cancelled")
                break

            except Exception as e:
                print(f"[RENDER] Fatal error in render loop: {e}")
                break

        print("[RENDER] Render and send loop stopped")

    async def send_payload(self, payload: RenderPayload):
        """
        Send encoded payload to Transport Service.

        Args:
            payload: Encoded render payload
        """
        # Pack to wire format
        data = pack_render_payload(payload)

        # Send
        self.video_writer.write(data)
        await self.video_writer.drain()

    def _save_debug_frame(self, frame_id: int, payload: RenderPayload):
        """
        Save rendered frame for debugging.

        Args:
            frame_id: Frame identifier
            payload: Encoded render payload
        """
        metadata = payload.metadata
        format_type = metadata.get('format_type', 0)

        if format_type == 0:  # JPEG
            # Extract JPEG and depth
            color_len = metadata.get('color_len', 0)
            depth_len = metadata.get('depth_len', 0)

            jpeg_data = payload.data[:color_len]
            depth_data = payload.data[color_len:color_len + depth_len]

            # Save JPEG
            jpeg_path = os.path.join(self.debug_output_dir, f"color_{frame_id:06d}.jpg")
            with open(jpeg_path, "wb") as f:
                f.write(jpeg_data)

            # Save depth
            depth_path = os.path.join(self.debug_output_dir, f"depth_{frame_id:06d}.bin")
            with open(depth_path, "wb") as f:
                f.write(depth_data)

            if frame_id % 60 == 0 or frame_id < 3:
                print(f"[DEBUG] Saved frame {frame_id}: color={len(jpeg_data)} bytes, depth={len(depth_data)} bytes")

        elif format_type == 1:  # H.264
            video_path = os.path.join(self.debug_output_dir, f"video_{frame_id:06d}.h264")
            with open(video_path, "wb") as f:
                f.write(payload.data)

            if frame_id % 60 == 0 or frame_id < 3:
                print(f"[DEBUG] Saved frame {frame_id}: video={len(payload.data)} bytes")

        elif format_type == 2:  # Raw
            raw_path = os.path.join(self.debug_output_dir, f"raw_{frame_id:06d}.pt")
            with open(raw_path, "wb") as f:
                f.write(payload.data)

            if frame_id % 60 == 0 or frame_id < 3:
                print(f"[DEBUG] Saved frame {frame_id}: raw={len(payload.data)} bytes")

    async def _handle_control_message(self, command: int):
        """
        Handle control message from Transport.

        Args:
            command: Control command code
        """
        if command == CONTROL_CMD_RESET_ENCODER:
            print("[CONTROL] Received encoder reset command")

            try:
                # Clear frame buffer (drop old frames from previous session)
                self.frame_buffer.clear()
                print("[CONTROL] Frame buffer cleared")

                # Shutdown encoder
                await self.encoder.on_shutdown()
                print("[CONTROL] Encoder shut down")

                # Reinitialize encoder
                if await self.encoder.on_init():
                    print("[CONTROL] Encoder reinitialized successfully")
                else:
                    print("[CONTROL] ERROR: Failed to reinitialize encoder")

            except Exception as e:
                print(f"[CONTROL] Encoder reset error: {e}")

        else:
            print(f"[CONTROL] Unknown control command: {command}")

    async def run(self):
        """
        Run the service (main event loop).

        Executes camera receive and render loops concurrently.
        """
        if not self.running:
            print("[ERROR] Service not initialized. Call initialize() first.")
            return

        print("[RUN] Starting Renderer Service...")

        try:
            # Run both loops concurrently
            await asyncio.gather(
                self.camera_receive_loop(),
                self.render_and_send_loop()
            )
        except KeyboardInterrupt:
            print("[RUN] Interrupted by user")
        finally:
            await self.shutdown()

    async def shutdown(self):
        """Clean up resources and close connections."""
        print("[SHUTDOWN] Shutting down Renderer Service...")

        self.running = False

        # Close sockets
        if self.camera_writer:
            self.camera_writer.close()
            await self.camera_writer.wait_closed()

        if self.video_writer:
            self.video_writer.close()
            await self.video_writer.wait_closed()

        # Cleanup renderer
        await self.scene_renderer.on_cleanup()

        # Cleanup encoder
        await self.encoder.on_shutdown()

        print("[SHUTDOWN] Renderer Service stopped")
