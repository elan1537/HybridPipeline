"""
Unix Socket Adapter for Backend (Renderer) communication.

Transport Service acts as Unix Socket SERVER, waiting for Renderer to connect.

Protocol:
- Camera: Transport → Renderer (204 bytes, Protocol v3)
- Video: Renderer → Transport (56 bytes header + data)
"""

import asyncio
import os
from typing import Optional

from .base import BaseBackendAdapter
from renderer.data_types import CameraFrame, RenderPayload
from renderer.utils.protocol import pack_camera_frame, parse_render_payload


class UnixSocketAdapter(BaseBackendAdapter):
    """
    Unix Socket Server adapter for Renderer communication.

    Architecture:
        Transport (Server) ← camera.sock ← Renderer (Client)
        Transport (Server) ← video.sock  ← Renderer (Client)

    Responsibilities:
    - Start Unix Socket servers for camera and video
    - Wait for Renderer to connect
    - Send camera data (204 bytes, Protocol v3)
    - Receive video data (56 bytes header + data)
    """

    def __init__(self,
                 camera_socket: str = "/run/ipc/camera.sock",
                 video_socket: str = "/run/ipc/video.sock",
                 save_debug_input: bool = False,
                 debug_input_dir: str = "backend/transport/input"):
        """
        Initialize Unix Socket Adapter.

        Args:
            camera_socket: Unix socket path for camera data
            video_socket: Unix socket path for video data
            save_debug_input: Save received data for debugging
            debug_input_dir: Directory to save debug input
        """
        self.camera_socket = camera_socket
        self.video_socket = video_socket
        self.save_debug_input = save_debug_input
        self.debug_input_dir = debug_input_dir

        # Create debug input directory if needed
        if self.save_debug_input:
            os.makedirs(self.debug_input_dir, exist_ok=True)
            print(f"[DEBUG] Saving transport input to: {self.debug_input_dir}")

        # Socket servers
        self.camera_server: Optional[asyncio.Server] = None
        self.video_server: Optional[asyncio.Server] = None

        # Renderer connections
        self.camera_reader: Optional[asyncio.StreamReader] = None
        self.camera_writer: Optional[asyncio.StreamWriter] = None
        self.video_reader: Optional[asyncio.StreamReader] = None
        self.video_writer: Optional[asyncio.StreamWriter] = None

        # State
        self._connected = False
        self._camera_connected = False
        self._video_connected = False
        self._frame_count = 0  # For debug logging

    async def connect(self, timeout: float = 10.0) -> bool:
        """
        Start Unix Socket servers and wait for Renderer to connect.

        Args:
            timeout: Connection timeout in seconds

        Returns:
            True if both sockets connected successfully
        """
        # Ensure socket directory exists
        socket_dir = os.path.dirname(self.camera_socket)
        if not os.path.exists(socket_dir):
            os.makedirs(socket_dir, exist_ok=True)
            print(f"[UnixSocket] Created socket directory: {socket_dir}")

        # Remove existing socket files
        for sock_path in [self.camera_socket, self.video_socket]:
            if os.path.exists(sock_path):
                os.remove(sock_path)
                print(f"[UnixSocket] Removed existing socket: {sock_path}")

        try:
            # Start camera socket server
            self.camera_server = await asyncio.start_unix_server(
                self._handle_camera_connection,
                self.camera_socket
            )
            print(f"[UnixSocket] Camera socket listening: {self.camera_socket}")

            # Start video socket server
            self.video_server = await asyncio.start_unix_server(
                self._handle_video_connection,
                self.video_socket
            )
            print(f"[UnixSocket] Video socket listening: {self.video_socket}")

            # Wait for Renderer to connect (with timeout)
            print(f"[UnixSocket] Waiting for Renderer to connect (timeout: {timeout}s)...")

            start_time = asyncio.get_event_loop().time()
            while not self._connected:
                if asyncio.get_event_loop().time() - start_time > timeout:
                    print(f"[UnixSocket] Connection timeout after {timeout}s")
                    return False

                await asyncio.sleep(0.1)

                # Check if both sockets are connected
                if self._camera_connected and self._video_connected:
                    self._connected = True
                    print(f"[UnixSocket] Renderer connected successfully")
                    return True

            return True

        except Exception as e:
            print(f"[UnixSocket] Failed to start servers: {e}")
            return False

    async def _handle_camera_connection(self,
                                       reader: asyncio.StreamReader,
                                       writer: asyncio.StreamWriter):
        """
        Handle Renderer connection to camera socket.

        Args:
            reader: Stream reader (not used for camera, Transport sends only)
            writer: Stream writer for sending camera data
        """
        peer = writer.get_extra_info('peername', 'unknown')
        print(f"[UnixSocket] Renderer connected to camera socket from {peer}")

        self.camera_reader = reader
        self.camera_writer = writer
        self._camera_connected = True

        # Keep connection alive (Transport sends camera data via send_camera())
        try:
            while True:
                await asyncio.sleep(1)
        except asyncio.CancelledError:
            pass
        finally:
            self._camera_connected = False
            print(f"[UnixSocket] Camera socket disconnected")

    async def _handle_video_connection(self,
                                      reader: asyncio.StreamReader,
                                      writer: asyncio.StreamWriter):
        """
        Handle Renderer connection to video socket.

        Args:
            reader: Stream reader for receiving video data
            writer: Stream writer (not used for video, Renderer sends only)
        """
        peer = writer.get_extra_info('peername', 'unknown')
        print(f"[UnixSocket] Renderer connected to video socket from {peer}")

        self.video_reader = reader
        self.video_writer = writer
        self._video_connected = True

        # Keep connection alive (Transport receives video data via recv_video())
        try:
            while True:
                await asyncio.sleep(1)
        except asyncio.CancelledError:
            pass
        finally:
            self._video_connected = False
            print(f"[UnixSocket] Video socket disconnected")

    async def send_camera(self, camera):
        """
        Send camera data or control message to Renderer.

        Args:
            camera: CameraFrame object or raw bytes (for control messages)

        Raises:
            ConnectionError: If camera socket is not connected
        """
        if not self._camera_connected or not self.camera_writer:
            raise ConnectionError("Camera socket not connected")

        try:
            # Handle both CameraFrame and raw bytes
            if isinstance(camera, bytes):
                # Raw control message
                data = camera
            else:
                # Pack camera frame to 204 bytes (Protocol v3)
                data = pack_camera_frame(camera)

                # Debug: Log packed size (first 3 frames)
                if not hasattr(self, '_send_count'):
                    self._send_count = 0
                self._send_count += 1

                if self._send_count <= 3:
                    print(f"[UnixSocket] DEBUG: Packed camera frame size: {len(data)} bytes")
                    print(f"  position: {camera.position}")
                    print(f"  target: {camera.target}")
                    print(f"  up: {camera.up}")

            # Send to Renderer
            self.camera_writer.write(data)
            await self.camera_writer.drain()

        except Exception as e:
            self._camera_connected = False
            raise ConnectionError(f"Failed to send camera: {e}")

    async def recv_video(self) -> Optional[RenderPayload]:
        """
        Receive video data from Renderer.

        Returns:
            RenderPayload if received, None if connection closed

        Raises:
            ConnectionError: If video socket is not connected
        """
        if not self._video_connected or not self.video_reader:
            raise ConnectionError("Video socket not connected")

        try:
            # Parse render payload (56 bytes header + data)
            payload = await parse_render_payload(self.video_reader)

            # Save debug input (if enabled)
            if self.save_debug_input:
                self._save_debug_frame(payload)

            return payload

        except EOFError:
            # Connection closed
            self._video_connected = False
            return None

        except Exception as e:
            self._video_connected = False
            raise ConnectionError(f"Failed to receive video: {e}")

    def _save_debug_frame(self, payload: RenderPayload):
        """Save received frame for debugging."""
        metadata = payload.metadata
        frame_id = metadata.get('frame_id', self._frame_count)
        format_type = metadata.get('format_type', 0)

        if format_type == 0:  # JPEG
            color_len = metadata.get('color_len', 0)
            depth_len = metadata.get('depth_len', 0)

            jpeg_data = payload.data[:color_len]
            depth_data = payload.data[color_len:color_len + depth_len]

            # Save JPEG
            jpeg_path = os.path.join(self.debug_input_dir, f"color_{frame_id:06d}.jpg")
            with open(jpeg_path, "wb") as f:
                f.write(jpeg_data)

            # Save depth
            depth_path = os.path.join(self.debug_input_dir, f"depth_{frame_id:06d}.bin")
            with open(depth_path, "wb") as f:
                f.write(depth_data)

            if self._frame_count % 60 == 0 or self._frame_count < 3:
                print(f"[DEBUG] Saved transport input {frame_id}: color={len(jpeg_data)} bytes, depth={len(depth_data)} bytes")

        elif format_type == 1:  # H.264
            video_path = os.path.join(self.debug_input_dir, f"video_{frame_id:06d}.h264")
            with open(video_path, "wb") as f:
                f.write(payload.data)

            if self._frame_count % 60 == 0 or self._frame_count < 3:
                print(f"[DEBUG] Saved transport input {frame_id}: video={len(payload.data)} bytes")

        elif format_type == 2:  # Raw
            raw_path = os.path.join(self.debug_input_dir, f"raw_{frame_id:06d}.pt")
            with open(raw_path, "wb") as f:
                f.write(payload.data)

            if self._frame_count % 60 == 0 or self._frame_count < 3:
                print(f"[DEBUG] Saved transport input {frame_id}: raw={len(payload.data)} bytes")

        self._frame_count += 1

    async def is_connected(self) -> bool:
        """
        Check if Renderer is connected.

        Returns:
            True if both camera and video sockets are connected
        """
        return self._connected and self._camera_connected and self._video_connected

    async def close(self):
        """Close Unix Socket servers and cleanup."""
        print("[UnixSocket] Closing adapter...")

        # Close connections
        if self.camera_writer:
            self.camera_writer.close()
            await self.camera_writer.wait_closed()

        if self.video_writer:
            self.video_writer.close()
            await self.video_writer.wait_closed()

        # Close servers
        if self.camera_server:
            self.camera_server.close()
            await self.camera_server.wait_closed()

        if self.video_server:
            self.video_server.close()
            await self.video_server.wait_closed()

        # Remove socket files
        for sock_path in [self.camera_socket, self.video_socket]:
            if os.path.exists(sock_path):
                try:
                    os.remove(sock_path)
                except Exception:
                    pass

        self._connected = False
        self._camera_connected = False
        self._video_connected = False

        print("[UnixSocket] Adapter closed")
