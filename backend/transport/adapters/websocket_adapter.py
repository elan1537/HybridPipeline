"""
WebSocket Adapter for Frontend communication.

Handles WebSocket connections from Frontend clients (browser, app, etc.)

Protocol:
- Camera: Frontend → Transport (260 bytes, Protocol v3)
- Video: Transport → Frontend (44/48 bytes header + data)
- Handshake: 4 bytes (width, height)
- Ping/Pong: 16 bytes
"""

import asyncio
import struct
import time
from typing import Optional

import websockets
from websockets.server import WebSocketServerProtocol

from .base import BaseFrontendAdapter
from renderer.data_types import CameraFrame, RenderPayload
from transport.protocol_converter import (
    parse_frontend_camera,
    parse_renderer_video_header,
    create_frontend_video_header
)


class WebSocketAdapter(BaseFrontendAdapter):
    """
    WebSocket Server adapter for Frontend communication.

    Architecture:
        Frontend (Client) ↔ WebSocket ↔ Transport (Server)

    Responsibilities:
    - Accept WebSocket connections from Frontend
    - Receive camera data (260 bytes, Protocol v3)
    - Send video data (44/48 bytes header + data)
    - Handle handshake and ping/pong
    """

    def __init__(self,
                 host: str = "0.0.0.0",
                 port: int = 8765,
                 width: int = 640,
                 height: int = 480,
                 on_reconnect=None):
        """
        Initialize WebSocket Adapter.

        Args:
            host: WebSocket server host
            port: WebSocket server port
            width: Default viewport width
            height: Default viewport height
            on_reconnect: Callback for reconnection events (async callable)
        """
        self.host = host
        self.port = port
        self.width = width
        self.height = height
        self.on_reconnect = on_reconnect

        # WebSocket connection
        self.ws: Optional[WebSocketServerProtocol] = None
        self._connected = False
        self.connection_count = 0  # Track reconnections

        # Camera queue (for recv_camera())
        self.camera_queue = asyncio.Queue(maxsize=2)

    async def start(self):
        """
        Start WebSocket server and accept connections.

        Note: This is a simplified MVP version that accepts only ONE client.
        For multiple clients, use websockets.serve() with handler registration.

        Raises:
            RuntimeError: If server fails to start
        """
        print(f"[WebSocket] Starting server on {self.host}:{self.port}...")

        try:
            # Start WebSocket server
            async with websockets.serve(
                self._handle_connection,
                self.host,
                self.port,
                max_size=None,  # No size limit
                ping_interval=None,  # Disable auto ping
                ping_timeout=None
            ):
                print(f"[WebSocket] Server started, waiting for connections...")
                # Keep server running
                await asyncio.Future()  # Run forever

        except Exception as e:
            raise RuntimeError(f"Failed to start WebSocket server: {e}")

    async def _handle_connection(self, ws: WebSocketServerProtocol):
        """
        Handle incoming WebSocket connection.

        Args:
            ws: WebSocket connection

        Note:
            In MVP mode, path is logged but not used for routing.
            All paths connect to the same Renderer (encoder type fixed at startup).
            Path is extracted from ws.request.path (websockets 10.0+)
        """
        # Extract path from request (websockets 10.0+ API)
        path = ws.request.path if hasattr(ws, 'request') else '/'
        peer = ws.remote_address

        # Detect reconnection
        self.connection_count += 1
        is_reconnect = self.connection_count > 1

        if is_reconnect:
            print(f"[WebSocket] Client reconnected: {peer}, path={path} (connection #{self.connection_count})")
        else:
            print(f"[WebSocket] Client connected: {peer}, path={path}")

        self.ws = ws
        self._connected = True

        # Notify about reconnection (before handshake)
        if is_reconnect and self.on_reconnect:
            try:
                await self.on_reconnect()
            except Exception as e:
                print(f"[WebSocket] Reconnect callback error: {e}")

        try:
            # Handshake: receive resolution (4 bytes)
            handshake = await ws.recv()
            if len(handshake) == 4:
                width, height = struct.unpack("<HH", handshake)
                self.width = width
                self.height = height
                print(f"[WebSocket] Handshake: resolution {width}x{height}")
            else:
                print(f"[WebSocket] Warning: Invalid handshake size {len(handshake)}")

            # Receive loop
            await self._recv_loop()

        except websockets.exceptions.ConnectionClosed:
            print(f"[WebSocket] Connection closed: {peer}")

        finally:
            self._connected = False
            self.ws = None
            print(f"[WebSocket] Client disconnected: {peer}")

    async def _recv_loop(self):
        """Receive camera data and ping/pong messages."""
        frame_count = 0

        while self._connected and self.ws:
            try:
                raw = await self.ws.recv()

                # Ping message (16 bytes)
                if len(raw) == 16:
                    await self._handle_ping(raw)
                    continue

                # Handshake/resolution change (4 bytes)
                elif len(raw) == 4:
                    width, height = struct.unpack("<HH", raw)
                    if width != self.width or height != self.height:
                        print(f"[WebSocket] Resolution changed: {width}x{height}")
                        self.width = width
                        self.height = height
                    continue

                # Control message (8 or 12 bytes)
                elif len(raw) == 8 or len(raw) == 12:
                    # Control message for Renderer (e.g., encoder change)
                    # Just forward to camera_queue with special marker
                    print(f"[WebSocket] Received control message ({len(raw)} bytes)")

                    # Create a special "control" camera frame with control data
                    # Use a marker in frame_id to indicate control message
                    control_frame = type('ControlFrame', (), {
                        'is_control': True,
                        'control_data': raw
                    })()

                    try:
                        self.camera_queue.put_nowait(control_frame)
                    except asyncio.QueueFull:
                        # Drop oldest frame
                        try:
                            self.camera_queue.get_nowait()
                        except asyncio.QueueEmpty:
                            pass
                        self.camera_queue.put_nowait(control_frame)
                    continue

                # Camera data (260 bytes, Protocol v3)
                elif len(raw) == 260:
                    server_timestamp = time.time() * 1000.0  # ms

                    # Parse Frontend camera (260 bytes → CameraFrame, Protocol v3)
                    camera = parse_frontend_camera(
                        raw,
                        width=self.width,
                        height=self.height,
                        device="cpu"  # Transport doesn't need GPU
                    )

                    # Put to queue (non-blocking, drop oldest if full)
                    try:
                        self.camera_queue.put_nowait(camera)
                    except asyncio.QueueFull:
                        # Drop oldest frame
                        try:
                            self.camera_queue.get_nowait()
                        except asyncio.QueueEmpty:
                            pass
                        self.camera_queue.put_nowait(camera)

                    # Log every 60 frames
                    frame_count += 1
                    if frame_count % 60 == 0:
                        print(f"[WebSocket] Received camera frame {camera.frame_id} "
                              f"(time_index={camera.time_index:.3f})")

                # Unknown message
                else:
                    print(f"[WebSocket] Warning: Unknown message size {len(raw)}")

            except websockets.exceptions.ConnectionClosed:
                break
            except Exception as e:
                print(f"[WebSocket] Receive error: {e}")
                break

    async def _handle_ping(self, raw: bytes):
        """
        Handle ping message and send pong response.

        Args:
            raw: 16 bytes ping message
        """
        message_type = struct.unpack_from("<B", raw, 0)[0]

        if message_type == 255:  # Ping
            client_time = struct.unpack_from("<d", raw, 8)[0]
            server_time = time.time() * 1000.0  # ms

            # Pong response (type=254)
            pong_response = struct.pack("<B7xdd", 254, client_time, server_time)
            await self.ws.send(pong_response)

    async def recv_camera(self) -> Optional[CameraFrame]:
        """
        Receive camera data from Frontend.

        Returns:
            CameraFrame if received, None if connection closed
        """
        if not self._connected:
            return None

        try:
            camera = await self.camera_queue.get()
            return camera
        except asyncio.CancelledError:
            return None

    async def send_video(self, payload: RenderPayload):
        """
        Send video data to Frontend.

        Args:
            payload: Encoded video payload from Renderer

        Raises:
            ConnectionError: If send fails
        """
        if not self._connected or not self.ws:
            raise ConnectionError("WebSocket not connected")

        try:
            # Parse Renderer metadata (from 56 bytes header)
            metadata = payload.metadata

            # Get current timestamp
            transport_send_timestamp = time.time() * 1000.0  # ms

            # Create Frontend video header (44 or 48 bytes)
            frontend_header = create_frontend_video_header(
                metadata,
                transport_send_timestamp
            )

            # Send header + data to Frontend
            await self.ws.send(frontend_header + payload.data)

        except websockets.exceptions.ConnectionClosed:
            self._connected = False
            raise ConnectionError("WebSocket connection closed")
        except Exception as e:
            raise ConnectionError(f"Failed to send video: {e}")

    async def is_connected(self) -> bool:
        """
        Check if Frontend is connected.

        Returns:
            True if connected
        """
        return self._connected and self.ws is not None

    async def close(self):
        """Close WebSocket connection."""
        print("[WebSocket] Closing adapter...")

        if self.ws:
            await self.ws.close()
            self.ws = None

        self._connected = False
        print("[WebSocket] Adapter closed")
