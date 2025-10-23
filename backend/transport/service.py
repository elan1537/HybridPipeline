"""
Transport Service - Main orchestration layer.

Mediates between Frontend (WebSocket) and Renderer (Unix Socket).

Responsibilities:
- Connect to Backend Renderer
- Accept Frontend connections
- Forward camera data: Frontend → Renderer
- Broadcast video data: Renderer → Frontend
"""

import asyncio
from typing import Optional

from transport.adapters.base import BaseFrontendAdapter, BaseBackendAdapter
from transport.adapters.websocket_adapter import WebSocketAdapter
from transport.adapters.unix_socket_adapter import UnixSocketAdapter
from renderer.utils.protocol import pack_control_message, CONTROL_CMD_RESET_ENCODER


class TransportService:
    """
    Main Transport Service orchestrating data flow.

    Architecture:
        Frontend (WebSocket) ↔ TransportService ↔ Renderer (Unix Socket)

    Data Flow:
        1. Camera: Frontend → Queue → Renderer
        2. Video: Renderer → Frontend (broadcast)

    MVP Features:
    - Single Frontend client
    - Single Renderer backend
    - Simple error handling
    - No reconnection logic
    """

    def __init__(self,
                 frontend_adapter: Optional[BaseFrontendAdapter] = None,
                 backend_adapter: Optional[BaseBackendAdapter] = None):
        """
        Initialize Transport Service.

        Args:
            frontend_adapter: Frontend communication adapter (default: WebSocket)
            backend_adapter: Backend communication adapter (default: Unix Socket)
        """
        # Use default adapters if not provided
        if frontend_adapter is None:
            # Create WebSocket adapter with reconnect callback
            frontend_adapter = WebSocketAdapter(on_reconnect=self._on_frontend_reconnect)
        self.frontend_adapter = frontend_adapter
        self.backend_adapter = backend_adapter or UnixSocketAdapter()

        # State
        self.running = False

    async def start(self):
        """
        Start Transport Service.

        Steps:
        1. Connect to Backend Renderer
        2. Start Frontend server
        3. Run forwarding loops
        """
        print("[Transport] Starting Transport Service...")

        # 1. Connect to Backend Renderer
        print("[Transport] Connecting to Renderer...")
        if not await self.backend_adapter.connect(timeout=30.0):
            print("[Transport] Failed to connect to Renderer")
            return

        print("[Transport] Connected to Renderer successfully")

        # 2. Start forwarding loops
        self.running = True

        try:
            # Run Frontend server and forwarding loops concurrently
            await asyncio.gather(
                self.frontend_adapter.start(),  # WebSocket server
                self.camera_forward_loop(),      # Frontend → Renderer
                self.video_broadcast_loop()      # Renderer → Frontend
            )

        except KeyboardInterrupt:
            print("[Transport] Interrupted by user")

        except Exception as e:
            print(f"[Transport] Error: {e}")

        finally:
            await self.shutdown()

    async def camera_forward_loop(self):
        """
        Forward camera data from Frontend to Renderer.

        Flow:
            Frontend → recv_camera() → send_camera() → Renderer

        Supports:
        - Wait for Frontend connection
        - Automatic reconnection
        """
        print("[Transport] Camera forward loop started")

        frame_count = 0

        while self.running:
            try:
                # Wait for Frontend connection
                if not await self.frontend_adapter.is_connected():
                    await asyncio.sleep(0.1)
                    continue

                # Receive camera from Frontend
                camera = await self.frontend_adapter.recv_camera()

                if camera is None:
                    # Frontend disconnected, wait for reconnection
                    print("[Transport] Frontend disconnected, waiting for reconnection...")
                    continue

                # Send camera to Renderer
                await self.backend_adapter.send_camera(camera)

                # Log every 60 frames
                frame_count += 1
                if frame_count % 60 == 0:
                    print(f"[Transport] Forwarded camera frame {camera.frame_id}")

            except ConnectionError as e:
                print(f"[Transport] Camera forward error: {e}")
                await asyncio.sleep(1)  # Wait before retry
                continue

            except asyncio.CancelledError:
                break

            except Exception as e:
                print(f"[Transport] Unexpected error in camera loop: {e}")
                await asyncio.sleep(1)
                continue

        print("[Transport] Camera forward loop stopped")

    async def video_broadcast_loop(self):
        """
        Broadcast video data from Renderer to Frontend.

        Flow:
            Renderer → recv_video() → send_video() → Frontend

        Supports:
        - Skip frames if Frontend not connected
        - Continue on Frontend errors
        """
        print("[Transport] Video broadcast loop started")

        frame_count = 0

        while self.running:
            try:
                # Receive video from Renderer
                payload = await self.backend_adapter.recv_video()

                if payload is None:
                    # Renderer disconnected
                    print("[Transport] Renderer disconnected")
                    break

                # Check if Frontend is connected
                if not await self.frontend_adapter.is_connected():
                    # No Frontend connected, skip frame (drop silently)
                    continue

                # Send video to Frontend
                try:
                    await self.frontend_adapter.send_video(payload)

                    # Log every 60 frames
                    frame_count += 1
                    if frame_count % 60 == 0:
                        frame_id = payload.metadata.get('frame_id', '?')
                        format_type = payload.metadata.get('format_type', '?')
                        print(f"[Transport] Sent video frame {frame_id} ({format_type})")

                except ConnectionError as e:
                    # Frontend send failed, but continue loop
                    # Debug: show metadata on first error
                    if frame_count <= 3:
                        print(f"[Transport] DEBUG: Received metadata = {payload.metadata}")
                    print(f"[Transport] Failed to send to Frontend: {e}")
                    continue

            except ConnectionError as e:
                print(f"[Transport] Video receive error: {e}")
                break

            except asyncio.CancelledError:
                break

            except Exception as e:
                print(f"[Transport] Unexpected error in video loop: {e}")
                await asyncio.sleep(1)
                continue

        print("[Transport] Video broadcast loop stopped")

    async def _on_frontend_reconnect(self):
        """
        Handle Frontend reconnection.

        Sends encoder reset command to Renderer to ensure clean H.264 stream.
        """
        print("[Transport] Frontend reconnected, sending encoder reset to Renderer...")

        try:
            # Pack control message
            control_msg = pack_control_message(CONTROL_CMD_RESET_ENCODER)

            # Send to Renderer via backend adapter
            await self.backend_adapter.send_camera(control_msg)

            print("[Transport] Encoder reset command sent")

        except Exception as e:
            print(f"[Transport] Failed to send encoder reset: {e}")

    async def shutdown(self):
        """Shutdown Transport Service and cleanup resources."""
        print("[Transport] Shutting down...")

        self.running = False

        # Close adapters
        await self.frontend_adapter.close()
        await self.backend_adapter.close()

        print("[Transport] Shutdown complete")


# Convenience function for simple usage
async def run_transport_service(
    websocket_host: str = "0.0.0.0",
    websocket_port: int = 8765,
    camera_socket: str = "/run/ipc/camera.sock",
    video_socket: str = "/run/ipc/video.sock",
    width: int = 640,
    height: int = 480,
    save_debug_input: bool = False,
    debug_input_dir: str = "backend/transport/input"
):
    """
    Run Transport Service with default configuration.

    Args:
        websocket_host: WebSocket server host
        websocket_port: WebSocket server port
        camera_socket: Unix socket path for camera
        video_socket: Unix socket path for video
        width: Default viewport width
        height: Default viewport height
        save_debug_input: Save received data for debugging
        debug_input_dir: Directory to save debug input
    """
    # Create service first (without adapters)
    service = TransportService(
        frontend_adapter=None,  # Let service create it with callbacks
        backend_adapter=None
    )

    # Override backend adapter with custom settings
    service.backend_adapter = UnixSocketAdapter(
        camera_socket=camera_socket,
        video_socket=video_socket,
        save_debug_input=save_debug_input,
        debug_input_dir=debug_input_dir
    )

    # Override frontend adapter settings
    service.frontend_adapter.host = websocket_host
    service.frontend_adapter.port = websocket_port
    service.frontend_adapter.width = width
    service.frontend_adapter.height = height

    await service.start()
