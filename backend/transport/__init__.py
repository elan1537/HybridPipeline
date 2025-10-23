"""
Transport Service - Mediates between Frontend (WebSocket) and Renderer (Unix Socket).

Components:
- TransportService: Main orchestrator
- Protocol Adapters: WebSocket, Unix Socket, etc.
- ProtocolConverter: Protocol translation
"""

from .service import TransportService, run_transport_service
from .adapters.base import BaseFrontendAdapter, BaseBackendAdapter
from .adapters.websocket_adapter import WebSocketAdapter
from .adapters.unix_socket_adapter import UnixSocketAdapter

__all__ = [
    'TransportService',
    'run_transport_service',
    'BaseFrontendAdapter',
    'BaseBackendAdapter',
    'WebSocketAdapter',
    'UnixSocketAdapter',
]
