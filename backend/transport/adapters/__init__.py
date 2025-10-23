"""
Transport Adapters - Protocol-agnostic communication interfaces.

Available Adapters:
- BaseFrontendAdapter: Abstract interface for Frontend communication (WebSocket, WebRTC, UDP)
- BaseBackendAdapter: Abstract interface for Backend communication (Unix Socket, Shared Memory)
- WebSocketAdapter: WebSocket implementation for Frontend
- UnixSocketAdapter: Unix Socket implementation for Backend (Renderer)
"""

from .base import BaseFrontendAdapter, BaseBackendAdapter

__all__ = [
    'BaseFrontendAdapter',
    'BaseBackendAdapter',
]
