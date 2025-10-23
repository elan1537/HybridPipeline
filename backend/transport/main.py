"""
Transport Service - Entry Point

Usage:
    python -m transport.main [OPTIONS]

Examples:
    # Default configuration
    python -m transport.main

    # Custom WebSocket port
    python -m transport.main --port 9000

    # Custom socket paths
    python -m transport.main \
        --camera-sock /tmp/camera.sock \
        --video-sock /tmp/video.sock
"""

import argparse
import asyncio
import sys

from transport.service import run_transport_service


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Transport Service - Mediates between Frontend and Renderer",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # WebSocket configuration
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="WebSocket server host (default: 0.0.0.0)"
    )

    parser.add_argument(
        "--port",
        type=int,
        default=8765,
        help="WebSocket server port (default: 8765)"
    )

    # Unix Socket configuration
    parser.add_argument(
        "--camera-sock",
        type=str,
        default="/run/ipc/camera.sock",
        help="Unix socket path for camera data (default: /run/ipc/camera.sock)"
    )

    parser.add_argument(
        "--video-sock",
        type=str,
        default="/run/ipc/video.sock",
        help="Unix socket path for video data (default: /run/ipc/video.sock)"
    )

    # Viewport configuration
    parser.add_argument(
        "--width",
        type=int,
        default=640,
        help="Default viewport width (default: 640)"
    )

    parser.add_argument(
        "--height",
        type=int,
        default=480,
        help="Default viewport height (default: 480)"
    )

    # Debug options
    parser.add_argument(
        "--save-debug-input",
        action="store_true",
        help="Save received data for debugging"
    )

    parser.add_argument(
        "--debug-input-dir",
        type=str,
        default="backend/transport/input",
        help="Directory to save debug input (default: backend/transport/input)"
    )

    return parser.parse_args()


def print_banner(args):
    """Print startup banner."""
    print("=" * 60)
    print("Transport Service - HybridPipeline")
    print("=" * 60)
    print()
    print("Configuration:")
    print(f"  WebSocket:     ws://{args.host}:{args.port}")
    print(f"  Camera Socket: {args.camera_sock}")
    print(f"  Video Socket:  {args.video_sock}")
    print(f"  Viewport:      {args.width}x{args.height}")
    print()
    print("=" * 60)
    print()


async def main():
    """Main entry point."""
    # Parse arguments
    args = parse_args()

    # Print banner
    print_banner(args)

    try:
        # Run Transport Service
        await run_transport_service(
            websocket_host=args.host,
            websocket_port=args.port,
            camera_socket=args.camera_sock,
            video_socket=args.video_sock,
            width=args.width,
            height=args.height,
            save_debug_input=args.save_debug_input,
            debug_input_dir=args.debug_input_dir
        )

    except KeyboardInterrupt:
        print("\n[Main] Interrupted by user")
        sys.exit(0)

    except Exception as e:
        print(f"\n[Main] Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
