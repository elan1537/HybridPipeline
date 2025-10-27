"""
Renderer Service Entry Point.

Simple CLI for starting the renderer service with different configurations.
"""

import argparse
import asyncio
import sys
import os

from renderer.renderer_service import RendererService
from renderer.scene_renderers.base import BaseSceneRenderer
from renderer.encoders.base import BaseEncoder


def create_scene_renderer(args) -> BaseSceneRenderer:
    """
    Factory function to create scene renderer based on CLI arguments.

    Args:
        args: Parsed command line arguments

    Returns:
        BaseSceneRenderer instance
    """
    scene_type = args.scene_type

    if scene_type == 'static':
        from renderer.scene_renderers.static_gaussian import GsplatRenderer
        renderer = GsplatRenderer()

    elif scene_type == 'streamable':
        from renderer.scene_renderers.streamable_gaussian import StreamableGaussian
        renderer = StreamableGaussian()

    else:
        raise ValueError(f"Unknown scene type: {scene_type}")

    return renderer


def create_renderer_config(args) -> dict:
    """
    Create renderer configuration from CLI arguments.

    Args:
        args: Parsed command line arguments

    Returns:
        Configuration dict
    """
    config = {
        'model_path': args.model_path,
        'device': 'cuda',
        'sh_degree': args.sh_degree,
        'gaussian_scale': args.gaussian_scale,
    }

    # Streamable-specific config
    if args.scene_type == 'streamable':
        config.update({
            'ntc_path': args.ntc_path,
            'ntc_config': args.ntc_config,
            'iterations_s1': args.iterations_s1,
            'iterations_s2': args.iterations_s2,
        })

    return config


def create_encoder(args) -> BaseEncoder:
    """
    Factory function to create encoder based on CLI arguments.

    Args:
        args: Parsed command line arguments

    Returns:
        BaseEncoder instance
    """
    encoder_type = args.encoder_type

    if encoder_type == 'jpeg':
        from renderer.encoders.jpeg import JpegEncoder
        encoder = JpegEncoder(quality=args.jpeg_quality)

    elif encoder_type == 'h264':
        from renderer.encoders.h264 import H264Encoder
        # H264 encoder will auto-initialize from first frame resolution
        encoder = H264Encoder()

    elif encoder_type == 'raw':
        from renderer.encoders.raw import RawEncoder
        encoder = RawEncoder()

    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}")

    return encoder


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Renderer Service - Real-time 3D scene rendering',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Required arguments
    parser.add_argument(
        '--scene-type',
        type=str,
        required=True,
        choices=['static', 'streamable'],
        help='Scene renderer type'
    )
    parser.add_argument(
        '--model-path',
        type=str,
        required=True,
        help='Path to scene model file (PLY)'
    )
    parser.add_argument(
        '--encoder-type',
        type=str,
        required=True,
        choices=['jpeg', 'h264', 'raw'],
        help='Output encoder type'
    )

    # Frame buffer
    parser.add_argument(
        '--buffer-type',
        type=str,
        default='latest',
        choices=['fifo', 'latest'],
        help='Frame buffering strategy'
    )

    # Socket paths
    parser.add_argument(
        '--camera-socket',
        type=str,
        default='/run/ipc/camera.sock',
        help='Unix socket path for camera data'
    )
    parser.add_argument(
        '--video-socket',
        type=str,
        default='/run/ipc/video.sock',
        help='Unix socket path for video data'
    )

    # Rendering resolution
    parser.add_argument(
        '--width',
        type=int,
        default=1280,
        help='Rendering width'
    )
    parser.add_argument(
        '--height',
        type=int,
        default=720,
        help='Rendering height'
    )

    # Debug options
    parser.add_argument(
        '--save-debug-output',
        action='store_true',
        help='Save rendered images for debugging'
    )
    parser.add_argument(
        '--debug-output-dir',
        type=str,
        default='backend/renderer/output',
        help='Directory to save debug output'
    )

    # Static Gaussian options
    parser.add_argument(
        '--sh-degree',
        type=int,
        default=3,
        help='Spherical harmonics degree (static only)'
    )
    parser.add_argument(
        '--gaussian-scale',
        type=float,
        default=1.0,
        help='Gaussian scale factor (static only)'
    )

    # Streamable Gaussian options
    parser.add_argument(
        '--ntc-path',
        type=str,
        default=None,
        help='Path to NTC model (streamable only)'
    )
    parser.add_argument(
        '--ntc-config',
        type=str,
        default=None,
        help='Path to NTC config JSON (streamable only)'
    )
    parser.add_argument(
        '--iterations-s1',
        type=int,
        default=50,
        help='Stage 1 iterations (streamable only)'
    )
    parser.add_argument(
        '--iterations-s2',
        type=int,
        default=50,
        help='Stage 2 iterations (streamable only)'
    )

    # Encoder options
    parser.add_argument(
        '--jpeg-quality',
        type=int,
        default=90,
        help='JPEG quality (0-100)'
    )

    return parser.parse_args()


def validate_args(args):
    """
    Validate command line arguments.

    Args:
        args: Parsed arguments

    Raises:
        ValueError: If arguments are invalid
    """
    # Check model file exists
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model file not found: {args.model_path}")

    # Streamable-specific validation
    if args.scene_type == 'streamable':
        if not args.ntc_path:
            raise ValueError("--ntc-path is required for streamable renderer")
        if not args.ntc_config:
            raise ValueError("--ntc-config is required for streamable renderer")

        if not os.path.exists(args.ntc_path):
            raise FileNotFoundError(f"NTC model not found: {args.ntc_path}")
        if not os.path.exists(args.ntc_config):
            raise FileNotFoundError(f"NTC config not found: {args.ntc_config}")

    # Validate resolution
    if args.width <= 0 or args.height <= 0:
        raise ValueError(f"Invalid resolution: {args.width}x{args.height}")

    # Note: H264 encoder will auto-adjust to even dimensions if needed

    # Validate JPEG quality
    if not 0 <= args.jpeg_quality <= 100:
        raise ValueError(f"JPEG quality must be 0-100, got {args.jpeg_quality}")


async def main():
    """Main entry point."""
    try:
        # Parse arguments
        args = parse_args()

        print("=" * 60)
        print("Renderer Service Starting")
        print("=" * 60)
        print(f"Scene Type: {args.scene_type}")
        print(f"Model Path: {args.model_path}")
        print(f"Encoder Type: {args.encoder_type}")
        print(f"Buffer Type: {args.buffer_type}")
        print(f"Resolution: {args.width}x{args.height}")
        print("=" * 60)

        # Validate
        validate_args(args)

        # Create scene renderer
        print("\n[MAIN] Creating scene renderer...")
        scene_renderer = create_scene_renderer(args)

        # Create renderer config
        renderer_config = create_renderer_config(args)

        # Create encoder
        print("[MAIN] Creating encoder...")
        encoder = create_encoder(args)

        # Create encoder config for dynamic switching
        encoder_config = {
            'jpeg_quality': args.jpeg_quality,
            # H264 encoder defaults (for dynamic switching)
            'h264_bitrate': 20_000_000,  # 20 Mbps
            'h264_fps': 60,
            'h264_preset': 'P3',
        }

        # Create service
        print("[MAIN] Creating renderer service...")

        service = RendererService(
            scene_renderer=scene_renderer,
            encoder=encoder,
            renderer_config=renderer_config,
            buffer_type=args.buffer_type,
            camera_socket=args.camera_socket,
            video_socket=args.video_socket,
            save_debug_output=args.save_debug_output,
            debug_output_dir=args.debug_output_dir,
            encoder_config=encoder_config
        )

        # Initialize and run
        print("\n")
        if await service.initialize():
            await service.run()
        else:
            print("[MAIN] Failed to initialize service")
            sys.exit(1)

    except KeyboardInterrupt:
        print("\n[MAIN] Interrupted by user")

    except FileNotFoundError as e:
        print(f"[MAIN] File not found: {e}")
        sys.exit(1)

    except ValueError as e:
        print(f"[MAIN] Invalid argument: {e}")
        sys.exit(1)

    except Exception as e:
        print(f"[MAIN] Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
