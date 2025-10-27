import websockets
import asyncio

class UserSession:
    def __init__(self, ws: websockets.WebSocketServerProtocol, width: int, height: int, use_encoder: bool = False):
        self.ws = ws
        self.width = width
        self.height = height
        self.encoder = None
        self.q = asyncio.Queue(maxsize=1)
        self.send_q = asyncio.Queue(maxsize=5)

        # Only create encoder if needed (for /ws/h264 and /ws/jpeg paths)
        if use_encoder:
            try:
                import PyNvVideoCodec as nvvc

                combined_height = height * 2
                encoder_params = {
                    "codec": "h264",
                    "preset": "P3",
                    "repeatspspps": 1,
                    "bitrate": 20000000,
                    # "tuning_info": "lossless",
                    "constqp": 0,
                    "gop": 1,
                    "fps": 60,
                }
                self.encoder = nvvc.CreateEncoder(
                    width=width, height=combined_height, fmt="NV12", usecpuinputbuffer=False, **encoder_params
                )
                print(f"✅ Encoder created for session {ws.remote_address} ({width}x{height})")
            except Exception as e:
                print(f"⚠️ Failed to create encoder: {e}")
                print(f"   Session will run without local encoding")
        else:
            print(f"✅ Session created without encoder for {ws.remote_address} ({width}x{height})")
        