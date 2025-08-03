import websockets
import asyncio
import PyNvVideoCodec as nvvc

class UserSession:
    def __init__(self, ws: websockets.WebSocketServerProtocol, width: int, height: int):
        self.ws = ws
        self.width = width
        self.height = height
        self.encoder = None
        self.q = asyncio.Queue(maxsize=1)
        self.send_q = asyncio.Queue(maxsize=5)

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
        