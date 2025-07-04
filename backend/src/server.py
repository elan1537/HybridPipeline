# server.py
import torch
import numpy as np
import time
import os
import cv2
import argparse
import imageio
import io
import asyncio, struct, datetime
import websockets
import socket

# ... (argparse 및 기타 설정은 기존과 동일) ...
parser = argparse.ArgumentParser()
parser.add_argument("--near_clip", type=float, default=1.0, help="Near clip")
parser.add_argument("--far_clip", type=float, default=30.0, help="Far clip")
parser.add_argument("--using3DGS", action="store_true", help="Using 3D Gaussian Scene")
parser.add_argument("--using4DGS", action="store_true", help="Using 4D Gaussian Scene")
args = parser.parse_args()

using3DGS = args.using3DGS
using4DGS = args.using4DGS
DEPTH_NEAR_CLIP = args.near_clip
DEPTH_FAR_CLIP = args.far_clip

from gaussian_scene import GaussianScene, Gaussian4DScene
import ws_handler
import PyNvVideoCodec as nvvc

# --- 설정값 ---
PLY_PATH = "./output_scene.ply"
WEBSOCKET_PORT = 8765
SAVE_FRAMES = True
CLIENT_ASSUMED_SCALE_Y = 1.0
# -------------

# --- 전역 변수 ---
scene: Gaussian4DScene = None
width: int = 0
height: int = 0
encoder = None # ✨ 이제 인코더는 하나만 사용합니다.
# -------------

if SAVE_FRAMES:
    os.makedirs("frames", exist_ok=True)

async def recv_loop(ws: websockets.WebSocketServerProtocol, q: asyncio.Queue):
    """ 클라이언트로부터 메시지(카메라 데이터)를 받아 큐에 넣는 루프 """
    global width, height
    # ... (이 함수는 기존과 동일하게 유지) ...
    print(f"Receive loop started for {ws.remote_address}")
    try:
        while True:
            raw_with_ts = await ws.recv()
            if isinstance(raw_with_ts, bytes):
                if len(raw_with_ts) == 4:
                    W, H = struct.unpack("<HH", raw_with_ts)
                    if width != W or height != H:
                        print(f"[+] Peer {ws.remote_address} resized to {W}x{H}")
                        width, height = W, H
                        while not q.empty():
                            try:
                                q.get_nowait()
                                q.task_done()
                            except asyncio.QueueEmpty:
                                break
                elif len(raw_with_ts) >= (32 * 4 + 8):
                    server_recv_timestamp_ms = time.time() * 1000
                    client_send_timestamp_ms = struct.unpack_from("<d", raw_with_ts, len(raw_with_ts) - 8)[0]
                    actual_payload = raw_with_ts[:-8]
                    if q.full():
                        try:
                            q.get_nowait()
                            q.task_done()
                        except asyncio.QueueEmpty:
                            pass
                    await q.put((actual_payload, client_send_timestamp_ms, server_recv_timestamp_ms))
    except websockets.exceptions.ConnectionClosed:
        print(f"Connection closed for {ws.remote_address}")
    finally:
        print(f"Receive loop finished for {ws.remote_address}")

async def render_loop(ws: websockets.WebSocketServerProtocol, q: asyncio.Queue, send_q: asyncio.Queue):
    """
    장면을 렌더링하고, 컬러와 뎁스를 하나의 프레임으로 합쳐 H.264로 인코딩한 후 전송합니다.
    """
    global width, height, scene, encoder
    import kornia

    print(f"Combined H.264 encoding loop started for {ws.remote_address}")

    video_file = None
    if SAVE_FRAMES:
        video_file = open(f"combined_video_{ws.remote_address[0]}_{ws.remote_address[1]}.h264", "wb")

    frame_count = 0
    try:
        while True:
            raw_payload, client_send_timestamp_ms, server_recv_timestamp_ms = await q.get()

            view_mat, intrinsics = ws_handler.parse_payload(raw_payload)

            # --- 렌더링 ---
            render_colors, render_alphas, _ = scene.render(
                viewmats=view_mat, ks=intrinsics, width=width, height=height,
                near=DEPTH_NEAR_CLIP, far=DEPTH_FAR_CLIP
            )

            # --- 프레임 합치기 ---
            # 1. 컬러 데이터 (RGBA, uint8)
            color_rgba_8bit = (render_colors[0, :, :, :4] * 255.0).clamp(0, 255).to(torch.uint8).contiguous()

            # 2. 뎁스 데이터 (Grayscale, uint8)
            depth_raw = render_colors[0, ..., -1].float()
            depth_normalized_01 = ((depth_raw - DEPTH_NEAR_CLIP) / (DEPTH_FAR_CLIP - DEPTH_NEAR_CLIP)).clamp(0.0, 1.0)
            depth_8bit_grayscale = (depth_normalized_01 * 255.0).to(torch.uint8)

            # 뎁스를 RGBA 프레임으로 변환 (A 채널은 255로 채움)
            depth_rgba_8bit = torch.stack([
                depth_8bit_grayscale, depth_8bit_grayscale, depth_8bit_grayscale,
                torch.full_like(depth_8bit_grayscale, 255)
            ], dim=-1).contiguous()

            # 3. 컬러와 뎁스를 세로로 합치기 -> [2*H, W, 4] 텐서 생성
            combined_frame_rgba = torch.cat([color_rgba_8bit, depth_rgba_8bit], dim=0)

            # --- 인코딩 ---
            video_bitstream = bytes(encoder.Encode(combined_frame_rgba))

            if SAVE_FRAMES and video_file:
                video_file.write(video_bitstream)

            # --- 전송 준비 ---
            server_process_end_timestamp_ms = time.time() * 1000
            header = struct.pack("<Iddd", len(video_bitstream),
                                 client_send_timestamp_ms, server_recv_timestamp_ms, server_process_end_timestamp_ms)

            await send_q.put((header, video_bitstream, frame_count))

            frame_count += 1
            q.task_done()

    except asyncio.CancelledError:
        print(f"Combined H.264 encoding loop cancelled.")
    except Exception as e:
        import traceback
        print(f"Error in combined H.264 encoding loop: {e}")
        traceback.print_exc()
    finally:
        if SAVE_FRAMES and video_file:
            video_file.close()

async def send_loop(ws: websockets.WebSocketServerProtocol, send_q: asyncio.Queue):
    """ 전송 전용 루프 """
    try:
        while True:
            header, video_bitstream, frame_count = await send_q.get()
            await ws.send(header + video_bitstream)
            send_q.task_done()
    except asyncio.CancelledError:
        print(f"Send loop cancelled.")
    except Exception as e:
        print(f"Error in send loop: {e}")

async def handler(ws: websockets.WebSocketServerProtocol):
    """ WebSocket 연결 핸들러 """
    global width, height, encoder
    remote_addr = ws.remote_address
    print(f"Connection opened from {remote_addr}")

    try:
        handshake = await ws.recv()
        if isinstance(handshake, bytes) and len(handshake) == 4:
            W, H = struct.unpack("<HH", handshake)
            width, height = W, H
            print(f"[+] Peer {remote_addr} => {W}x{H}")
        else:
            await ws.close()
            return

        combined_height = height * 2
        encoder_params = {
            "codec": "h264", "preset": "P1", "profile": "baseline",
            "bitrate": 12000000, # 비트레이트 증가 (프레임이 2배 커졌으므로)
            "gop": 1, # 모든 프레임을 키프레임으로 설정
        }
        encoder = nvvc.CreateEncoder(
            width=width, height=combined_height, fmt="ARGB", usecpuinputbuffer=False, **encoder_params
        )

        q = asyncio.Queue(maxsize=2)
        send_q = asyncio.Queue(maxsize=5)

        recv_task = asyncio.create_task(recv_loop(ws, q))
        render_task = asyncio.create_task(render_loop(ws, q, send_q))
        send_task = asyncio.create_task(send_loop(ws, send_q))

        done, pending = await asyncio.wait(
            [recv_task, render_task, send_task], return_when=asyncio.FIRST_COMPLETED
        )
        for task in pending:
            task.cancel()

    except Exception as e:
        print(f"Handler error for {remote_addr}: {e}")
    finally:
        print(f"Connection handler finished for {remote_addr}")

async def main():
    global scene
    print("Loading Gaussian Scene...")
    scene = GaussianScene(PLY_PATH)
    scene.upload_to_gpu()
    print("Gaussian Scene loaded and uploaded to GPU.")
    async with websockets.serve(handler, "0.0.0.0", WEBSOCKET_PORT, max_size=None):
        print(f"WebSocket server listening on ws://0.0.0.0:{WEBSOCKET_PORT}")
        await asyncio.Future()

if __name__ == "__main__":
    torch.cuda.init()
    asyncio.run(main())
