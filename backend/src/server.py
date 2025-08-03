# server.py
import torch, math
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

# --- 설정 (기존과 동일) ---
parser = argparse.ArgumentParser()
parser.add_argument("--near_clip", type=float, default=1.0, help="Near clip")
parser.add_argument("--far_clip", type=float, default=30.0, help="Far clip")
parser.add_argument("--using3DGS", action="store_true", help="Using 3D Gaussian Scene")
parser.add_argument("--using4DGS", action="store_true", help="Using 4D Gaussian Scene")
parser.add_argument("--ply_path", type=str, default="./output_scene.ply", help="PLY file path")
parser.add_argument("--gaussian_scale", type=float, default=1.0, help="Scale factor for Gaussian Splatting")
args = parser.parse_args()

using3DGS = args.using3DGS
using4DGS = args.using4DGS
DEPTH_NEAR_CLIP = args.near_clip
DEPTH_FAR_CLIP = args.far_clip
GAUSSIAN_SCALE = args.gaussian_scale

from gaussian_scene import GaussianScene, Gaussian4DScene
from nvidia import nvimgcodec as nvc
import cupy as cp
import ws_handler
import PyNvVideoCodec as nvvc

PLY_PATH = args.ply_path
WEBSOCKET_PORT = 8765
SAVE_FRAMES = True

params = nvc.EncodeParams(quality=90)
img_encoder = nvc.Encoder()

# --- 전역 변수 ---
scene: Gaussian4DScene = None
width: int = 0
height: int = 0
encoder = None

if SAVE_FRAMES:
    os.makedirs("frames", exist_ok=True)

def convert_rgb_to_nv12(rgb_frame: torch.Tensor) -> torch.Tensor:
    """
    주어진 RGB 텐서를 NV12 포맷 텐서로 변환합니다.
    """
    # 입력 텐서에서 높이와 너비를 직접 가져옵니다.
    frame_height, frame_width, _ = rgb_frame.shape

    r = rgb_frame[..., 0].float()
    g = rgb_frame[..., 1].float()
    b = rgb_frame[..., 2].float()

    # BT.601 변환 행렬 (SD 비디오 표준, 더 일반적)
    y = 0.299 * r + 0.587 * g + 0.114 * b
    u = -0.168736 * r - 0.331264 * g + 0.5 * b + 128
    v = 0.5 * r - 0.418688 * g - 0.081312 * b + 128

    y = torch.clamp(y, 0, 255).to(torch.uint8)
    u = torch.clamp(u, 0, 255).to(torch.uint8)
    v = torch.clamp(v, 0, 255).to(torch.uint8)

    y_plane = y.contiguous()

    # UV 서브샘플링: 2x2 블록의 평균 계산
    # .view()를 사용할 때 실제 프레임의 높이와 너비를 사용합니다.
    u_reshaped = u.view(frame_height // 2, 2, frame_width // 2, 2).float().mean(dim=(1, 3)).to(torch.uint8)
    v_reshaped = v.view(frame_height // 2, 2, frame_width // 2, 2).float().mean(dim=(1, 3)).to(torch.uint8)

    # UV 인터리빙
    uv_plane = torch.zeros(frame_height // 2, frame_width, dtype=torch.uint8, device=rgb_frame.device)
    uv_plane[:, 0::2] = u_reshaped
    uv_plane[:, 1::2] = v_reshaped

    # NV12 포맷으로 연결
    nv12 = torch.cat([y_plane, uv_plane], dim=0)
    return nv12

async def recv_loop(ws: websockets.WebSocketServerProtocol, q: asyncio.Queue):
    """ 클라이언트로부터 메시지(카메라 데이터)를 받아 큐에 넣는 루프 """
    global width, height
    print(f"Receive loop started for {ws.remote_address}")
    try:
        while True:
            raw_with_ts = await ws.recv()

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


def vis_depth(tensor_data: torch.Tensor, filename: str):
    with open(filename, "wb") as f:
        nv_img = nvc.as_image(tensor_data)
        encoded = img_encoder.encode(nv_img, ".jpeg", params=nvc.EncodeParams(quality=95))
        f.write(encoded)
        
def depth_to_8bit_one_channel(depth: torch.Tensor, alpha: torch.Tensor, near=DEPTH_NEAR_CLIP, far=DEPTH_FAR_CLIP) -> torch.Tensor:
    def log_depth(depth: torch.Tensor, alpha: torch.Tensor, near=DEPTH_NEAR_CLIP, far=DEPTH_FAR_CLIP) -> torch.Tensor:
        depth[alpha < 0.08] = far
        z = depth.cuda().clamp(min=near, max=far)

        log_ratio = torch.log(z / near)
        log_den = math.log(far / near)

        return (log_ratio / log_den) * 0.80823
    
    def nonlinear_depth(depth: torch.Tensor, alpha: torch.Tensor, near=DEPTH_NEAR_CLIP, far=DEPTH_FAR_CLIP) -> torch.Tensor:
        depth[alpha < 0.08] = far
        
        term_A = (far + near) / (far - near)
        term_B = 2 * far * near
        term_B_factor = (far - near) * depth
        
        d = term_A - (term_B / (term_B_factor * depth))
        d = torch.clamp(d, 0, 1)

        return d

    # d = nonlinear_depth(depth, alpha)
    # d_uint8 = (d * 255.0).to(torch.uint8)
    # vis_depth(d_uint8, "nonlinear_depth.jpg")
    
    d = log_depth(depth, alpha)
    d_uint8 = (d * 255.0).to(torch.uint8)
    # vis_depth(d_uint8, "normalized_log_depth.jpg")

    return d_uint8

async def render_loop(ws: websockets.WebSocketServerProtocol, q: asyncio.Queue, send_q: asyncio.Queue):
    """
    장면을 렌더링하고, 컬러와 뎁스를 하나의 프레임으로 합쳐 H.264로 인코딩한 후 전송합니다.
    """
    global width, height, scene, encoder

    from collections import OrderedDict

    print(f"Combined H.264 encoding loop started for {ws.remote_address}")

    video_file = None
    if SAVE_FRAMES:
        video_file = open(f"combined_video.h264", "wb")

    frame_count = 0
    try:
        while True:
            raw_payload, client_send_timestamp_ms, server_recv_timestamp_ms = await q.get()
            start_time = time.perf_counter()
            view_mat, intrinsics = ws_handler.parse_payload(raw_payload)

            # --- 렌더링 ---
            render_colors, render_alphas, _ = scene.render(
                viewmats=view_mat, ks=intrinsics, width=width, height=height,
                near=DEPTH_NEAR_CLIP, far=DEPTH_FAR_CLIP
            )

            color_rgb_8bit = (render_colors[0, :, :, :3] * 255.0).clamp(0, 255).to(torch.uint8).contiguous()

            depth_raw = render_colors[0, ..., -1].float()
            alpha_cuda = render_alphas[0, ..., 0].float()

            alpha_8bit = alpha_cuda.mul(255.0).to(torch.uint8)

            processed_depth = depth_to_8bit_one_channel(depth_raw, alpha_cuda)
            depth_rgb_8bit = processed_depth.unsqueeze(-1).expand(-1, -1, 3).contiguous()

            # with open("depth_rgb_8bit.jpg", "wb") as f:
            #     nv_img = nvc.as_image(d_uint8)
            #     encoded = img_encoder.encode(nv_img, ".jpeg", params=nvc.EncodeParams(quality=95))
            #     f.write(encoded)

            combined_frame_rgb = torch.cat([
                color_rgb_8bit, 
                # depth_rgb_8bit,
                depth_rgb_8bit], dim=0)

            # rgba_color = torch.cat([color_rgb_8bit, alpha_8bit.unsqueeze(-1)], dim=-1)

            combined_frame_nv12 = convert_rgb_to_nv12(combined_frame_rgb)

            # --- 인코딩 ---
            video_bitstream = bytes(encoder.Encode(combined_frame_nv12))

            if SAVE_FRAMES and video_file:
                video_file.write(video_bitstream)

            # --- 전송 준비 ---
            server_process_end_timestamp_ms = time.time() * 1000
            header = struct.pack("<Iddd", len(video_bitstream),
                                 client_send_timestamp_ms, server_recv_timestamp_ms, server_process_end_timestamp_ms)

            await send_q.put((header, video_bitstream, frame_count))
            print(f"{frame_count}: {((server_process_end_timestamp_ms - client_send_timestamp_ms) / 1000):.2f}ms")

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
            bitstream = encoder.EndEncode()
            video_file.write(bitstream)
            video_file.close()


async def render_loop_jpeg(ws: websockets.WebSocketServerProtocol, q: asyncio.Queue, send_q: asyncio.Queue):
    """
    장면을 렌더링하고, 컬러와 뎁스를 하나의 프레임으로 합쳐 H.264로 인코딩한 후 전송합니다.
    """
    global width, height, scene, encoder

    from collections import OrderedDict

    print(f"Combined JPEG loop started for {ws.remote_address}")

    frame_count = 0
    try:
        while True:
            raw_payload, client_send_timestamp_ms, server_recv_timestamp_ms = await q.get()
            start_time = time.perf_counter() * 1000
            view_mat, intrinsics = ws_handler.parse_payload(raw_payload)

            render_colors, render_alphas, _ = scene.render(
                viewmats=view_mat,
                ks=intrinsics,
                width=width, 
                height=height,
                near=DEPTH_NEAR_CLIP,
                far=DEPTH_FAR_CLIP
            )

            rgb_cuda = render_colors[0, ..., :3].mul(255).clamp(0, 255).to(torch.uint8).contiguous()
            depth_cuda_raw = render_colors[0, ..., -1].float()  # 정밀도를 위해 float32 사용
            alpha_cuda = render_alphas[0, ..., 0].float()

            img_nv = nvc.as_image(rgb_cuda)
            jpeg_bytes = encoder.encode(img_nv, "jpeg", params=nvc.EncodeParams(quality=100))

            ALPHA_CUTOFF = 0.1  # 알파 값 임계치
            depth_cuda_raw[alpha_cuda < ALPHA_CUTOFF] = torch.nan  # 알파 낮은 부분 NaN 처리

            term_A = (DEPTH_FAR_CLIP + DEPTH_NEAR_CLIP) / (DEPTH_FAR_CLIP - DEPTH_NEAR_CLIP)
            term_B_num = 2 * DEPTH_FAR_CLIP * DEPTH_NEAR_CLIP
            term_B_den_factor = (DEPTH_FAR_CLIP - DEPTH_NEAR_CLIP)

            denominator_term_B = term_B_den_factor * depth_cuda_raw
                
            calculated_ndc_webgl = term_A - (term_B_num / denominator_term_B)

            final_ndc_webgl = torch.clamp(calculated_ndc_webgl, -1.0, 1.0)
            final_ndc_to_send = final_ndc_webgl

            depth_ndc_f16 = final_ndc_to_send.to(torch.float16) 
            depth_numpy = depth_ndc_f16.contiguous().cpu().numpy()
            depth_bytes = depth_numpy.tobytes()

            server_process_end_timestamp_ms = time.perf_counter() * 1000 - start_time

            header = struct.pack("<IIddd", len(jpeg_bytes), len(depth_bytes), 
                                client_send_timestamp_ms, server_recv_timestamp_ms, server_process_end_timestamp_ms)
        
            await send_q.put((header, jpeg_bytes + depth_bytes, frame_count))
            frame_count += 1

            q.task_done()


    except asyncio.CancelledError:
        print(f"Combined JPEG loop cancelled.")
    except Exception as e:
        print(f"Error in combined JPEG loop: {e}")
    finally:
        print(f"Combined JPEG loop finished for {ws.remote_address}")


async def send_loop(ws: websockets.WebSocketServerProtocol, send_q: asyncio.Queue):
    """ 전송 전용 루프 """

    target_fps = 60
    frame_interval = 1 / target_fps
    last_send = time.perf_counter()

    try:
        while True:
            header, video_bitstream, frame_count = await send_q.get()

            now = time.perf_counter()
            elapsed = now - last_send

            if elapsed < frame_interval:
                await asyncio.sleep(frame_interval - elapsed)
            
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

        # ✨ 인코더 생성 시 fmt를 'NV12'로, 높이는 합쳐진 높이로 설정
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
        encoder = nvvc.CreateEncoder(
            width=width, height=combined_height, fmt="NV12", usecpuinputbuffer=False, **encoder_params
        )
        print("✅ Combined frame encoder created with NV12 input format.")

        q = asyncio.Queue(maxsize=1)
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
    
    # Gaussian 스케일 적용
    if GAUSSIAN_SCALE != 1.0:
        print(f"Applying Gaussian scale factor: {GAUSSIAN_SCALE}")
        scene.scale(GAUSSIAN_SCALE)
    
    scene.upload_to_gpu()
    print("Gaussian Scene loaded and uploaded to GPU.")
    async with websockets.serve(handler, "0.0.0.0", WEBSOCKET_PORT, max_size=None, ping_interval=None, ping_timeout=None):
        print(f"WebSocket server listening on ws://0.0.0.0:{WEBSOCKET_PORT}")
        await asyncio.Future()

if __name__ == "__main__":
    torch.cuda.init()
    asyncio.run(main())