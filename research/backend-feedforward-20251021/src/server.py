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

from session import UserSession

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

# --- 전역 변수 ---
scene: Gaussian4DScene = None

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


async def recv_loop(session: UserSession):
    """ 클라이언트로부터 메시지(카메라 데이터)를 받아 큐에 넣는 루프 """
    ws = session.ws
    q = session.q
    width = session.width
    height = session.height

    print(f"Receive loop started for {ws.remote_address}")
    try:
        while True:
            raw_with_ts = await ws.recv()

            # Ping 메시지 처리 (type(1) + padding(7) + clientTime(8) = 16 bytes)
            if len(raw_with_ts) == 16:
                # 첫 번째 바이트가 메시지 타입, 8번째 바이트부터 clientTime
                message_type = struct.unpack_from("<B", raw_with_ts, 0)[0]
                if message_type == 255:  # ping message
                    client_time = struct.unpack_from("<d", raw_with_ts, 8)[0]
                    server_time = time.time_ns() / 1_000_000  # 나노초 정밀도
                    # Pong 응답 (type(1) + padding(7) + clientTime(8) + serverTime(8) = 24 bytes)
                    pong_response = struct.pack("<B7xdd", 254, client_time, server_time)
                    await ws.send(pong_response)
                    continue

            # 기존 핸드셰이크 처리
            if len(raw_with_ts) == 4:
                W, H = struct.unpack("<HH", raw_with_ts)
                if width != W or height != H:
                    print(f"[+] Peer {ws.remote_address} resized to {W}x{H}")
                    width, height = W, H
                    session.width = width
                    session.height = height
                    while not q.empty():
                        try:
                            q.get_nowait()
                            q.task_done()
                        except asyncio.QueueEmpty:
                            break
            # 확장된 카메라 데이터 (128 + 4 + 4(패딩) + 8 + 4 = 148 bytes, 160으로 패딩)
            elif len(raw_with_ts) == 160:
                server_recv_timestamp_ms = time.time_ns() / 1_000_000  # 나노초 정밀도

                # frameId 추출 (128번째 바이트부터 4바이트, uint32)
                frame_id = struct.unpack_from("<I", raw_with_ts, 128)[0]
                # client timestamp 추출 (136번째 바이트부터 8바이트, float64, 8바이트 정렬)
                client_send_timestamp_ms = struct.unpack_from("<d", raw_with_ts, 136)[0]
                # time_index 추출 (144번째 바이트부터 4바이트, float32)
                time_index = struct.unpack_from("<f", raw_with_ts, 144)[0]
                # 실제 카메라 데이터 (처음 128바이트)
                actual_payload = raw_with_ts[:128]
                
                if q.full():
                    try:
                        q.get_nowait()
                        q.task_done()
                    except asyncio.QueueEmpty:
                        pass

                await q.put((actual_payload, client_send_timestamp_ms, server_recv_timestamp_ms, time_index, frame_id))
            # 기존 프로토콜 호환성 (32 * 4 + 8 = 136 bytes)
            elif len(raw_with_ts) == 136:
                server_recv_timestamp_ms = time.time_ns() / 1_000_000
                client_send_timestamp_ms = struct.unpack_from("<d", raw_with_ts, len(raw_with_ts) - 8)[0]
                actual_payload = raw_with_ts[:-8]
                if q.full():
                    try:
                        q.get_nowait()
                        q.task_done()
                    except asyncio.QueueEmpty:
                        pass
                await q.put((actual_payload, client_send_timestamp_ms, server_recv_timestamp_ms, 0, 0))  # frameId = 0
                
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
        depth[alpha < 0.5] = far
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

    # print(alpha.min(), alpha.max())
    
    d = log_depth(depth, alpha)
    d_uint8 = (d * 255.0).to(torch.uint8)
    # vis_depth(d_uint8, "normalized_log_depth.jpg")

    return d_uint8


async def render_loop(session: UserSession):
    """
    장면을 렌더링하고, 컬러와 뎁스를 하나의 프레임으로 합쳐 H.264로 인코딩한 후 전송합니다.
    """
    from collections import OrderedDict

    ws = session.ws
    q = session.q
    send_q = session.send_q
    encoder = session.encoder
    width = session.width
    height = session.height

    print(f"Combined H.264 encoding loop started for {ws.remote_address}")

    video_file = None
    if SAVE_FRAMES:
        video_file = open(f"combined_video.h264", "wb")

    frame_count = 0
    try:
        while True:
            queue_data = await q.get()
            
            # 확장된 프로토콜 지원
            if len(queue_data) == 5:
                raw_payload, client_send_timestamp_ms, server_recv_timestamp_ms, _, frame_id = queue_data
            else:
                # 기존 프로토콜 호환성
                raw_payload, client_send_timestamp_ms, server_recv_timestamp_ms = queue_data
                frame_id = frame_count
            
            server_process_start_ms = time.time() * 1000
            view_mat, intrinsics = ws_handler.parse_payload(raw_payload)

            # --- 렌더링 ---
            render_colors, render_alphas, _ = scene.render(
                viewmats=view_mat, ks=intrinsics, width=width, height=height,
                near=DEPTH_NEAR_CLIP, far=DEPTH_FAR_CLIP
            )

            color_rgb_8bit = (render_colors[0, :, :, :3] * 255.0).clamp(0, 255).to(torch.uint8).contiguous()

            depth_raw = render_colors[0, ..., -1].float()
            alpha_cuda = render_alphas[0, ..., 0].float()

            # alpha_8bit = alpha_cuda.mul(255.0).to(torch.uint8)

            processed_depth = depth_to_8bit_one_channel(depth_raw, alpha_cuda)
            depth_rgb_8bit = processed_depth.unsqueeze(-1).expand(-1, -1, 3).contiguous()

            combined_frame_rgb = torch.cat([
                color_rgb_8bit, 
                depth_rgb_8bit], dim=0)

            combined_frame_nv12 = convert_rgb_to_nv12(combined_frame_rgb)

            # --- 인코딩 ---
            video_bitstream = bytes(encoder.Encode(combined_frame_nv12))

            if SAVE_FRAMES and video_file:
                video_file.write(video_bitstream)

            # --- 전송 준비 (확장된 헤더) ---
            server_process_end_timestamp_ms = time.time_ns() / 1_000_000
            
            # 새로운 헤더 형식: videoLen(4) + frameId(4) + clientSendTime(8) + serverReceiveTime(8) + serverProcessEndTime(8) + serverSendTime(8)
            # 실제 전송 시점은 send_loop에서 측정하므로 임시값 사용
            header = struct.pack("<IIdddd", 
                               len(video_bitstream),
                               frame_id,
                               client_send_timestamp_ms, 
                               server_recv_timestamp_ms, 
                               server_process_end_timestamp_ms,
                               0.0)  # 실제 전송 시점은 send_loop에서 설정

            await send_q.put((header, video_bitstream, frame_count, server_process_end_timestamp_ms))
            
            total_latency = server_process_end_timestamp_ms - client_send_timestamp_ms
            print(f"Frame {frame_id}: {total_latency:.1f}ms total latency")

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


async def render_loop_jpeg(session: UserSession):
    """
    장면을 렌더링하고, 컬러와 뎁스를 JPEG + Float16로 인코딩한 후 전송합니다.
    """
    ws = session.ws
    q = session.q
    send_q = session.send_q
    encoder = session.encoder
    width = session.width
    height = session.height

    print(f"Combined JPEG loop started for {ws.remote_address}")

    params = nvc.EncodeParams(quality=90)
    img_encoder = nvc.Encoder()

    frame_count = 0
    try:
        while True:
            queue_data = await q.get()
            
            # 확장된 프로토콜 지원
            if len(queue_data) == 5:
                raw_payload, client_send_timestamp_ms, server_recv_timestamp_ms, _, frame_id = queue_data
            else:
                # 기존 프로토콜 호환성
                raw_payload, client_send_timestamp_ms, server_recv_timestamp_ms = queue_data
                frame_id = frame_count
                
            server_process_start_ms = time.time() * 1000
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
            jpeg_bytes = img_encoder.encode(img_nv, "jpeg", params=params)

            # Use same [0,1] normalization as H264 path for consistency
            processed_depth_01 = depth_to_8bit_one_channel(depth_cuda_raw, alpha_cuda)
            # Convert back to [0,1] float range instead of 8-bit
            depth_01_normalized = processed_depth_01.float() / 255.0

            depth_01_f16 = depth_01_normalized.to(torch.float16) 
            depth_numpy = depth_01_f16.contiguous().cpu().numpy()
            depth_bytes = depth_numpy.tobytes()

            server_process_end_timestamp_ms = time.time_ns() / 1_000_000

            # 새로운 헤더 형식: jpegLen(4) + depthLen(4) + frameId(4) + clientSendTime(8) + serverReceiveTime(8) + serverProcessEndTime(8) + serverSendTime(8)
            # 실제 전송 시점은 send_loop에서 측정하므로 임시값 사용
            header = struct.pack("<IIIdddd", 
                               len(jpeg_bytes), 
                               len(depth_bytes),
                               frame_id,
                               client_send_timestamp_ms, 
                               server_recv_timestamp_ms, 
                               server_process_end_timestamp_ms,
                               0.0)  # 실제 전송 시점은 send_loop에서 설정
        
            await send_q.put((header, jpeg_bytes + depth_bytes, frame_count, server_process_end_timestamp_ms))
            
            total_latency = server_process_end_timestamp_ms - client_send_timestamp_ms
            print(f"JPEG Frame {frame_id}: {total_latency:.1f}ms total latency")
            
            frame_count += 1
            q.task_done()

    except asyncio.CancelledError:
        print(f"Combined JPEG loop cancelled.")
    except Exception as e:
        print(f"Error in combined JPEG loop: {e}")
    finally:
        print(f"Combined JPEG loop finished for {ws.remote_address}")


async def send_loop(session: UserSession):
    """ 전송 전용 루프 """

    target_fps = 60
    frame_interval = 1 / target_fps
    last_send = time.perf_counter()

    ws = session.ws
    send_q = session.send_q

    try:
        while True:
            queue_item = await send_q.get()
            
            # 확장된 큐 아이템 형식 지원
            if len(queue_item) >= 4:
                header, video_bitstream, frame_count, server_process_end_ms = queue_item[:4]
                
                # 실제 전송 시점 측정
                server_send_timestamp_ms = time.time_ns() / 1_000_000
                
                # 헤더의 마지막 8바이트(serverSendTime)를 실제 전송 시점으로 업데이트
                header_bytes = bytearray(header)
                send_time_offset = len(header_bytes) - 8
                struct.pack_into("<d", header_bytes, send_time_offset, server_send_timestamp_ms)
                header = bytes(header_bytes)
                
            else:
                # 기존 형식 호환성
                header, video_bitstream, frame_count = queue_item
            
            now = time.perf_counter()
            elapsed = now - last_send

            if elapsed < frame_interval:
                await asyncio.sleep(frame_interval - elapsed)
            
            await ws.send(header + video_bitstream)
            send_q.task_done()
            last_send = time.perf_counter()

    except asyncio.CancelledError:
        print(f"Send loop cancelled.")
    except Exception as e:
        print(f"Error in send loop: {e}")


async def render_feedforward_loop(session: UserSession, fifo_path="/run/ipc/video_stream.fifo"):
    """
    Named Pipe에서 H.264 프레임을 읽어 frontend로 전송하는 루프 (Feed-Forward 전용)
    """
    ws = session.ws
    send_q = session.send_q

    print(f"FIFO read loop started for {ws.remote_address}")
    print(f"Waiting for FIFO: {fifo_path}")

    loop = asyncio.get_event_loop()
    fifo = None
    frame_count = 0

    try:
        # FIFO가 생성될 때까지 대기
        timeout = 30
        start_time = time.time()
        while not os.path.exists(fifo_path):
            if time.time() - start_time > timeout:
                print(f"Timeout: FIFO not found after {timeout}s")
                return
            await asyncio.sleep(0.1)

        # FIFO 연결 (blocking open을 executor에서 실행)
        fifo = await loop.run_in_executor(None, open, fifo_path, "rb")
        print(f"Connected to FIFO: {fifo_path}")

        while True:
            # Header 읽기 (16 bytes): frame_id(8) + time_index(4) + frame_size(4)
            header_bytes = await loop.run_in_executor(None, fifo.read, 16)

            if len(header_bytes) < 16:
                print("FIFO closed or incomplete header")
                break

            frame_id, time_index, frame_size = struct.unpack("<Qfi", header_bytes)

            # 유효성 검사
            if frame_size <= 0 or frame_size > 10 * 1024 * 1024:
                print(f"Invalid frame size: {frame_size}, skipping")
                continue

            # Video bitstream 읽기
            video_bitstream = await loop.run_in_executor(None, fifo.read, frame_size)

            if len(video_bitstream) < frame_size:
                print(f"Incomplete frame: expected {frame_size}, got {len(video_bitstream)}")
                break

            # Transport에서 타임스탬프 측정
            transport_recv_time = time.time_ns() / 1_000_000

            # Frontend 전송용 헤더 구성
            # 형식: videoLen(4) + frameId(4) + clientSendTime(8) + serverReceiveTime(8) +
            #       serverProcessEndTime(8) + serverSendTime(8)
            header = struct.pack("<IIdddd",
                len(video_bitstream),     # videoLen
                int(frame_id),            # frameId
                0.0,                      # clientSendTime (feed-forward에는 없음)
                0.0,                      # serverReceiveTime (feed-forward에는 없음)
                transport_recv_time,      # serverProcessEndTime
                0.0)                      # serverSendTime (send_loop에서 설정)

            await send_q.put((header, video_bitstream, frame_count, transport_recv_time))

            # 진행 상황 출력 (60프레임마다)
            if frame_count % 60 == 0:
                print(f"[FIFO Feedforward] Frame {frame_id}: {frame_size} bytes, time_index={time_index:.4f}")

            frame_count += 1

    except FileNotFoundError:
        print(f"FIFO not found: {fifo_path}")
    except asyncio.CancelledError:
        print(f"FIFO read loop cancelled.")
    except Exception as e:
        import traceback
        print(f"Error in FIFO read loop: {e}")
        traceback.print_exc()
    finally:
        if fifo:
            fifo.close()
        print(f"FIFO read loop finished for {ws.remote_address}")


def load_static_gs():
    global scene
    print("Loading Gaussian Scene...")
    scene = GaussianScene(PLY_PATH)
    
    # Gaussian 스케일 적용
    if GAUSSIAN_SCALE != 1.0:
        print(f"Applying Gaussian scale factor: {GAUSSIAN_SCALE}")
        scene.scale(GAUSSIAN_SCALE)
    
    scene.upload_to_gpu()
    print("Gaussian Scene loaded and uploaded to GPU.")


def load_dynamic_gs():
    global scene_4d
    print("Loading Gaussian Scene")
    scene_4d = Gaussian4DScene()
    

async def handler(ws: websockets.WebSocketServerProtocol):
    """ WebSocket 연결 핸들러 """
    remote_addr = ws.remote_address
    
    print(f"Connection opened from {remote_addr}")
    session = None

    try:
        handshake = await ws.recv()
        if isinstance(handshake, bytes) and len(handshake) == 4:
            width, height = struct.unpack("<HH", handshake)
            session = UserSession(ws, width, height)
            print(f"[+] Session created for {remote_addr} => {width}x{height}")
        else:
            await ws.close()
            return

        recv_task = asyncio.create_task(recv_loop(session))
        send_task = asyncio.create_task(send_loop(session))


        print(ws.request.path)

        match ws.request.path:
            case "/ws/h264":
                render_task = asyncio.create_task(render_loop(session))
                print(f"[+] H.264 loop started for {remote_addr}")
            case "/ws/jpeg":
                render_task = asyncio.create_task(render_loop_jpeg(session))
                print(f"[+] JPEG loop started for {remote_addr}")
            case "/ws/feedforward":
                render_task = asyncio.create_task(render_feedforward_loop(session, fifo_path="/run/ipc/video_stream.fifo"))
                print(f"[+] Feedforward loop started for {remote_addr}")
            case _:
                await ws.close()

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
    async with websockets.serve(handler, "0.0.0.0", WEBSOCKET_PORT, max_size=None, ping_interval=None, ping_timeout=None):
        print(f"WebSocket server listening on ws://0.0.0.0:{WEBSOCKET_PORT}")
        await asyncio.Future()


if __name__ == "__main__":
    torch.cuda.init()
    asyncio.run(main())