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
import traceback
import kornia
import torch.nn.functional as Fn

parser = argparse.ArgumentParser()
parser.add_argument("--near_clip", type=float, default=1.0, help="Near clip")
parser.add_argument("--far_clip", type=float, default=30.0, help="Far clip")
parser.add_argument("--using3DGS", action="store_true", help="Using 3D Gaussian Scene")
parser.add_argument("--using4DGS", action="store_true", help="Using 4D Gaussian Scene")
parser.add_argument("--usingH264", action="store_true", help="Using H.264")
parser.add_argument("--raw_rgb", action="store_true", help="Send raw RGB data instead of JPEG/H.264")
parser.add_argument("--webgpu_depth", action="store_true", help="Using WebGPU Depth")
args = parser.parse_args()

using3DGS = args.using3DGS
using4DGS = args.using4DGS
usingH264 = args.usingH264
usingRawRGB = args.raw_rgb # True로 설정 시 RAW RGB 데이터 전송
OUTPUT_WEBGPU_DEPTH_FOR_3DGS = args.webgpu_depth # True로 설정 시 render_loop (3DGS)에서 [0,1] NDC 뎁스 출력
DEPTH_NEAR_CLIP = args.near_clip # 뎁스 최소 클리핑 값
DEPTH_FAR_CLIP = args.far_clip # 뎁스 최대 클리핑 값 (클라이언트와 일치 확인)

print(f"Using 3DGS: {using3DGS}")
print(f"Using 4DGS: {using4DGS}")
print(f"Using H.264: {usingH264}")
print(f"Using Raw RGB: {usingRawRGB}")
print(f"WebGPU Depth: {OUTPUT_WEBGPU_DEPTH_FOR_3DGS}")
print(f"Near Clip: {DEPTH_NEAR_CLIP}")
print(f"Far Clip: {DEPTH_FAR_CLIP}")

# nvimgcodec 임포트 확인
try:
    from nvidia import nvimgcodec as nvc
    encoder = nvc.Encoder()
    # decoder = nvc.Decoder() # 디코더는 현재 사용 안 함
    NVC_AVAILABLE = True
except ImportError:
    print("Warning: nvidia-nvimgcodec not found. JPEG encoding will be slower.")
    NVC_AVAILABLE = False
    # 대체 인코딩/디코딩 로직 필요 시 여기에 추가 (예: Pillow)
    try:
        from PIL import Image
        import io
        PILLOW_AVAILABLE = True
    except ImportError:
        print("Error: Pillow is not installed. Cannot encode/decode images.")
        PILLOW_AVAILABLE = False

def rgb_to_yuv420(rgb_tensor):
    """
    RGB 텐서를 YUV420 포맷으로 변환
    Args:
        rgb_tensor: [H, W, 3] 형태의 RGB 텐서 (uint8)
    Returns:
        yuv420_tensor: [H*3/2, W] 형태의 YUV420 텐서 (uint8)
    """
    H, W, _ = rgb_tensor.shape
    
    # RGB를 YUV로 변환 (벡터화된 연산)
    r = rgb_tensor[..., 0].float()
    g = rgb_tensor[..., 1].float()
    b = rgb_tensor[..., 2].float()
    
    # BT.709 변환 행렬
    y = 0.299 * r + 0.587 * g + 0.114 * b
    u = -0.169 * r - 0.331 * g + 0.500 * b + 128
    v = 0.500 * r - 0.419 * g - 0.081 * b + 128
    
    # 클램핑
    y = torch.clamp(y, 0, 255).to(torch.uint8)
    u = torch.clamp(u, 0, 255).to(torch.uint8)
    v = torch.clamp(v, 0, 255).to(torch.uint8)
    
    # Y 평면
    y_plane = y.contiguous()
    
    # UV 서브샘플링 (벡터화된 방식)
    # 2x2 블록의 평균을 계산 (float로 변환 후 계산)
    u_reshaped = u.view(H//2, 2, W//2, 2).float().mean(dim=(1, 3)).to(torch.uint8)
    v_reshaped = v.view(H//2, 2, W//2, 2).float().mean(dim=(1, 3)).to(torch.uint8)
    
    # YUV420 포맷: Y 평면 + U 평면 + V 평면
    yuv420 = torch.cat([y_plane, u_reshaped, v_reshaped], dim=0)
    
    return yuv420


from gaussian_scene import GaussianScene, Gaussian4DScene

import importlib
import ws_handler # 가정: 수동 행렬 재구성 버전 사용

# --- 설정값 ---
PLY_PATH = "./output_scene.ply" # PLY 파일 경로
# PLY_PATH = "./livinglab-no-ur5.ply"
# PLY_PATH = "./livinglab-3.ply"
WEBSOCKET_PORT = 8765
SAVE_FRAMES = True # 프레임 저장 여부
JPEG_QUALITY = 100   # JPEG 압축 품질
CLIENT_ASSUMED_SCALE_Y = 1.0 
# -------------

# --- 전역 변수 ---
# scene: GaussianScene = None
scene: Gaussian4DScene = None
width: int = 0
height: int = 0

# -------------


# 프레임 저장 디렉토리 생성
if SAVE_FRAMES:
    os.makedirs("frames", exist_ok=True)

import PyNvVideoCodec as nvvc

encoder = None
depth_encoder = None


class SeparatedSendPipeline:
    def __init__(self, ws):
        self.ws = ws
        self.color_queue = asyncio.Queue(maxsize=10)
        self.depth_queue = asyncio.Queue(maxsize=10)
        self.frame_id_counter = 0
        self.color_task = None
        self.depth_task = None
    
    async def start(self):
        """분리된 전송 태스크 시작"""
        self.color_task = asyncio.create_task(self.color_send_loop())
        self.depth_task = asyncio.create_task(self.depth_send_loop())
        print(f"[SEND] Separated send pipeline started for {self.ws.remote_address}")
    
    async def stop(self):
        """전송 태스크 정리"""
        if self.color_task:
            self.color_task.cancel()
        if self.depth_task:
            self.depth_task.cancel()
        print(f"[SEND] Separated send pipeline stopped for {self.ws.remote_address}")
    
    async def color_send_loop(self):
        """Color 데이터만 전송하는 독립적인 루프"""
        while True:
            try:
                color_data, frame_id, timestamp = await self.color_queue.get()
                
                # Color 전용 헤더 (더 작은 크기)
                color_header = struct.pack("<IIIddd", 
                    len(color_data),  # color_length
                    0,                # depth_length (항상 0)
                    frame_id,         # frame_id
                    timestamp,        # timestamp
                    0.0,              # placeholder
                    0.0               # placeholder
                )
                
                await self.ws.send(color_header + color_data)
                print(f"[COLOR] Sent frame {frame_id}, size: {len(color_data)} bytes")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"[ERROR] Color send error: {e}")
    
    async def depth_send_loop(self):
        """Depth 데이터만 전송하는 독립적인 루프"""
        while True:
            try:
                depth_data, frame_id, timestamp = await self.depth_queue.get()
                
                # Depth 전용 헤더
                depth_header = struct.pack("<IIIddd", 
                    0,                # color_length (항상 0)
                    len(depth_data),  # depth_length
                    frame_id,         # frame_id
                    timestamp,        # timestamp
                    0.0,              # placeholder
                    0.0               # placeholder
                )
                
                await self.ws.send(depth_header + depth_data)
                print(f"[DEPTH] Sent frame {frame_id}, size: {len(depth_data)} bytes")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"[ERROR] Depth send error: {e}")
    
    async def enqueue_color(self, color_data, timestamp):
        """Color 데이터를 큐에 추가"""
        frame_id = self.frame_id_counter
        await self.color_queue.put((color_data, frame_id, timestamp))
        return frame_id
    
    async def enqueue_depth(self, depth_data, timestamp):
        """Depth 데이터를 큐에 추가"""
        frame_id = self.frame_id_counter
        await self.depth_queue.put((depth_data, frame_id, timestamp))
        self.frame_id_counter += 1
        return frame_id


async def recv_loop(ws: websockets.WebSocketServerProtocol, q: asyncio.Queue):
    """ 클라이언트로부터 메시지(카메라 데이터)를 받아 큐에 넣는 루프 """
    global width, height
    print(f"Receive loop started for {ws.remote_address}")
    try:
        while True:
            raw_with_ts = await ws.recv()
            if isinstance(raw_with_ts, bytes):
                if len(raw_with_ts) == 4:
                    W, H = struct.unpack("<HH", raw_with_ts)
                    if width != W or height != H: # 크기가 실제로 변경되었을 때만 업데이트
                        print(f"[+] Peer {ws.remote_address} resized to {W}x{H}")
                        width, height = W, H # 전역 변수 업데이트
                        while not q.empty():
                            try:
                                q.get_nowait()
                                q.task_done() # 비워진 작업에 대해 task_done 호출
                            except asyncio.QueueEmpty:
                                break
                        print(f"Render queue cleared due to resize for {ws.remote_address}")


                elif len(raw_with_ts) >= (32 * 4 + 8): # 최소 payload (legacy) + timestamp (double)
                    server_recv_timestamp_ms = time.time() * 1000 # 서버 수신 시점 (Unix Timestamp, ms)
                    client_send_timestamp_ms = struct.unpack_from("<d", raw_with_ts, len(raw_with_ts) - 8)[0] # 클라이언트 송신 시점 (Unix Timestamp, ms)
                    actual_payload = raw_with_ts[:-8] # 타임스탬프를 제외한 실제 페이로드
                    
                    # Unix 타임스탬프(ms)를 datetime 객체로 변환
                    client_dt = datetime.datetime.fromtimestamp(client_send_timestamp_ms / 1000.0)
                    server_dt = datetime.datetime.fromtimestamp(server_recv_timestamp_ms / 1000.0)
                    
                    # 사람이 읽을 수 있는 형태로 포맷팅
                    client_ts_str = client_dt.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3] # 밀리초 3자리까지
                    server_ts_str = server_dt.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3] # 밀리초 3자리까지

                    if q.full():
                        try:
                            q.get_nowait() # 이전 항목 제거 (블로킹 없이)
                            q.task_done()
                        except asyncio.QueueEmpty:
                            pass
                    # 큐에 (페이로드, 클라이언트 송신 TS (ms), 서버 수신 TS (ms)) 저장
                    await q.put((actual_payload, client_send_timestamp_ms, server_recv_timestamp_ms))
                else:
                    print(f"Warning: Received unexpected byte message length: {len(raw_with_ts)} from {ws.remote_address}")
            elif isinstance(raw_with_ts, str):
                # 텍스트 메시지 처리 (디코더 준비 알림 등)
                if raw_with_ts == "DECODERS_READY":
                    print(f"[+] Client {ws.remote_address} reports decoders are ready")
                    # 디코더 준비 메시지를 다시 전송하여 handler에서 처리하도록 함
                    await ws.send("DECODERS_READY_ACK")
                else:
                    print(f"Info: Received text message from {ws.remote_address}: {raw_with_ts}")
            else:
                print(f"Warning: Received unexpected data type: {type(raw_with_ts)} from {ws.remote_address}")

    except websockets.exceptions.ConnectionClosedOK:
        print(f"Connection closed normally for {ws.remote_address}")
    except websockets.exceptions.ConnectionClosedError as e:
        print(f"Connection closed with error for {ws.remote_address}: {e}")
    except Exception as e:
        print(f"Error in receive loop for {ws.remote_address}: {e}")
    finally:
        print(f"Receive loop finished for {ws.remote_address}")


async def encode_color_async(rgb_cuda, alpha_cuda, height, width, encoder):
    """Color 인코딩을 비동기로 실행"""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None, 
        lambda: encode_h264(rgb_cuda, alpha_cuda, height, width, encoder)
    )

async def encode_depth_async(depth_cuda, alpha_cuda, depth_encoder):
    """Depth 인코딩을 비동기로 실행"""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None, 
        lambda: encode_depth(depth_cuda, alpha_cuda, depth_encoder)
    )

async def render_separated_loop(ws: websockets.WebSocketServerProtocol, q: asyncio.Queue):
    """분리된 렌더링 및 전송 파이프라인"""
    global width, height, scene, encoder, depth_encoder
    
    print(f"Separated rendering loop started for {ws.remote_address}")
    
    # 분리된 전송 파이프라인 초기화
    send_pipeline = SeparatedSendPipeline(ws)
    await send_pipeline.start()
    
    frame_count = 0
    
    try:
        while True:
            raw_payload, client_send_timestamp_ms, server_recv_timestamp_ms = await q.get()
            
            # 렌더링 (Color와 Depth 동시에)
            view_mat, intrinsics = parse_camera_params(raw_payload)
            rgb_cuda, alpha_cuda, depth_cuda = render_frame(view_mat, intrinsics, width, height, scene)
            
            # Color와 Depth 인코딩을 병렬로 실행
            color_task = asyncio.create_task(
                encode_color_async(rgb_cuda, alpha_cuda, height, width, encoder)
            )
            depth_task = asyncio.create_task(
                encode_depth_async(depth_cuda, alpha_cuda, depth_encoder)
            )
            
            # 두 인코딩이 완료될 때까지 대기
            color_bytes, depth_bytes = await asyncio.gather(color_task, depth_task)
            
            # 각각 독립적으로 전송 큐에 추가
            timestamp = time.time() * 1000
            
            if len(color_bytes) > 0:
                await send_pipeline.enqueue_color(color_bytes, timestamp)
            
            if len(depth_bytes) > 0:
                await send_pipeline.enqueue_depth(depth_bytes, timestamp)
            
            frame_count += 1
            q.task_done()
            
    except asyncio.CancelledError:
        print(f"Separated render loop cancelled for {ws.remote_address}")
    except Exception as e:
        print(f"Error in separated render loop for {ws.remote_address}: {e}")
    finally:
        await send_pipeline.stop()
        print(f"Separated render loop finished for {ws.remote_address}")

def split_hevc_header(data: bytes):
    """
    HEVC 비트스트림에서 헤더(VPS, SPS, PPS)와 프레임 데이터를 분리합니다.

    :param data: 인코더에서 나온 전체 바이트 데이터
    :return: (헤더 바이트, 프레임 바이트) 튜플
    """
    start_code = b'\x00\x00\x00\x01'
    nal_units = data.split(start_code)
    nal_units = [unit for unit in nal_units if unit] # 빈 조각 제거

    print(f"[DEBUG] Found {len(nal_units)} NAL units in bitstream")

    header_nals = []
    frame_nals_start_index = -1

    for i, unit in enumerate(nal_units):
        # NAL 유닛 타입은 첫 바이트의 특정 비트에 있습니다.
        nal_type = (unit[0] >> 1) & 0b00111111
        print(f"[DEBUG] NAL unit {i}: type={nal_type}, length={len(unit)} bytes")

        # VPS (32), SPS (33), PPS (34)는 헤더로 간주합니다.
        if nal_type in [32, 33, 34]:
            header_nals.append(start_code + unit)
            print(f"[DEBUG] Added NAL type {nal_type} to headers")
        else:
            # 헤더가 아닌 NAL 유닛(예: 프레임 데이터)이 시작되는 위치를 기록
            frame_nals_start_index = i
            print(f"[DEBUG] Found non-header NAL type {nal_type}, stopping header search")
            break

    print(f"[DEBUG] Header NALs found: {len(header_nals)}")
    print(f"[DEBUG] Frame NALs start at index: {frame_nals_start_index}")

    if not header_nals or frame_nals_start_index == -1:
        raise ValueError("스트림에서 헤더(VPS/SPS/PPS)를 찾을 수 없습니다.")

    # 분리된 헤더와 프레임 데이터를 조합합니다.
    header_bytes = b''.join(header_nals)
    frame_bytes = start_code + start_code.join(nal_units[frame_nals_start_index:])
    
    return header_bytes, frame_bytes

def show_recv_server_timestamp(client_send_timestamp_ms, server_recv_timestamp_ms):
    client_dt = datetime.datetime.fromtimestamp(client_send_timestamp_ms / 1000.0)
    server_dt = datetime.datetime.fromtimestamp(server_recv_timestamp_ms / 1000.0)
    client_ts_str = client_dt.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
    server_ts_str = server_dt.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
    print(f"client_ts_str: {client_ts_str}, server_ts_str: {server_ts_str}")

def parse_camera_params(raw_payload: bytes, device="cuda"):
    # 클라이언트에서 전송하는 데이터: 32 floats + 1 double (타임스탬프)
    # 타임스탬프를 제외한 카메라 데이터만 파싱
    camera_data_size = 32 * 4  # 32 floats * 4 bytes
    camera_data = raw_payload[:camera_data_size]
    
    vals = struct.unpack("<32f", camera_data)
    eye = torch.tensor([[vals[0], -vals[1], vals[2]]], dtype=torch.float32, device=device)
    target = torch.tensor([[vals[3], -vals[4], vals[5]]], dtype=torch.float32, device=device)
    up = torch.tensor([[0., 1., 0]], dtype=torch.float32, device=device)

    zaxis = Fn.normalize(target - eye, dim=-1)
    xaxis = Fn.normalize(torch.cross(up, zaxis, dim=-1), dim=-1)
    yaxis = torch.cross(zaxis, xaxis, dim=-1)

    R_w2c = torch.stack([xaxis.squeeze(0), yaxis.squeeze(0), zaxis.squeeze(0)], dim=0).unsqueeze(0)

    t = -R_w2c @ eye.unsqueeze(-1)
    view_mat = kornia.geometry.conversions.Rt_to_matrix4x4(R_w2c, t)
    
    intrinsics_vals = vals[6:]
    intrinsics = torch.tensor([[
        [intrinsics_vals[0], intrinsics_vals[1], intrinsics_vals[2]],
        [intrinsics_vals[3], intrinsics_vals[4], intrinsics_vals[5]],
        [intrinsics_vals[6], intrinsics_vals[7], intrinsics_vals[8]] 
    ]], device=device)

    return view_mat, intrinsics

def encode_rawrgb(rgb_cuda, alpha_uint8):
    rgba_cuda = torch.cat([rgb_cuda, alpha_uint8.unsqueeze(-1)], dim=-1)  # [H, W, 4]
    rgba_cuda = rgba_cuda.to(torch.uint8)
    rgba_bytes = rgba_cuda.contiguous().cpu().numpy().tobytes()
    return rgba_bytes

def encode_h264(rgb_cuda: torch.Tensor, alpha_cuda_uint8: torch.Tensor, height: int, width: int, encoder):
    r = rgb_cuda[..., 0].float()
    g = rgb_cuda[..., 1].float()
    b = rgb_cuda[..., 2].float()
    
    # BT.709 변환 행렬
    y = 0.299 * r + 0.587 * g + 0.114 * b
    u = -0.169 * r - 0.331 * g + 0.500 * b + 128
    v = 0.500 * r - 0.419 * g - 0.081 * b + 128
    
    # 클램핑
    y = torch.clamp(y, 0, 255).to(torch.uint8)
    u = torch.clamp(u, 0, 255).to(torch.uint8)
    v = torch.clamp(v, 0, 255).to(torch.uint8)
    
    # Y 평면
    y_plane = y.contiguous()
    
    # UV 서브샘플링 (벡터화된 방식)
    # 2x2 블록의 평균을 계산 (float로 변환 후 계산)
    u_reshaped = u.view(height//2, 2, width//2, 2).float().mean(dim=(1, 3)).to(torch.uint8)
    v_reshaped = v.view(height//2, 2, width//2, 2).float().mean(dim=(1, 3)).to(torch.uint8)
    
    # UV 인터리빙 (U, V가 번갈아가며 나오도록)
    # [H//2, W//2] -> [H//2, W] (U와 V가 번갈아가며)
    uv_plane = torch.zeros(height//2, width, dtype=torch.uint8, device=rgb_cuda.device)
    uv_plane[:, 0::2] = u_reshaped  # 짝수 열에 U
    uv_plane[:, 1::2] = v_reshaped  # 홀수 열에 V
    
    # NV12 포맷으로 연결
    nv12 = torch.cat([y_plane, uv_plane], dim=0)

    try:
        video_bitstream = encoder.Encode(nv12)
        video_bitstream = bytes(video_bitstream)
        
        if len(video_bitstream) == 0:
            print("[WARNING] H.264 encoder returned empty bitstream")
            return b''
    
        return video_bitstream
    except Exception as e:
        print(f"[ERROR] H.264 encoding failed: {e}")
        return b''

def encode_depth(depth_cuda_raw, alpha_cuda_uint8, depth_encoder):
    alpha_float = alpha_cuda_uint8.float()
    depth_cuda_mod = depth_cuda_raw.clone()

    # ALPHA_CUTOFF = 0.5
    # depth_cuda_mod[alpha_float < ALPHA_CUTOFF] = torch.nan

    N, F = DEPTH_NEAR_CLIP, DEPTH_FAR_CLIP
    term_A = (F + N) / (F - N)
    term_B_num = 2 * F * N
    denominator_term_B = (F - N) * depth_cuda_mod
    calculated_ndc_webgl = term_A - (term_B_num / denominator_term_B)
    final_ndc_webgl = torch.clamp(calculated_ndc_webgl, -1.0, 1.0)

    depth_uint10 = (
        ((final_ndc_webgl + 1.0) * 0.5) * 1023.0
    ).round().clamp(0, 1023).to(torch.uint16)
    
    depth_y_plane = (depth_uint10.to(torch.int32) * 64).to(torch.uint16).contiguous()
    H, W = depth_uint10.shape

    # P010(YUV420_10BIT) 포맷: Y [H, W], UV [H//2, W] (interleaved, 0으로 채움)
    uv_zero = torch.zeros((H // 2, W), dtype=torch.uint16, device=depth_y_plane.device)
    depth_p010 = torch.cat([depth_y_plane, uv_zero], dim=0)   # shape [(H*3/2), W]

    try:
        depth_bitstream = depth_encoder.Encode(depth_p010)
        depth_bytes = bytes(depth_bitstream)

        if len(depth_bytes) == 0:
            print("[WARNING] Depth encoder returned empty bitstream")
            return b''
                
        return depth_bytes
    except Exception as e:
        print(f"[ERROR] Depth encoding failed: {e}")
        return b''

def render_frame(view_mat, intrinsics, width, height, scene):
    current_render_width = width
    server_correction_y = 1.0 / CLIENT_ASSUMED_SCALE_Y if CLIENT_ASSUMED_SCALE_Y > 0 else 1.0
    corrected_render_height = round(height * server_correction_y)

    render_colors, render_alphas, _ = scene.render(
        viewmats=view_mat,
        ks=intrinsics,
        width=current_render_width,
        height=corrected_render_height,
        near=DEPTH_NEAR_CLIP,
        far=DEPTH_FAR_CLIP
    )

    last_mat = view_mat.clone()

    # 명시적으로 RGB 3채널만 추출
    rgb_cuda = render_colors[0, ..., :3]  # [H, W, 3]
    rgb_cuda = (rgb_cuda * 255.0).clamp(0, 255).to(torch.uint8).contiguous()

    alpha_cuda = render_alphas[0, ..., 0]  # [H, W]
    alpha_cuda_uint8 = (alpha_cuda * 255.0).clamp(0, 255).to(torch.uint8).contiguous()

    depth_cuda_raw = render_colors[0, ..., -1].float()

    # background = torch.zeros_like(rgb_cuda)
    # rgb_cuda = rgb_cuda * alpha_cuda.unsqueeze(-1) + background * (1 - alpha_cuda.unsqueeze(-1))

    return rgb_cuda, alpha_cuda_uint8, depth_cuda_raw

def encode_data(rgb_cuda, alpha_cuda_uint8, depth_cuda_raw, height, width, encoder, depth_encoder):
    if usingRawRGB:
        rgba_bytes = encode_rawrgb(rgb_cuda, alpha_cuda_uint8)
        color_bytes = rgba_bytes
    else:
        video_bitstream = encode_h264(rgb_cuda, alpha_cuda_uint8, height, width, encoder)
        color_bytes = video_bitstream

    depth_bytes = encode_depth(depth_cuda_raw, alpha_cuda_uint8, depth_encoder)

    # 디버깅을 위한 로그 추가
    if len(color_bytes) == 0:
        print(f"[DEBUG] Color encoding result: {len(color_bytes)} bytes")
    if len(depth_bytes) == 0:
        print(f"[DEBUG] Depth encoding result: {len(depth_bytes)} bytes")

    return color_bytes, depth_bytes

async def send_initial_frame_separated(ws: websockets.WebSocketServerProtocol, camera_payload: bytes):
    """분리된 초기 프레임 전송"""
    print(f"Separated initial frame generation started for {ws.remote_address}")
    
    try:
        # 분리된 전송 파이프라인 초기화
        send_pipeline = SeparatedSendPipeline(ws)
        await send_pipeline.start()
        
        # 카메라 데이터 파싱
        view_mat, intrinsics = parse_camera_params(camera_payload)
        
        # 렌더링
        rgb_cuda, alpha_cuda, depth_cuda = render_frame(view_mat, intrinsics, width, height, scene)
        
        # 병렬 인코딩
        color_task = asyncio.create_task(
            encode_color_async(rgb_cuda, alpha_cuda, height, width, encoder)
        )
        depth_task = asyncio.create_task(
            encode_depth_async(depth_cuda, alpha_cuda, depth_encoder)
        )
        
        color_bytes, depth_bytes = await asyncio.gather(color_task, depth_task)
        
        # 분리된 전송
        timestamp = time.time() * 1000
        
        if len(color_bytes) > 0:
            await send_pipeline.enqueue_color(color_bytes, timestamp)
        
        if len(depth_bytes) > 0:
            await send_pipeline.enqueue_depth(depth_bytes, timestamp)
        
        print(f"[+] Separated initial frame sent successfully")
        
        await send_pipeline.stop()
        
    except Exception as e:
        print(f"Error in separated initial frame generation: {e}")


async def send_loop(ws: websockets.WebSocketServerProtocol, send_q: asyncio.Queue):
    """ 전송 전용 루프 - 렌더링과 독립적으로 데이터 전송 """
    print(f"Send loop started for {ws.remote_address}")
    
    try:
        while True:
            # 전송할 데이터를 큐에서 가져옴
            send_data = await send_q.get()
            header, color_data, depth_bytes, frame_count = send_data
            
            send_call_start_ms = time.perf_counter() * 1000
            
            # 실제 전송 (depth_bytes가 None이면 빈 바이트로 처리)
            depth_data = depth_bytes if depth_bytes is not None else b''
            await ws.send(header + color_data + depth_data)
            
            send_call_end_ms = time.perf_counter() * 1000
            send_call_duration_ms = send_call_end_ms - send_call_start_ms
            
            # WebSocket 버퍼링 상태 확인
            depth_size = len(depth_bytes) if depth_bytes is not None else 0
            total_data_size = len(header) + len(color_data) + depth_size
            data_size_mb = total_data_size / (1024 * 1024)
            
            buffered_amount = 0
            try:
                if hasattr(ws.transport, 'get_write_buffer_size'):
                    buffered_amount = ws.transport.get_write_buffer_size()
                elif hasattr(ws, 'bufferedAmount'):
                    buffered_amount = ws.bufferedAmount
            except:
                pass
            
            buffered_mb = buffered_amount / (1024 * 1024)
            
            if buffered_mb > 5.0:
                print(f"[WARNING] High buffering detected: {buffered_mb:.2f}MB")
            
            # print(f"[{frame_count:03d}] SEND: {send_call_duration_ms:.2f}ms, data: {data_size_mb:.2f}MB, buffered: {buffered_mb:.2f}MB")
            
            send_q.task_done()
            
    except asyncio.CancelledError:
        print(f"Send loop cancelled for {ws.remote_address}")
    except Exception as e:
        print(f"Error in send loop for {ws.remote_address}: {e}")
    finally:
        print(f"Send loop finished for {ws.remote_address}")


async def handler(ws: websockets.WebSocketServerProtocol):
    """ WebSocket 연결 핸들러 (분리된 파이프라인 사용) """
    global width, height, using4DGS, encoder, depth_encoder
    remote_addr = ws.remote_address
    print(f"Connection opened from {remote_addr}")

    transport = ws.transport
    if transport and hasattr(transport, "get_extra_info"):
        raw_socket = transport.get_extra_info("socket")
        if raw_socket:
            raw_socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

    try:
        # 핸드셰이크: 클라이언트로부터 너비, 높이 받기
        handshake = await ws.recv()
        if isinstance(handshake, bytes) and len(handshake) == 4:
            W, H = struct.unpack("<HH", handshake)
            width, height = W, H
            print(f"[+] Peer {remote_addr} => {W}x{H}")
        else:
            print(f"Error: Invalid handshake from {remote_addr}")
            await ws.close()
            return
        
        if not usingRawRGB:
            color_encoder_params = {
                "codec": "h264",
                "gop": 1,
                "preset": "P1",
                "colorspace": "bt709",
                "color_range": "full",  # 또는 "limited"
                "bitrate": 5000000,     # 5Mbps (빠른 전송을 위해 적절한 비트레이트)
                "max_bitrate": 10000000, # 10Mbps 최대
                "profile": "baseline",   # 빠른 디코딩을 위한 baseline 프로파일
                "level": "4.1",          # 적절한 레벨
            }

            encoder = nvvc.CreateEncoder(
                width=width,
                height=height,
                fmt="NV12",
                usecpuinputbuffer=False,
                **color_encoder_params
            )

            depth_encoder_params = {
                "codec": "hevc",
                "profile": "main10",
                "level": "5.1",
                "rc": "constqp",
                "constqp": 0,     # lossless
                "gop": 1,         # 모든 프레임을 IDR로
                "bf": 0,
                "insertSEI": 1,
                "insertVUI": 1,
            }

            depth_encoder = nvvc.CreateEncoder(
                width=width,
                height=height,
                fmt="P010",  # YUV420_10BIT 대신 YUV444_10BIT 사용
                usecpuinputbuffer=False,
                **depth_encoder_params
            )

            ### Supported formats:
            # - NV12, YUV420, ARGB, ABGR (always supported)
            # - YUV444
            # - P010
            # - YUV444_10BIT, YUV444_16BIT

        # 첫 카메라 패킷을 반드시 수신 (디코더 초기화용)
        print(f"[+] Waiting for initial camera packet from client...")
        initial_camera_payload = await ws.recv()
        if not isinstance(initial_camera_payload, bytes):
            print(f"Error: Initial camera packet is not bytes from {remote_addr}")
            await ws.close()
            return
        
        print(f"[+] Received initial camera packet from {remote_addr}, size: {len(initial_camera_payload)} bytes")

        q = asyncio.Queue(maxsize=2)  # 큐 크기를 2로 증가 (수신과 렌더링 병렬화)

        # 전송용 큐 추가
        send_q = asyncio.Queue(maxsize=5)  # 전송용 큐
        
        # 첫 카메라 패킷으로 initial_frame 생성 및 전송
        print(f"[+] Generating initial frame from client camera packet...")
        print(f"[+] Initial camera packet size: {len(initial_camera_payload)} bytes")
        
        # 카메라 데이터 파싱 테스트
        try:
            view_mat, intrinsics = parse_camera_params(initial_camera_payload)
            print(f"[+] Camera parsing successful - view_mat shape: {view_mat.shape}, intrinsics shape: {intrinsics.shape}")
        except Exception as e:
            print(f"[ERROR] Camera parsing failed: {e}")
            import traceback
            traceback.print_exc()
        
        await send_initial_frame_separated(ws, initial_camera_payload)
        
        # 클라이언트의 디코더 초기화 완료를 기다림 (최대 5초)
        print(f"[+] Waiting for client decoder initialization...")
        decoder_init_timeout = 5.0  # 5초 타임아웃
        decoder_init_start = time.time()
        
        # 디코더 준비 메시지를 받기 위한 이벤트
        decoder_ready_event = asyncio.Event()
        
        # 디코더 준비 메시지 수신을 위한 임시 핸들러
        async def wait_for_decoder_ready():
            try:
                while time.time() - decoder_init_start < decoder_init_timeout:
                    try:
                        message = await asyncio.wait_for(ws.recv(), timeout=0.1)
                        if isinstance(message, str) and message == "DECODERS_READY":
                            print(f"[+] Client decoders initialized successfully")
                            decoder_ready_event.set()
                            return
                    except asyncio.TimeoutError:
                        continue
                    except Exception as e:
                        print(f"[WARNING] Error waiting for decoder ready message: {e}")
                        break
            except Exception as e:
                print(f"[WARNING] Error in decoder ready waiter: {e}")
        
        # 디코더 준비 대기 태스크 시작
        decoder_wait_task = asyncio.create_task(wait_for_decoder_ready())
        
        # 디코더 준비 이벤트 대기 또는 타임아웃
        try:
            await asyncio.wait_for(decoder_ready_event.wait(), timeout=decoder_init_timeout)
            print(f"[+] Client decoder initialization completed in {time.time() - decoder_init_start:.2f}s")
        except asyncio.TimeoutError:
            print(f"[WARNING] Client decoder initialization timeout after {decoder_init_timeout}s, proceeding anyway")
        finally:
            # 대기 태스크 정리
            if not decoder_wait_task.done():
                decoder_wait_task.cancel()
                try:
                    await decoder_wait_task
                except asyncio.CancelledError:
                    pass
        
        # 수신 루프와 분리된 렌더링 루프 실행
        recv_task = asyncio.create_task(recv_loop(ws, q))
        render_task = asyncio.create_task(render_separated_loop(ws, q))

        # 기존 send_loop는 제거 (분리된 파이프라인으로 대체)

        # 태스크 관리
        done, pending = await asyncio.wait(
            [recv_task, render_task],
            return_when=asyncio.FIRST_COMPLETED,
        )

        for task in pending:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    except websockets.exceptions.ConnectionClosed as e:
        print(f"Connection closed for {remote_addr}: {e}")
    except Exception as e:
        print(f"Handler error for {remote_addr}: {e}")
    finally:
        print(f"Connection handler finished for {remote_addr}")

async def main():
    """ 메인 함수: GaussianScene 로드 및 WebSocket 서버 시작 """
    global scene, using4DGS
    print("Loading Gaussian Scene...")

    if not using4DGS:
        scene = GaussianScene(PLY_PATH)
        scene.upload_to_gpu() # GPU에 미리 업로드
        print("Gaussian Scene loaded and uploaded to GPU.")
    else:
        scene = Gaussian4DScene()

    print(f"Starting WebSocket server on port {WEBSOCKET_PORT}...")

    async with websockets.serve(handler, "0.0.0.0", WEBSOCKET_PORT, max_size=None, ping_interval=20, ping_timeout=20):
        print(f"WebSocket server listening on ws://0.0.0.0:{WEBSOCKET_PORT}")
        await asyncio.Future()  # 서버 계속 실행

if __name__ == "__main__":
    torch.cuda.init() # 명시적 초기화 시도
    print(f"CUDA initialized. Device: {torch.cuda.get_device_name(0)}")

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Server stopped by user.")

