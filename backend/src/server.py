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
    image_encoder = nvc.Encoder()
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
            else:
                print(f"Warning: Received non-byte data type: {type(raw_with_ts)} from {ws.remote_address}")

    except websockets.exceptions.ConnectionClosedOK:
        print(f"Connection closed normally for {ws.remote_address}")
    except websockets.exceptions.ConnectionClosedError as e:
        print(f"Connection closed with error for {ws.remote_address}: {e}")
    except Exception as e:
        print(f"Error in receive loop for {ws.remote_address}: {e}")
    finally:
        print(f"Receive loop finished for {ws.remote_address}")


async def render_loop(ws: websockets.WebSocketServerProtocol, q: asyncio.Queue):
    """ 큐에서 카메라 데이터를 가져와 렌더링하고 결과를 클라이언트에 전송하는 루프 """
    global width, height, scene, using3DGS
    print(f"Render loop started for {ws.remote_address}")

    last_mat = None
    frame_idx_total = 0
    frame_idx_collected = 0

    # prev_width/height는 마지막으로 *렌더링*한 크기를 저장
    prev_width = -1 # 초기값 설정 (첫 프레임은 무조건 렌더링되도록)
    prev_height = -1

    # 각 단계별 시간 저장을 위한 딕셔너리 및 리스트
    timings = {
        "parse": [],
        "render": [],
        "jpeg_encode": [],
        "depth": [],
        "send": [],
        "total": []
    }

    # 1분 평균 시간 측정을 위한 변수
    last_print_time = time.time() # 이 부분을 time.time()으로 변경
    one_minute_timings = {
        "parse": 0.0,
        "render": 0.0,
        "jpeg_encode": 0.0,
        "depth": 0.0,
        "send": 0.0,
        "send_call_duration": 0.0, # ws.send() 호출 시간
        "total_cycle": 0.0,  # 루프 전체 시간
        "jpeg_size": 0.0, # JPEG 크기 누적
        "depth_size": 0.0   # Depth 크기 누적
    }
    one_minute_frame_count = 0

    try:
        while True:
            loop_start_time_ms = time.time() * 1000
            raw_payload, client_send_ts_ms, server_recv_ts_ms = await q.get()

            loop_start_time_ms = time.time() * 1000

            try:
                importlib.reload(ws_handler)
                # 실제 페이로드(raw_payload)로 파싱
                view_mat, intrinsics = ws_handler.parse_payload(raw_payload)
            except Exception as e:
                print(f"Error parsing payload: {e}")
                q.task_done()
                continue

            # 현재 전역 width/height 와 마지막 렌더링 width/height 비교
            size_changed = (prev_width != width or prev_height != height)
            matrix_changed = (last_mat is None or not torch.allclose(view_mat, last_mat, atol=1e-5))

            should_skip_render = (not size_changed and not matrix_changed)

            # 첫 프레임 디버깅 로그
            if frame_idx_total == 0:
                print(f"[DEBUG] First frame - size_changed: {size_changed}, matrix_changed: {matrix_changed}")
                print(f"[DEBUG] width: {width}, height: {height}, prev_width: {prev_width}, prev_height: {prev_height}")
                print(f"[DEBUG] last_mat is None: {last_mat is None}")

            # if should_skip_render:
            #     q.task_done()
            #     await asyncio.sleep(0.001) # Prevent busy-waiting
            #     continue # Skip rendering

            # --- 렌더링 ---
            current_render_width = width
            current_render_height = height

            client_requested_height = height # 현재 전역 height 사용

            # 서버에서 보정할 Y 스케일 팩터
            server_correction_y = 1.0 / CLIENT_ASSUMED_SCALE_Y if CLIENT_ASSUMED_SCALE_Y > 0 else 1.0

            # 실제 렌더링에 사용할 보정된 높이
            # 정수 값으로 변환 필요할 수 있음
            corrected_render_height = round(client_requested_height * server_correction_y)
            current_render_width = width # 너비는 그대로 사용 (또는 X 스케일도 고려)

            # 첫 프레임에서 크기가 0인 경우 기본값 사용
            if frame_idx_total == 0 and (current_render_width == 0 or corrected_render_height == 0):
                print(f"[DEBUG] First frame with zero dimensions, using defaults")
                current_render_width = 1920
                corrected_render_height = 1080

            t_parse = time.time() * 1000 # 근사치, t_parse가 perf_counter이므로 정확하지 않음.
                                                                # 정확하려면 t_parse 시점도 time.time()*1000으로 기록해야 함.
            
            if using3DGS:
                render_colors, render_alphas, _ = scene.render(
                    viewmats=view_mat,
                    ks=intrinsics,
                    width=current_render_width, 
                    height=corrected_render_height,
                    near=DEPTH_NEAR_CLIP,
                    far=DEPTH_FAR_CLIP
                )
            else:
                render_colors, render_alphas, render_normals, surf_normals, render_distort, render_median, meta = scene.render_2dgs(
                    viewmats=view_mat,
                    ks=intrinsics,
                    width=current_render_width,
                    height=corrected_render_height,
                    near=DEPTH_NEAR_CLIP,
                    far=DEPTH_FAR_CLIP
                )

            t_render = time.time() * 1000 # 위와 동일

            # --- 상태 업데이트 (다음 루프 비교용) ---
            # 렌더링에 사용된 정보 저장
            last_mat = view_mat.clone()
            prev_width = current_render_width # 이번에 렌더링한 크기 저장
            prev_height = current_render_height # 이번에 렌더링한 크기 저장

            # --- 컬러 이미지 처리 (JPEG 인코딩) ---
            rgb_cuda = render_colors[0, ..., :3].mul(255).clamp(0, 255).to(torch.uint8).contiguous()

            if True:
                img_nv = nvc.as_image(rgb_cuda)
                jpeg_bytes = encoder.encode(img_nv, "jpeg", params=nvc.EncodeParams(quality=JPEG_QUALITY))
                with open("rendered_color.jpg", "wb") as f:
                    f.write(jpeg_bytes)
            else:
                jpeg_bytes = rgb_cuda.cpu().numpy().tobytes()
                
            t_jpeg = time.time() * 1000 # 위와 동일

            # --- 뎁스 데이터 처리 (Float16, NDC [-1, 1] 범위) ---
            depth_cuda_raw = render_colors[0, ..., -1].float()  # 정밀도를 위해 float32 사용
            alpha_cuda = render_alphas[0, ..., 0].float()

            depth_cuda_mod = depth_cuda_raw.clone()

            if not OUTPUT_WEBGPU_DEPTH_FOR_3DGS:
                ALPHA_CUTOFF = 0.5  # 알파 값 임계치
                depth_cuda_mod[alpha_cuda < ALPHA_CUTOFF] = torch.nan  # 알파 낮은 부분 NaN 처리

                N = DEPTH_NEAR_CLIP
                F = DEPTH_FAR_CLIP
                
                # WebGL NDC Z 변환: z_ndc = ( (F+N)/(F-N) ) - ( 2*F*N / ( (F-N) * z_eye ) )
                # z_eye = N -> z_ndc = -1
                # z_eye = F -> z_ndc = 1
                
                term_A = (F + N) / (F - N)
                term_B_num = 2 * F * N
                term_B_den_factor = (F - N)
                
                # 분모 계산 (NaN 전파, 0인 경우 inf 발생 가능)
                denominator_term_B = term_B_den_factor * depth_cuda_mod
                
                calculated_ndc_webgl = term_A - (term_B_num / denominator_term_B)
                
                # inf 값을 NaN으로 변경 후 [-1, 1]로 클램핑 (NaN은 유지됨)
                # torch.clamp는 NaN을 NaN으로, inf를 min/max 값으로 자동 처리 (PyTorch 1.8+).
                final_ndc_webgl = torch.clamp(calculated_ndc_webgl, -1.0, 1.0)
                final_ndc_to_send = final_ndc_webgl

                depth_ndc_f16 = final_ndc_to_send.to(torch.float16) # 2953212
                # depth_ndc_f16 = (depth_ndc_f16 + 1.0) / 2.0
                depth_numpy = depth_ndc_f16.contiguous().cpu().numpy()

                depth_bytes = depth_numpy.tobytes()

                # print(len(jpeg_bytes),len(depth_bytes)) # 1k,1k -> 1201328 2097152

            else:
                ALPHA_CUTOFF = 0.5

                border_mask      = (alpha_cuda < ALPHA_CUTOFF) & (alpha_cuda > 0.0)
                depth_cuda_mod[border_mask]  = torch.nan      # 경계도 전부 NaN
                depth_cuda_mod[alpha_cuda <= 0.0] = torch.nan

                depth_01_scale = (DEPTH_FAR_CLIP * (depth_cuda_mod - DEPTH_NEAR_CLIP)) / (depth_cuda_mod * (DEPTH_FAR_CLIP - DEPTH_NEAR_CLIP))
                depth_01_scale[alpha_cuda <= 0.0] = torch.nan

                # result = cv2.imwrite("rendered_depth.jpg", depth_01_scale.mul(255).to(torch.uint8).cpu().numpy())
                
                depth_01_scale = depth_01_scale.to(torch.float16)

                depth_bytes = depth_01_scale.contiguous().cpu().detach().numpy().tobytes()
                # print(len(jpeg_bytes),len(depth_bytes)) # 2,953,212

            
            t_depth = time.time() * 1000 # 위와 동일

            server_process_end_timestamp_ms = time.time() * 1000 # 서버 처리 완료 및 송신 직전 시점 (Unix Timestamp, ms)

            # 헤더: RGB 길이(I), Depth 길이(I), clientSendTS(d, ms), serverRecvTS(d, ms), serverProcessEndTS(d, ms)
            header = struct.pack("<IIddd", len(jpeg_bytes), len(depth_bytes), 
                                 client_send_ts_ms, server_recv_ts_ms, server_process_end_timestamp_ms)
            
            send_call_start_ms = time.time() * 1000
            await ws.send(header + jpeg_bytes + depth_bytes)
            send_call_end_ms = time.time() * 1000
            send_call_duration_ms = send_call_end_ms - send_call_start_ms

            # t_send = time.perf_counter() # Unix Timestamp 기준으로 변경 고려
            send_time_ms = time.time() * 1000

            frame_idx_total += 1

            # 현재 프레임의 각 단계별 시간 계산 (ms 단위)
            # time_parse = t_parse - t_loop_start # 아래에서 재계산
            # time_render = t_render - t_parse
            # time_jpeg = t_jpeg - t_render
            # time_depth = t_depth - t_jpeg
            # time_send = t_send - t_depth
            # total_cycle_time = t_send - t_loop_start

            # Unix 타임스탬프 기준 시간 계산 (ms)
            # t_parse, t_render, t_jpeg, t_depth는 perf_counter 기반이므로 그대로 사용하거나 Unix 시간으로 변환 필요.
            # 여기서는 루프 전체 시간과 주요 단계 시간을 Unix 시간 기준으로 다시 로깅.
            
            # perf_counter 기반 시간 (단위: 초) -> ms로 변환하여 로깅하거나, Unix 시간 기준으로 새로 측정.
            # 기존 t_parse 등은 perf_counter 기반이므로, Unix 시간과 혼용 시 주의.
            # 여기서는 주요 이벤트 시점만 Unix 시간으로 로깅하고, 기존 성능 카운터 기반 시간도 유지.

            time_parse_unix = (t_parse * 1000) - loop_start_time_ms # 근사치, t_parse가 perf_counter이므로 정확하지 않음.
                                                                # 정확하려면 t_parse 시점도 time.time()*1000으로 기록해야 함.
            time_render_unix = (t_render * 1000) - (t_parse*1000) # 위와 동일
            # ... (이하 생략, 또는 아래처럼 주요 시점만 Unix 시간으로)

            total_cycle_time_unix = send_time_ms - loop_start_time_ms
            # 서버 내부 처리 시간 (큐에서 나온 시점부터 보내기 직전까지)
            server_internal_processing_unix = server_process_end_timestamp_ms - server_recv_ts_ms 
            # (참고: server_recv_ts_ms는 recv_loop에서 기록, loop_start_time_ms는 render_loop 시작점)

            # 모든 시간 단위를 초(seconds)로 통일하여 누적
            one_minute_timings["parse"] += (t_parse - loop_start_time_ms) / 1000.0
            one_minute_timings["render"] += (t_render - t_parse) / 1000.0
            one_minute_timings["jpeg_encode"] += (t_jpeg - t_render) / 1000.0
            one_minute_timings["depth"] += (t_depth - t_jpeg) / 1000.0
            # "send"는 depth 처리 완료부터 ws.send() 완료까지 (헤더 패킹 포함)
            one_minute_timings["send"] += (send_time_ms - t_depth) / 1000.0 
            one_minute_timings["send_call_duration"] += send_call_duration_ms / 1000.0
            one_minute_timings["total_cycle"] += total_cycle_time_unix / 1000.0 # total_cycle_time_unix는 (send_time_ms - loop_start_time_ms)로 이미 ms 단위

            one_minute_timings["jpeg_size"] += len(jpeg_bytes)
            one_minute_timings["depth_size"] += len(depth_bytes)
            one_minute_frame_count += 1
            
            # print(f"total: {total_cycle_time:.4f} | parse: {time_parse:.4f} | render: {time_render:.4f} | jpeg: {time_jpeg:.4f} | depth: {time_depth:.4f} | send: {time_send:.4f} | jpeg_size: {len(jpeg_bytes)} | depth_size: {len(depth_bytes)}")
            # print(f"UnixTsPerf (ms): TotalCycle={total_cycle_time_unix:.2f}, ServerInternalProc={server_internal_processing_unix:.2f} | jpeg_size: {len(jpeg_bytes)}, depth_size: {len(depth_bytes)}")
            
            # 상세 로그에 send_call_duration_ms 추가 (주석 처리된 기존 로그를 활용하거나 새 로그 라인 추가)
            # 예시:
            # print(f"UnixTsPerf (ms): TotalCycle={total_cycle_time_unix:.2f}, ServerInternalProc={server_internal_processing_unix:.2f}, SendCallDuration={send_call_duration_ms:.2f} | jpeg_size: {len(jpeg_bytes)}, depth_size: {len(depth_bytes)}")

            # 1분 경과 시 평균 출력 (기존 로직 유지, 시간 단위 일관성 확인 필요)
            current_time_unix_ms = time.time()

            if current_time_unix_ms - last_print_time >= 60.0: # last_print_time이 Unix 시간(초) 기준
                if one_minute_frame_count > 0:
                    print("--- 1-Minute Average Performance Statistics (seconds for time, bytes for size) ---")
                    print(f"{'Stage/Data':<18} | {'Avg Value':<10}")
                    print("-" * 40)
                    # 현재 one_minute_timings에 perf_counter와 Unix ms가 혼재되어 있을 수 있음. 통일 필요.
                    # 예시: 모든 시간을 초 단위로 저장 및 계산한다고 가정.
                    for stage, total_value in one_minute_timings.items():
                        avg_value = total_value / one_minute_frame_count 
                        if "size" in stage:
                            print(f"{stage:<18} | {avg_value:<10.2f}")
                        else:
                            print(f"{stage:<18} | {avg_value:<10.4f}") # 시간은 초 단위로 출력 가정
                    
                    avg_fps_one_min = (one_minute_frame_count / ((current_time_unix_ms - last_print_time) / 1000.0))
                    print(f"Average FPS (last 1 min): {avg_fps_one_min:.2f}")
                    print("-" * 40)

                one_minute_timings = {key: 0.0 for key in one_minute_timings}
                one_minute_frame_count = 0
                last_print_time = time.time() # Unix 시간(초)으로 업데이트

            q.task_done() # 큐 작업 완료 알림

    except asyncio.CancelledError:
         print(f"Render loop cancelled for {ws.remote_address}")
    except Exception as e:
        print(f"Error in render loop for {ws.remote_address}: {e}")
    finally:
        print(f"Render loop finished for {ws.remote_address}")


async def render_jpeg_test_loop(ws: websockets.WebSocketServerProtocol, q: asyncio.Queue, send_q: asyncio.Queue):
    """
    Renders the scene, encodes the color image to H.264 or sends raw RGB data,
    and sends the video bitstream/depth data over WebSocket.
    """
    global width, height, scene, encoder, depth_encoder
    import struct
    import traceback
    import io
    import tempfile
    import kornia

    print(f"TorchAudio H.264 encoding loop started for {ws.remote_address}")

    last_mat = None
    prev_width, prev_height = width, height
    frame_count = 0

    # 성능 측정을 위한 변수들
    last_print_time = time.time()
    one_minute_timings = {
        "parse": 0.0,
        "render": 0.0,
        "tensor_conversion": 0.0,
        "h264_encode": 0.0,
        "raw_rgb_process": 0.0,
        "depth": 0.0,
        "send": 0.0,
        "send_call_duration": 0.0,
        "total_cycle": 0.0,
        "h264_size": 0.0,
        "raw_rgb_size": 0.0,
        "depth_size": 0.0
    }
    one_minute_frame_count = 0
    # 하나의 StreamWriter를 재사용
    writer = None
    output_buffer = None
    
    # 비디오 파일 저장을 위한 변수
    video_file = None
    depth_video_file = None  # None으로 초기화

    # ws_handler.py
    import struct
    import torch
    import torch.nn.functional as Fn
    import kornia

    print(width, height)

    firstRender = False
    firstDepthRender = False
    frame_count = 0
    depth_header_sent = False  # 헤더 전송 상태 추적

    try:
        # 비디오 파일 열기 (연결별로 고유 파일명)
        # video_file = open(f"session_video_{ws.remote_address[0]}_{ws.remote_address[1]}.h264", "wb")
        
        # Depth H.265 파일 열기 (연결별로 고유 파일명)
        depth_video_file = open(f"depth_video.h264", "wb")
        print(f"[+] Depth H.264 file opened: depth_video.h264")
        
        while True:
            raw_payload, client_send_timestamp_ms, server_recv_timestamp_ms = await q.get()

            # # Unix 타임스탬프(ms)를 datetime 객체로 변환
            # client_dt = datetime.datetime.fromtimestamp(client_send_timestamp_ms / 1000.0)
            # server_dt = datetime.datetime.fromtimestamp(server_recv_timestamp_ms / 1000.0)
            
            # # 사람이 읽을 수 있는 형태로 포맷팅
            # client_ts_str = client_dt.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3] # 밀리초 3자리까지
            # server_ts_str = server_dt.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3] # 밀리초 3자리까지

            # print(f"client_ts_str: {client_ts_str}, server_ts_str: {server_ts_str}")
            
            
            device = "cuda"
            loop_start_time_ms = time.perf_counter() * 1000
            vals = struct.unpack("<32f", raw_payload) 
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

            parse_time_ms = time.perf_counter() * 1000

            # 크기 변경 감지
            # size_changed = (prev_width != width or prev_height != height)
            # if size_changed and (width > 0 and height > 0) and not usingRawRGB:
            #     print(f"Size changed: {prev_width}x{prev_height} -> {width}x{height}")
            #     # 크기가 변경되면 StreamWriter 재생성
            #     encoder = nvvc.CreateEncoder(
            #         width=width,
            #         height=height,
            #         fmt="NV12",
            #         usecpuinputbuffer=False,
            #         **color_encoder_params
            #     )

            #     depth_encoder = nvvc.CreateEncoder(
            #         width=width,
            #         height=height,
            #         fmt="P010",  # P010 대신 YUV420_10BIT 사용
            #         usecpuinputbuffer=False,
            #         **depth_encoder_params
            #     )
            #     prev_width, prev_height = width, height

            # matrix_changed = (last_mat is None or not torch.allclose(view_mat, last_mat, atol=1e-5))

            # should_skip_render = (not size_changed and not matrix_changed)

            # if should_skip_render:
            #     q.task_done()
            #     await asyncio.sleep(0.001) # Prevent busy-waiting
            #     continue # Skip rendering


            # --- Rendering ---
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

            render_time_ms = time.perf_counter() * 1000
    
            # render_colors: [1, H, W, 4] (RGBA) -> RGB만 추출
        
            # 명시적으로 RGB 3채널만 추출
            rgb_cuda = render_colors[0, :, :, :3]  # [H, W, 3]
            rgb_cuda = (rgb_cuda * 255.0).clamp(0, 255).to(torch.uint8).contiguous()

            alpha_cuda = render_alphas[0, ..., 0]  # [H, W]
            alpha_uint8 = (alpha_cuda * 255.0).clamp(0, 255).to(torch.uint8).contiguous()

            # background = torch.zeros_like(rgb_cuda)
            # rgb_cuda = rgb_cuda * alpha_cuda.unsqueeze(-1) + background * (1 - alpha_cuda.unsqueeze(-1))

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

            video_bitstream = encoder.Encode(nv12)
            video_bitstream = bytes(video_bitstream)
            
            # # 인코딩된 비디오 비트스트림을 파일에 append
            # video_file.write(video_bitstream)

            h264_time_ms = time.perf_counter() * 1000
            color_data = video_bitstream
            color_data_size = len(video_bitstream)

            # --- Depth Data Processing ---
            # ALPHA_CUTOFF = 0.5
            # depth_cuda_mod[alpha_float < ALPHA_CUTOFF] = torch.nan

            # ####################################################################
            # ### START: Depth to 8-bit Grayscale H.264 Encoding ###
            # ####################################################################
            # --- 1. Get 16-bit float depth map ---
            depth_cuda_raw = render_colors[0, ..., -1].float()
            H, W = depth_cuda_raw.shape

            # --- 2. Normalize to 0-1 range ---
            # Inverse of WebGL NDC transform to get linear depth [0, 1]
            # This part depends on how depth is packed. Assuming it's linear depth for now.
            # If it's already in a specific range, adjust accordingly.
            # Let's assume `depth_cuda_raw` is linear depth from near to far clip.
            depth_normalized_01 = (depth_cuda_raw - DEPTH_NEAR_CLIP) / (DEPTH_FAR_CLIP - DEPTH_NEAR_CLIP)
            depth_normalized_01 = torch.clamp(depth_normalized_01, 0.0, 1.0) # Clamp to be safe

            # --- 3. Convert to 8-bit grayscale (0-255) ---
            depth_8bit_grayscale = (depth_normalized_01 * 255.0).to(torch.uint8).contiguous()

            # --- 4. Prepare NV12 frame for H.264 encoder ---
            # Y plane is our 8-bit depth map
            depth_y_plane_8bit = depth_8bit_grayscale

            # U and V planes are black (value 128 for neutral chroma)
            # YUV420 has chroma planes at half resolution
            uv_plane_black_8bit = torch.full(
                (H // 2, W), 
                128, # Neutral gray for chroma
                dtype=torch.uint8, 
                device=depth_y_plane_8bit.device
            )

            # Concatenate to form NV12 format (Y plane followed by interleaved UV plane)
            nv12_depth = torch.cat([depth_y_plane_8bit, uv_plane_black_8bit], dim=0)

            # --- 5. Encode with H.264 depth encoder ---
            depth_bitstream = bytes(depth_encoder.Encode(nv12_depth))

            # Save to file for debugging
            if depth_bitstream:
                depth_video_file.write(depth_bitstream)

            # depth_grayscale = nvc.as_image(depth_uint8)
            # jpeg_bytes = image_encoder.encode(depth_grayscale, "jpeg", params=nvc.EncodeParams(quality=JPEG_QUALITY))

            # with open("depth_grayscale.jpeg", "wb") as f:
            #     f.write(jpeg_bytes)

            depth_time_ms = time.perf_counter() * 1000

            # --- Data Preparation for Transmission ---
            server_process_end_timestamp_ms = time.time() * 1000
            header = struct.pack("<IIddd", color_data_size, len(depth_bitstream),
                                client_send_timestamp_ms, server_recv_timestamp_ms, server_process_end_timestamp_ms)

            # 전송 큐에 데이터 추가 (렌더링 루프는 블로킹되지 않음)
            await send_q.put((header, color_data, depth_bitstream, frame_count))
            
            depth_size = len(depth_bitstream) if depth_bitstream is not None else 0
            print(f"[{frame_count:03d}] RENDER: {depth_time_ms - loop_start_time_ms:.2f}ms, color_data: {(color_data_size)}Bytes, depth_data: {depth_size}Bytes")

            # 성능 통계 업데이트
            frame_count += 1
                
            # one_minute_timings["parse"] += (parse_time_ms - loop_start_time_ms) / 1000.0
            # one_minute_timings["render"] += (render_time_ms - parse_time_ms) / 1000.0
            # if usingRawRGB:
            #     one_minute_timings["raw_rgb_process"] += (raw_rgb_time_ms - render_time_ms) / 1000.0
            #     one_minute_timings["depth"] += (depth_time_ms - raw_rgb_time_ms) / 1000.0
            #     one_minute_timings["raw_rgb_size"] += color_data_size
            # else:
            #     one_minute_timings["h264_encode"] += (h264_time_ms - render_time_ms) / 1000.0
            #     one_minute_timings["depth"] += (depth_time_ms - h264_time_ms) / 1000.0
            #     one_minute_timings["h264_size"] += color_data_size
            # one_minute_timings["send"] += (send_call_end_ms - depth_time_ms) / 1000.0
            # one_minute_timings["send_call_duration"] += send_call_duration_ms / 1000.0
            # one_minute_timings["total_cycle"] += total_cycle_time_ms / 1000.0
            # one_minute_timings["depth_size"] += len(depth_bytes)
            # one_minute_frame_count += 1

            # 1분 통계 출력
            # current_time = time.time()
            # if current_time - last_print_time >= 1.0:
            #     if one_minute_frame_count > 0:
            #         print("--- 1-Minute TorchAudio H.264 Performance Statistics ---")
            #         print(f"{'Stage/Data':<18} | {'Avg Value(s,Bytes)':<10}")
            #         print("-" * 40)
            #         for stage, total_value in one_minute_timings.items():
            #             avg_value = total_value / one_minute_frame_count
            #             if "size" in stage:
            #                 print(f"{stage:<18} | {avg_value:<10.2f}")
            #             else:
            #                 print(f"{stage:<18} | {avg_value:<10.4f}")
                    
            #         avg_fps = one_minute_frame_count / 1.0
            #         print(f"Average FPS: {avg_fps:.2f}")
            #         print("-" * 40)

            #     # 통계 초기화
            #     one_minute_timings = {key: 0.0 for key in one_minute_timings}
            #     one_minute_frame_count = 0
            #     last_print_time = current_time

            frame_count += 1
            q.task_done()

    except asyncio.CancelledError:
        print(f"TorchAudio H.264 encoding loop cancelled for {ws.remote_address}")
    except Exception as e:
        print(f"Error in TorchAudio H.264 encoding loop for {ws.remote_address}: {e}")
        traceback.print_exc()
    finally:
        print(f"TorchAudio H.264 encoding loop finished for {ws.remote_address}")
        print(f"Total frames encoded with TorchAudio: {frame_count}")
        
        # 파일 안전하게 닫기
        if depth_video_file is not None:
            depth_video_file.close()
            print(f"[+] Depth H.264 file closed: depth_video.h264")


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
            
            print(f"[{frame_count:03d}] SEND: {send_call_duration_ms:.2f}ms, data: {data_size_mb:.2f}MB, buffered: {buffered_mb:.2f}MB")
            
            send_q.task_done()
            
    except asyncio.CancelledError:
        print(f"Send loop cancelled for {ws.remote_address}")
    except Exception as e:
        print(f"Error in send loop for {ws.remote_address}: {e}")
    finally:
        print(f"Send loop finished for {ws.remote_address}")


async def handler(ws: websockets.WebSocketServerProtocol):
    """ WebSocket 연결 핸들러 """
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
        
        color_encoder_params = {
            "codec": "h264",
            "preset": "P1", # P1: Fastest
            "bitrate": 8000000, # 8Mbps
            "profile": "baseline",
            "gop": 1,
        }
        encoder = nvvc.CreateEncoder(
            width=width,
            height=height,
            fmt="NV12", # RGB to NV12,
            usecpuinputbuffer=False,
            **color_encoder_params
        )

        # --- H.264 Depth Encoder (8-bit Grayscale) ---
        depth_encoder_params = {
            "codec": "h264",
            "preset": "P1",
            "bitrate": 4000000, # 4Mbps for depth
            "profile": "baseline",
            "gop": 1,
        }
        depth_encoder = nvvc.CreateEncoder(
            width=width,
            height=height,
            fmt="NV12", # Grayscale data will be put into Y plane of NV12
            usecpuinputbuffer=False,
            **depth_encoder_params
        )

            ### Supported formats:
            # - NV12, YUV420, ARGB, ABGR (always supported)
            # - YUV444
            # - P010
            # - YUV444_10BIT, YUV444_16BIT

        q = asyncio.Queue(maxsize=2)  # 큐 크기를 2로 증가 (수신과 렌더링 병렬화)

        # 전송용 큐 추가
        send_q = asyncio.Queue(maxsize=5)  # 전송용 큐
        
        # 수신 루프, 렌더링 루프, 전송 루프를 동시에 실행
        recv_task = asyncio.create_task(recv_loop(ws, q))

        render_task = asyncio.create_task(render_jpeg_test_loop(ws, q, send_q))
            
        send_task = asyncio.create_task(send_loop(ws, send_q))

        # 세 태스크 중 하나라도 완료되면 (오류 또는 정상 종료) 다른 태스크도 취소
        done, pending = await asyncio.wait(
            [recv_task, 
             render_task, 
             send_task],
            return_when=asyncio.FIRST_COMPLETED,
        )

        for task in pending:
            task.cancel()
            try:
                await task # 작업 취소 대기
            except asyncio.CancelledError:
                pass # 예상된 취소

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