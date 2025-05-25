# server.py
import torch
import numpy as np
import time
import os
import cv2
import argparse
import imageio
import io

parser = argparse.ArgumentParser()
parser.add_argument("--near_clip", type=float, default=1.0, help="Near clip")
parser.add_argument("--far_clip", type=float, default=30.0, help="Far clip")
parser.add_argument("--using3DGS", action="store_true", help="Using 3D Gaussian Scene")
parser.add_argument("--using4DGS", action="store_true", help="Using 4D Gaussian Scene")
parser.add_argument("--webgpu_depth", action="store_true", help="Using WebGPU Depth")
args = parser.parse_args()

using3DGS = args.using3DGS
using4DGS = args.using4DGS
OUTPUT_WEBGPU_DEPTH_FOR_3DGS = args.webgpu_depth # True로 설정 시 render_loop (3DGS)에서 [0,1] NDC 뎁스 출력
DEPTH_NEAR_CLIP = args.near_clip # 뎁스 최소 클리핑 값
DEPTH_FAR_CLIP = args.far_clip # 뎁스 최대 클리핑 값 (클라이언트와 일치 확인)

print(f"Using 3DGS: {using3DGS}")
print(f"Using 4DGS: {using4DGS}")
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


from gaussian_scene import GaussianScene, Gaussian4DScene

import importlib
import ws_handler # 가정: 수동 행렬 재구성 버전 사용

import asyncio, struct
import websockets

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

async def recv_loop(ws: websockets.WebSocketServerProtocol, q: asyncio.Queue):
    """ 클라이언트로부터 메시지(카메라 데이터)를 받아 큐에 넣는 루프 """
    global width, height
    print(f"Receive loop started for {ws.remote_address}")
    try:
        while True:
            raw = await ws.recv()
            if isinstance(raw, bytes):
                if len(raw) == 4:
                    W, H = struct.unpack("<HH", raw)
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

                elif len(raw) >= 60: 
                    if q.full():
                        await q.get()
                    await q.put(raw)
                else:
                    print(f"Warning: Received unexpected byte message length: {len(raw)} from {ws.remote_address}")
            else:
                print(f"Warning: Received non-byte data type: {type(raw)} from {ws.remote_address}")

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
    last_print_time = time.perf_counter()
    one_minute_timings = {
        "parse": 0.0,
        "render": 0.0,
        "jpeg_encode": 0.0,
        "depth": 0.0,
        "send": 0.0,
        "total_cycle": 0.0  # 루프 전체 시간
    }
    one_minute_frame_count = 0

    try:
        while True:
            t_loop_start = time.perf_counter()
            raw = await q.get()

            try:
                importlib.reload(ws_handler)
                view_mat, intrinsics = ws_handler.parse_payload(raw)
            except Exception as e:
                print(f"Error parsing payload: {e}")
                q.task_done()
                continue

            # 현재 전역 width/height 와 마지막 렌더링 width/height 비교
            size_changed = (prev_width != width or prev_height != height)
            matrix_changed = (last_mat is None or not torch.allclose(view_mat, last_mat, atol=1e-5))

            should_skip_render = (not size_changed and not matrix_changed)

            if should_skip_render:
                q.task_done()
                await asyncio.sleep(0.001) # Prevent busy-waiting
                continue # Skip rendering

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

            t_parse = time.perf_counter()
            
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

            t_render = time.perf_counter()

            # --- 상태 업데이트 (다음 루프 비교용) ---
            # 렌더링에 사용된 정보 저장
            last_mat = view_mat.clone()
            prev_width = current_render_width # 이번에 렌더링한 크기 저장
            prev_height = current_render_height # 이번에 렌더링한 크기 저장

            # --- 컬러 이미지 처리 (JPEG 인코딩) ---
            rgb_cuda = render_colors[0, ..., :3].mul(255).clamp(0, 255).to(torch.uint8).contiguous()

            img_nv = nvc.as_image(rgb_cuda)
            jpeg_bytes = encoder.encode(img_nv, "jpeg", params=nvc.EncodeParams(quality=JPEG_QUALITY))
                
            t_jpeg = time.perf_counter()

            # --- 뎁스 데이터 처리 (Float16, NDC [-1, 1] 범위) ---
            depth_cuda_raw = render_colors[0, ..., -1].float()  # 정밀도를 위해 float32 사용
            alpha_cuda = render_alphas[0, ..., 0].float()

            depth_cuda_mod = depth_cuda_raw.clone()

            if not OUTPUT_WEBGPU_DEPTH_FOR_3DGS:
                ALPHA_CUTOFF = 0.2  # 알파 값 임계치
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
                depth_ndc_f16 = (depth_ndc_f16 + 1.0) / 2.0
                depth_numpy = depth_ndc_f16.contiguous().cpu().numpy()

                depth_bytes = depth_numpy.tobytes()

                # print(len(jpeg_bytes),len(depth_bytes)) # 1k,1k -> 1201328 2097152

            else:
                ALPHA_CUTOFF = 0.1

                border_mask      = (alpha_cuda < ALPHA_CUTOFF) & (alpha_cuda > 0.0)
                depth_cuda_mod[border_mask]  = torch.nan      # 경계도 전부 NaN
                depth_cuda_mod[alpha_cuda <= 0.0] = torch.nan

                depth_01_scale = (DEPTH_FAR_CLIP * (depth_cuda_mod - DEPTH_NEAR_CLIP)) / (depth_cuda_mod * (DEPTH_FAR_CLIP - DEPTH_NEAR_CLIP))
                depth_01_scale[alpha_cuda <= 0.0] = torch.nan

                result = cv2.imwrite("rendered_depth.jpg", depth_01_scale.mul(255).to(torch.uint8).cpu().numpy())
                
                depth_01_scale = depth_01_scale.to(torch.float16)

                depth_bytes = depth_01_scale.contiguous().cpu().detach().numpy().tobytes()
                # print(len(jpeg_bytes),len(depth_bytes)) # 2,953,212

            
            t_depth = time.perf_counter()

            with open("rendered_color.jpg", "wb") as f:
                f.write(jpeg_bytes)

            header = struct.pack("<II", len(jpeg_bytes), len(depth_bytes))
            await ws.send(header + jpeg_bytes + depth_bytes)

            t_send = time.perf_counter()

            frame_idx_total += 1

            # 현재 프레임의 각 단계별 시간 계산
            time_parse = t_parse - t_loop_start
            time_render = t_render - t_parse
            time_jpeg = t_jpeg - t_render
            time_depth = t_depth - t_jpeg
            time_send = t_send - t_depth
            total_cycle_time = t_send - t_loop_start

            # 1분 통계 누적
            one_minute_timings["parse"] += time_parse
            one_minute_timings["render"] += time_render
            one_minute_timings["jpeg_encode"] += time_jpeg
            one_minute_timings["depth"] += time_depth
            one_minute_timings["send"] += time_send
            one_minute_timings["total_cycle"] += total_cycle_time
            one_minute_frame_count += 1
            
            # print(f"total: {total_cycle_time:.4f} | parse: {time_parse:.4f} | render: {time_render:.4f} | jpeg: {time_jpeg:.4f} | depth: {time_depth:.4f} | send: {time_send:.4f}")
            # total: 0.4113 | parse: 0.0939 | render: 0.0026 | jpeg: 0.0019 | depth: 0.2120 | send: 0.1009
            # total: 0.2747 | parse: 0.1165 | render: 0.0026 | jpeg: 0.0018 | depth: 0.0293 | send: 0.1245
            
            # 1분 경과 시 평균 출력
            current_time = time.perf_counter()
            # print(f"current_time: {current_time} | last_print_time: {last_print_time}")
            if current_time - last_print_time >= 60.0:
                if one_minute_frame_count > 0:
                    print("--- 1-Minute Average Performance Statistics (seconds) ---")
                    print(f"{'Stage':<18} | {'Avg Time':<10}")
                    print("-" * 40)
                    for stage, total_time in one_minute_timings.items():
                        avg_time = total_time / one_minute_frame_count
                        print(f"{stage:<18} | {avg_time:<10.4f}")
                    
                    avg_fps_one_min = one_minute_frame_count / (current_time - last_print_time)
                    print(f"Average FPS (last 1 min): {avg_fps_one_min:.2f}")
                    print("-" * 40)

                # 변수 초기화
                one_minute_timings = {key: 0.0 for key in one_minute_timings}
                one_minute_frame_count = 0
                last_print_time = current_time

            q.task_done() # 큐 작업 완료 알림

    except asyncio.CancelledError:
         print(f"Render loop cancelled for {ws.remote_address}")
    except Exception as e:
        print(f"Error in render loop for {ws.remote_address}: {e}")
    finally:
        print(f"Render loop finished for {ws.remote_address}")


async def render_4dgs_loop(ws: websockets.WebSocketServerProtocol, q: asyncio.Queue):
    global width, height
    import torch
    import torch.nn.functional as F
    import kornia
    
    device = "cuda"
        
    global width, height, scene
    print(f"Render loop started for {ws.remote_address}")

    last_mat = None
    frame_idx_total = 0
    frame_idx_collected = 0

    prev_width = -1 
    prev_height = -1
    last_time = None

    # 1분 평균 시간 측정을 위한 변수
    last_print_time = time.perf_counter()
    one_minute_timings = {
        "parse": 0.0,
        "render": 0.0,
        "jpeg_encode": 0.0,
        "depth": 0.0,
        "send": 0.0,
        "total_cycle": 0.0  # 루프 전체 시간
    }
    one_minute_frame_count = 0

    def legacy_view_mat(vals):
        eye = torch.tensor([[vals[0], -vals[1], vals[2]]], dtype=torch.float32, device=device)
        target = torch.tensor([[vals[3], -vals[4], vals[5]]], dtype=torch.float32, device=device)
        up = torch.tensor([[0., -1., 0]], dtype=torch.float32, device=device)

        zaxis = F.normalize(target - eye, dim=-1)
        xaxis = F.normalize(torch.cross(up, zaxis, dim=-1), dim=-1)
        yaxis = torch.cross(zaxis, xaxis, dim=-1)

        R_w2c = torch.stack([xaxis.squeeze(0), yaxis.squeeze(0), zaxis.squeeze(0)], dim=0).unsqueeze(0)

        t = -R_w2c @ eye.unsqueeze(-1)
        view = kornia.geometry.conversions.Rt_to_matrix4x4(R_w2c, t).squeeze(0)
        view_mat = torch.tensor([[
            [view[0, 0], view[1, 0], view[2, 0], 0],
            [view[0, 1], view[1, 1], view[2, 1], 0],
            [view[0, 2], view[1, 2], view[2, 2], 0],
            [view[0, 3], view[1, 3], view[2, 3], 1]
        ]], device=device).squeeze(0)

        return view_mat
    
    legacy_way = False

    while True:
        t_start = time.perf_counter()
        raw = await q.get()
        
        vals = struct.unpack(f"<{16+9+1+16+6}f", raw) 
        view_mat = torch.tensor([
            [vals[0], vals[1], vals[2], vals[3]],
            [vals[4], vals[5], vals[6], vals[7]],
            [vals[8], vals[9], vals[10], vals[11]],
            [vals[12], vals[13], vals[14], vals[15]]
        ], device=device, dtype=torch.float32)

        # print(view_mat)

        now_time = vals[25]
        print(now_time)
        proj_matrix = torch.tensor([
            [vals[26], vals[27], vals[28], vals[29]],
            [vals[30], vals[31], vals[32], vals[33]],
            [vals[34], vals[35], vals[36], vals[37]],
            [vals[38], vals[39], vals[40], vals[41]]
        ], device=device, dtype=torch.float32)

        # print(proj_matrix)

        t_parse = time.perf_counter()
        size_changed = (prev_width != width or prev_height != height)
        matrix_changed = (last_mat is None or not torch.allclose(view_mat, last_mat, atol=1e-5))
        time_changed = (last_time is None or not last_time == now_time)

        should_skip_render = (not size_changed and not matrix_changed and not time_changed)

        if should_skip_render:
            q.task_done()
            await asyncio.sleep(0.001) # Prevent busy-waiting
            continue # Skip rendering

    
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

    
        rendered_color, rendered_depth, rendered_alpha = scene.render_4dgs(viewmat=view_mat, 
                                                            projmat=proj_matrix, 
                                                            width=current_render_width, 
                                                            height=corrected_render_height, 
                                                            near=DEPTH_NEAR_CLIP, 
                                                            far=DEPTH_FAR_CLIP, 
                                                            scene_time=now_time)
        t_render = time.perf_counter()

        last_mat = view_mat.clone()
        last_time = now_time
        prev_width = current_render_width # 이번에 렌더링한 크기 저장
        prev_height = current_render_height # 이번에 렌더링한 크기 저장

        # --- 컬러 이미지 처리 (JPEG 인코딩) ---
        rgb_cuda = rendered_color.permute(1, 2, 0).mul(255).clamp(0, 255).to(torch.uint8).contiguous()
        rgb_cuda = rgb_cuda.flip(0)
        # rgb_cuda = rgb_cuda.flip(1)

        # rendered_depth = rendered_depth.flip(1)
        # rendered_depth = rendered_depth.flip(2)

        # rendered_alpha = rendered_alpha.flip(1)
        # rendered_alpha = rendered_alpha.flip(2)

        img_nv = nvc.as_image(rgb_cuda)
        jpeg_bytes = encoder.encode(img_nv, "jpeg", params=nvc.EncodeParams(quality=JPEG_QUALITY))

        with open("rendered_color.jpg", "wb") as f:
            f.write(jpeg_bytes)

        t_jpeg = time.perf_counter()
        depth_cuda = rendered_depth.clone()
        alpha_cuda = rendered_alpha

        depth_cuda_mod = depth_cuda.clone()
        
        if not OUTPUT_WEBGPU_DEPTH_FOR_3DGS:
            ALPHA_CUTOFF = 0.05  # 알파 값 임계치
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

            depth_ndc_f16 = final_ndc_to_send.to(torch.float16)

            depth_numpy = depth_ndc_f16.contiguous().cpu().detach().numpy()
            depth_vis_debug_normalized = (depth_numpy + 1.0) / 2.0
            
            depth_vis_debug_img = np.nan_to_num(depth_vis_debug_normalized * 255.0, nan=0.0).astype(np.uint8) # NaN은 0으로
            result = cv2.imwrite("rendered_depth.jpg", depth_vis_debug_img)
        
            depth_bytes = depth_numpy.tobytes()
        else:
            # ALPHA_CUTOFF = 0.1

            # border_mask      = (alpha_cuda < ALPHA_CUTOFF) & (alpha_cuda > 0.0)
            # depth_cuda_mod[border_mask]  = torch.nan      # 경계도 전부 NaN
            # depth_cuda_mod[alpha_cuda <= 0.0] = torch.nan

            depth_01_scale = (DEPTH_FAR_CLIP * (depth_cuda_mod - DEPTH_NEAR_CLIP)) / (depth_cuda_mod * (DEPTH_FAR_CLIP - DEPTH_NEAR_CLIP))
            depth_01_scale[alpha_cuda <= 0.0] = torch.nan

            result = cv2.imwrite("rendered_depth.jpg", depth_01_scale.mul(255).to(torch.uint8).cpu().numpy())
            
            depth_01_scale = depth_01_scale.to(torch.float16)
            depth_bytes = depth_01_scale.contiguous().cpu().detach().numpy().tobytes()

        t_depth = time.perf_counter()
        
        header = struct.pack("<II", len(jpeg_bytes), len(depth_bytes))
        await ws.send(header + jpeg_bytes + depth_bytes)

        t_send = time.perf_counter()

        frame_idx_total += 1

        q.task_done() # 큐 작업 완료 알림

        # 현재 프레임의 각 단계별 시간 계산
        time_parse = t_parse - t_start
        time_render = t_render - t_parse
        time_jpeg = t_jpeg - t_render
        time_depth = t_depth - t_jpeg
        time_send = t_send - t_depth
        total_cycle_time = t_send - t_start

        # 1분 통계 누적
        one_minute_timings["parse"] += time_parse
        one_minute_timings["render"] += time_render
        one_minute_timings["jpeg_encode"] += time_jpeg
        one_minute_timings["depth"] += time_depth
        one_minute_timings["send"] += time_send
        one_minute_timings["total_cycle"] += total_cycle_time
        one_minute_frame_count += 1

        print(f"parse: {time_parse:.4f}, render: {time_render:.4f}, jpeg: {time_jpeg:.4f}, depth: {time_depth:.4f}, send: {time_send:.4f}")

        # 1분 경과 시 평균 출력
        current_time = time.perf_counter()
        if current_time - last_print_time >= 60.0:
            if one_minute_frame_count > 0:
                print("--- 1-Minute Average Performance Statistics (4DGS) (seconds) ---")
                print(f"{'Stage':<18} | {'Avg Time':<10}")
                print("-" * 40)
                for stage, total_time in one_minute_timings.items():
                    avg_time = total_time / one_minute_frame_count
                    print(f"{stage:<18} | {avg_time:<10.4f}")
                
                avg_fps_one_min = one_minute_frame_count / (current_time - last_print_time)
                print(f"Average FPS (last 1 min): {avg_fps_one_min:.2f}")
                print("-" * 40)

            # 변수 초기화
            one_minute_timings = {key: 0.0 for key in one_minute_timings}
            one_minute_frame_count = 0
            last_print_time = current_time


async def handler(ws: websockets.WebSocketServerProtocol):
    """ WebSocket 연결 핸들러 """
    global width, height, using4DGS
    remote_addr = ws.remote_address
    print(f"Connection opened from {remote_addr}")

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

        # 데이터 처리 큐 생성 (버퍼 크기 1)
        q = asyncio.Queue(maxsize=1)

        # 수신 루프와 렌더링 루프를 동시에 실행
        recv_task = asyncio.create_task(recv_loop(ws, q))

        if using4DGS:
            render_task = asyncio.create_task(render_4dgs_loop(ws, q))
        else:
            render_task = asyncio.create_task(render_loop(ws, q))

        # 두 태스크 중 하나라도 완료되면 (오류 또는 정상 종료) 다른 태스크도 취소
        done, pending = await asyncio.wait(
            [recv_task, render_task],
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



            # 프레임 저장 (선택 사항)
            # if SAVE_FRAMES:
            #     try:
            #         with open(f"frames/frame_color.jpg", "wb") as f:
            #             f.write(jpeg_bytes)
            #         # Optionally save depth map (might need visualization conversion)
            #         # Example: Convert linear depth to grayscale image
            #         # depth_vis = DEPTH_FAR_CLIP * (depth_cuda_mod - DEPTH_NEAR_CLIP) / clamped_depth_cuda * (DEPTH_FAR_CLIP - DEPTH_NEAR_CLIP)
                    
            #         depth_scaling_cuda = depth_linear_fp16.copy()
            #         depth_scaling_cuda = (depth_scaling_cuda - DEPTH_NEAR_CLIP) / (DEPTH_FAR_CLIP - DEPTH_NEAR_CLIP)
            #         depth_scaling_cuda = (depth_scaling_cuda * 255).astype(np.uint8)
            #         depth_viz_bytes = encoder.encode(depth_scaling_cuda, "jpeg", params=nvc.EncodeParams(quality=100))

            #         with open(f"frames/frame_depth.png", "wb") as f:
            #             f.write(depth_viz_bytes)
            #         # if PILLOW_AVAILABLE:
            #         #     Image.fromarray(depth_vis, 'L').save(f"frames/frame_{frame_idx:05d}_depth.png")
            #     except Exception as e:
            #         print(f"Error saving frame: {e}")

            # --- 데이터 전송 ---