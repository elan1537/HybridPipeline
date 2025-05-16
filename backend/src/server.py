# server.py
import torch
import numpy as np
import time
import os

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


from gaussian_scene import GaussianScene # gsplat.utils 사용 버전
from gaussian_renderer import GaussianModel

def render(self, viewmats, ks, width, height, near, far, frame_time):
        viewmats = viewmats.cuda()
        ks = ks.cuda()

        
        # return rasterization(
        #     means=self.means_gpu,
        #     quats=self.quats_gpu,
        #     scales=self.scales_gpu,
        #     opacities=self.ops_gpu,
        #     colors=self.shs_gpu,
        #     viewmats=viewmats,
        #     Ks=ks,
        #     width=width,
        #     height=height,
        #     near_plane=near,
        #     far_plane=far,
        #     packed=False,
        #     radius_clip=0.1,
        #     sh_degree=3,
        #     eps2d=0.3,
        #     render_mode="RGB+D",
        #     rasterize_mode="antialiased",
        #     camera_model="pinhole"
        # )

import importlib
import ws_handler # 가정: 수동 행렬 재구성 버전 사용

import asyncio, struct
import websockets

# --- 설정값 ---
# PLY_PATH = "./output_scene.ply" # PLY 파일 경로
PLY_PATH = "./livinglab-no-ur5.ply"
# PLY_PATH = "./livinglab-3.ply"
WEBSOCKET_PORT = 8765
SAVE_FRAMES = True # 프레임 저장 여부
JPEG_QUALITY = 100   # JPEG 압축 품질
DEPTH_NEAR_CLIP = 0.1 # 뎁스 최소 클리핑 값
DEPTH_FAR_CLIP = 100 # 뎁스 최대 클리핑 값 (클라이언트와 일치 확인)
CLIENT_ASSUMED_SCALE_Y = 1.0 
# -------------

# --- 전역 변수 ---
scene: GaussianScene = None
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

                elif len(raw) == 60: 
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
        # Optionally signal render_loop to stop or handle cleanup

async def render_loop(ws: websockets.WebSocketServerProtocol, q: asyncio.Queue):
    """ 큐에서 카메라 데이터를 가져와 렌더링하고 결과를 클라이언트에 전송하는 루프 """
    global width, height, scene, NVC_AVAILABLE, NVCOMP_AVAILABLE
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

            t_parse = time.perf_counter()

            # --- 렌더링 건너뛰기 조건 확인 ---
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


            using3DGS = True
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

            # --- 뎁스 데이터 처리 (Float16 + Padding) ---
            depth_cuda = render_colors[0, ..., -1] # 마지막 채널이 뎁스라고 가정
            alpha_cuda = render_alphas[0, ..., 0]

            depth_cuda_mod = depth_cuda.clone()
            
            ALPHA_CUTOFF = 0.1

            border_mask      = (alpha_cuda < ALPHA_CUTOFF) & (alpha_cuda > 0.0)
            depth_cuda_mod[border_mask]  = torch.nan      # 경계도 전부 NaN
            depth_cuda_mod[alpha_cuda <= 0.0] = torch.nan

            depth_01_scale = (DEPTH_FAR_CLIP * (depth_cuda_mod - DEPTH_NEAR_CLIP)) / (depth_cuda_mod * (DEPTH_FAR_CLIP - DEPTH_NEAR_CLIP))
            depth_01_scale[alpha_cuda <= 0.0] = torch.nan
            # print(depth_01_scale.min(), depth_01_scale.max(), depth_cuda_mod.min(), depth_cuda_mod.max())

            # with open("frames/frame_color.jpg", "wb") as f:
            #     f.write(jpeg_bytes)

            # with open("frames/frame_alpha.jpg", "wb") as f:
            #     alpha_save = alpha_cuda.to(torch.float16)
            #     alpha_save = (alpha_save * 255).to(torch.uint8)
            #     alpha_nv = nvc.as_image(alpha_save)
            #     alpha_jpeg_bytes = encoder.encode(alpha_nv, "jpeg", params=nvc.EncodeParams(quality=JPEG_QUALITY))
            #     f.write(alpha_jpeg_bytes)

            # with open("frames/frame_depth.jpg", "wb") as f:
            #     depth_save = (depth_cuda_mod - DEPTH_NEAR_CLIP) / (DEPTH_FAR_CLIP - DEPTH_NEAR_CLIP)    
            #     depth_save = (depth_save * 255).to(torch.uint8)
            #     depth_nv = nvc.as_image(depth_save)
            #     depth_jpeg_bytes = encoder.encode(depth_nv, "jpeg", params=nvc.EncodeParams(quality=JPEG_QUALITY))
            #     f.write(depth_jpeg_bytes)

            depth_01_scale = depth_01_scale.to(torch.float16)

            depth_bytes = depth_01_scale.contiguous().cpu().numpy().tobytes()
            t_depth = time.perf_counter()
            
            header = struct.pack("<II", len(jpeg_bytes), len(depth_bytes))
            await ws.send(header + jpeg_bytes + depth_bytes)

            t_send = time.perf_counter()

            frame_idx_total += 1

            q.task_done() # 큐 작업 완료 알림

    except asyncio.CancelledError:
         print(f"Render loop cancelled for {ws.remote_address}")
    except Exception as e:
        print(f"Error in render loop for {ws.remote_address}: {e}")
    finally:
        print(f"Render loop finished for {ws.remote_address}")

        # --- 통계 계산 및 출력 ---
        if frame_idx_collected > 0:
            print("\n--- Performance Statistics (seconds) ---")
            print(f"{'Stage':<18} | {'Avg':<7} | {'Std':<7} | {'Min':<7} | {'Max':<7} | {'Median':<7} | {'P95':<7}")
            print("-" * 80)
            for stage_name, times_list in timings.items():
                if not times_list: continue # 데이터 없는 스테이지는 스킵 (예: 압축 안 쓴 경우)
                
                np_times = np.array(times_list)
                avg_time = np.mean(np_times)
                std_time = np.std(np_times)
                min_time = np.min(np_times)
                max_time = np.max(np_times)
                median_time = np.median(np_times)
                p95_time = np.percentile(np_times, 95)
                
                print(f"{stage_name:<18} | {avg_time:<7.4f} | {std_time:<7.4f} | "
                      f"{min_time:<7.4f} | {max_time:<7.4f} | "
                      f"{median_time:<7.4f} | {p95_time:<7.4f}")
            
            # 평균 FPS 계산 (total_cycle 기준)
            avg_total_cycle_time = np.mean(timings["total"])
            if avg_total_cycle_time > 0:
                avg_fps = 1.0 / avg_total_cycle_time
                print(f"\nAverage Server Effective FPS (based on total_cycle): {avg_fps:.2f}")
        else:
            print("No frames collected for statistics.")


async def handler(ws: websockets.WebSocketServerProtocol):
    """ WebSocket 연결 핸들러 """
    global width, height
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
    global scene
    print("Loading Gaussian Scene...")
    try:
        scene = GaussianScene(PLY_PATH)
        scene.upload_to_gpu() # GPU에 미리 업로드
        print("Gaussian Scene loaded and uploaded to GPU.")
    except FileNotFoundError:
        print(f"Error: PLY file not found at {PLY_PATH}")
        return
    except Exception as e:
        print(f"Error loading Gaussian Scene: {e}")
        return

    # Pillow 설치 여부 확인
    if not NVC_AVAILABLE and not PILLOW_AVAILABLE:
        print("Error: Cannot proceed without nvimgcodec or Pillow for image encoding.")
        return

    print(f"Starting WebSocket server on port {WEBSOCKET_PORT}...")
    try:
        async with websockets.serve(handler, "0.0.0.0", WEBSOCKET_PORT, max_size=None, ping_interval=20, ping_timeout=20):
            print(f"WebSocket server listening on ws://0.0.0.0:{WEBSOCKET_PORT}")
            await asyncio.Future()  # 서버 계속 실행
    except OSError as e:
        print(f"Error starting WebSocket server (Port {WEBSOCKET_PORT} likely in use): {e}")
    except Exception as e:
        print(f"WebSocket server error: {e}")

if __name__ == "__main__":
    # CUDA 초기화 확인
    if torch.cuda.is_available():
        try:
            torch.cuda.init() # 명시적 초기화 시도
            print(f"CUDA initialized. Device: {torch.cuda.get_device_name(0)}")
        except Exception as e:
            print(f"Error initializing CUDA: {e}")
            # CUDA 사용 불가 시 종료 또는 CPU 모드 전환 로직 필요
            exit()
    else:
        print("Error: CUDA is not available. Exiting.")
        exit()

    # 비동기 메인 함수 실행
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