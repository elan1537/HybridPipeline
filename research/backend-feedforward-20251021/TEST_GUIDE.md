# Feedforward Pipeline Test Guide

브라우저 없이 전체 파이프라인을 테스트하는 방법입니다.

## 아키텍처

```
test_feedforward.py (Frontend Mock)
    ↓ WebSocket: ws://localhost:8765/ws/feedforward
server-fifo.py (Transport)
    ↓ Unix Socket: /run/ipc/camera.sock
    ↓ Unix Socket: /run/ipc/video.sock
feed-forward-renderer-socket.py (Renderer)
```

## 사전 준비

### 1. Transport 인스턴스 설정

```bash
# Python 환경 확인
python3 --version

# websockets 패키지 설치
pip install websockets

# 또는 conda 환경
conda install -c conda-forge websockets
```

### 2. Renderer 인스턴스 설정

3DGS 모델 파일 준비:
```bash
cd /workspace/research/3DGStream

# 모델 파일 경로 확인
ls test/flame_steak_suite/frame000000/point_cloud/iteration_15000/point_cloud.ply
```

## 실행 순서

### Step 1: Transport 서비스 시작

```bash
# Terminal 1 (Transport 인스턴스)
cd /home/wrl-ubuntu/workspace/HybridPipeline/backend/src
python server-fifo.py
```

**예상 출력:**
```
Created socket directory: /run/ipc
Transport WebSocket server listening on ws://0.0.0.0:8765
Waiting for Renderer to connect to Unix sockets...
```

### Step 2: Renderer 서비스 시작

```bash
# Terminal 2 (Renderer 인스턴스)
cd /workspace/research/3DGStream
python feed-forward-renderer-socket.py
```

**예상 출력:**
```
Loading Gaussian Scene from test/flame_steak_suite/...
Gaussian Scene loaded and uploaded to GPU
Connecting to Transport...
⏳ Waiting for camera socket... (1/30)
⏳ Waiting for video socket... (1/30)
```

**Note:** Renderer는 Transport의 소켓이 준비될 때까지 대기합니다.

### Step 3: 테스트 클라이언트 실행

```bash
# Terminal 3 (Transport 또는 별도 머신)
cd /home/wrl-ubuntu/workspace/HybridPipeline/backend
python test_feedforward.py
```

**예상 출력:**
```
============================================================
Feedforward Pipeline Test
============================================================
Target: ws://localhost:8765/ws/feedforward
Resolution: 1280x720
Frames: 100
============================================================

Connecting to ws://localhost:8765/ws/feedforward...
✅ Connected to ws://localhost:8765/ws/feedforward
📤 Sent handshake: 1280x720
📹 Starting camera data transmission...
📺 Starting video reception...
📤 Sent camera frame 0, time_index=0.000
📺 Received frame 1: 45678 bytes, total_latency=125.3ms, transport=2.1ms
📤 Sent camera frame 10, time_index=0.101
📺 Received frame 10: 46234 bytes, total_latency=128.7ms, transport=1.8ms
...
✅ Received all 100 frames, stopping...

=== Test Summary ===
Sent frames: 100
Received frames: 100
Success rate: 100.0%

💾 Video saved to test_output.h264
🔌 Disconnected

============================================================
Test completed!
Check test_output.h264 for received video
Play with: ffplay test_output.h264
============================================================
```

## 결과 확인

### 비디오 재생

```bash
# ffplay로 재생
ffplay test_output.h264

# 또는 ffmpeg로 정보 확인
ffprobe test_output.h264
```

### 로그 확인

**Transport (server-fifo.py):**
```
Connection opened from ('127.0.0.1', 54321)
✅ Session created without encoder for ('127.0.0.1', 54321) (1280x720)
/ws/feedforward
[+] Feedforward mode started for ('127.0.0.1', 54321)
Receive loop started for ('127.0.0.1', 54321)
Camera server listening on /run/ipc/camera.sock
Video server listening on /run/ipc/video.sock
Camera client connected: None
Video client connected: None
[Video] Frame 0: 45678 bytes, time_index=0.0000
[Video] Frame 60: 46123 bytes, time_index=0.6061
```

**Renderer (feed-forward-renderer-socket.py):**
```
✅ Connected to camera socket
✅ Connected to video socket
🚀 Starting render loop...
Camera receive loop started
Render and send loop started
[Render] Frame 0: 45678 bytes, time_index=0.000
[Render] Frame 60: 46123 bytes, time_index=0.606
```

## 문제 해결

### 1. Connection refused (Transport → Renderer)

**증상:**
```
⏳ Waiting for camera socket... (30/30)
❌ Failed to connect to camera socket after 30 retries
```

**해결:**
- Transport (server-fifo.py)가 먼저 실행되었는지 확인
- `/run/ipc/` 디렉토리 권한 확인: `ls -la /run/ipc/`

### 2. WebSocket connection failed

**증상:**
```
❌ Test failed: [Errno 111] Connection refused
```

**해결:**
- Transport가 8765 포트에서 리스닝 중인지 확인: `netstat -tlnp | grep 8765`
- 방화벽 확인

### 3. No frames received

**증상:**
```
Sent frames: 100
Received frames: 0
Success rate: 0.0%
```

**해결:**
- Renderer 로그에서 에러 확인
- GPU 메모리 확인: `nvidia-smi`
- Gaussian 모델 파일 경로 확인

### 4. NVENC error (Transport)

**증상:**
```
RuntimeError: Failed to load NVENC library
```

**해결:**
- 이미 해결됨! `session.py`에서 `use_encoder=False` 사용
- Transport는 GPU 불필요

## 성능 모니터링

### 레이턴시 측정

test_feedforward.py 출력에서:
- `total_latency`: Frontend → Transport → Renderer → Transport → Frontend
- `transport`: Transport 내부 처리 시간

### 프레임 레이트

```bash
# 실시간 FPS 확인 (Transport 로그에서)
grep "Video.*Frame" | tail -20
```

## 커스터마이징

### 해상도 변경

```python
# test_feedforward.py
WIDTH = 1920
HEIGHT = 1080
```

### 프레임 수 변경

```python
# test_feedforward.py
NUM_FRAMES = 300  # 더 많은 프레임 테스트
```

### 카메라 움직임 추가

```python
# test_feedforward.py의 send_camera_loop() 함수에서
view_matrix[0, 3] = np.sin(frame_id * 0.1)  # X축 이동
view_matrix[1, 3] = np.cos(frame_id * 0.1)  # Y축 이동
```

## 다음 단계

테스트 성공 후:
1. 실제 Frontend 통합
2. Docker Compose로 배포
3. 성능 최적화
4. 에러 복구 로직 추가
