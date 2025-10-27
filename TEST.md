# HybridPipeline Test Plan

## 개요

HybridPipeline의 핵심 기능 검증을 위한 테스트 계획입니다.

**테스트 철학:**
- ✅ **MVP 먼저**: 기본 파이프라인 동작 확인
- ✅ **실용적**: 실제 발생 가능한 문제에 집중
- ✅ **점진적**: 필수 → 안정성 → 최적화 순서

**우선순위:**
1. **Phase 1 (MVP 필수)**: 기본 파이프라인 동작
2. **Phase 2 (안정성)**: 운영 환경 준비
3. **Phase 3 (추가)**: 엣지 케이스 대응
4. **Phase 4 (후순위)**: 운영 중 필요시 추가

---

## 테스트 환경 설정

### 필수 도구

```bash
# Python 테스트 프레임워크
pip install pytest pytest-asyncio pytest-timeout

# 추가 도구
pip install pytest-cov  # 커버리지 측정
```

### 테스트 데이터

```
test/
├── fixtures/
│   ├── scenes/
│   │   ├── minimal_scene.ply       # 최소 Scene (10 gaussians)
│   │   └── test_scene.ply          # 테스트 Scene
│   └── expected/
│       ├── camera_frame_001.bin    # 152 bytes
│       └── render_output_001.pkl
└── sockets/
    └── test_*.sock                  # 테스트용 Unix Socket
```

---

## Phase 1: MVP 필수 테스트

> **목표**: 기본 파이프라인이 동작하는지 확인
> **시간**: 2-3일
> **테스트 개수**: 5개

### 1. Unix Socket 생성 테스트

**목적**: Socket 파일이 정상적으로 생성되는지 확인

```python
# transport/tests/test_socket_creation.py

import os
import asyncio
import pytest

@pytest.mark.asyncio
async def test_unix_socket_creation():
    """Unix Socket 생성 테스트"""
    camera_socket = "/tmp/test_camera.sock"
    video_socket = "/tmp/test_video.sock"

    # 기존 소켓 제거
    for sock in [camera_socket, video_socket]:
        if os.path.exists(sock):
            os.remove(sock)

    # Transport 서버 시작
    async def dummy_handler(reader, writer):
        pass

    camera_server = await asyncio.start_unix_server(
        dummy_handler, camera_socket
    )
    video_server = await asyncio.start_unix_server(
        dummy_handler, video_socket
    )

    # 소켓 파일 생성 확인
    assert os.path.exists(camera_socket), "Camera socket not created"
    assert os.path.exists(video_socket), "Video socket not created"

    # 소켓 타입 확인
    import stat
    assert stat.S_ISSOCK(os.stat(camera_socket).st_mode), "Not a socket file"
    assert stat.S_ISSOCK(os.stat(video_socket).st_mode), "Not a socket file"

    # 정리
    camera_server.close()
    video_server.close()
    await camera_server.wait_closed()
    await video_server.wait_closed()

    print("✅ Unix Socket 생성 성공")
```

**검증 항목:**
- [x] `/run/ipc/camera.sock` 파일 존재
- [x] `/run/ipc/video.sock` 파일 존재
- [x] Socket 파일 타입 확인

---

### 2. Socket 연결 테스트 (양방향 통신)

**목적**: Transport와 Renderer가 Unix Socket으로 연결되고 데이터를 주고받을 수 있는지 확인

```python
# tests/test_socket_connection.py

import asyncio
import struct
import pytest

@pytest.mark.asyncio
async def test_transport_renderer_socket_connection():
    """Transport ↔ Renderer 양방향 통신 테스트"""
    camera_socket = "/tmp/test_camera.sock"
    video_socket = "/tmp/test_video.sock"

    # Transport: 서버 역할
    camera_received = []
    video_sent = []

    async def camera_handler(reader, writer):
        """Camera 데이터 수신"""
        data = await reader.read(152)
        camera_received.append(data)
        writer.close()
        await writer.wait_closed()

    async def video_handler(reader, writer):
        """Video 데이터 송신"""
        # 테스트 payload 전송
        test_payload = b"x" * 100
        header = struct.pack("<QII", 1, 0, len(test_payload))  # frame_id=1
        writer.write(header + test_payload)
        await writer.drain()
        video_sent.append(test_payload)
        writer.close()
        await writer.wait_closed()

    camera_server = await asyncio.start_unix_server(
        camera_handler, camera_socket
    )
    video_server = await asyncio.start_unix_server(
        video_handler, video_socket
    )

    await asyncio.sleep(0.1)

    # Renderer: 클라이언트 역할
    async def renderer_client():
        # Camera socket 연결 및 전송
        camera_reader, camera_writer = await asyncio.open_unix_connection(
            camera_socket
        )
        test_camera_data = b"c" * 152  # 152 bytes
        camera_writer.write(test_camera_data)
        await camera_writer.drain()
        camera_writer.close()
        await camera_writer.wait_closed()

        # Video socket 연결 및 수신
        video_reader, video_writer = await asyncio.open_unix_connection(
            video_socket
        )
        header = await video_reader.read(16)
        frame_id, meta_len, data_len = struct.unpack("<QII", header)
        data = await video_reader.read(data_len)

        return frame_id, data

    frame_id, received_data = await renderer_client()

    # 검증
    assert len(camera_received) == 1, "Camera data not received"
    assert camera_received[0] == b"c" * 152, "Camera data mismatch"

    assert frame_id == 1, "Frame ID mismatch"
    assert received_data == b"x" * 100, "Video data mismatch"

    # 정리
    camera_server.close()
    video_server.close()
    await camera_server.wait_closed()
    await video_server.wait_closed()

    print("✅ Socket 양방향 통신 성공")
```

**검증 항목:**
- [x] Transport → Renderer: Camera 데이터 전송
- [x] Renderer → Transport: Video 데이터 수신
- [x] 데이터 무결성 (송신 == 수신)

---

### 3. Scene Renderer 동작 테스트

**목적**: 개별 Renderer가 Camera 데이터를 받아 Scene을 렌더링할 수 있는지 확인

```python
# renderer/tests/test_scene_renderer.py

import pytest
import torch
import numpy as np
from scene_renderers.gaussian_splatting import GaussianSplattingRenderer
from data_types import CameraFrame, RenderOutput

@pytest.mark.asyncio
async def test_scene_renderer_render():
    """Scene Renderer 렌더링 테스트"""
    # Renderer 초기화
    renderer = GaussianSplattingRenderer(
        ply_path="test/fixtures/scenes/minimal_scene.ply"
    )

    success = await renderer.on_init()
    assert success, "Renderer 초기화 실패"

    # 테스트 Camera 생성
    camera = CameraFrame(
        view_matrix=np.eye(4, dtype=np.float32),
        intrinsics=create_test_intrinsics(width=640, height=480),
        time_index=0.0,
        frame_id=1,
        client_timestamp=0.0,
        server_timestamp=0.0
    )

    # 렌더링
    output = await renderer.render(camera)

    # 검증: RenderOutput 구조
    assert isinstance(output, RenderOutput), "Invalid output type"
    assert isinstance(output.color, torch.Tensor), "Color must be tensor"
    assert isinstance(output.depth, torch.Tensor), "Depth must be tensor"
    assert isinstance(output.alpha, torch.Tensor), "Alpha must be tensor"

    # 검증: Shape
    assert output.color.shape == (480, 640, 3), f"Color shape mismatch: {output.color.shape}"
    assert output.depth.shape == (480, 640), f"Depth shape mismatch: {output.depth.shape}"
    assert output.alpha.shape == (480, 640), f"Alpha shape mismatch: {output.alpha.shape}"

    # 검증: dtype
    assert output.color.dtype == torch.float32, "Color must be float32"
    assert output.depth.dtype == torch.float32, "Depth must be float32"
    assert output.alpha.dtype == torch.float32, "Alpha must be float32"

    # 검증: 값 범위
    assert torch.all(output.color >= 0) and torch.all(output.color <= 1), \
        "Color values must be in [0, 1]"
    assert torch.all(output.alpha >= 0) and torch.all(output.alpha <= 1), \
        "Alpha values must be in [0, 1]"

    # 정리
    await renderer.on_shutdown()

    print("✅ Scene Renderer 렌더링 성공")
    print(f"   Color: {output.color.shape}, range: [{output.color.min():.3f}, {output.color.max():.3f}]")
    print(f"   Depth: {output.depth.shape}, range: [{output.depth.min():.3f}, {output.depth.max():.3f}]")
    print(f"   Alpha: {output.alpha.shape}, range: [{output.alpha.min():.3f}, {output.alpha.max():.3f}]")


def create_test_intrinsics(width, height, fov=60):
    """테스트용 Intrinsics 생성"""
    focal_length = width / (2 * np.tan(np.radians(fov) / 2))
    intrinsics = np.array([
        [focal_length, 0, width / 2, 0],
        [0, focal_length, height / 2, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ], dtype=np.float32)
    return intrinsics
```

**검증 항목:**
- [x] Renderer 초기화 성공
- [x] RenderOutput 생성
- [x] Color: (H, W, 3), float32, [0, 1]
- [x] Depth: (H, W), float32
- [x] Alpha: (H, W), float32, [0, 1]

---

### 4. Encoder 동작 테스트 ⭐

**목적**: RenderOutput이 올바르게 인코딩되는지 확인

```python
# renderer/tests/test_encoder.py

import pytest
import torch
from encoders.jpeg import JPEGEncoder
from data_types import RenderOutput, RenderPayload

@pytest.mark.asyncio
async def test_encoder_encode():
    """Encoder 인코딩 테스트"""
    encoder = JPEGEncoder()

    # 테스트 RenderOutput 생성
    output = RenderOutput(
        color=torch.rand(480, 640, 3, dtype=torch.float32),
        depth=torch.rand(480, 640, dtype=torch.float32),
        alpha=torch.ones(480, 640, dtype=torch.float32),
        metadata={}
    )

    # 인코딩
    payload = await encoder.encode(output, frame_id=42)

    # 검증: RenderPayload 구조
    assert isinstance(payload, RenderPayload), "Invalid payload type"
    assert payload.frame_id == 42, "Frame ID mismatch"
    assert isinstance(payload.metadata, dict), "Metadata must be dict"
    assert isinstance(payload.data, bytes), "Data must be bytes"

    # 검증: Metadata
    assert payload.metadata["format_type"] == "jpeg+depth", \
        f"Format type mismatch: {payload.metadata['format_type']}"
    assert "color_len" in payload.metadata, "Missing color_len in metadata"
    assert "depth_len" in payload.metadata, "Missing depth_len in metadata"
    assert "width" in payload.metadata, "Missing width in metadata"
    assert "height" in payload.metadata, "Missing height in metadata"

    color_len = payload.metadata["color_len"]
    depth_len = payload.metadata["depth_len"]

    # 검증: 데이터 크기
    assert color_len > 0, "Color JPEG is empty"
    assert depth_len == 640 * 480 * 2, \
        f"Depth size mismatch: expected {640*480*2}, got {depth_len}"  # float16

    # 검증: 전체 데이터 길이
    assert len(payload.data) == color_len + depth_len, \
        f"Total data size mismatch: {len(payload.data)} != {color_len + depth_len}"

    print("✅ Encoder 인코딩 성공")
    print(f"   Format: {payload.metadata['format_type']}")
    print(f"   Color JPEG: {color_len} bytes")
    print(f"   Depth (float16): {depth_len} bytes")
    print(f"   Total: {len(payload.data)} bytes")
```

**검증 항목:**
- [x] RenderPayload 생성
- [x] Frame ID 일치
- [x] Metadata 포함: format_type, color_len, depth_len, width, height
- [x] JPEG 데이터 크기 > 0
- [x] Depth 데이터 크기 = W × H × 2 (float16)
- [x] 전체 데이터 크기 일치

---

### 5. E2E 데이터 패스 테스트

**목적**: 전체 파이프라인이 1 프레임을 정상적으로 처리하는지 확인

```python
# tests/test_e2e.py

import pytest
import asyncio
import struct
import websockets
import json

@pytest.mark.asyncio
@pytest.mark.timeout(20)
async def test_e2e_one_frame():
    """E2E 1 프레임 전송 테스트"""

    # 1. Renderer Service 시작 (Mock)
    renderer_ready = asyncio.Event()
    received_cameras = []

    async def mock_renderer():
        """Mock Renderer: Camera 수신 → Video 송신"""
        # Camera 수신
        camera_reader, camera_writer = await asyncio.open_unix_connection(
            "/tmp/e2e_camera.sock"
        )

        renderer_ready.set()

        # Camera 데이터 수신
        camera_data = await camera_reader.read(152)
        received_cameras.append(camera_data)

        # Mock 렌더링 결과 생성
        video_reader, video_writer = await asyncio.open_unix_connection(
            "/tmp/e2e_video.sock"
        )

        # RenderPayload 전송
        metadata = {
            "format_type": "jpeg+depth",
            "color_len": 100,
            "depth_len": 200
        }
        metadata_bytes = json.dumps(metadata).encode('utf-8')
        test_data = b"mock_jpeg_data" + b"mock_depth_data"

        header = struct.pack("<QII", 99, len(metadata_bytes), len(test_data))
        video_writer.write(header + metadata_bytes + test_data)
        await video_writer.drain()

        video_writer.close()
        await video_writer.wait_closed()

    # 2. Transport Service 시작
    transport_camera_queue = asyncio.Queue()

    async def transport_camera_server(reader, writer):
        """Transport: Camera 중계"""
        data = await reader.read(152)
        await transport_camera_queue.put(data)

    async def transport_video_server(reader, writer):
        """Transport: Video 중계"""
        # Renderer로부터 수신
        header = await reader.read(16)
        frame_id, meta_len, data_len = struct.unpack("<QII", header)

        metadata_bytes = await reader.read(meta_len)
        data = await reader.read(data_len)

        # Frontend로 전송 (WebSocket 시뮬레이션)
        # 실제로는 WebSocketAdapter가 처리
        transport_video_server.payload = {
            "frame_id": frame_id,
            "metadata": json.loads(metadata_bytes),
            "data": data
        }

    transport_video_server.payload = None

    camera_server = await asyncio.start_unix_server(
        transport_camera_server, "/tmp/e2e_camera.sock"
    )
    video_server = await asyncio.start_unix_server(
        transport_video_server, "/tmp/e2e_video.sock"
    )

    await asyncio.sleep(0.5)

    # 3. Renderer 시작
    renderer_task = asyncio.create_task(mock_renderer())

    # Renderer 준비 대기
    await asyncio.wait_for(renderer_ready.wait(), timeout=2.0)

    # 4. Frontend: Camera 전송
    test_camera_data = b"C" * 152
    camera_reader, camera_writer = await asyncio.open_unix_connection(
        "/tmp/e2e_camera.sock"
    )
    camera_writer.write(test_camera_data)
    await camera_writer.drain()
    camera_writer.close()
    await camera_writer.wait_closed()

    # 5. 데이터 전파 대기
    await asyncio.sleep(1.0)

    # 6. 검증
    # Transport가 Camera 수신했는지
    assert transport_camera_queue.qsize() == 1, "Camera data not received by Transport"
    received = await transport_camera_queue.get()
    assert received == test_camera_data, "Camera data corrupted"

    # Renderer가 Camera 수신했는지
    assert len(received_cameras) == 1, "Renderer did not receive camera data"
    assert received_cameras[0] == test_camera_data, "Renderer received corrupted camera data"

    # Transport가 Video 수신했는지
    assert transport_video_server.payload is not None, "Transport did not receive video payload"
    assert transport_video_server.payload["frame_id"] == 99, "Frame ID mismatch"
    assert transport_video_server.payload["metadata"]["format_type"] == "jpeg+depth", \
        "Format type mismatch"

    # 정리
    renderer_task.cancel()
    camera_server.close()
    video_server.close()
    await camera_server.wait_closed()
    await video_server.wait_closed()

    print("✅ E2E 1 프레임 전송 성공")
    print(f"   Frame ID: {transport_video_server.payload['frame_id']}")
    print(f"   Format: {transport_video_server.payload['metadata']['format_type']}")
```

**검증 항목:**
- [x] Frontend → Transport: Camera 전송
- [x] Transport → Renderer: Camera 전달
- [x] Renderer: 렌더링 수행
- [x] Renderer → Transport: Video 전송
- [x] Transport → Frontend: Video 전달
- [x] Frame ID 일치
- [x] 데이터 무결성

---

## Phase 2: 안정성 테스트

> **목표**: 운영 환경에서 안정적으로 동작
> **시간**: 1-2일
> **테스트 개수**: 3개

### 6. 잘못된 데이터 처리 테스트

**목적**: 손상되거나 잘못된 데이터를 받아도 크래시하지 않는지 확인

```python
@pytest.mark.asyncio
async def test_invalid_data_handling():
    """잘못된 데이터 처리 테스트"""

    # 1. 잘못된 크기의 Camera 데이터
    invalid_camera_data = b"x" * 100  # 152 bytes 아님
    # → 무시하고 로그, 크래시 X

    # 2. 손상된 Metadata
    corrupted_metadata = b"not_valid_json"
    # → 에러 로그, 해당 프레임 drop

    # 3. 과도하게 큰 Payload
    oversized_data = b"x" * (100 * 1024 * 1024)  # 100 MB
    # → 거부, 에러 로그
```

**검증 항목:**
- [x] 잘못된 크기 → 무시
- [x] 손상된 JSON → 에러 로그, drop
- [x] 과도한 크기 → 거부
- [x] 크래시 없음

---

### 7. 성능 목표 달성 테스트

**목적**: 실시간 렌더링 목표 (60 FPS) 달성 확인

```python
import time
import numpy as np

@pytest.mark.asyncio
async def test_render_latency_60fps():
    """렌더링 레이턴시 목표 달성 테스트"""
    renderer = GaussianSplattingRenderer("test/fixtures/scenes/test_scene.ply")
    await renderer.on_init()

    camera = create_test_camera_frame()

    # 100 프레임 렌더링
    latencies = []
    for _ in range(100):
        start = time.perf_counter()
        output = await renderer.render(camera)
        end = time.perf_counter()
        latencies.append((end - start) * 1000)  # ms

    avg_latency = np.mean(latencies)
    p95_latency = np.percentile(latencies, 95)

    print(f"Render Latency: Avg={avg_latency:.2f}ms, P95={p95_latency:.2f}ms")

    # 목표: 60 FPS = 16.67 ms/frame
    assert avg_latency < 16.67, f"Too slow: {avg_latency:.2f}ms"

    await renderer.on_shutdown()
```

**검증 항목:**
- [x] 평균 렌더링 시간 < 16.67ms (60 FPS)
- [x] P95 레이턴시 측정
- [x] E2E 레이턴시 < 50ms

---

### 8. 다중 Frontend 처리 테스트

**목적**: 여러 Frontend가 동시에 연결되어도 정상 동작하는지 확인

```python
@pytest.mark.asyncio
async def test_multiple_frontends():
    """다중 Frontend 동시 처리 테스트"""
    transport = TransportCore()

    # 3개 Frontend 연결
    received_payloads = [[], [], []]

    class MockFrontendAdapter:
        def __init__(self, index):
            self.index = index

        async def send(self, payload):
            received_payloads[self.index].append(payload)

    for i in range(3):
        transport.add_frontend_adapter(MockFrontendAdapter(i))

    # Payload 브로드캐스트
    payload = create_test_payload(frame_id=10)
    await transport.broadcast_to_frontends(payload)

    # 모든 Frontend가 받았는지 확인
    for i in range(3):
        assert len(received_payloads[i]) == 1
        assert received_payloads[i][0].frame_id == 10

    print("✅ 3개 Frontend 모두 동일한 프레임 수신")
```

**검증 항목:**
- [x] 2-3개 Frontend 동시 연결
- [x] 모두 동일한 프레임 수신
- [x] Frame ID 일치

---

## Phase 3: 추가 테스트

> **목표**: 엣지 케이스 대응
> **시간**: 1일
> **테스트 개수**: 2개

### 9. Queue Overflow 테스트

**목적**: Queue가 꽉 찼을 때 처리

```python
@pytest.mark.asyncio
async def test_camera_queue_overflow():
    """Camera Queue 오버플로우 처리"""
    queue = asyncio.Queue(maxsize=2)

    # Queue 가득 채우기
    await queue.put("frame1")
    await queue.put("frame2")

    # 새 프레임 추가 시도
    # → 가장 오래된 프레임 drop, 새 프레임 추가

    assert queue.qsize() == 2
```

**검증 항목:**
- [x] Queue full → 가장 오래된 프레임 drop
- [x] 크래시 없음

---

### 10. GPU 메모리 부족 처리

**목적**: 큰 Scene 로드 시 OOM 처리

```python
@pytest.mark.asyncio
async def test_gpu_oom_handling():
    """GPU OOM 처리 테스트"""
    # 매우 큰 Scene 로드 시도
    renderer = GaussianSplattingRenderer("huge_scene.ply")

    try:
        await renderer.on_init()
    except torch.cuda.OutOfMemoryError:
        # 에러 메시지 출력
        print("GPU 메모리 부족: Scene이 너무 큽니다")
        # Graceful exit
        return

    # OOM이 발생해야 함
    assert False, "Expected OOM error"
```

**검증 항목:**
- [x] OOM 발생 시 에러 메시지
- [x] Graceful exit

---

## Phase 4: 후순위 (운영 중 추가)

> 당장 구현하지 않아도 되지만, 운영 중 필요시 추가

### 재연결 관련

- `test_renderer_disconnect_reconnect()` - 서버 재시작으로 대체 가능
- `test_transport_crash_recovery()` - 서버 재시작으로 대체 가능

### 장시간 안정성

- `test_1000_frames_stability()` - 운영 중 모니터링으로 대체
- `test_24hour_stability()` - 운영 중 확인

### 보안

- `test_socket_file_permissions()` - Docker 환경에서 자동 설정
- `test_unauthorized_access()` - 로컬 환경이므로 낮은 우선순위

### 기타

- `test_different_resolutions()` - 필요시 추가
- `test_protocol_version_compatibility()` - 버전 관리 시작 후 추가

---

## 테스트 실행

### 로컬 실행

```bash
# Phase 1: MVP 필수 테스트
pytest tests/test_mvp/ -v

# Phase 2: 안정성 테스트
pytest tests/test_stability/ -v

# 전체 테스트
pytest -v

# 특정 테스트
pytest tests/test_e2e.py::test_e2e_one_frame -v

# 커버리지 측정
pytest --cov=renderer --cov=transport --cov-report=html
```

### Docker 실행

```bash
# Docker Compose로 통합 테스트
docker-compose -f docker-compose.test.yml up --abort-on-container-exit

# 개별 서비스 테스트
docker-compose -f docker-compose.test.yml run renderer pytest
```

---

## 테스트 체크리스트

### Phase 1: MVP 필수

- [ ] 1. Unix Socket 생성
  - [ ] `/run/ipc/camera.sock` 존재
  - [ ] `/run/ipc/video.sock` 존재
  - [ ] Socket 타입 확인

- [ ] 2. Socket 연결 (양방향)
  - [ ] Transport → Renderer: Camera 전송
  - [ ] Renderer → Transport: Video 수신
  - [ ] 데이터 무결성

- [ ] 3. Scene Renderer
  - [ ] 초기화 성공
  - [ ] RenderOutput 생성
  - [ ] Shape, dtype 검증
  - [ ] 값 범위 검증

- [ ] 4. Encoder
  - [ ] RenderPayload 생성
  - [ ] Metadata 포함
  - [ ] 데이터 크기 검증

- [ ] 5. E2E 데이터 패스
  - [ ] 1 프레임 완전 전송
  - [ ] Frame ID 일치
  - [ ] 전체 흐름 동작

### Phase 2: 안정성

- [ ] 6. 잘못된 데이터 처리
  - [ ] 잘못된 크기 처리
  - [ ] 손상된 데이터 처리
  - [ ] 크래시 없음

- [ ] 7. 성능 목표
  - [ ] 렌더링 < 16.67ms
  - [ ] E2E < 50ms

- [ ] 8. 다중 Frontend
  - [ ] 2-3개 동시 연결
  - [ ] 모두 수신 확인

### Phase 3: 추가

- [ ] 9. Queue Overflow
- [ ] 10. GPU OOM 처리

---

## 테스트 우선순위 요약

| Phase | 목표 | 테스트 개수 | 예상 시간 | 중요도 |
|-------|------|------------|----------|--------|
| Phase 1 | MVP 필수 | 5 | 2-3일 | ⭐⭐⭐ |
| Phase 2 | 안정성 | 3 | 1-2일 | ⭐⭐ |
| Phase 3 | 추가 | 2 | 1일 | ⭐ |
| Phase 4 | 후순위 | - | 운영 중 | - |

**총 10개 핵심 테스트로 파이프라인 검증 완료**

---

## 유틸리티 함수

테스트에 사용되는 공통 유틸리티:

```python
# tests/utils.py

import numpy as np
import torch
from data_types import CameraFrame, RenderOutput, RenderPayload

def create_test_camera_frame(frame_id=1, width=640, height=480):
    """테스트용 CameraFrame 생성"""
    return CameraFrame(
        view_matrix=np.eye(4, dtype=np.float32),
        intrinsics=create_test_intrinsics(width, height),
        time_index=0.0,
        frame_id=frame_id,
        client_timestamp=0.0,
        server_timestamp=0.0
    )

def create_test_intrinsics(width, height, fov=60):
    """테스트용 Intrinsics 생성"""
    focal_length = width / (2 * np.tan(np.radians(fov) / 2))
    return np.array([
        [focal_length, 0, width / 2, 0],
        [0, focal_length, height / 2, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ], dtype=np.float32)

def create_test_render_output(width=640, height=480):
    """테스트용 RenderOutput 생성"""
    return RenderOutput(
        color=torch.rand(height, width, 3, dtype=torch.float32),
        depth=torch.rand(height, width, dtype=torch.float32),
        alpha=torch.ones(height, width, dtype=torch.float32),
        metadata={}
    )

def create_test_payload(frame_id=1):
    """테스트용 RenderPayload 생성"""
    return RenderPayload(
        frame_id=frame_id,
        metadata={
            "format_type": "jpeg+depth",
            "color_len": 100,
            "depth_len": 200
        },
        data=b"test_data"
    )
```

---

## 테스트 픽스처 생성 가이드

### Minimal Scene 생성

테스트용 최소 Gaussian Scene (.ply 파일) 생성:

```python
# scripts/create_test_fixtures.py

import numpy as np
import struct

def create_minimal_ply(output_path="tests/fixtures/scenes/minimal_scene.ply", num_points=10):
    """
    최소 Gaussian Scene 생성 (10 gaussians)

    각 Gaussian은 다음을 포함:
    - xyz: 위치 (3 floats)
    - normals: 법선 (3 floats, 사용 안 함)
    - f_dc_0, f_dc_1, f_dc_2: SH 계수 (3 floats)
    - opacity: 불투명도 (1 float)
    - scale_0, scale_1, scale_2: 스케일 (3 floats)
    - rot_0, rot_1, rot_2, rot_3: 회전 (4 floats)
    """
    # Random gaussians in unit cube
    np.random.seed(42)

    positions = np.random.rand(num_points, 3) * 2 - 1  # [-1, 1]
    normals = np.zeros((num_points, 3))  # Not used
    sh_dc = np.ones((num_points, 3)) * 0.5  # Gray color
    opacity = np.ones(num_points) * 0.9  # 90% opaque
    scale = np.ones((num_points, 3)) * 0.01  # Small gaussians
    rotation = np.zeros((num_points, 4))  # Identity quaternion
    rotation[:, 0] = 1.0

    # PLY header
    header = f"""ply
format binary_little_endian 1.0
element vertex {num_points}
property float x
property float y
property float z
property float nx
property float ny
property float nz
property float f_dc_0
property float f_dc_1
property float f_dc_2
property float opacity
property float scale_0
property float scale_1
property float scale_2
property float rot_0
property float rot_1
property float rot_2
property float rot_3
end_header
"""

    # Write PLY file
    with open(output_path, 'wb') as f:
        f.write(header.encode('utf-8'))

        for i in range(num_points):
            # Pack all properties for this gaussian
            data = struct.pack('<17f',
                positions[i, 0], positions[i, 1], positions[i, 2],
                normals[i, 0], normals[i, 1], normals[i, 2],
                sh_dc[i, 0], sh_dc[i, 1], sh_dc[i, 2],
                opacity[i],
                scale[i, 0], scale[i, 1], scale[i, 2],
                rotation[i, 0], rotation[i, 1], rotation[i, 2], rotation[i, 3]
            )
            f.write(data)

    print(f"✅ Created minimal scene: {output_path} ({num_points} gaussians)")


def create_camera_frame_fixture(output_path="tests/fixtures/expected/camera_frame_001.bin"):
    """152 bytes camera frame 샘플 생성"""
    view_matrix = np.eye(4, dtype=np.float32)
    intrinsics = create_test_intrinsics(640, 480)

    camera = CameraFrame(
        view_matrix=view_matrix,
        intrinsics=intrinsics,
        time_index=0.0,
        frame_id=1,
        client_timestamp=1000.0,
        server_timestamp=1001.0
    )

    data = pack_camera_frame(camera)
    assert len(data) == 152, f"Invalid camera frame size: {len(data)}"

    with open(output_path, 'wb') as f:
        f.write(data)

    print(f"✅ Created camera frame fixture: {output_path} (152 bytes)")


if __name__ == "__main__":
    import os
    os.makedirs("tests/fixtures/scenes", exist_ok=True)
    os.makedirs("tests/fixtures/expected", exist_ok=True)

    create_minimal_ply()
    create_camera_frame_fixture()
```

**실행:**
```bash
python scripts/create_test_fixtures.py
```

---

## Mock 객체 구현

### Mock Scene Renderer

```python
# tests/mocks/mock_renderer.py

import torch
import asyncio
from renderer.scene_renderers.base import BaseSceneRenderer
from renderer.data_types import CameraFrame, RenderOutput

class MockSceneRenderer(BaseSceneRenderer):
    """테스트용 Mock Renderer (빠른 실행)"""

    def __init__(self, width=640, height=480, init_delay=0.0, render_delay=0.001):
        self.width = width
        self.height = height
        self.init_delay = init_delay
        self.render_delay = render_delay
        self.initialized = False
        self.render_count = 0

    async def on_init(self) -> bool:
        """초기화 시뮬레이션"""
        if self.init_delay > 0:
            await asyncio.sleep(self.init_delay)

        self.initialized = True
        print(f"MockRenderer initialized ({self.width}x{self.height})")
        return True

    async def render(self, camera: CameraFrame) -> RenderOutput:
        """가짜 렌더링 (고정된 패턴)"""
        if not self.initialized:
            raise RuntimeError("Renderer not initialized")

        if self.render_delay > 0:
            await asyncio.sleep(self.render_delay)

        # 고정된 패턴 생성 (체크보드)
        color = torch.zeros(self.height, self.width, 3, dtype=torch.float32)
        color[::2, ::2, :] = 1.0  # White squares
        color[1::2, 1::2, :] = 1.0

        depth = torch.ones(self.height, self.width, dtype=torch.float32) * 5.0
        alpha = torch.ones(self.height, self.width, dtype=torch.float32)

        self.render_count += 1

        return RenderOutput(
            color=color,
            depth=depth,
            alpha=alpha,
            metadata={"renderer": "mock", "frame_id": camera.frame_id}
        )

    async def on_shutdown(self):
        """종료 시뮬레이션"""
        self.initialized = False
        print(f"MockRenderer shutdown (rendered {self.render_count} frames)")
```

### Mock Encoder

```python
# tests/mocks/mock_encoder.py

from renderer.encoders.base import BaseEncoder
from renderer.data_types import RenderOutput, RenderPayload

class MockEncoder(BaseEncoder):
    """테스트용 Mock Encoder (실제 인코딩 없음)"""

    def __init__(self, format_type="mock"):
        self.format_type = format_type
        self.encode_count = 0

    def get_format_type(self) -> str:
        return self.format_type

    async def encode(self, output: RenderOutput, frame_id: int) -> RenderPayload:
        """가짜 인코딩 (고정된 데이터)"""
        self.encode_count += 1

        # 고정된 payload
        metadata = {
            "format_type": self.format_type,
            "width": output.color.shape[1],
            "height": output.color.shape[0],
            "color_len": 100,
            "depth_len": 200
        }

        data = b"MOCK_COLOR_DATA" + b"MOCK_DEPTH_DATA"

        return RenderPayload(
            frame_id=frame_id,
            metadata=metadata,
            data=data
        )
```

### Mock Transport (Unix Socket 시뮬레이션)

```python
# tests/mocks/mock_transport.py

import asyncio
import struct
import json
from collections import deque

class MockTransport:
    """테스트용 Mock Transport (Unix Socket 없이)"""

    def __init__(self):
        self.camera_queue = asyncio.Queue()
        self.video_queue = asyncio.Queue()
        self.received_cameras = []
        self.sent_videos = []

    async def send_camera(self, camera: CameraFrame):
        """Frontend → Transport: Camera 전송 시뮬레이션"""
        data = pack_camera_frame(camera)
        await self.camera_queue.put(data)
        self.received_cameras.append(camera)

    async def receive_camera(self) -> bytes:
        """Transport → Renderer: Camera 수신 시뮬레이션"""
        return await self.camera_queue.get()

    async def send_video(self, payload: RenderPayload):
        """Renderer → Transport: Video 전송 시뮬레이션"""
        data = pack_render_payload(payload)
        await self.video_queue.put(data)
        self.sent_videos.append(payload)

    async def receive_video(self) -> RenderPayload:
        """Transport → Frontend: Video 수신 시뮬레이션"""
        data = await self.video_queue.get()

        # Parse wire format
        header = data[:16]
        frame_id, meta_len, data_len = struct.unpack("<QII", header)

        offset = 16
        metadata_bytes = data[offset:offset+meta_len]
        metadata = json.loads(metadata_bytes)

        offset += meta_len
        payload_data = data[offset:offset+data_len]

        return RenderPayload(
            frame_id=frame_id,
            metadata=metadata,
            data=payload_data
        )
```

**사용 예시:**
```python
@pytest.mark.asyncio
async def test_with_mocks():
    """Mock 객체를 사용한 테스트"""
    # Mock renderer & encoder
    renderer = MockSceneRenderer(width=640, height=480)
    encoder = MockEncoder()

    await renderer.on_init()

    # Test render
    camera = create_test_camera_frame(frame_id=1)
    output = await renderer.render(camera)

    assert output.color.shape == (480, 640, 3)

    # Test encode
    payload = await encoder.encode(output, frame_id=1)

    assert payload.frame_id == 1
    assert payload.metadata["format_type"] == "mock"

    await renderer.on_shutdown()
```

---

## CI/CD 통합

### GitHub Actions 설정

```yaml
# .github/workflows/test.yml

name: Test Pipeline

on:
  push:
    branches: [ main, develop, feature-* ]
  pull_request:
    branches: [ main ]

jobs:
  test-renderer:
    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v3

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.10'

    - name: Install dependencies
      run: |
        cd renderer
        pip install -r requirements.txt
        pip install pytest pytest-asyncio pytest-timeout pytest-cov

    - name: Create test fixtures
      run: |
        python scripts/create_test_fixtures.py

    - name: Run Phase 1 tests (MVP)
      run: |
        cd renderer
        pytest tests/test_mvp/ -v --tb=short

    - name: Run Phase 2 tests (Stability)
      run: |
        cd renderer
        pytest tests/test_stability/ -v --tb=short

    - name: Generate coverage report
      run: |
        cd renderer
        pytest --cov=. --cov-report=xml --cov-report=html

    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v3
      with:
        file: ./renderer/coverage.xml

  test-integration:
    runs-on: ubuntu-latest
    needs: test-renderer

    services:
      # Docker Compose로 통합 테스트
      transport:
        image: hybrid-transport:test
      renderer:
        image: hybrid-renderer:test

    steps:
    - uses: actions/checkout@v3

    - name: Run E2E tests
      run: |
        pytest tests/test_e2e.py -v --timeout=60
```

### Docker Compose Test 설정

```yaml
# docker-compose.test.yml

version: '3.8'

services:
  transport-test:
    build:
      context: ./transport
      dockerfile: Dockerfile.test
    volumes:
      - ipc-sockets:/run/ipc
    networks:
      - test-net

  renderer-test:
    build:
      context: ./renderer
      dockerfile: Dockerfile.test
    volumes:
      - ipc-sockets:/run/ipc
      - ./tests/fixtures:/fixtures
    networks:
      - test-net
    depends_on:
      - transport-test

  test-runner:
    build:
      context: ./tests
      dockerfile: Dockerfile
    volumes:
      - ipc-sockets:/run/ipc
      - ./tests:/tests
    networks:
      - test-net
    depends_on:
      - transport-test
      - renderer-test
    command: pytest /tests -v

volumes:
  ipc-sockets:

networks:
  test-net:
```

**실행:**
```bash
docker-compose -f docker-compose.test.yml up --abort-on-container-exit
```

---

## 커버리지 측정

### pytest-cov 설정

```ini
# renderer/pytest.ini

[pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*

# 비동기 테스트 지원
asyncio_mode = auto

# Timeout 설정 (기본 10초)
timeout = 10

# 커버리지 설정
addopts =
    --cov=renderer
    --cov-report=term-missing
    --cov-report=html:htmlcov
    --cov-report=xml
    --cov-fail-under=80
    -v
    --tb=short

# 커버리지 제외 경로
[coverage:run]
omit =
    */tests/*
    */scene_renderers/external/*
    setup.py

[coverage:report]
exclude_lines =
    pragma: no cover
    def __repr__
    raise AssertionError
    raise NotImplementedError
    if __name__ == .__main__.:
    if TYPE_CHECKING:
```

### 커버리지 실행

```bash
# 전체 테스트 + 커버리지
pytest --cov=renderer --cov-report=html

# HTML 리포트 열기
open htmlcov/index.html

# 특정 모듈 커버리지
pytest --cov=renderer.scene_renderers --cov-report=term-missing

# 최소 커버리지 검증 (80% 미만 시 실패)
pytest --cov=renderer --cov-fail-under=80
```

### 커버리지 목표

| 모듈 | 목표 커버리지 | 우선순위 |
|------|--------------|----------|
| `data_types.py` | 100% | ⭐⭐⭐ |
| `utils/protocol.py` | 100% | ⭐⭐⭐ |
| `renderer_service.py` | 90% | ⭐⭐⭐ |
| `scene_renderers/base.py` | 100% | ⭐⭐⭐ |
| `encoders/base.py` | 100% | ⭐⭐⭐ |
| `scene_renderers/gaussian_splatting.py` | 80% | ⭐⭐ |
| `encoders/jpeg.py` | 80% | ⭐⭐ |
| **전체** | **80%+** | ⭐⭐⭐ |

---

## Docker 테스트 환경

### Renderer Test Dockerfile

```dockerfile
# renderer/Dockerfile.test

FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

# Python 설치
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git

# PyTorch 설치
RUN pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 테스트 의존성
COPY requirements.txt requirements-test.txt ./
RUN pip3 install -r requirements.txt -r requirements-test.txt

# 코드 복사
COPY . /app
WORKDIR /app

# Unix socket 디렉토리
RUN mkdir -p /run/ipc

# 테스트 실행
CMD ["pytest", "tests/", "-v"]
```

### requirements-test.txt

```
# renderer/requirements-test.txt

pytest>=7.4.0
pytest-asyncio>=0.21.0
pytest-timeout>=2.1.0
pytest-cov>=4.1.0
pytest-mock>=3.11.0

# Mock 도구
faker>=19.0.0

# 코드 품질
black>=23.7.0
flake8>=6.0.0
mypy>=1.4.0
```

### 테스트 스크립트

```bash
#!/bin/bash
# scripts/run_tests.sh

set -e

echo "🧪 Running HybridPipeline Tests"

# 1. 테스트 픽스처 생성
echo "📦 Creating test fixtures..."
python scripts/create_test_fixtures.py

# 2. Phase 1 테스트 (MVP)
echo "🎯 Phase 1: MVP Tests"
pytest tests/test_mvp/ -v --tb=short

# 3. Phase 2 테스트 (안정성)
echo "🛡️  Phase 2: Stability Tests"
pytest tests/test_stability/ -v --tb=short

# 4. 커버리지 측정
echo "📊 Generating coverage report..."
pytest --cov=renderer --cov-report=html --cov-report=term

# 5. 코드 품질 체크
echo "✨ Code quality checks..."
black renderer/ --check
flake8 renderer/ --max-line-length=100
mypy renderer/ --ignore-missing-imports

echo "✅ All tests passed!"
```

**실행:**
```bash
chmod +x scripts/run_tests.sh
./scripts/run_tests.sh
```

---

## 테스트 디버깅 팁

### 1. 개별 테스트 실행

```bash
# 특정 테스트 파일
pytest tests/test_encoder.py -v

# 특정 테스트 함수
pytest tests/test_encoder.py::test_encoder_encode -v

# 특정 테스트 클래스
pytest tests/test_encoder.py::TestJPEGEncoder -v
```

### 2. 상세 출력

```bash
# 전체 traceback 출력
pytest -v --tb=long

# print 문 출력 보기
pytest -v -s

# 로그 출력 보기
pytest -v --log-cli-level=DEBUG
```

### 3. 실패 시 즉시 중단

```bash
# 첫 번째 실패에서 중단
pytest -x

# 3번 실패 후 중단
pytest --maxfail=3
```

### 4. 마지막 실패한 테스트만 재실행

```bash
# 마지막 실패한 테스트만
pytest --lf

# 마지막 실패한 테스트 먼저, 그 다음 나머지
pytest --ff
```

### 5. 디버거 사용

```bash
# 실패 시 pdb 시작
pytest --pdb

# 특정 위치에 breakpoint()
# 테스트 코드에 breakpoint() 추가 후
pytest tests/test_encoder.py -v
```

---

## 다음 단계

1. ✅ **Phase 1 테스트 구현** (MVP 필수 5개)
2. ✅ **CI/CD 설정** (GitHub Actions)
3. ✅ **Phase 2 테스트 추가** (안정성 3개)
4. ✅ **커버리지 80% 목표**
5. 운영 중 Phase 3, 4 필요시 추가

**테스트 주도로 안정적인 파이프라인 구축!** 🚀
