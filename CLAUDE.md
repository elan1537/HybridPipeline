# HybridPipeline - Project Memory

## 프로젝트 개요

**목표**: 실시간 3D Gaussian Splatting 렌더링 시스템 구축

**핵심 기능**:
- 다양한 렌더러 지원 (3DGS, 4DGS/3DGStream)
- 실시간 스트리밍 (WebSocket)
- 인코딩 지원 (JPEG+Depth, H.264, Raw)
- 확장 가능한 아키텍처 (Protocol Adapter Pattern)
- Feed-forward 렌더링 (프레임별 PLY 동적 로딩, 시간 제어)

---

## 아키텍처

### 3-Tier 구조

```
Frontend (Browser/TypeScript)
    ↕ WebSocket (Camera: 160 bytes, Video: 44 bytes header)
Transport Service (Python/asyncio)
    ↕ Unix Socket (Camera: 168 bytes, Video: 56 bytes header)
Renderer Service (Python/PyTorch/CUDA)
```

### 설계 원칙

1. **관심사 분리 (Separation of Concerns)**
   - Transport: 데이터 전송만 담당 (프로토콜 변환)
   - Renderer: 렌더링 + 인코딩만 담당

2. **조합 가능성 (Composability)**
   ```python
   RendererService(
       scene_renderer=GaussianSplattingRenderer(...),
       encoder=JPEGEncoder()
   )
   ```

3. **확장 가능성 (Extensibility)**
   - BaseSceneRenderer 인터페이스로 새로운 렌더러 추가
   - BaseEncoder 인터페이스로 새로운 인코더 추가
   - Protocol Adapter Pattern으로 WebSocket/WebRTC/UDP 지원

### Frontend 아키텍처 (Application + Systems 패턴)

```
Application (core/Application.ts)
  ├─ WebSocketSystem: WebSocket 통신 및 디코딩
  ├─ CameraController: 카메라 제어 및 상태 관리
  ├─ RenderingSystem: 렌더링 루프 및 모드 전환
  ├─ TextureManager: 텍스처 생성 및 업데이트
  ├─ UISystem: UI 컴포넌트 관리
  └─ PhysicsSystem: Mesh-Gaussian 충돌 감지

UISystem
  ├─ panels/: ControlPanel, DebugPanel, StatsPanel, FPSTestPanel, RecordingPanel
  ├─ managers/: CameraStateManager, SettingsManager
  └─ components/: FrameScrubber, TimeControlUI
```

**설계 원칙**:
- **명확한 책임 분리**: 각 System은 단일 책임 원칙 준수
- **느슨한 결합**: System 간 의존성 최소화
- **높은 응집도**: 관련 기능을 System 내부에 캡슐화
- **확장 가능성**: 새로운 System 추가가 용이

---

## 프로토콜 명세

### Camera Frame (Frontend ↔ Renderer)

**Frontend → Transport (160 bytes)**:
- eye (3 floats), target (3 floats), intrinsics (9 floats), unused (17 floats)
- frame_id (uint32), client_timestamp (float64), time_index (float32)

**Transport → Renderer (168 bytes)**:
- view_matrix (4×4, 64 bytes)
- intrinsics (3×3, 36 bytes)
- width, height, near, far (16 bytes)
- frame_id, timestamps, time_index (28 bytes)
- reserved (24 bytes)

### Video Payload (Renderer → Frontend)

**Renderer → Transport (56 bytes header + data)**:
```
Offset  Size  Type     Field
------  ----  -------  ------------------
0       4     uint32   frame_id
4       1     uint8    format_type (0=JPEG, 1=H264, 2=Raw)
5       3     -        padding
8       4     uint32   color_len (or video_len)
12      4     uint32   depth_len (or 0)
16      4     uint32   width
20      4     uint32   height
24      8     float64  client_timestamp
32      8     float64  server_timestamp
40      8     float64  render_start_timestamp
48      8     float64  encode_end_timestamp
56      var   bytes    data (color + depth or video)
```

**Transport → Frontend (44 bytes header + data)**:
```
JPEG format:
0-3:   jpegLen (uint32)
4-7:   depthLen (uint32)
8-11:  frameId (uint32)
12-19: clientSendTime (float64)
20-27: serverReceiveTime (float64)
28-35: serverProcessEndTime (float64)
36-43: serverSendTime (float64)

H.264/Raw format:
0-3:   videoLen (uint32)
4-7:   frameId (uint32)
8-15:  clientSendTime (float64)
16-23: serverReceiveTime (float64)
24-31: serverProcessEndTime (float64)
32-39: serverSendTime (float64)
```

---

## 구현 상태

### ✅ Phase 1: MVP Complete (2025-10-22)

#### Renderer Service

**Scene Renderers**:
- ✅ `static_gaussian.py`: GsplatRenderer (gsplat 라이브러리)
- ✅ `streamable_gaussian.py`: StreamableGaussian (3DGStream + NTC)
  - Training mode: NTC 기반 실시간 최적화
  - Inference mode: 프레임별 PLY 동적 로딩 (feed-forward)
- ✅ `base.py`: BaseSceneRenderer 추상 인터페이스

**Encoders**:
- ✅ `jpeg.py`: JPEG + Float16 Depth (~3MB/frame, nvimgcodec/OpenCV)
- ✅ `h264.py`: H.264 color+depth stack (~145KB/frame, NVENC)
- ✅ `raw.py`: torch.save lossless (~18MB/frame)
- ✅ `base.py`: BaseEncoder 추상 인터페이스

**Core Service**:
- ✅ `renderer_service.py`: Unix Socket client, 비동기 렌더링 루프
- ✅ `main.py`: CLI 진입점, factory pattern
- ✅ `data_types.py`: CameraFrame, RenderOutput, RenderPayload
- ✅ `utils/protocol.py`: 168/56 bytes binary protocol
- ✅ `utils/frame_buffer.py`: FIFO/Latest buffer

**디버깅 지원**:
- ✅ `--save-debug-output`: 렌더링 결과 저장 (`backend/renderer/output/`)

#### Transport Service

**Core Service**:
- ✅ `service.py`: TransportService 메인 로직
- ✅ `main.py`: CLI 진입점
- ✅ `protocol_converter.py`: Frontend(160) ↔ Renderer(168) 변환

**Protocol Adapters**:
- ✅ `adapters/base.py`: BaseFrontendAdapter, BaseBackendAdapter
- ✅ `adapters/websocket_adapter.py`: WebSocket server (Frontend)
- ✅ `adapters/unix_socket_adapter.py`: Unix Socket server (Renderer)

**주요 기능**:
- ✅ Frontend 자동 재연결 지원
- ✅ torch 의존성 제거 (numpy 사용)
- ✅ format_type 정수화 (0=JPEG, 1=H264, 2=Raw)

**디버깅 지원**:
- ✅ `--save-debug-input`: 수신 데이터 저장 (`backend/transport/input/`)

#### Test Client

- ✅ `test_client.py`: Mock WebSocket client
- ✅ 카메라 데이터 생성 (circular motion)
- ✅ 비디오 수신 및 저장 (`output/`)

#### E2E 통합 테스트

**테스트 결과 (2025-10-22)**:
```
Sent:     10 frames
Received: 10 frames
Success:  100.0%
```

**검증 완료**:
- ✅ Client → Transport → Renderer (카메라 전송)
- ✅ Renderer → Transport → Client (비디오 수신)
- ✅ 모든 계층에서 데이터 저장 완료
- ✅ 프레임 손실 없음 (100% success)

**저장 위치**:
- `backend/renderer/output/`: Renderer 렌더링 결과
- `backend/transport/input/`: Transport 수신 데이터
- `output/`: Client 수신 데이터

#### Frontend (Browser/TypeScript)

**Core Systems**:
- ✅ `Application.ts`: 메인 애플리케이션 및 System 관리
- ✅ `WebSocketSystem.ts`: WebSocket 통신 및 비디오 디코딩
- ✅ `CameraController.ts`: 카메라 제어 및 상태 관리
- ✅ `RenderingSystem.ts`: 렌더링 루프 및 모드 전환
- ✅ `TextureManager.ts`: 텍스처 생성 및 업데이트
- ✅ `UISystem.ts`: UI 컴포넌트 관리
- ✅ `PhysicsSystem.ts`: Mesh-Gaussian 충돌 감지

**Render Modes**:
- ✅ `fusion`: Gaussian + Local Mesh 합성
- ✅ `gaussian-only`: Gaussian Splatting만 표시
- ✅ `local-only`: Local Mesh만 표시
- ✅ `depth-fusion`: Depth 기반 합성
- ✅ `feed-forward`: 프레임별 렌더링 (시간 제어)

**UI Components**:
- ✅ `panels/`: ControlPanel, DebugPanel, StatsPanel, FPSTestPanel, RecordingPanel
- ✅ `managers/`: CameraStateManager, SettingsManager
- ✅ `ui/`: FrameScrubber, TimeControlUI
- ✅ `physics/`: MeshGaussianCollision

**주요 기능**:
- ✅ 실시간/Feed-forward 모드 전환
- ✅ 프레임 Scrubbing (시간 제어)
- ✅ 카메라 상태 저장/불러오기
- ✅ 화면 녹화 및 FPS 측정
- ✅ WebSocket 자동 재연결

---

## 사용 방법

### Renderer Service 실행

```bash
# Static Gaussian (3DGS)
python -m renderer.main \
  --scene-type static \
  --model-path /path/to/scene.ply \
  --encoder-type jpeg \
  --save-debug-output

# Streamable Gaussian (Training mode)
python -m renderer.main \
  --scene-type streamable \
  --model-path /path/to/scene.ply \
  --ntc-path /path/to/ntc.pth \
  --ntc-config /path/to/config.json \
  --encoder-type h264

# Streamable Gaussian (Inference mode - Feed-forward)
python -m renderer.main \
  --scene-type streamable \
  --model-path /path/to/initial_scene.ply \
  --weight-path-pattern '/path/to/checkpoints/frame_{frame_id:06d}/gaussian.ply' \
  --checkpoints-dir /path/to/checkpoints \
  --encoder-type h264

# 옵션
--buffer-type latest|fifo       # Frame buffer strategy
--width 1280 --height 720       # Rendering resolution
--jpeg-quality 90               # JPEG quality (0-100)
--save-debug-output             # Save rendered images
--debug-output-dir DIR          # Debug output directory
```

### Transport Service 실행

```bash
python -m transport.main \
  --host 0.0.0.0 \
  --port 8765 \
  --camera-sock /run/ipc/camera.sock \
  --video-sock /run/ipc/video.sock \
  --width 640 --height 480 \
  --save-debug-input \
  --debug-input-dir backend/transport/input
```

### Test Client 실행

```bash
python backend/test_client.py \
  --host localhost \
  --port 8765 \
  --frames 10 \
  --output-dir output \
  --width 640 --height 480
```

---

## 파일 구조

```
backend/
├── renderer/
│   ├── main.py                         # CLI 진입점
│   ├── renderer_service.py             # 메인 서비스
│   ├── data_types.py                   # 데이터 타입
│   ├── scene_renderers/
│   │   ├── base.py                     # 추상 인터페이스
│   │   ├── static_gaussian.py          # GsplatRenderer
│   │   ├── streamable_gaussian.py      # StreamableGaussian
│   │   ├── gaussian_state.py           # FastGaussianState
│   │   ├── optimization_params.py      # Optimization parameters
│   │   └── external/
│   │       └── 3DGStream/              # 3DGStream submodule
│   ├── encoders/
│   │   ├── base.py                     # 추상 인터페이스
│   │   ├── jpeg.py                     # JPEG encoder
│   │   ├── h264.py                     # H.264 encoder
│   │   └── raw.py                      # Raw encoder
│   └── utils/
│       ├── protocol.py                 # Binary protocol (168/56 bytes)
│       ├── frame_buffer.py             # FIFO/Latest buffer
│       ├── depth_utils.py              # Depth encoding utilities
│       └── color_utils.py              # Color conversion utilities
│
├── transport/
│   ├── main.py                         # CLI 진입점
│   ├── service.py                      # TransportService
│   ├── protocol_converter.py           # Frontend ↔ Renderer 변환
│   └── adapters/
│       ├── base.py                     # 추상 인터페이스
│       ├── websocket_adapter.py        # WebSocket server
│       └── unix_socket_adapter.py      # Unix Socket server
│
└── test_client.py                      # Mock WebSocket client

frontend/
├── src/
│   ├── core/
│   │   └── Application.ts              # 메인 애플리케이션
│   ├── systems/
│   │   ├── WebSocketSystem.ts          # WebSocket 통신
│   │   ├── CameraController.ts         # 카메라 제어
│   │   ├── RenderingSystem.ts          # 렌더링 루프
│   │   ├── TextureManager.ts           # 텍스처 관리
│   │   ├── UISystem.ts                 # UI 컴포넌트 관리
│   │   └── PhysicsSystem.ts            # 충돌 감지
│   ├── ui/
│   │   ├── panels/                     # UI 패널들
│   │   ├── managers/                   # 상태 관리자들
│   │   ├── FrameScrubber.ts            # 프레임 scrubber
│   │   └── TimeControlUI.ts            # 시간 제어 UI
│   ├── physics/
│   │   └── MeshGaussianCollision.ts    # 충돌 감지
│   ├── main.ts                         # 진입점
│   └── types/index.ts                  # 타입 정의
└── index.html                          # HTML 진입점
```

---

## 성능 목표

- **렌더링 시간**: < 16.67ms (60 FPS)
- **E2E 레이턴시**: < 50ms (Client → Renderer → Client)
- **처리량**: 60 FPS @ 1280×720
- **JPEG 인코딩**: < 5ms

---

## 로깅 전략

**성능 최적화를 위한 로그 출력 빈도 제한**:
- 처음 3개 프레임: 모두 출력 (초기 동작 확인)
- 이후 60프레임마다: 1번 출력 (장기 실행 모니터링)

```python
if frame_count % 60 == 0 or frame_count < 3:
    print(f"[DEBUG] Frame {frame_id}: ...")
```

**이유**:
- 60 FPS에서 매 프레임 로그 출력 시 I/O 병목 발생
- 1시간 실행 시 216,000줄 → 2,160줄로 축소 (100배 감소)
- 모든 프레임은 파일로 저장되어 사후 분석 가능

---

## 의존성

### Python 패키지

```
# Renderer Service
torch>=2.5.1+cu121
numpy
opencv-python
gsplat
nvimgcodec  # Optional: GPU JPEG encoding
PyNvVideoCodec  # Optional: GPU H.264 encoding

# Transport Service
numpy
websockets

# Test Client
websockets
```

### Docker

- Base: `nvidia/cuda:12.1.0-devel-ubuntu22.04`
- GPU: NVIDIA GPU 필요
- Volumes: `/run/ipc` (Unix Socket 공유)

---

## 주요 결정 사항

### 1. Binary Protocol 선택
- **이유**: JSON 대비 60% 크기 감소, 파싱 속도 향상
- **트레이드오프**: 디버깅 어려움 → debug 저장 기능으로 해결

### 2. format_type 정수화
- **변경**: `"jpeg+depth"` → `0`, `"h264"` → `1`, `"raw"` → `2`
- **이유**: Protocol efficiency, type safety

### 3. numpy in Transport, torch in Renderer
- **이유**: Transport는 가벼운 numpy로 충분, Renderer만 GPU 필요
- **효과**: Transport Service 메모리 ~60MB 감소

### 4. Protocol Adapter Pattern
- **이유**: WebSocket/WebRTC/UDP 등 다양한 프로토콜 지원
- **확장성**: 새로운 adapter 추가만으로 기능 확장

### 5. Latest Frame Buffer (기본값)
- **이유**: 실시간 스트리밍에서는 최신 프레임이 중요
- **대안**: FIFO buffer (순차 처리 필요 시)

### 6. Frontend Application + Systems 패턴
- **이유**: main.ts (1481줄 감소) 복잡도 해소, 유지보수성 향상
- **효과**:
  - 각 System의 명확한 책임 분리
  - 테스트 용이성 증가
  - 새로운 기능 추가 시 영향 범위 최소화
- **트레이드오프**: 초기 구조 설계 비용 증가

### 7. Feed-forward Rendering Mode
- **이유**: 프리렌더링된 시퀀스 재생 및 시간 제어 필요
- **구현**:
  - Backend: weight_path_pattern으로 프레임별 PLY 동적 로딩
  - Frontend: time_index를 통한 프레임 제어
  - Training/Inference mode 명확히 분리
- **효과**: 실시간 최적화 없이 고품질 프리렌더링 시퀀스 재생

---

## 알려진 이슈

### H.264 초기 3프레임 버퍼링

**증상**: 처음 3프레임이 즉시 출력되지 않음

**원인**: GOP (Group of Pictures) 내부 최적화

**영향**: 시작 지연 ~50ms (연속 스트리밍에서는 무시 가능)

**해결**: 필요 시 Flush() 메커니즘 추가 가능

---

## 다음 단계 (Future Work)

### 우선순위 1: Docker Compose 통합
- [ ] `docker-compose.yaml` 작성
- [ ] Multi-container 환경 구성
- [ ] Volume mount 설정 (`/run/ipc`)

### 우선순위 2: 성능 최적화
- [ ] Zero-copy buffer sharing
- [ ] GPU Direct communication
- [ ] Batch processing (multiple cameras)
- [ ] PLY 로딩 캐싱 (feed-forward mode)

### 우선순위 3: 테스트 강화
- [ ] 유닛 테스트 작성 (pytest, jest)
- [ ] 부하 테스트 (sustained 60 FPS)
- [ ] E2E 테스트 자동화
- [ ] 에러 복구 테스트

### 우선순위 4: 기능 확장
- [ ] WebRTC adapter 추가
- [ ] Multi-view 렌더링
- [ ] 실시간 편집 기능
- [ ] 품질 프로파일 (Low/Medium/High)

---

## 진행 기록

### 2025-10-21: Scene Renderer 구현 완료 ✅
- GsplatRenderer (Static 3DGS)
- StreamableGaussian (4DGS/3DGStream + NTC)
- Helper classes (gaussian_state, optimization_params)

### 2025-10-21: Encoder 구현 완료 ✅
- JpegEncoder (nvimgcodec/OpenCV)
- H264Encoder (NVENC)
- RawEncoder (torch.save)
- Depth/Color utilities

### 2025-10-22: RendererService 및 Protocol 구현 완료 ✅
- Protocol Utils (168/56 bytes binary)
- Frame Buffer (FIFO/Latest)
- RendererService (Unix Socket client)
- Main entry point (CLI)

### 2025-10-22: Transport Service 구현 완료 ✅
- Protocol Adapter Pattern
- WebSocket adapter (Frontend)
- Unix Socket adapter (Renderer)
- torch 의존성 제거 (numpy 사용)
- format_type 정수화 (0, 1, 2)

### 2025-10-22: E2E 통합 테스트 성공 ✅
- Test Client 구현
- 디버그 저장 기능 추가 (Renderer/Transport/Client)
- 10 frames 테스트 → 100% success
- 전체 파이프라인 검증 완료

### 2025-10-30: Frontend 모듈화 및 Feed-forward 렌더링 통합 ✅

**Frontend 리팩토링** (Commit: 1cfda79):
- main.ts → Application + Systems 패턴으로 분리 (1481줄 감소)
- 새로운 시스템 추가:
  - UISystem: UI 컴포넌트 관리 및 이벤트 처리
  - PhysicsSystem: Mesh-Gaussian 충돌 감지
  - TextureManager 렌더링 파이프라인 통합
- UI 컴포넌트화:
  - panels/: 5개 패널 (Control, Debug, Stats, FPSTest, Recording)
  - managers/: CameraStateManager, SettingsManager
  - ui/: FrameScrubber, TimeControlUI
  - physics/: MeshGaussianCollision
- 타입 안정성 및 코드 품질 개선

**Feed-forward 렌더링 통합** (Commit: e58a3fe):
- Backend:
  - StreamableGaussian inference mode 추가
  - weight_path_pattern: 프레임별 PLY 동적 로딩
  - Training/Inference mode 명확히 분리
  - CLI: --weight-path-pattern, --checkpoints-dir 옵션
- Frontend:
  - feed-forward render mode 추가
  - Time index 제어 및 프레임 scrubbing
  - 실시간/프리로드 모드 전환 UI
- Protocol:
  - time_index 전송 안정성 개선
  - 프레임 타이밍 정확도 향상

**현재 진행률**: Phase 2 완료 (Frontend 통합 + Feed-forward 렌더링)

---

## 참고 문서

- `architecture.md`: 상세 아키텍처 설계 문서
- `TEST.md`: 테스트 계획 및 결과
- `FEED_FORWARD_INTEGRATION.md`: Feed-forward 렌더링 통합 가이드
- `REFACTORING.camera.md`: 카메라 리팩토링 문서
- `TROUBLESHOOTING.md`: 문제 해결 가이드
- `feature.collision.md`: 충돌 감지 기능 문서
- `research/`: 연구용 코드 및 실험

---

## 프로젝트 상태

- **프로젝트**: HybridPipeline
- **최종 업데이트**: 2025-10-30
- **현재 상태**: Phase 2 Complete
- **브랜치**: feature/collision-detect
- **최근 커밋**:
  - `e58a3fe`: feat: Feed-forward 렌더링 통합
  - `1cfda79`: refactor: Frontend 모듈화 아키텍처 구축
  - `d8e1ab2`: refactor: Extract renderLoop to RenderingSystem

**완료된 Phase**:
- ✅ Phase 1: MVP (Backend 3-Tier 아키텍처)
- ✅ Phase 2: Frontend 통합 + Feed-forward 렌더링

**다음 Phase**:
- Phase 3: 성능 최적화 및 Docker Compose 통합