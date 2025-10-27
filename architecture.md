# HybridPipeline Architecture

## 개요

### 프로젝트 목적

HybridPipeline은 실시간 3D 장면 렌더링을 위한 모듈화된 파이프라인입니다.

**핵심 목표:**
- 🎯 **렌더러 교체 가능**: 3DGS, 4DGS, NeRF 등 다양한 렌더러를 쉽게 교체
- 🔌 **프로토콜 독립성**: WebSocket, FIFO 등 다양한 전송 프로토콜 지원
- 🧩 **모듈화**: 렌더링, 인코딩, 전송 로직의 명확한 분리
- ⚡ **고성능**: Unix Socket 기반 저지연 통신

**문제점:**
기존 코드는 렌더링 로직과 전송 로직이 혼재되어 있어 렌더러 교체가 어렵고, 불필요한 기능이 많아 유지보수가 힘듭니다.

**해결책:**
Transport Service와 Renderer Service를 분리하고, 각 서비스 내부를 모듈화하여 확장 가능하고 유지보수가 쉬운 아키텍처를 구축합니다.

---

## 시스템 구조

### 서비스 구성

```
┌─────────────────┐
│ Frontend        │  브라우저/앱 (TypeScript)
│ Service         │  - Three.js 렌더링
└────────┬────────┘  - 카메라 제어
         │
         │ WebSocket (Camera Data)
         │
┌────────▼────────┐
│ Transport       │  Python/asyncio
│ Service         │  - Frontend ↔ Renderer 브릿지
│                 │  - 프로토콜 변환
└────────┬────────┘  - 여러 프로토콜 지원
         │
         │ Unix Socket (IPC)
         │
┌────────▼────────┐
│ Renderer        │  Python/PyTorch/CUDA
│ Service         │  - 장면 렌더링
│                 │  - 데이터 인코딩
└─────────────────┘  - 렌더러 교체 가능
```

### 1. Frontend Service

사용자 인터페이스 및 카메라 제어:
- Three.js 기반 3D 뷰어
- 카메라 파라미터 생성 (view matrix, intrinsics)
- WebSocket으로 Transport와 통신

### 2. Transport Service

Frontend와 Renderer를 연결하는 **순수 브릿지**:
- ✅ 데이터 전달만 수행 (렌더링 X)
- ✅ 프로토콜 변환 (WebSocket ↔ Unix Socket)
- ✅ 여러 Frontend 클라이언트 동시 지원
- ✅ 여러 프로토콜 동시 지원 (WebSocket, FIFO)

**역할:**
- Frontend → Renderer: Camera 데이터 전달
- Renderer → Frontend: 렌더링 결과 전달
- 타임스탬프 추가/관리

### 3. Renderer Service

실제 렌더링 수행:
- ✅ **Scene Renderer**: 장면 렌더링 (3DGS, 4DGS, NeRF 등)
- ✅ **Output Encoder**: 데이터 포맷 변환 (JPEG, H.264, Raw)
- ✅ **Hook 시스템**: 초기화, 렌더링, 종료 Hook
- ✅ **렌더러 교체 가능**: git clone으로 새 렌더러 추가

---

## 데이터 타입

### CameraFrame

카메라 파라미터 (Frontend → Transport → Renderer):

```python
class CameraFrame:
    view_matrix: np.ndarray    # (4, 4) float32 - Camera view matrix
    intrinsics: np.ndarray     # (4, 4) float32 - Camera intrinsics
    time_index: float          # Temporal index (for 4DGS)
    frame_id: int              # Frame identifier
    client_timestamp: float    # Client send time (ms)
    server_timestamp: float    # Server receive time (ms)
```

### RenderOutput

렌더링 결과 (인코딩 전, Scene Renderer → Encoder):

```python
class RenderOutput:
    color: torch.Tensor        # (H, W, 3) RGB float32 [0, 1]
    depth: torch.Tensor        # (H, W) float32
    alpha: torch.Tensor        # (H, W) float32 [0, 1]
    metadata: dict             # 추가 메타데이터
```

### RenderPayload

인코딩된 최종 결과 (Encoder → Transport → Frontend):

```python
class RenderPayload:
    frame_id: int              # Frame identifier
    metadata: dict             # Format-specific metadata
    data: bytes                # Encoded payload (opaque)
```

**Metadata 예시:**

JPEG + Depth:
```json
{
  "format_type": "jpeg+depth",
  "color_len": 12345,
  "depth_len": 67890,
  "depth_encoding": "float16",
  "width": 1280,
  "height": 720
}
```

H.264:
```json
{
  "format_type": "h264",
  "codec": "h264",
  "width": 1280,
  "height": 1440
}
```

---

## 데이터 흐름

### 전체 흐름

```
┌─────────────────┐
│ Frontend        │
└────────┬────────┘
         │ 1. Camera Data (WebSocket)
         │    - view_matrix, intrinsics, time_index
         ▼
┌─────────────────────────────┐
│ Transport Service           │
│                             │
│ ┌─────────────────────────┐ │
│ │ Frontend Adapters       │ │
│ │ - WebSocketAdapter      │ │  2. Camera Data 수신
│ │ - FIFOAdapter           │ │
│ └───────────┬─────────────┘ │
│             │                │
│ ┌───────────▼─────────────┐ │
│ │ Transport Core          │ │  3. Camera Queue에 추가
│ │ - Camera Queue          │ │
│ │ - Renderer 통신         │ │
│ └───────────┬─────────────┘ │
└─────────────┼───────────────┘
              │ 4. Camera Data (Unix Socket)
              │    /run/ipc/camera.sock
              ▼
┌─────────────────────────────┐
│ Renderer Service            │
│                             │
│ ┌─────────────────────────┐ │
│ │ Camera Receive Loop     │ │  5. Camera Data 수신
│ └───────────┬─────────────┘ │
│             │                │
│ ┌───────────▼─────────────┐ │
│ │ Scene Renderer          │ │  6. 장면 렌더링
│ │ (3DGS/4DGS/NeRF)        │ │     → RenderOutput
│ └───────────┬─────────────┘ │
│             │                │
│ ┌───────────▼─────────────┐ │
│ │ Output Encoder          │ │  7. 인코딩
│ │ (JPEG/H264/Raw)         │ │     → RenderPayload
│ └───────────┬─────────────┘ │
└─────────────┼───────────────┘
              │ 8. RenderPayload (Unix Socket)
              │    /run/ipc/video.sock
              ▼
┌─────────────────────────────┐
│ Transport Service           │  9. Payload 수신
│ - Broadcast to all clients  │
└─────────────┬───────────────┘
              │ 10. Encoded data (WebSocket)
              │     - JPEG+Depth or H.264
              ▼
┌─────────────────┐
│ Frontend        │  11. 디코딩 및 렌더링
└─────────────────┘
```

### 실행 흐름

#### 1. 초기화 단계

```
Transport Service 시작
    ↓
Unix Socket 리스닝 시작
(/run/ipc/camera.sock, /run/ipc/video.sock)
    ↓
Renderer Service 시작
    ↓
Renderer가 Unix Socket 연결
    ↓
Renderer: on_init() 실행
- Scene 로드
- GPU 업로드
- Encoder 초기화
    ↓
Transport에 초기화 완료 신호 (handshake)
    ↓
Transport: Frontend 연결 대기 (WebSocket)
    ↓
시스템 준비 완료
```

#### 2. 렌더링 루프

```
Frontend: Camera 데이터 생성
    ↓
WebSocket으로 Transport에 전송
    ↓
Transport: Camera Queue에 추가
    ↓
Transport: Unix Socket으로 Renderer에 전송
    ↓
Renderer: Camera 데이터 수신
    ↓
Scene Renderer: render(camera) 실행
    ↓
Output Encoder: encode(render_output) 실행
    ↓
Renderer: RenderPayload를 Unix Socket으로 전송
    ↓
Transport: Payload 수신
    ↓
Transport: 모든 연결된 Frontend에 브로드캐스트
    ↓
Frontend: 디코딩 및 화면 렌더링
    ↓
(반복)
```

#### 3. 종료 단계

```
Frontend 연결 종료
    ↓
Transport: Renderer에 종료 신호
    ↓
Renderer: on_shutdown() 실행
- GPU 메모리 해제
- 리소스 정리
    ↓
Unix Socket 연결 종료
    ↓
Transport Service 종료
```

---

## 클래스 구조

### Renderer Service

#### 1. BaseSceneRenderer (추상 클래스)

장면 렌더링 인터페이스:

```python
class BaseSceneRenderer:
    """Scene 렌더링 추상 클래스"""

    async def on_init(self) -> bool:
        """
        렌더러 초기화 Hook
        - Scene 로드
        - GPU 업로드
        - 모델 준비
        Returns: 성공 여부
        """
        raise NotImplementedError

    async def render(self, camera: CameraFrame) -> RenderOutput:
        """
        카메라 파라미터로 렌더링 수행
        Args:
            camera: 카메라 파라미터
        Returns:
            RenderOutput(color, depth, alpha, metadata)
        """
        raise NotImplementedError

    async def on_shutdown(self):
        """
        렌더러 종료 Hook
        - GPU 메모리 해제
        - 리소스 정리
        """
        raise NotImplementedError
```

**구현 예시:**

```python
class GaussianSplattingRenderer(BaseSceneRenderer):
    """3D Gaussian Splatting 렌더러"""

    def __init__(self, ply_path: str):
        self.ply_path = ply_path
        self.scene = None

    async def on_init(self) -> bool:
        print(f"Loading Gaussian Scene from {self.ply_path}")

        # Scene 로드
        self.scene = GaussianScene(self.ply_path)
        self.scene.upload_to_gpu()

        # Pipeline 설정
        self.pipe = PipelineParams()
        self.background = torch.tensor([0, 0, 0], dtype=torch.float32).cuda()

        print(f"Loaded {self.scene.get_xyz.shape[0]} Gaussians")
        return True

    async def render(self, camera: CameraFrame) -> RenderOutput:
        # View matrix와 intrinsics로 Camera 객체 생성
        cam = create_camera(
            camera.view_matrix,
            camera.intrinsics,
            width=1280,
            height=720
        )

        # Gaussian Splatting 렌더링
        render_pkg = gaussian_render(
            cam,
            self.scene,
            self.pipe,
            self.background
        )

        return RenderOutput(
            color=render_pkg["render"],       # (3, H, W) → (H, W, 3)
            depth=render_pkg["depth"],         # (1, H, W) → (H, W)
            alpha=render_pkg["alpha"],         # (1, H, W) → (H, W)
            metadata={"renderer": "3dgs"}
        )

    async def on_shutdown(self):
        del self.scene
        torch.cuda.empty_cache()
        print("Gaussian Scene unloaded")


class StreamableGaussianRenderer(BaseSceneRenderer):
    """3DGStream 렌더러 (NTC + 2-stage training)"""

    def __init__(self, ply_path: str, ntc_path: str, ntc_config: str):
        self.ply_path = ply_path
        self.ntc_path = ntc_path
        self.ntc_config = ntc_config

    async def on_init(self) -> bool:
        # Gaussian 로드
        self.gaussians = TemporalGaussianModel(sh_degree=1)
        self.gaussians.load_ply(self.ply_path)

        # NTC 모델 로드
        with open(self.ntc_config) as f:
            ntc_conf = json.load(f)

        self.ntc = NeuralTransformationCache(...)
        self.ntc.load_state_dict(torch.load(self.ntc_path))
        self.ntc_optimizer = torch.optim.Adam(self.ntc.parameters(), lr=0.002)

        # State 관리
        self.state_manager = GaussianStateManager()
        self.state_manager.save_state(0, self.gaussians)

        return True

    async def render(self, camera: CameraFrame) -> RenderOutput:
        # 이전 프레임 state 복원
        if camera.frame_id > 0:
            self.gaussians = self.state_manager.load_state(camera.frame_id - 1)

        # Stage 1: NTC transformation
        for _ in range(50):
            self.gaussians.query_ntc()
            render_pkg = gaussian_render(...)
            loss = compute_loss(render_pkg, gt_image)
            loss.backward()
            self.ntc_optimizer.step()

        # Stage 2: Gaussian refinement
        self.gaussians.update_by_ntc()
        for _ in range(50):
            render_pkg = gaussian_render(...)
            loss = compute_loss(...)
            loss.backward()
            self.gaussians.optimizer.step()

        # 현재 프레임 state 저장
        self.state_manager.save_state(camera.frame_id, self.gaussians)

        return RenderOutput(
            color=render_pkg["render"],
            depth=render_pkg["depth"],
            alpha=render_pkg["alpha"],
            metadata={"renderer": "3dgstream"}
        )
```

**외부 렌더러 추가 방법:**

```bash
# 새로운 렌더러를 git clone으로 추가
cd renderer/scene_renderers/
git clone https://github.com/user/custom-renderer.git

# Wrapper 클래스 작성
# custom_renderer_wrapper.py
from .custom_renderer import CustomRenderer as _CustomRenderer

class CustomRenderer(BaseSceneRenderer):
    def __init__(self):
        self.renderer = _CustomRenderer()

    async def on_init(self) -> bool:
        return self.renderer.init()

    async def render(self, camera: CameraFrame) -> RenderOutput:
        result = self.renderer.render(camera)
        return RenderOutput(color=result.color, ...)
```

#### 2. BaseEncoder (추상 클래스)

출력 포맷 인코딩 인터페이스:

```python
class BaseEncoder:
    """출력 인코더 추상 클래스"""

    def get_format_type(self) -> str:
        """포맷 타입 반환 (예: 'jpeg+depth', 'h264')"""
        raise NotImplementedError

    async def encode(self, output: RenderOutput, frame_id: int) -> RenderPayload:
        """
        RenderOutput → RenderPayload 변환
        Args:
            output: 렌더링 결과
            frame_id: 프레임 ID
        Returns:
            RenderPayload (metadata + data)
        """
        raise NotImplementedError
```

**구현 예시:**

```python
class JPEGEncoder(BaseEncoder):
    """JPEG + Float16 Depth Encoder"""

    def get_format_type(self) -> str:
        return "jpeg+depth"

    async def encode(self, output: RenderOutput, frame_id: int) -> RenderPayload:
        # Color → JPEG
        color_uint8 = (output.color * 255).clamp(0, 255).to(torch.uint8)
        color_np = color_uint8.cpu().numpy()

        _, color_jpeg = cv2.imencode('.jpg',
            cv2.cvtColor(color_np, cv2.COLOR_RGB2BGR),
            [cv2.IMWRITE_JPEG_QUALITY, 90])
        color_bytes = color_jpeg.tobytes()

        # Depth → Float16
        depth_normalized = normalize_depth(output.depth, output.alpha)
        depth_f16 = depth_normalized.to(torch.float16)
        depth_bytes = depth_f16.cpu().numpy().tobytes()

        return RenderPayload(
            frame_id=frame_id,
            metadata={
                "format_type": "jpeg+depth",
                "color_len": len(color_bytes),
                "depth_len": len(depth_bytes),
                "width": output.color.shape[1],
                "height": output.color.shape[0]
            },
            data=color_bytes + depth_bytes
        )


class H264Encoder(BaseEncoder):
    """H.264 Video Stream Encoder (color + depth combined)"""

    def __init__(self, width: int, height: int):
        import PyNvVideoCodec as nvvc

        self.width = width
        self.height = height

        # H.264 인코더 생성 (combined height)
        self.encoder = nvvc.CreateEncoder(
            width=width,
            height=height * 2,  # color + depth vertically stacked
            fmt="NV12",
            codec="h264",
            preset="P3",
            bitrate=20000000,
            constqp=0,
            gop=1,
            fps=60,
            usecpuinputbuffer=False
        )

    def get_format_type(self) -> str:
        return "h264"

    async def encode(self, output: RenderOutput, frame_id: int) -> RenderPayload:
        # Color RGB → uint8
        color_uint8 = (output.color * 255).clamp(0, 255).to(torch.uint8)

        # Depth → 8bit visualization
        depth_vis = depth_to_8bit(output.depth, output.alpha)
        depth_rgb = depth_vis.unsqueeze(-1).expand(-1, -1, 3)

        # Vertical stack: [color, depth]
        combined = torch.cat([color_uint8, depth_rgb], dim=0)

        # RGB → NV12
        nv12 = rgb_to_nv12(combined)

        # H.264 인코딩
        h264_bitstream = bytes(self.encoder.Encode(nv12))

        return RenderPayload(
            frame_id=frame_id,
            metadata={
                "format_type": "h264",
                "codec": "h264",
                "width": self.width,
                "height": self.height * 2
            },
            data=h264_bitstream
        )


class RawEncoder(BaseEncoder):
    """Raw tensor data (디버깅용)"""

    def get_format_type(self) -> str:
        return "raw"

    async def encode(self, output: RenderOutput, frame_id: int) -> RenderPayload:
        import pickle

        data = {
            "color": output.color.cpu().numpy(),
            "depth": output.depth.cpu().numpy(),
            "alpha": output.alpha.cpu().numpy()
        }

        serialized = pickle.dumps(data)

        return RenderPayload(
            frame_id=frame_id,
            metadata={
                "format_type": "raw",
                "width": output.color.shape[1],
                "height": output.color.shape[0]
            },
            data=serialized
        )
```

#### 3. RendererService (조합)

Scene Renderer와 Encoder를 조합:

```python
class RendererService:
    """Renderer + Encoder 조합 서비스"""

    def __init__(self,
                 scene_renderer: BaseSceneRenderer,
                 encoder: BaseEncoder,
                 camera_socket: str = "/run/ipc/camera.sock",
                 video_socket: str = "/run/ipc/video.sock"):
        self.scene_renderer = scene_renderer
        self.encoder = encoder
        self.camera_socket = camera_socket
        self.video_socket = video_socket

    async def initialize(self) -> bool:
        """초기화"""
        # Renderer 초기화
        success = await self.scene_renderer.on_init()
        if not success:
            return False

        # Unix Socket 연결
        await self.connect_to_transport()

        return True

    async def connect_to_transport(self):
        """Transport Service에 연결"""
        # Camera data 수신용
        self.camera_reader, self.camera_writer = \
            await asyncio.open_unix_connection(self.camera_socket)

        # Video data 송신용
        self.video_reader, self.video_writer = \
            await asyncio.open_unix_connection(self.video_socket)

        print(f"Connected to Transport Service")

    async def camera_receive_loop(self, camera_queue: asyncio.Queue):
        """Camera 데이터 수신"""
        while True:
            # 152 bytes 수신
            packet = await self.camera_reader.read(152)

            if len(packet) < 152:
                print("Incomplete camera packet")
                break

            camera = parse_camera_frame(packet)
            await camera_queue.put(camera)

    async def render_and_send_loop(self, camera_queue: asyncio.Queue):
        """렌더링 및 전송"""
        while True:
            # Camera 데이터 가져오기
            camera = await camera_queue.get()

            # 1. 장면 렌더링
            render_output = await self.scene_renderer.render(camera)

            # 2. 인코딩
            payload = await self.encoder.encode(
                render_output,
                camera.frame_id
            )

            # 3. Wire format으로 전송
            await self.send_payload(payload)

            camera_queue.task_done()

    async def send_payload(self, payload: RenderPayload):
        """RenderPayload를 Transport로 전송"""
        # Metadata를 JSON으로 직렬화
        metadata_bytes = json.dumps(payload.metadata).encode('utf-8')

        # Header: frame_id(8) + metadata_len(4) + data_len(4)
        header = struct.pack("<QII",
            payload.frame_id,
            len(metadata_bytes),
            len(payload.data)
        )

        # 전송
        self.video_writer.write(header + metadata_bytes + payload.data)
        await self.video_writer.drain()

    async def run(self):
        """메인 루프"""
        camera_queue = asyncio.Queue(maxsize=2)

        # 초기화
        if not await self.initialize():
            print("Failed to initialize renderer")
            return

        # 동시 실행
        await asyncio.gather(
            self.camera_receive_loop(camera_queue),
            self.render_and_send_loop(camera_queue)
        )

    async def shutdown(self):
        """종료"""
        await self.scene_renderer.on_shutdown()
        self.camera_writer.close()
        self.video_writer.close()
```

**사용 예시:**

```python
# main.py

# 3DGS + JPEG
renderer = RendererService(
    scene_renderer=GaussianSplattingRenderer(ply_path="scene.ply"),
    encoder=JPEGEncoder()
)

# 3DGS + H.264
renderer = RendererService(
    scene_renderer=GaussianSplattingRenderer(ply_path="scene.ply"),
    encoder=H264Encoder(width=1280, height=720)
)

# 3DGStream + JPEG
renderer = RendererService(
    scene_renderer=StreamableGaussianRenderer(
        ply_path="frame000000.ply",
        ntc_path="ntc_model.pth",
        ntc_config="ntc_config.json"
    ),
    encoder=JPEGEncoder()
)

# 실행
asyncio.run(renderer.run())
```

### Transport Service

#### 1. TransportCore

프로토콜 독립적인 핵심 로직:

```python
class TransportCore:
    """Transport 핵심 로직 (프로토콜 독립적)"""

    def __init__(self,
                 camera_socket: str = "/run/ipc/camera.sock",
                 video_socket: str = "/run/ipc/video.sock"):
        self.camera_socket = camera_socket
        self.video_socket = video_socket

        # Renderer 연결
        self.camera_writer = None
        self.video_reader = None

        # Frontend adapters
        self.frontend_adapters: List[BaseFrontendAdapter] = []

        # Camera queue (Frontend → Renderer)
        self.camera_queue = asyncio.Queue(maxsize=2)

    async def start_renderer_listener(self):
        """Renderer 연결 대기 (Unix Socket Server)"""
        # Camera socket (Transport → Renderer)
        camera_server = await asyncio.start_unix_server(
            self.handle_camera_connection,
            self.camera_socket
        )

        # Video socket (Renderer → Transport)
        video_server = await asyncio.start_unix_server(
            self.handle_video_connection,
            self.video_socket
        )

        print(f"Listening for Renderer on {self.camera_socket}")
        print(f"Listening for Renderer on {self.video_socket}")

        await asyncio.gather(
            camera_server.serve_forever(),
            video_server.serve_forever()
        )

    async def handle_camera_connection(self, reader, writer):
        """Camera socket 연결 처리"""
        print("Renderer connected to camera socket")
        self.camera_writer = writer

        # Camera 전송 루프
        await self.camera_send_loop()

    async def handle_video_connection(self, reader, writer):
        """Video socket 연결 처리"""
        print("Renderer connected to video socket")
        self.video_reader = reader

        # Video 수신 루프
        await self.video_receive_loop()

    async def camera_send_loop(self):
        """Frontend → Renderer로 Camera 전송"""
        while True:
            camera = await self.camera_queue.get()

            # CameraFrame → bytes (152 bytes)
            data = pack_camera_frame(camera)

            self.camera_writer.write(data)
            await self.camera_writer.drain()

            self.camera_queue.task_done()

    async def video_receive_loop(self):
        """Renderer → Frontend로 Video 전송"""
        while True:
            # Wire format 파싱
            header = await read_exact(self.video_reader, 16)
            frame_id, meta_len, data_len = struct.unpack("<QII", header)

            metadata_bytes = await read_exact(self.video_reader, meta_len)
            data = await read_exact(self.video_reader, data_len)

            metadata = json.loads(metadata_bytes)

            payload = RenderPayload(
                frame_id=frame_id,
                metadata=metadata,
                data=data
            )

            # 모든 Frontend에 브로드캐스트
            await self.broadcast_to_frontends(payload)

    async def broadcast_to_frontends(self, payload: RenderPayload):
        """모든 연결된 Frontend에 전송"""
        for adapter in self.frontend_adapters:
            try:
                await adapter.send(payload)
            except Exception as e:
                print(f"Failed to send to frontend: {e}")

    def add_frontend_adapter(self, adapter: 'BaseFrontendAdapter'):
        """Frontend adapter 추가"""
        self.frontend_adapters.append(adapter)

    def remove_frontend_adapter(self, adapter: 'BaseFrontendAdapter'):
        """Frontend adapter 제거"""
        self.frontend_adapters.remove(adapter)
```

#### 2. BaseFrontendAdapter (추상 클래스)

Frontend 프로토콜 어댑터:

```python
class BaseFrontendAdapter:
    """Frontend 프로토콜 어댑터 추상 클래스"""

    async def send(self, payload: RenderPayload):
        """Frontend로 렌더링 결과 전송"""
        raise NotImplementedError

    async def recv(self) -> CameraFrame:
        """Frontend에서 카메라 데이터 수신"""
        raise NotImplementedError
```

**구현 예시:**

```python
class WebSocketAdapter(BaseFrontendAdapter):
    """WebSocket 프로토콜 어댑터"""

    def __init__(self, ws: websockets.WebSocketServerProtocol,
                 transport_core: TransportCore):
        self.ws = ws
        self.transport_core = transport_core

    async def send(self, payload: RenderPayload):
        """RenderPayload → WebSocket 프로토콜"""
        now = time.time_ns() / 1_000_000  # ms

        if payload.metadata["format_type"] == "h264":
            # H.264 헤더 형식
            header = struct.pack("<IIdddd",
                len(payload.data),           # videoLen
                payload.frame_id,            # frameId
                0.0,                         # clientSendTime
                0.0,                         # serverReceiveTime
                now,                         # serverProcessEndTime
                now                          # serverSendTime
            )
            await self.ws.send(header + payload.data)

        elif payload.metadata["format_type"] == "jpeg+depth":
            # JPEG + Depth 헤더 형식
            header = struct.pack("<IIIdddd",
                payload.metadata["color_len"],
                payload.metadata["depth_len"],
                payload.frame_id,
                0.0,                         # clientSendTime
                0.0,                         # serverReceiveTime
                now,                         # serverProcessEndTime
                now                          # serverSendTime
            )
            await self.ws.send(header + payload.data)

    async def recv(self) -> CameraFrame:
        """WebSocket → CameraFrame"""
        raw = await self.ws.recv()

        # Ping/Pong 처리
        if len(raw) == 16:
            # Ping message
            return None

        # Camera data (160 bytes)
        if len(raw) == 160:
            frame_id = struct.unpack_from("<I", raw, 128)[0]
            client_ts = struct.unpack_from("<d", raw, 136)[0]
            server_ts = time.time_ns() / 1_000_000
            time_index = struct.unpack_from("<f", raw, 144)[0]

            payload = raw[:128]
            floats = struct.unpack("<32f", payload)

            view_matrix = np.array(floats[:16]).reshape(4, 4)
            intrinsics = np.array(floats[16:32]).reshape(4, 4)

            return CameraFrame(
                view_matrix=view_matrix,
                intrinsics=intrinsics,
                time_index=time_index,
                frame_id=frame_id,
                client_timestamp=client_ts,
                server_timestamp=server_ts
            )

    async def recv_loop(self):
        """Camera 수신 루프"""
        try:
            while True:
                camera = await self.recv()
                if camera:
                    await self.transport_core.camera_queue.put(camera)
        except websockets.exceptions.ConnectionClosed:
            print(f"WebSocket closed: {self.ws.remote_address}")
        finally:
            self.transport_core.remove_frontend_adapter(self)


class FIFOAdapter(BaseFrontendAdapter):
    """Named Pipe (FIFO) 프로토콜 어댑터 (송신 전용)"""

    def __init__(self, fifo_path: str):
        self.fifo_path = fifo_path
        self.fifo = None

    async def send(self, payload: RenderPayload):
        """RenderPayload → FIFO"""
        if not self.fifo:
            loop = asyncio.get_event_loop()
            self.fifo = await loop.run_in_executor(
                None, open, self.fifo_path, 'wb'
            )

        # FIFO는 간단한 헤더만
        header = struct.pack("<QI",
            payload.frame_id,
            len(payload.data)
        )

        await asyncio.get_event_loop().run_in_executor(
            None, self.fifo.write, header + payload.data
        )
        await asyncio.get_event_loop().run_in_executor(
            None, self.fifo.flush
        )

    async def recv(self) -> CameraFrame:
        """FIFO는 단방향 (송신 전용)"""
        raise NotImplementedError("FIFO adapter is send-only")
```

#### 3. Transport Service Main

```python
# transport/main.py

async def websocket_handler(ws: websockets.WebSocketServerProtocol,
                           transport_core: TransportCore):
    """WebSocket 연결 핸들러"""
    print(f"Frontend connected: {ws.remote_address}")

    # Handshake
    handshake = await ws.recv()
    if len(handshake) != 4:
        await ws.close()
        return

    width, height = struct.unpack("<HH", handshake)
    print(f"Frontend resolution: {width}x{height}")

    # Adapter 생성 및 등록
    adapter = WebSocketAdapter(ws, transport_core)
    transport_core.add_frontend_adapter(adapter)

    try:
        # Camera 수신 루프
        await adapter.recv_loop()
    finally:
        transport_core.remove_frontend_adapter(adapter)
        print(f"Frontend disconnected: {ws.remote_address}")


async def main():
    # Transport Core 생성
    transport_core = TransportCore()

    # Renderer listener 시작
    renderer_task = asyncio.create_task(
        transport_core.start_renderer_listener()
    )

    # WebSocket 서버 시작
    async with websockets.serve(
        lambda ws: websocket_handler(ws, transport_core),
        "0.0.0.0",
        8765,
        max_size=None,
        ping_interval=None
    ):
        print("Transport Service started")
        print("  WebSocket: ws://0.0.0.0:8765")
        print("  Renderer: /run/ipc/*.sock")

        await renderer_task


if __name__ == "__main__":
    asyncio.run(main())
```

---

## 통신 프로토콜

### Frontend ↔ Transport (WebSocket)

#### Camera Data (Frontend → Transport)

**프로토콜**: 160 bytes

| Offset | Size | Type    | Field              |
|--------|------|---------|--------------------|
| 0      | 64   | float32 | view_matrix (4×4)  |
| 64     | 64   | float32 | intrinsics (4×4)   |
| 128    | 4    | uint32  | frame_id           |
| 132    | 4    | -       | padding            |
| 136    | 8    | float64 | client_timestamp   |
| 144    | 4    | float32 | time_index         |
| 148    | 12   | -       | padding            |

#### Video Data (Transport → Frontend)

**H.264 프로토콜**: 40 bytes header + data

| Offset | Size | Type    | Field                   |
|--------|------|---------|-------------------------|
| 0      | 4    | uint32  | video_len               |
| 4      | 4    | uint32  | frame_id                |
| 8      | 8    | float64 | client_send_time        |
| 16     | 8    | float64 | server_receive_time     |
| 24     | 8    | float64 | server_process_end_time |
| 32     | 8    | float64 | server_send_time        |
| 40     | var  | bytes   | h264_bitstream          |

**JPEG + Depth 프로토콜**: 44 bytes header + data

| Offset | Size | Type    | Field                   |
|--------|------|---------|-------------------------|
| 0      | 4    | uint32  | jpeg_len                |
| 4      | 4    | uint32  | depth_len               |
| 8      | 4    | uint32  | frame_id                |
| 12     | 8    | float64 | client_send_time        |
| 20     | 8    | float64 | server_receive_time     |
| 28     | 8    | float64 | server_process_end_time |
| 36     | 8    | float64 | server_send_time        |
| 44     | var  | bytes   | jpeg_data               |
| var    | var  | bytes   | depth_data (float16)    |

### Transport ↔ Renderer (Unix Socket)

#### Camera Frame (Transport → Renderer)

**프로토콜**: 152 bytes

| Offset | Size | Type    | Field              |
|--------|------|---------|--------------------|
| 0      | 64   | float32 | view_matrix (4×4)  |
| 64     | 64   | float32 | intrinsics (4×4)   |
| 128    | 8    | float64 | client_timestamp   |
| 136    | 8    | float64 | server_timestamp   |
| 144    | 4    | float32 | time_index         |
| 148    | 4    | uint32  | frame_id           |

#### Render Payload (Renderer → Transport)

**프로토콜**: 16 bytes header + metadata (JSON) + data

| Offset | Size | Type   | Field         |
|--------|------|--------|---------------|
| 0      | 8    | uint64 | frame_id      |
| 8      | 4    | uint32 | metadata_len  |
| 12     | 4    | uint32 | data_len      |
| 16     | var  | bytes  | metadata (JSON)|
| var    | var  | bytes  | data (opaque) |

---

## Docker 구성

### docker-compose.yml

```yaml
version: '3.8'

services:
  transport-service:
    build: ./transport
    container_name: hybrid-transport
    ports:
      - "8765:8765"  # WebSocket port
    volumes:
      - ipc-sockets:/run/ipc  # Unix socket 공유
    networks:
      - hybrid-net
    depends_on:
      - renderer-service

  renderer-service:
    build: ./renderer
    container_name: hybrid-renderer
    volumes:
      - ipc-sockets:/run/ipc  # Unix socket 공유
      - ./data:/data          # Scene data
    networks:
      - hybrid-net
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

volumes:
  ipc-sockets:  # Unix Socket 공유 볼륨

networks:
  hybrid-net:
```

### Renderer Dockerfile

```dockerfile
# renderer/Dockerfile
FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

# Python 및 의존성 설치
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git

# PyTorch 설치
RUN pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 프로젝트 의존성
COPY requirements.txt .
RUN pip3 install -r requirements.txt

# 코드 복사
COPY . /app
WORKDIR /app

# Unix socket 디렉토리 생성
RUN mkdir -p /run/ipc

CMD ["python3", "main.py"]
```

### Transport Dockerfile

```dockerfile
# transport/Dockerfile
FROM python:3.10-slim

# 의존성 설치
COPY requirements.txt .
RUN pip install -r requirements.txt

# 코드 복사
COPY . /app
WORKDIR /app

# Unix socket 디렉토리 생성
RUN mkdir -p /run/ipc

CMD ["python3", "main.py"]
```

---

## 디렉토리 구조

```
HybridPipeline/
├── architecture.md              # 이 문서
│
├── frontend/                    # Frontend Service
│   ├── index.html
│   ├── src/
│   │   ├── main.ts
│   │   ├── scene-setup.ts
│   │   └── decode-worker.ts
│   └── package.json
│
├── transport/                   # Transport Service
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── main.py                  # Entry point
│   ├── transport_core.py        # TransportCore
│   ├── frontend_adapters/
│   │   ├── base.py              # BaseFrontendAdapter
│   │   ├── websocket.py         # WebSocketAdapter
│   │   └── fifo.py              # FIFOAdapter
│   └── utils/
│       └── protocol.py          # 프로토콜 파싱 유틸
│
├── renderer/                    # Renderer Service
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── main.py                  # Entry point
│   │
│   ├── scene_renderers/         # Scene Renderers
│   │   ├── base.py              # BaseSceneRenderer
│   │   ├── gaussian_splatting.py
│   │   ├── streamable_gaussian.py
│   │   │
│   │   └── external/            # git clone으로 추가되는 렌더러
│   │       ├── 3d-gaussian-splatting/  (git submodule)
│   │       ├── 3dgstream/              (git submodule)
│   │       └── custom-renderer/        (git submodule)
│   │
│   ├── encoders/                # Output Encoders
│   │   ├── base.py              # BaseEncoder
│   │   ├── jpeg.py              # JPEGEncoder
│   │   ├── h264.py              # H264Encoder
│   │   └── raw.py               # RawEncoder
│   │
│   ├── renderer_service.py      # RendererService
│   └── utils/
│       ├── protocol.py          # 프로토콜 파싱
│       └── image_utils.py       # 이미지 변환 유틸
│
├── docker-compose.yml
└── README.md
```

---

## 설계 원칙

### 1. 관심사의 분리 (Separation of Concerns)

- **Scene Renderer**: 장면 렌더링만 담당
- **Encoder**: 데이터 포맷 변환만 담당
- **Transport**: 데이터 전달만 담당 (렌더링 X)

### 2. 조합 가능성 (Composability)

렌더러와 인코더를 자유롭게 조합:

```python
# 3DGS + JPEG
GaussianSplattingRenderer + JPEGEncoder

# 3DGS + H.264
GaussianSplattingRenderer + H264Encoder

# 3DGStream + JPEG
StreamableGaussianRenderer + JPEGEncoder

# Custom + Raw
CustomRenderer + RawEncoder
```

### 3. 확장 가능성 (Extensibility)

- **새로운 Scene Renderer 추가**: `BaseSceneRenderer` 상속
- **새로운 Encoder 추가**: `BaseEncoder` 상속
- **새로운 프로토콜 추가**: `BaseFrontendAdapter` 상속

**외부 렌더러 통합**:

```bash
# Git submodule로 추가
cd renderer/scene_renderers/external/
git submodule add https://github.com/user/custom-renderer.git

# Wrapper 작성
# renderer/scene_renderers/custom_wrapper.py
from .external.custom_renderer import Renderer as _CustomRenderer
from .base import BaseSceneRenderer

class CustomRenderer(BaseSceneRenderer):
    def __init__(self):
        self.renderer = _CustomRenderer()

    async def on_init(self) -> bool:
        return self.renderer.initialize()

    async def render(self, camera) -> RenderOutput:
        result = self.renderer.render(camera)
        return RenderOutput(color=result.rgb, depth=result.depth, ...)
```

### 4. 프로토콜 독립성 (Protocol Independence)

Transport Core는 Frontend 프로토콜과 독립적:
- WebSocket, FIFO 등을 Adapter로 추상화
- 여러 프로토콜 동시 지원 가능

Transport ↔ Renderer는 Unix Socket 고정:
- 같은 호스트 환경 (Docker Volume Mount)
- 최고 성능 (40-50 GB/s)
- 간단한 구현

### 5. YAGNI (You Aren't Gonna Need It)

**Phase 1**: 핵심 기능만 구현
- Unix Socket만 지원
- 기본 렌더러 (3DGS)
- 기본 인코더 (JPEG)

**Phase 2**: 필요시 확장
- 추가 렌더러 (3DGStream, NeRF)
- 추가 인코더 (H.264, Raw)
- 추가 프로토콜 (FIFO)

---

## 구현 우선순위

### Phase 1: MVP (Minimum Viable Product)

1. **Renderer Service**
   - ✅ `BaseSceneRenderer` 추상 클래스
   - ✅ `GaussianSplattingRenderer` 구현
   - ✅ `JPEGEncoder` 구현
   - ✅ `RendererService` 조합
   - ✅ Unix Socket 통신

2. **Transport Service**
   - ✅ `TransportCore` 구현
   - ✅ `WebSocketAdapter` 구현
   - ✅ Unix Socket 서버

3. **통합 테스트**
   - ✅ Frontend → Transport → Renderer 데이터 흐름
   - ✅ Docker Compose 환경

### Phase 2: 확장 기능

1. **Encoder 추가**
   - `H264Encoder` 구현
   - `RawEncoder` 구현 (디버깅용)

2. **Renderer 추가**
   - `StreamableGaussianRenderer` 구현 (3DGStream)

3. **프로토콜 추가**
   - `FIFOAdapter` 구현

### Phase 3: 최적화

1. **성능 최적화**
   - Zero-copy 전송
   - Batch processing
   - GPU Direct 전송

2. **모니터링**
   - 레이턴시 측정
   - FPS 모니터링
   - 리소스 사용량

---

## 기존 코드 문제점

### 1. feed-forward-renderer-socket.py (807줄)

**문제점:**
- 3DGStream 특화 로직 (NTC, 2-stage training)과 일반 렌더링 혼재
- 렌더러 교체 불가능한 구조
- H.264 인코딩 로직이 렌더링 로직과 섞여있음
- State 관리, training 로직이 복잡하게 얽혀있음

**개선:**
- Scene Renderer와 Encoder 분리
- `StreamableGaussianRenderer`로 3DGStream 로직 캡슐화
- `H264Encoder`로 인코딩 로직 분리

### 2. server.py

**문제점:**
- Transport 역할 + 렌더링 역할 혼재
- 3가지 렌더링 루프 중복 (`render_loop`, `render_loop_jpeg`, `render_feedforward_loop`)
- WebSocket과 FIFO 통신 로직이 혼재

**개선:**
- Transport는 데이터 전달만 수행
- 렌더링은 Renderer Service로 완전 분리
- Protocol Adapter 패턴으로 WebSocket/FIFO 추상화

### 3. session.py

**문제점:**
- Encoder 생성 로직이 Session에 포함
- 불필요한 결합

**개선:**
- Encoder는 Renderer Service에서만 관리
- Transport는 Encoder에 대해 알 필요 없음

---

## 참고 자료

### 프로토콜 설계

- WebSocket: RFC 6455
- Unix Socket: POSIX IPC
- H.264: ITU-T H.264 / MPEG-4 AVC

### 렌더링 엔진

- 3D Gaussian Splatting: https://github.com/graphdeco-inria/gaussian-splatting
- 3DGStream: (프로젝트 내부 구현)
- diff-gaussian-rasterization: CUDA 기반 rasterizer

### 인코딩

- NVENC (H.264): PyNvVideoCodec
- JPEG: OpenCV, nvImageCodec
- Depth Encoding: Log-depth normalization

---

## 마이그레이션 가이드

기존 코드에서 새 아키텍처로 전환:

### 1. feed-forward-renderer-socket.py → Renderer Service

```python
# Before: 807줄의 복잡한 코드

# After: 명확한 분리
renderer = RendererService(
    scene_renderer=StreamableGaussianRenderer(...),
    encoder=H264Encoder(...)
)
```

### 2. server.py → Transport Service

```python
# Before: 렌더링 + 전송 혼재

# After: 전송만 수행
transport = TransportCore()
# 렌더링은 Renderer Service가 담당
```

### 3. 새로운 렌더러 추가

```bash
# Git submodule로 추가
cd renderer/scene_renderers/external/
git submodule add https://github.com/user/nerf-renderer.git

# Wrapper 작성
class NeRFRenderer(BaseSceneRenderer):
    async def render(self, camera):
        # NeRF 렌더링 로직
        ...
```

---

## FAQ

### Q: 왜 Transport와 Renderer를 분리하나요?

**A**: 관심사의 분리와 유연성을 위해서입니다.
- Transport는 프로토콜 변환만 담당
- Renderer는 렌더링만 담당
- 렌더러 교체 시 Transport 수정 불필요
- 각 서비스를 독립적으로 스케일링 가능

### Q: Unix Socket vs TCP Socket?

**A**: Docker 환경에서는 Unix Socket이 최적입니다.
- Volume Mount로 컨테이너 간 공유 가능
- TCP 대비 2배 이상 빠름 (40-50 GB/s)
- 간단한 구현
- 로컬 환경에서 충분

### Q: 새로운 렌더러 추가는 어떻게 하나요?

**A**: Git submodule + Wrapper 패턴:
```bash
git submodule add <repo-url> renderer/scene_renderers/external/my-renderer
```
그리고 `BaseSceneRenderer`를 상속하는 Wrapper 작성

### Q: JPEG vs H.264?

**A**: 용도에 따라 선택:
- **JPEG**: 프레임 독립, 디버깅 쉬움, 압축률 낮음
- **H.264**: 높은 압축률, 복잡한 디버깅, 시간적 종속성

### Q: 기존 코드는 삭제하나요?

**A**: 아니요, `research/` 폴더에 보관:
```
research/
├── 3DGStream/
│   ├── feed-forward-renderer-socket.py  # 참고용
│   └── feed-forward-renderer.py
└── backend/
    └── src/
        └── server.py  # 참고용
```

---

## Renderer Service 구현 세부사항

### 프로토콜 파싱 구현

#### Camera Frame 파싱 (152 bytes)

```python
# renderer/utils/protocol.py

import struct
import numpy as np
from data_types import CameraFrame

def parse_camera_frame(data: bytes) -> CameraFrame:
    """
    152 bytes → CameraFrame 변환

    Layout:
    0-64:     view_matrix (16 float32)
    64-128:   intrinsics (16 float32)
    128-136:  client_timestamp (float64)
    136-144:  server_timestamp (float64)
    144-148:  time_index (float32)
    148-152:  frame_id (uint32)
    """
    if len(data) != 152:
        raise ValueError(f"Invalid camera frame size: {len(data)} (expected 152)")

    # View matrix (64 bytes)
    view_floats = struct.unpack("<16f", data[0:64])
    view_matrix = np.array(view_floats, dtype=np.float32).reshape(4, 4)

    # Intrinsics (64 bytes)
    intrinsics_floats = struct.unpack("<16f", data[64:128])
    intrinsics = np.array(intrinsics_floats, dtype=np.float32).reshape(4, 4)

    # Metadata
    client_timestamp = struct.unpack("<d", data[128:136])[0]
    server_timestamp = struct.unpack("<d", data[136:144])[0]
    time_index = struct.unpack("<f", data[144:148])[0]
    frame_id = struct.unpack("<I", data[148:152])[0]

    return CameraFrame(
        view_matrix=view_matrix,
        intrinsics=intrinsics,
        time_index=time_index,
        frame_id=frame_id,
        client_timestamp=client_timestamp,
        server_timestamp=server_timestamp
    )


def pack_camera_frame(camera: CameraFrame) -> bytes:
    """CameraFrame → 152 bytes 변환 (Transport에서 사용)"""
    view_bytes = camera.view_matrix.astype(np.float32).tobytes()
    intrinsics_bytes = camera.intrinsics.astype(np.float32).tobytes()

    metadata_bytes = struct.pack("<ddfi",
        camera.client_timestamp,
        camera.server_timestamp,
        camera.time_index,
        camera.frame_id
    )

    return view_bytes + intrinsics_bytes + metadata_bytes
```

#### Render Payload 파싱 (16 + metadata + data)

```python
import json

def pack_render_payload(payload: RenderPayload) -> bytes:
    """
    RenderPayload → Wire format

    Header (16 bytes):
    0-8:   frame_id (uint64)
    8-12:  metadata_len (uint32)
    12-16: data_len (uint32)

    Metadata: JSON bytes (UTF-8)
    Data: opaque bytes
    """
    metadata_bytes = json.dumps(payload.metadata).encode('utf-8')

    header = struct.pack("<QII",
        payload.frame_id,
        len(metadata_bytes),
        len(payload.data)
    )

    return header + metadata_bytes + payload.data


async def read_exact(reader: asyncio.StreamReader, n: int) -> bytes:
    """정확히 n bytes 읽기 (incomplete 방지)"""
    data = b""
    while len(data) < n:
        chunk = await reader.read(n - len(data))
        if not chunk:
            raise EOFError(f"Incomplete read: expected {n}, got {len(data)}")
        data += chunk
    return data


async def parse_render_payload(reader: asyncio.StreamReader) -> RenderPayload:
    """Wire format → RenderPayload"""
    # Header 읽기
    header = await read_exact(reader, 16)
    frame_id, meta_len, data_len = struct.unpack("<QII", header)

    # Metadata 읽기
    metadata_bytes = await read_exact(reader, meta_len)
    metadata = json.loads(metadata_bytes.decode('utf-8'))

    # Data 읽기
    data = await read_exact(reader, data_len)

    return RenderPayload(
        frame_id=frame_id,
        metadata=metadata,
        data=data
    )
```

### 에러 핸들링 및 복구 전략

#### 1. Socket 연결 에러

```python
class RendererService:
    async def connect_to_transport(self, max_retries=5):
        """Transport Service에 연결 (재시도 로직 포함)"""
        for attempt in range(max_retries):
            try:
                # Camera socket 연결
                self.camera_reader, self.camera_writer = \
                    await asyncio.open_unix_connection(self.camera_socket)

                # Video socket 연결
                self.video_reader, self.video_writer = \
                    await asyncio.open_unix_connection(self.video_socket)

                print(f"✅ Connected to Transport Service")
                return True

            except FileNotFoundError:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff
                    print(f"⚠️  Transport not ready, retrying in {wait_time}s...")
                    await asyncio.sleep(wait_time)
                else:
                    print(f"❌ Failed to connect after {max_retries} attempts")
                    return False
            except Exception as e:
                print(f"❌ Connection error: {e}")
                return False
```

#### 2. 렌더링 에러

```python
async def render_and_send_loop(self, camera_queue: asyncio.Queue):
    """렌더링 및 전송 (에러 복구)"""
    while True:
        try:
            # Camera 가져오기
            camera = await camera_queue.get()

            try:
                # 1. 장면 렌더링
                render_output = await self.scene_renderer.render(camera)

                # 2. 인코딩
                payload = await self.encoder.encode(
                    render_output,
                    camera.frame_id
                )

                # 3. 전송
                await self.send_payload(payload)

            except torch.cuda.OutOfMemoryError:
                # GPU OOM: 캐시 정리 후 재시도
                print(f"⚠️  GPU OOM at frame {camera.frame_id}, clearing cache...")
                torch.cuda.empty_cache()
                # 해당 프레임 drop

            except Exception as e:
                # 기타 렌더링 에러: 로그 후 skip
                print(f"❌ Render error at frame {camera.frame_id}: {e}")
                # 해당 프레임 drop

            finally:
                camera_queue.task_done()

        except asyncio.CancelledError:
            print("Render loop cancelled")
            break
        except Exception as e:
            print(f"❌ Fatal error in render loop: {e}")
            break
```

#### 3. 데이터 검증

```python
def validate_camera_frame(camera: CameraFrame) -> bool:
    """Camera 데이터 검증"""
    # View matrix 체크
    if camera.view_matrix.shape != (4, 4):
        print(f"❌ Invalid view_matrix shape: {camera.view_matrix.shape}")
        return False

    # Intrinsics 체크
    if camera.intrinsics.shape != (4, 4):
        print(f"❌ Invalid intrinsics shape: {camera.intrinsics.shape}")
        return False

    # Frame ID 체크
    if camera.frame_id < 0:
        print(f"❌ Invalid frame_id: {camera.frame_id}")
        return False

    return True


def validate_render_output(output: RenderOutput, expected_size: tuple) -> bool:
    """RenderOutput 검증"""
    H, W = expected_size

    # Shape 검증
    if output.color.shape != (H, W, 3):
        print(f"❌ Invalid color shape: {output.color.shape}, expected ({H}, {W}, 3)")
        return False

    if output.depth.shape != (H, W):
        print(f"❌ Invalid depth shape: {output.depth.shape}, expected ({H}, {W})")
        return False

    # 값 범위 검증
    if not (torch.all(output.color >= 0) and torch.all(output.color <= 1)):
        print(f"⚠️  Color values out of range [0, 1]")
        output.color = torch.clamp(output.color, 0, 1)

    if not (torch.all(output.alpha >= 0) and torch.all(output.alpha <= 1)):
        print(f"⚠️  Alpha values out of range [0, 1]")
        output.alpha = torch.clamp(output.alpha, 0, 1)

    return True
```

#### 4. Queue Overflow 처리

```python
async def handle_camera_with_overflow(self, camera_queue: asyncio.Queue, camera: CameraFrame):
    """Queue overflow 처리 (오래된 프레임 drop)"""
    try:
        camera_queue.put_nowait(camera)
    except asyncio.QueueFull:
        # 가장 오래된 프레임 제거
        try:
            dropped = camera_queue.get_nowait()
            print(f"⚠️  Queue full, dropping frame {dropped.frame_id}")
        except asyncio.QueueEmpty:
            pass

        # 새 프레임 추가
        camera_queue.put_nowait(camera)
```

---

## 성능 최적화 가이드

### 1. GPU 메모리 관리

```python
class GaussianSplattingRenderer(BaseSceneRenderer):
    async def on_init(self) -> bool:
        # Scene 로드
        self.scene = GaussianScene(self.ply_path)

        # GPU 메모리 사용량 확인
        total_memory = torch.cuda.get_device_properties(0).total_memory
        print(f"Total GPU memory: {total_memory / 1e9:.2f} GB")

        # Scene 업로드
        self.scene.upload_to_gpu()

        # 메모리 사용량 확인
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        print(f"GPU memory: allocated={allocated:.2f}GB, reserved={reserved:.2f}GB")

        return True

    async def on_shutdown(self):
        # GPU 메모리 해제
        del self.scene
        torch.cuda.empty_cache()

        # 해제 확인
        allocated = torch.cuda.memory_allocated() / 1e9
        print(f"GPU memory after cleanup: {allocated:.2f}GB")
```

### 2. 비동기 처리 최적화

```python
class RendererService:
    async def run(self):
        """최적화된 메인 루프"""
        # Queue 크기 조정 (레이턴시 vs 처리량)
        camera_queue = asyncio.Queue(maxsize=2)  # 작을수록 낮은 레이턴시

        # 동시 처리
        await asyncio.gather(
            self.camera_receive_loop(camera_queue),
            self.render_and_send_loop(camera_queue),
            # 추가 워커 가능 (다중 GPU 환경)
            # self.render_and_send_loop(camera_queue),
        )
```

### 3. 인코딩 최적화

```python
class JPEGEncoder(BaseEncoder):
    def __init__(self, quality=90, optimize_cpu=True):
        self.quality = quality
        self.optimize_cpu = optimize_cpu

        # JPEG 인코더 파라미터
        self.encode_params = [
            cv2.IMWRITE_JPEG_QUALITY, quality,
            cv2.IMWRITE_JPEG_OPTIMIZE, 1 if optimize_cpu else 0,
            cv2.IMWRITE_JPEG_PROGRESSIVE, 0  # Progressive 끄기 (속도 향상)
        ]

    async def encode(self, output: RenderOutput, frame_id: int) -> RenderPayload:
        # GPU → CPU 전송 최소화
        color_uint8 = (output.color * 255).clamp(0, 255).to(torch.uint8)

        # CPU로 이동 (한번만)
        color_np = color_uint8.cpu().numpy()

        # JPEG 인코딩 (OpenCV는 CPU에서 실행)
        _, color_jpeg = cv2.imencode('.jpg',
            cv2.cvtColor(color_np, cv2.COLOR_RGB2BGR),
            self.encode_params)

        # Depth는 GPU에서 float16 변환 후 CPU로
        depth_f16 = output.depth.to(torch.float16)
        depth_bytes = depth_f16.cpu().numpy().tobytes()

        # ...
```

### 4. Zero-Copy 전송 (고급)

```python
# PyTorch Tensor → Unix Socket 직접 전송 (중간 복사 제거)

async def send_payload_zerocopy(self, payload: RenderPayload):
    """Zero-copy 전송 (memoryview 사용)"""
    # Header
    header = struct.pack("<QII",
        payload.frame_id,
        len(payload.metadata),
        len(payload.data)
    )

    # Metadata
    metadata_bytes = json.dumps(payload.metadata).encode('utf-8')

    # Zero-copy write
    self.video_writer.write(header)
    self.video_writer.write(metadata_bytes)

    # Data를 memoryview로 전송 (복사 없음)
    if isinstance(payload.data, bytes):
        self.video_writer.write(memoryview(payload.data))

    await self.video_writer.drain()
```

### 5. 프로파일링

```python
import time

class ProfilingRendererService(RendererService):
    """성능 프로파일링이 포함된 Renderer"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.metrics = {
            "render_times": [],
            "encode_times": [],
            "send_times": [],
            "total_times": []
        }

    async def render_and_send_loop(self, camera_queue: asyncio.Queue):
        while True:
            camera = await camera_queue.get()

            total_start = time.perf_counter()

            # 1. Render
            render_start = time.perf_counter()
            render_output = await self.scene_renderer.render(camera)
            render_time = (time.perf_counter() - render_start) * 1000

            # 2. Encode
            encode_start = time.perf_counter()
            payload = await self.encoder.encode(render_output, camera.frame_id)
            encode_time = (time.perf_counter() - encode_start) * 1000

            # 3. Send
            send_start = time.perf_counter()
            await self.send_payload(payload)
            send_time = (time.perf_counter() - send_start) * 1000

            total_time = (time.perf_counter() - total_start) * 1000

            # Metrics 기록
            self.metrics["render_times"].append(render_time)
            self.metrics["encode_times"].append(encode_time)
            self.metrics["send_times"].append(send_time)
            self.metrics["total_times"].append(total_time)

            # 주기적으로 출력 (100 프레임마다)
            if camera.frame_id % 100 == 0:
                self.print_metrics()

            camera_queue.task_done()

    def print_metrics(self):
        """성능 메트릭 출력"""
        import numpy as np

        print(f"\n📊 Performance Metrics (last 100 frames):")
        print(f"  Render:  avg={np.mean(self.metrics['render_times'][-100:]):.2f}ms")
        print(f"  Encode:  avg={np.mean(self.metrics['encode_times'][-100:]):.2f}ms")
        print(f"  Send:    avg={np.mean(self.metrics['send_times'][-100:]):.2f}ms")
        print(f"  Total:   avg={np.mean(self.metrics['total_times'][-100:]):.2f}ms")
        print(f"  Target:  16.67ms (60 FPS)\n")
```

---

## 디버깅 및 모니터링

### 1. 로깅 설정

```python
# renderer/main.py

import logging

def setup_logging(level=logging.INFO):
    """로깅 설정"""
    logging.basicConfig(
        level=level,
        format='[%(asctime)s] %(levelname)s - %(name)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # 각 모듈별 로거
    logging.getLogger('renderer.service').setLevel(logging.DEBUG)
    logging.getLogger('renderer.scene').setLevel(logging.INFO)
    logging.getLogger('renderer.encoder').setLevel(logging.INFO)

if __name__ == "__main__":
    setup_logging()
    # ...
```

### 2. Frame ID 추적

```python
class RendererService:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_received_frame_id = -1
        self.last_sent_frame_id = -1

    async def camera_receive_loop(self, camera_queue: asyncio.Queue):
        while True:
            packet = await self.camera_reader.read(152)
            camera = parse_camera_frame(packet)

            # Frame ID 연속성 체크
            if camera.frame_id != self.last_received_frame_id + 1:
                print(f"⚠️  Frame skip detected: {self.last_received_frame_id} → {camera.frame_id}")

            self.last_received_frame_id = camera.frame_id
            await camera_queue.put(camera)

    async def send_payload(self, payload: RenderPayload):
        # ...

        # Frame ID 추적
        self.last_sent_frame_id = payload.frame_id
        print(f"📤 Sent frame {payload.frame_id}")
```

### 3. 중간 결과 저장 (디버깅)

```python
class DebugJPEGEncoder(JPEGEncoder):
    """디버깅용 Encoder (중간 결과 저장)"""

    def __init__(self, *args, save_dir="/tmp/debug_frames", **kwargs):
        super().__init__(*args, **kwargs)
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

    async def encode(self, output: RenderOutput, frame_id: int) -> RenderPayload:
        # 정상 인코딩
        payload = await super().encode(output, frame_id)

        # 중간 결과 저장 (10 프레임마다)
        if frame_id % 10 == 0:
            # Color 저장
            color_uint8 = (output.color * 255).clamp(0, 255).to(torch.uint8).cpu().numpy()
            cv2.imwrite(f"{self.save_dir}/color_{frame_id:06d}.png",
                       cv2.cvtColor(color_uint8, cv2.COLOR_RGB2BGR))

            # Depth 저장 (visualization)
            depth_vis = (output.depth.cpu().numpy() * 255).astype(np.uint8)
            cv2.imwrite(f"{self.save_dir}/depth_{frame_id:06d}.png", depth_vis)

            print(f"💾 Saved debug frames: {frame_id}")

        return payload
```

### 4. Health Check

```python
class RendererService:
    async def health_check_loop(self, interval=10):
        """주기적인 헬스 체크"""
        while True:
            await asyncio.sleep(interval)

            # GPU 메모리 체크
            allocated = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9

            # Queue 상태 체크
            queue_size = self.camera_queue.qsize()

            # Socket 연결 상태
            camera_connected = self.camera_writer and not self.camera_writer.is_closing()
            video_connected = self.video_writer and not self.video_writer.is_closing()

            print(f"💚 Health Check:")
            print(f"  GPU: {allocated:.2f}GB / {reserved:.2f}GB")
            print(f"  Queue: {queue_size} / {self.camera_queue.maxsize}")
            print(f"  Sockets: camera={camera_connected}, video={video_connected}")

    async def run(self):
        await asyncio.gather(
            self.camera_receive_loop(self.camera_queue),
            self.render_and_send_loop(self.camera_queue),
            self.health_check_loop(interval=10)  # 10초마다
        )
```

---

## 배포 체크리스트

### Phase 1 (MVP) 배포 전 확인사항

**코드:**
- [ ] `renderer/data_types.py` 구현 완료
- [ ] `renderer/scene_renderers/base.py` 구현 완료
- [ ] `renderer/scene_renderers/gaussian_splatting.py` 구현 완료
- [ ] `renderer/encoders/base.py` 구현 완료
- [ ] `renderer/encoders/jpeg.py` 구현 완료
- [ ] `renderer/utils/protocol.py` 구현 완료
- [ ] `renderer/renderer_service.py` 구현 완료
- [ ] `renderer/main.py` 진입점 구현

**테스트:**
- [ ] 테스트 1: Unix Socket 생성 (PASS)
- [ ] 테스트 2: Socket 양방향 통신 (PASS)
- [ ] 테스트 3: Scene Renderer 단독 (PASS)
- [ ] 테스트 4: Encoder 단독 (PASS)
- [ ] 테스트 5: E2E 데이터 패스 (PASS)

**성능:**
- [ ] 렌더링 레이턴시 < 16.67ms (60 FPS)
- [ ] E2E 레이턴시 < 50ms
- [ ] GPU 메모리 사용량 확인
- [ ] 100 프레임 연속 렌더링 안정성

**Docker:**
- [ ] `renderer/Dockerfile` 작성
- [ ] `renderer/requirements.txt` 작성
- [ ] Docker 이미지 빌드 성공
- [ ] Docker Compose 통합 테스트

**문서:**
- [ ] README.md 업데이트
- [ ] API 문서 작성
- [ ] 배포 가이드 작성

**통합:**
- [ ] Transport Service와 연결 확인
- [ ] Frontend와 E2E 테스트
- [ ] Frame ID 일치 확인
- [ ] 데이터 무결성 확인

---

## 결론

이 아키텍처는 **모듈화**, **확장성**, **성능**을 모두 고려한 설계입니다.

**핵심 원칙:**
- ✅ 관심사의 분리
- ✅ 조합 가능성
- ✅ 확장 가능성
- ✅ YAGNI (필요시 확장)

**다음 단계:**
1. Phase 1 구현 (MVP)
2. 통합 테스트
3. 성능 측정
4. Phase 2 확장 기능 추가
