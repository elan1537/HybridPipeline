# Frontend Architecture

> **Last Updated**: 2025-01-13 (Phase 6 완료)

이 문서는 `frontend/src/main.ts`의 모듈 의존성과 데이터 흐름을 설명합니다.

---

## 목차

1. [시스템 개요](#시스템-개요)
2. [초기화 흐름](#초기화-흐름)
3. [카메라 전송 흐름](#카메라-전송-흐름-camera--server)
4. [비디오 수신 흐름](#비디오-수신-흐름-server--render)
5. [렌더 루프](#렌더-루프-render-loop)
6. [UI 이벤트 흐름](#ui-이벤트-흐름)
7. [주요 모듈 설명](#주요-모듈-설명)
8. [WebSocketSystem API](#websocketsystem-api)

---

## 시스템 개요

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              Frontend Architecture                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   main.ts                                                                   │
│     │                                                                       │
│     ├── Application ─────────────────────────────────────────────────────┐ │
│     │     ├── RenderingContext    (THREE.js 객체 래핑)                   │ │
│     │     ├── EventBus            (시스템간 이벤트 통신)                  │ │
│     │     ├── ApplicationState    (전역 상태 관리)                       │ │
│     │     ├── TimeController      (프레임 시간 제어)                     │ │
│     │     │                                                              │ │
│     │     └── Systems:                                                   │ │
│     │         ├── WebSocketSystem   → decode-worker.ts                   │ │
│     │         ├── TextureManager    → 텍스처 생성/업데이트               │ │
│     │         ├── CameraController  → 카메라 상태 관리                   │ │
│     │         └── RenderingSystem   → 렌더 루프 실행                     │ │
│     │                                                                    │ │
│     └── UISystem ────────────────────────────────────────────────────────┤ │
│           ├── ControlPanel      (연결/모드 제어)                         │ │
│           ├── DebugPanel        (카메라 디버그)                          │ │
│           ├── StatsPanel        (FPS/레이턴시 표시)                      │ │
│           ├── FPSTestPanel      (성능 측정)                              │ │
│           └── RecordingPanel    (화면 녹화)                              │ │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 초기화 흐름

```
main.ts
  │
  ├─ [전역 선언] ─────────────────────────────────────────────────────────────────
  │   ├─ camera, renderer, controls                    ← THREE.js 기본 객체
  │   ├─ localScene, fusionScene, debugScene, ...      ← 6개 Scene
  │   ├─ fusionMaterial, debugMaterial, ...            ← 6개 ShaderMaterial
  │   ├─ latencyTracker = new LatencyTracker()         ← 레이턴시 측정
  │   ├─ app: Application | null = null                ← line 404
  │   └─ uiSystem: UISystem | null = null              ← line 405
  │
  ├─ initScene() ─────────────────────────────────────────────────────────────────
  │   ├─ new THREE.PerspectiveCamera()                 ← camera 생성
  │   ├─ new THREE.WebGLRenderer()                     ← renderer 생성
  │   ├─ new OrbitControls()                           ← controls 생성
  │   ├─ new THREE.Scene() × 6                         ← 각 렌더 모드용 Scene
  │   ├─ new THREE.ShaderMaterial() × 6                ← 각 렌더 모드용 Material
  │   ├─ new THREE.WebGLRenderTarget()                 ← 오프스크린 렌더링
  │   ├─ object_setup()                                ← 씬 오브젝트 배치
  │   └─ window.addEventListener('resize', ...)        ← 리사이즈 핸들러
  │
  └─ initScene().then(async () => { ... }) ───────────────────────────────────────
      │
      ├─ app = new Application({ config })
      │   │
      │   └─ Application 내부:
      │       ├─ this.eventBus = new EventBus()
      │       ├─ this.state = new ApplicationState()
      │       └─ this.timeController = new TimeController()
      │
      ├─ app.initializeWithExistingObjects(scene, camera, renderer, controls)
      │   │
      │   └─ Application.initializeWithExistingObjects():
      │       ├─ this.renderingContext = new RenderingContext(...)
      │       │
      │       ├─ this.websocketSystem = new WebSocketSystem()
      │       ├─ this.textureManager = new TextureManager()
      │       ├─ this.cameraController = new CameraController()
      │       ├─ this.renderingSystem = new RenderingSystem()
      │       │
      │       ├─ systems.set("websocket", websocketSystem)
      │       ├─ systems.set("texture", textureManager)
      │       ├─ systems.set("camera", cameraController)
      │       ├─ systems.set("rendering", renderingSystem)
      │       │
      │       └─ Promise.all(systems.map(s => s.initialize(context)))
      │           │
      │           ├─ WebSocketSystem.initialize(context)
      │           │   ├─ this.worker = new Worker("decode-worker.ts")
      │           │   ├─ this.setupWorkerHandlers()
      │           │   │   ├─ worker.onmessage = handleWorkerMessage
      │           │   │   └─ worker.onerror = console.error
      │           │   └─ context.state.subscribe("connection:state", ...)
      │           │
      │           ├─ TextureManager.initialize(context)
      │           │   ├─ this.createColorTexture()
      │           │   └─ this.createDepthTexture()
      │           │
      │           ├─ CameraController.initialize(context)
      │           │   └─ context.state.subscribe("camera:config", ...)
      │           │
      │           └─ RenderingSystem.initialize(context)
      │               └─ (설정 대기 상태)
      │
      ├─ uiSystem = new UISystem(app, latencyTracker)
      │   │
      │   └─ UISystem 내부:
      │       ├─ this.control = new ControlPanel(callbacks)
      │       ├─ this.debug = new DebugPanel(callbacks)
      │       ├─ this.stats = new StatsPanel()
      │       ├─ this.fpsTest = new FPSTestPanel(callbacks)
      │       └─ this.recording = new RecordingPanel()
      │
      ├─ uiSystem.initialize(context)
      │
      ├─ wsSystem = app.getWebSocketSystem()
      │   │
      │   ├─ wsSystem.setCallbacks({...})              ← UI/레이턴시 콜백 연결
      │   │   ├─ onConnectionStateChange → uiSystem.setConnectionState()
      │   │   ├─ onDecodeStats → uiSystem.updateDecodeFPS()
      │   │   ├─ onFrameReceive → latencyTracker.recordFrameReceive()
      │   │   ├─ onFrameDecoded → latencyTracker.recordDecodeComplete()
      │   │   └─ onClockSync → latencyTracker.recordClockSync()
      │   │
      │   └─ wsSystem.connect(wsURL, width, height)    ← 초기 WebSocket 연결
      │       │
      │       └─ WebSocketSystem.connect():
      │           └─ this.worker.postMessage({ type: 'init', ... })
      │
      ├─ renderingContext.registerMaterial('fusion', fusionMaterial)
      ├─ renderingContext.registerMaterial('debug', debugMaterial)
      ├─ renderingContext.registerMaterial('depthFusion', depthFusionMaterial)
      │
      ├─ texManager.initializeShaderMaterials()        ← 텍스처 → Material 연결
      │
      ├─ renderingSystem.configure({...})
      │   ├─ scenes: localScene, fusionScene, ...
      │   ├─ cameras: camera, orthoCamera, ...
      │   ├─ onCameraUpdate: () => app.sendCameraFrame()
      │   └─ onUpdate: () => { robot_animation(); updateLatencyStats(); }
      │
      └─ app.start()                                   ← 렌더 루프 시작
          │
          └─ Application.start():
              └─ requestAnimationFrame(this.loop)
```

---

## 카메라 전송 흐름 (Camera → Server)

```
RenderingSystem.render()
  │
  └─ onCameraUpdate()                                  ← 매 프레임 콜백
      │
      └─ Application.sendCameraFrame()
          │
          ├─ CameraController.getCurrentFrame()        ← CameraFrame 생성
          │   └─ { eye, target, intrinsics, frameId, clientTimestamp, timeIndex }
          │
          └─ WebSocketSystem.sendCameraFrame(frame)
              │
              └─ this.worker.postMessage({ type: 'camera', frame })
                  │
                  └─ decode-worker.ts:
                      └─ ws.send(binaryData)           ← 260 bytes to Backend
```

### CameraFrame 구조

| 필드 | 타입 | 설명 |
|------|------|------|
| `eye` | `Vector3` | 카메라 위치 |
| `target` | `Vector3` | 카메라 타겟 |
| `intrinsics` | `Matrix3` | 카메라 내부 파라미터 |
| `frameId` | `number` | 프레임 ID |
| `clientTimestamp` | `number` | 클라이언트 타임스탬프 |
| `timeIndex` | `number` | 시간 인덱스 (Feed-forward용) |

---

## 비디오 수신 흐름 (Server → Render)

```
Backend
  │
  └─ WebSocket binary frame
      │
      └─ decode-worker.ts (ws.onmessage)
          │
          ├─ parseHeader()                             ← 44/56 bytes 헤더
          ├─ decodeH264() or decodeJPEG()              ← color + depth
          │
          └─ self.postMessage({ type: 'video-frame', color, depth, ... })
              │
              └─ WebSocketSystem (worker.onmessage)
                  │
                  └─ handleWorkerMessage(data)
                      │
                      ├─ case 'video-frame':
                      │   │
                      │   └─ handleVideoFrame(msg)
                      │       │
                      │       ├─ VideoFrame 생성 { colorBitmap, depthBitmap, ... }
                      │       │
                      │       ├─ eventBus.emit('frame:received', { data: frame })
                      │       │   │
                      │       │   └─ Application.setupEventListeners()에서 수신
                      │       │       │
                      │       │       └─ TextureManager.updateFromVideoFrame(frame)
                      │       │           ├─ colorTexture.image = colorBitmap
                      │       │           ├─ depthTexture.image.data = depthData
                      │       │           └─ texture.needsUpdate = true
                      │       │
                      │       ├─ callbacks.onFrameReceive(frameId, timestamps)
                      │       │   └─ latencyTracker.recordFrameReceive()
                      │       │
                      │       └─ callbacks.onFrameDecoded(frameId)
                      │           └─ latencyTracker.recordDecodeComplete()
                      │
                      ├─ case 'pure-decode-stats':
                      │   └─ callbacks.onDecodeStats(stats)
                      │       ├─ uiSystem.updateDecodeFPS()
                      │       └─ latencyTracker.recordPureDecodeFPS()
                      │
                      ├─ case 'pong-received':
                      │   └─ callbacks.onClockSync(...)
                      │       └─ latencyTracker.recordClockSync()
                      │
                      └─ case 'ws-state':
                          └─ handleConnectionState(state)
                              └─ callbacks.onConnectionStateChange(state)
                                  └─ uiSystem.setConnectionState()
```

### VideoFrame 구조

| 필드 | 타입 | 설명 |
|------|------|------|
| `colorBitmap` | `ImageBitmap` | 컬러 이미지 (GPU 최적화) |
| `depthBitmap` | `ImageBitmap?` | Depth 이미지 (H.264 모드) |
| `depthRaw` | `Uint16Array?` | Depth 데이터 (JPEG 모드) |
| `frameId` | `number` | 프레임 ID |
| `width` | `number` | 이미지 너비 |
| `height` | `number` | 이미지 높이 |
| `formatType` | `FormatType` | 인코딩 형식 (JPEG/H264) |
| `timestamps` | `object` | 서버 타임스탬프 |

---

## 렌더 루프 (Render Loop)

```
Application.loop() [requestAnimationFrame]
  │
  ├─ this.update(deltaTime)
  │   │
  │   └─ systems.forEach(s => s.update(deltaTime))
  │       ├─ WebSocketSystem.update()        ← (no-op)
  │       ├─ TextureManager.update()         ← (no-op)
  │       ├─ CameraController.update()       ← 카메라 상태 업데이트
  │       └─ RenderingSystem.update()        ← (no-op)
  │
  └─ this.render()
      │
      └─ RenderingSystem.render()
          │
          ├─ onUpdate()                      ← 매 프레임 콜백
          │   ├─ robot_animation()
          │   ├─ updateLatencyStats()
          │   └─ uiSystem.updateFPSTestUI()
          │
          ├─ renderer.setRenderTarget(localRenderTarget)
          ├─ renderer.render(localScene, camera)          ← Local 씬 렌더
          │
          ├─ renderer.setRenderTarget(null)
          ├─ switch(renderMode):
          │   ├─ FUSION       → renderer.render(fusionScene, orthoCamera)
          │   ├─ GAUSSIAN_ONLY → renderer.render(gaussianOnlyScene, ...)
          │   ├─ LOCAL_ONLY   → renderer.render(localOnlyScene, ...)
          │   ├─ DEPTH_FUSION → renderer.render(depthFusionScene, ...)
          │   └─ DEBUG        → renderer.render(debugScene, debugCamera)
          │
          └─ onCameraUpdate()                ← 카메라 변경 시 서버 전송
              └─ app.sendCameraFrame()
```

### 렌더 모드

| 모드 | Scene | 설명 |
|------|-------|------|
| `FUSION` | fusionScene | Gaussian + Local Mesh 합성 |
| `GAUSSIAN_ONLY` | gaussianOnlyScene | Gaussian만 표시 |
| `LOCAL_ONLY` | localOnlyScene | Local Mesh만 표시 |
| `DEPTH_FUSION` | depthFusionScene | Depth 기반 합성 |
| `FEED_FORWARD` | fusionScene | 프레임별 PLY 렌더링 |
| `DEBUG` | debugScene | 4분할 디버그 뷰 |

---

## UI 이벤트 흐름

### Connect 버튼 클릭

```
User Click (Connect Button)
  │
  └─ ControlPanel.handleConnect()
      │
      └─ UISystem.onConnect()
          │
          └─ WebSocketSystem.connect(url, width, height)
              │
              └─ worker.postMessage({ type: 'init', ... })
```

### JPEG 체크박스 변경

```
User Click (JPEG Checkbox)
  │
  └─ ControlPanel.handleEncoderChange()
      │
      └─ UISystem.onEncoderChange(isJpeg)
          │
          ├─ TextureManager.setJpegMode(isJpeg)
          │
          └─ WebSocketSystem.changeEncoderType('jpeg' | 'h264')
              │
              └─ reconnect(newURL, width, height)
                  ├─ disconnect()
                  │   └─ worker.postMessage({ type: 'ws-close' })
                  │
                  └─ connect(newURL, ...)
                      └─ worker.postMessage({ type: 'init', ... })
```

### 렌더 모드 변경

```
User Click (Render Mode Radio)
  │
  └─ ControlPanel.handleRenderModeChange(mode)
      │
      └─ UISystem.onRenderModeChange(mode)
          │
          └─ Application.setRenderMode(mode)
              │
              └─ RenderingSystem.setMode(mode)
```

---

## 주요 모듈 설명

### Application (`src/core/Application.ts`)

중앙 조율자 역할. 모든 시스템을 생성하고 관리합니다.

```typescript
class Application {
  // Core
  renderingContext: RenderingContext;
  eventBus: EventBus;
  state: ApplicationState;
  timeController: TimeController;

  // Systems
  systems: Map<string, System>;

  // Lifecycle
  start(): void;
  stop(): void;

  // API
  sendCameraFrame(): void;
  setRenderMode(mode: RenderMode): void;
  getWebSocketSystem(): WebSocketSystem;
  getTextureManager(): TextureManager;
  getRenderingSystem(): RenderingSystem;
}
```

### WebSocketSystem (`src/systems/WebSocketSystem.ts`)

Web Worker를 통한 WebSocket 통신 관리. **decode-worker.ts**를 내부에서 생성하고 관리합니다.

### TextureManager (`src/systems/TextureManager.ts`)

GPU 텍스처 생성 및 업데이트. 비디오 프레임을 THREE.js 텍스처로 변환합니다.

### RenderingSystem (`src/systems/RenderingSystem.ts`)

렌더 루프 실행 및 렌더 모드 전환 관리.

### UISystem (`src/systems/UISystem.ts`)

UI 패널들의 상위 관리자. Application과 UI 패널 간 이벤트 브릿지 역할.

---

## WebSocketSystem API

### Lifecycle

| 메서드 | 설명 |
|--------|------|
| `initialize(context)` | Worker 생성, 핸들러 설정, 상태 구독 |
| `update(deltaTime)` | (사용 안함 - WebSocket은 이벤트 기반) |
| `dispose()` | Worker 종료, 핸들러 정리 |

### Connection 관리

| 메서드 | 설명 |
|--------|------|
| `connect(url, width, height)` | WebSocket 연결 시작 |
| `disconnect()` | WebSocket 연결 종료 |
| `reconnect(url, width, height)` | disconnect() → 100ms 후 connect() |
| `getConnectionState()` | 현재 연결 상태 반환 |
| `isConnected()` | ConnectionState.Connected 여부 |
| `getURL()` | 현재 WebSocket URL |
| `getResolution()` | { width, height } |

### 데이터 송수신

| 메서드 | 설명 |
|--------|------|
| `sendCameraFrame(frame)` | 카메라 프레임을 서버로 전송 |
| `onVideoFrame(handler)` | 비디오 프레임 수신 핸들러 등록 |
| `onConnectionState(handler)` | 연결 상태 변경 핸들러 등록 |

### 설정 변경

| 메서드 | 설명 |
|--------|------|
| `changeEncoderType('jpeg'\|'h264')` | 인코더 타입 변경 (재연결) |
| `toggleJPEGFallback(enabled)` | JPEG 모드 전환 (재연결) |
| `startFPSMeasurement()` | FPS 측정 시작 |
| `stopFPSMeasurement()` | FPS 측정 중지 |

### 콜백 설정

```typescript
wsSystem.setCallbacks({
  onConnectionStateChange?: (state) => void;  // 연결 상태 변경
  onDecodeStats?: (stats) => void;            // 디코드 통계 수신
  onFrameReceive?: (frameId, timestamps) => void;  // 프레임 수신
  onFrameDecoded?: (frameId, time) => void;   // 프레임 디코드 완료
  onClockSync?: (client, serverRx, serverTx) => void;  // 시계 동기화
});
```

---

## 주요 객체 생성 시점

| 객체 | 선언 위치 | 생성 시점 | 접근 방법 |
|------|----------|----------|----------|
| `app` | main.ts:404 | initScene().then() | 전역 변수 |
| `uiSystem` | main.ts:405 | app 생성 후 | 전역 변수 |
| `wsSystem` | Application 내부 | initializeWithExistingObjects() | `app.getWebSocketSystem()` |
| `texManager` | Application 내부 | initializeWithExistingObjects() | `app.getTextureManager()` |
| `renderingSystem` | Application 내부 | initializeWithExistingObjects() | `app.getRenderingSystem()` |
| Worker | WebSocketSystem 내부 | initialize() | (직접 접근 불가) |

---

## 파일 구조

```
frontend/src/
├── main.ts                          # 진입점, 초기화 로직
├── decode-worker.ts                 # Web Worker (WebSocket + 디코딩)
├── debug-logger.ts                  # 로깅 유틸리티
├── ui-controller.ts                 # UI 가시성 제어 (H키, 자동숨김)
├── latency-tracker.ts               # 레이턴시 측정
│
├── core/
│   ├── Application.ts               # 메인 애플리케이션
│   ├── RenderingContext.ts          # THREE.js 객체 래핑
│   ├── EventBus.ts                  # 이벤트 통신
│   └── ApplicationState.ts          # 상태 관리
│
├── systems/
│   ├── WebSocketSystem.ts           # WebSocket + Worker 관리
│   ├── TextureManager.ts            # 텍스처 관리
│   ├── CameraController.ts          # 카메라 제어
│   ├── RenderingSystem.ts           # 렌더 루프
│   └── UISystem.ts                  # UI 시스템
│
├── ui/
│   ├── panels/
│   │   ├── BasePanel.ts             # 패널 추상 클래스
│   │   ├── ControlPanel.ts          # 연결/모드 제어
│   │   ├── DebugPanel.ts            # 카메라 디버그
│   │   ├── StatsPanel.ts            # FPS/레이턴시 표시
│   │   ├── FPSTestPanel.ts          # 성능 측정
│   │   └── RecordingPanel.ts        # 화면 녹화
│   └── managers/
│       ├── CameraStateManager.ts    # 카메라 상태 저장/불러오기
│       └── SettingsManager.ts       # 설정 관리
│
├── shaders/
│   ├── fusionVertexShader.vs
│   ├── fusionColorShader.fs
│   ├── debugVertexShader.vs
│   ├── debugColorShader.fs
│   └── depthFusionShader.fs
│
├── types/
│   └── index.ts                     # 타입 정의
│
└── state/
    └── scene-state.ts               # 씬 상태 (로봇/메시 애니메이션)
```

---

## 변경 이력

| 날짜 | Phase | 설명 |
|------|-------|------|
| 2025-01-13 | Phase 6 | Worker 통신 완전 통합 (main.ts → WebSocketSystem) |
| 2025-01-13 | Phase 5 | BasePanel 추상 클래스, 5개 패널 리팩토링 |
| 2025-01-13 | Phase 4.5 | Depth Fusion 버그 수정 |
| 2025-01-13 | Phase 4 | 워커 메시지 핸들러 통합 |
| 2025-01-13 | Phase 3 | DebugSystem, PhysicsSystem, physics/ 삭제 |
| 2025-01-13 | Phase 1-2 | test/ 디렉토리 삭제, dead code 제거 |
