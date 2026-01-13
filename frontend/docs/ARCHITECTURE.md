# Frontend Architecture

> **Last Updated**: 2025-01-14 (kebab-case 리팩토링 완료)

이 문서는 HybridPipeline Frontend의 전체 아키텍처를 설명합니다.

---

## 목차

1. [전체 구조 개요](#1-전체-구조-개요)
2. [변수간 종속 관계](#2-변수간-종속-관계-dependency-graph)
3. [프레임워크 구조](#3-프레임워크-구조)
4. [데이터 흐름](#4-데이터-흐름-data-flow)
5. [함수 호출 깊이](#5-함수-호출-깊이-call-depth)
6. [이벤트 시스템](#6-이벤트-시스템)
7. [렌더링 모드](#7-렌더링-모드)
8. [주요 타입 정의](#8-주요-타입-정의)
9. [초기화 흐름](#9-초기화-흐름)
10. [API Reference](#10-api-reference)

---

## 1. 전체 구조 개요

### 1.1 디렉토리 구조

```
src/
├── main.ts                       # Entry Point (레거시 + Application 브릿지)
│
├── core/                         # 핵심 인프라
│   ├── application.ts            # 메인 오케스트레이터
│   ├── application-state.ts      # 중앙 상태 관리 (122개 전역변수 대체)
│   ├── event-bus.ts              # 이벤트 기반 통신
│   ├── rendering-context.ts      # THREE.js 객체 래퍼
│   ├── debug-logger.ts           # 로깅 유틸리티
│   └── latency-tracker.ts        # 성능 측정
│
├── systems/                      # 핵심 시스템 (System 인터페이스 구현)
│   ├── websocket-system.ts       # WebSocket + Worker 통신
│   ├── texture-manager.ts        # 텍스처 관리
│   ├── camera-controller.ts      # 카메라 제어
│   ├── rendering-system.ts       # 렌더 루프
│   └── ui-system.ts              # UI 조율
│
├── ui/                           # 사용자 인터페이스
│   ├── ui-controller.ts          # UI 가시성 제어 (H키, 자동숨김)
│   ├── time-control-ui.ts        # Dynamic Gaussian 시간 제어
│   ├── frame-scrubber.ts         # 타임라인 스크러버
│   ├── panels/                   # UI 패널들 (BasePanel 상속)
│   │   ├── base-panel.ts         # 패널 추상 클래스
│   │   ├── control-panel.ts      # 연결/모드 제어
│   │   ├── debug-panel.ts        # 카메라 디버그
│   │   ├── stats-panel.ts        # FPS/레이턴시 표시
│   │   ├── fps-test-panel.ts     # 성능 측정
│   │   └── recording-panel.ts    # 화면 녹화
│   └── managers/                 # UI 헬퍼
│       ├── camera-state-manager.ts  # 카메라 상태 저장/불러오기
│       └── settings-manager.ts      # 설정 관리
│
├── scene/                        # 3D 씬 관리
│   ├── scene-setup.ts            # 씬 초기화 (로봇, 오브젝트)
│   └── scene-state.ts            # 씬 상태 컨테이너
│
├── workers/                      # Web Worker
│   └── decode-worker.ts          # H.264/JPEG 디코딩
│
├── shaders/                      # GLSL 셰이더
│   ├── fusionVertexShader.vs
│   ├── fusionColorShader.fs
│   ├── debugVertexShader.vs
│   ├── debugColorShader.fs
│   ├── depthFusionShader.fs
│   ├── gaussianOnlyShader.fs
│   └── localOnlyShader.fs
│
└── types/                        # TypeScript 타입 정의
    └── index.ts
```

### 1.2 시스템 개요도

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
│           ├── RecordingPanel    (화면 녹화)                              │ │
│           ├── TimeControlUI     (시간 제어)                              │ │
│           └── FrameScrubber     (타임라인)                               │ │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. 변수간 종속 관계 (Dependency Graph)

### 2.1 클래스 계층 구조

```
Application (Root Orchestrator)
    │
    ├─→ EventBus (owns)           ← 모든 시스템이 이벤트 통신에 사용
    │
    ├─→ ApplicationState (owns)   ← 모든 시스템이 상태 읽기/쓰기
    │
    ├─→ TimeController (owns)     ← Dynamic Gaussian 시간 제어
    │
    ├─→ RenderingContext (owns)   ← THREE.js 객체 래퍼
    │
    └─→ Systems Map (owns)
         │
         ├─ WebSocketSystem
         │   ├─→ Worker (decode-worker.ts)
         │   ├─→ EventBus (emit: frame:received)
         │   └─→ ApplicationState (write: connection:state)
         │
         ├─ TextureManager
         │   ├─→ RenderingContext (텍스처 등록)
         │   ├─→ ApplicationState (구독: resolution 변경)
         │   └─→ ShaderMaterial uniforms (텍스처 참조 업데이트)
         │
         ├─ CameraController
         │   ├─→ RenderingContext.camera & controls
         │   └─→ EventBus (emit: camera-moved)
         │
         ├─ RenderingSystem
         │   ├─→ RenderingContext.renderer
         │   └─→ ApplicationState (read: rendering:mode)
         │
         └─ UISystem
             ├─→ Application (시스템 접근용)
             ├─→ LatencyTracker
             └─→ All UI Panels
```

### 2.2 의존성 매트릭스

```
                  EVB  STATE  RCtx   WS   TM   CC   RS   UI   Worker
Application       own   own   own   own  own  own  own  own   ---
EventBus          base  ----  ----  use  use  use  use  use   ---
ApplicationState  ----  base  ----  use  use  use  use  use   ---
RenderingContext  ----  ----  base  use  use  use  use  use   ---
WebSocketSystem   emit  write read  base ----  ---- ----  ----  own
TextureManager    emit  sub   reg   ----  base ---- ----  ----  ---
CameraController  emit  sub   read  ----  ---- base ----  ----  ---
RenderingSystem   emit  read  read  ----  ---- ---- base  ----  ---
UISystem          use   use   use   use  use  use  use   base  ---

Legend:
  own  = 소유 (생성 및 관리)
  use  = 사용 (읽기/호출)
  emit = 이벤트 발행
  sub  = 상태 구독
  reg  = 리소스 등록
  read = 읽기 전용
  write = 쓰기
  base = 기본 클래스/인터페이스
  ---  = 의존성 없음
```

### 2.3 순환 의존성 방지 패턴

```
✓ 단방향 의존성 (No Cycles)
    Application → Systems → EventBus/State/RenderingContext

✓ 이벤트 기반 통신 (Decoupled)
    WebSocketSystem ──emit("frame:received")──→ TextureManager (listens)
    직접 호출 대신 이벤트로 통신하여 순환 의존성 방지

✓ 콜백 패턴 (Explicit Contracts)
    RenderingSystem.onCameraUpdate = () => app.sendCameraFrame()
    시스템이 외부 동작을 알 필요 없이 콜백만 실행

✓ 인터페이스 분리
    System interface: initialize(), update(), dispose()
    모든 시스템이 동일한 생명주기 인터페이스 구현
```

### 2.4 상세 의존성 트리

```
Application (root orchestrator)
    │
    ├─→ EventBus (owns)
    │    └─ used by all systems for decoupled communication
    │
    ├─→ ApplicationState (owns)
    │    └─ read/write by all systems
    │
    ├─→ TimeController (owns)
    │    ├─ read by CameraController (timeIndex)
    │    ├─ read by RenderingSystem
    │    └─ controlled by TimeControlUI
    │
    ├─→ RenderingContext (owns)
    │    ├─ wraps existing THREE.js objects
    │    ├─ registry for render targets, textures, meshes, materials
    │    └─ read by all systems
    │
    └─→ Systems Map (owns)
         │
         ├─ WebSocketSystem
         │   ├─→ Worker (create or accept external)
         │   ├─→ EventBus (emit frame:received)
         │   ├─→ ApplicationState (write connection:state)
         │   └─→ Callbacks (onDecodeStats, onFrameDecoded, etc.)
         │
         ├─ TextureManager
         │   ├─→ RenderingContext (register textures)
         │   ├─→ ApplicationState (subscribe to resolution changes)
         │   ├─→ EventBus (via updateFromVideoFrame listener)
         │   └─→ ShaderMaterial uniforms (update texture references)
         │
         ├─ CameraController
         │   ├─→ RenderingContext.camera & controls
         │   ├─→ ApplicationState (resolution, camera:config)
         │   └─→ EventBus (emit camera-moved, camera-config-updated)
         │
         ├─ RenderingSystem
         │   ├─→ RenderingContext.renderer
         │   ├─→ RenderingConfig (scenes, cameras, render targets)
         │   ├─→ ApplicationState (rendering:mode)
         │   ├─→ EventBus (emit camera:update)
         │   └─→ Callbacks (onCameraUpdate, onUpdate)
         │
         └─ UISystem
             ├─→ Application (for system access)
             ├─→ LatencyTracker (passed via constructor)
             ├─→ All UI Panels
             │   ├─ StatsPanel (display only)
             │   ├─ ControlPanel (→ WebSocketSystem, RenderingSystem)
             │   ├─ DebugPanel (→ RenderingSystem)
             │   ├─ RecordingPanel (canvas capture)
             │   ├─ FPSTestPanel (→ WebSocketSystem, LatencyTracker)
             │   └─ TimeControlUI/FrameScrubber (→ TimeController)
             │
             └─→ Managers
                 ├─ CameraStateManager (localStorage)
                 └─ SettingsManager
```

---

## 3. 프레임워크 구조

### 3.1 Phase 1 하이브리드 아키텍처

```
┌─────────────────────────────────┬─────────────────────────────────┐
│     Legacy Code (main.ts)       │     Modular Systems Layer       │
├─────────────────────────────────┼─────────────────────────────────┤
│ Global variables                │ ApplicationState                │
│   scene, camera, renderer       │        ↑                        │
│         │                       │        │ (wrapped)              │
│         ↓                       │        │                        │
│    THREE.js objects        ─────→   RenderingContext              │
│                                 │        │                        │
│                                 │        ↓                        │
│                                 │   System instances              │
│                                 │   (병렬 실행)                    │
└─────────────────────────────────┴─────────────────────────────────┘

장점: 레거시 코드 수정 없이 새 시스템 추가 가능
```

### 3.2 모듈 관계도

```
APPLICATION LAYER
    └─ main.ts
        ├─ Creates Application instance
        ├─ Creates THREE.js objects (scene, camera, renderer)
        └─ Coordinates initialization & render loop

CORE LAYER
    ├─ Application
    │   ├─ Owns: EventBus, ApplicationState, Systems Map, TimeController
    │   ├─ Coordinates: system initialization, update/render cycles
    │   └─ Provides: system access, public API
    │
    ├─ ApplicationState
    │   ├─ Centralized key-value store
    │   ├─ Pub/sub listeners for state changes
    │   └─ Replaces ~122 global variables
    │
    ├─ EventBus
    │   ├─ Typed event emit/subscribe
    │   ├─ Supports sync & async handlers
    │   └─ Enables decoupled system communication
    │
    ├─ RenderingContext
    │   ├─ Wraps existing THREE.js objects (Phase 1)
    │   └─ Registry for render targets, textures, meshes, materials
    │
    └─ System Interface (base for all systems)
        ├─ initialize(context): setup
        ├─ update(deltaTime): per-frame logic
        └─ dispose(): cleanup

SYSTEMS LAYER
    ├─ WebSocketSystem
    │   ├─ Owns: Worker, connection state
    │   ├─ Manages: WebSocket lifecycle, camera/control messages
    │   └─ Emits: frame:received events
    │
    ├─ TextureManager
    │   ├─ Owns: colorTexture, depthTexture
    │   ├─ Manages: texture creation, updates, mode switching
    │   └─ Updates: shader material uniforms
    │
    ├─ CameraController
    │   ├─ Owns: camera, controls, config
    │   ├─ Provides: camera frames, movement detection
    │   └─ Handles: position saving/loading
    │
    ├─ RenderingSystem
    │   ├─ Owns: render config, state
    │   ├─ Manages: render loop, render mode selection
    │   └─ Emits: camera:update events
    │
    └─ UISystem
        ├─ Owns: all UI panels
        ├─ Coordinates: panel interactions with systems
        └─ Provides: UI update methods

UI LAYER
    ├─ UISystem (coordinator)
    │
    ├─ Panels (independent components)
    │   ├─ StatsPanel (read-only display)
    │   ├─ ControlPanel (connection, encoder, render mode)
    │   ├─ DebugPanel (logging, camera debug)
    │   ├─ RecordingPanel (video capture)
    │   ├─ FPSTestPanel (performance measurement)
    │   └─ TimeControlUI (gaussian time playback)
    │
    └─ Managers (utilities)
        ├─ CameraStateManager (localStorage)
        └─ SettingsManager

WORKER LAYER
    └─ decode-worker.ts
        ├─ H.264/JPEG decoding
        ├─ Frame assembly
        ├─ Clock sync with server
        └─ Pure decode FPS measurement
```

---

## 4. 데이터 흐름 (Data Flow)

### 4.1 전체 파이프라인

```
┌─────────────────────────────────────────────────────────────────┐
│                     HYBRID RENDERING PIPELINE                   │
└─────────────────────────────────────────────────────────────────┘

                          SERVER
                             │
                    (WebSocket binary stream)
                             │
                             ▼
         ┌──────────────────────────────────┐
         │   DECODE-WORKER.TS               │
         │   (Web Worker in browser)        │
         │                                  │
         │  ┌─────────────────────────────┐ │
         │  │ WebSocket Listener          │ │
         │  │ - Listen on ws connection   │ │
         │  │ - Receive video frame data  │ │
         │  └──────────┬──────────────────┘ │
         │             │                    │
         │  ┌──────────▼──────────────────┐ │
         │  │ Frame Parser/Assembler      │ │
         │  │ - Parse binary header       │ │
         │  │ - Handle H.264/JPEG         │ │
         │  │ - Extract color + depth     │ │
         │  └──────────┬──────────────────┘ │
         │             │                    │
         │  ┌──────────▼──────────────────┐ │
         │  │ Decoder (hw accelerated)    │ │
         │  │ - Canvas decode             │ │
         │  │ - ImageBitmap creation      │ │
         │  └──────────┬──────────────────┘ │
         │             │                    │
         │  ┌──────────▼──────────────────┐ │
         │  │ Send to Main Thread         │ │
         │  │ - postMessage({             │ │
         │  │   type: "frame",            │ │
         │  │   image: ImageBitmap,       │ │
         │  │   depth: ImageBitmap|Raw    │ │
         │  │ })                          │ │
         │  └─────────────────────────────┘ │
         └──────────────────────────────────┘
                      │
                      │ postMessage (ImageBitmap)
                      ▼
    ┌─────────────────────────────────────────┐
    │      WEBSOCKETSYSTEM.TS                  │
    │      (Main Thread - System)              │
    │                                          │
    │  ┌───────────────────────────────────┐   │
    │  │ Worker Message Handler            │   │
    │  │ - Receive frame data from worker  │   │
    │  │ - Convert to VideoFrame interface │   │
    │  │ - Emit "frame:received" event     │   │
    │  └──────────┬────────────────────────┘   │
    │             │                            │
    │  ┌──────────▼────────────────────────┐   │
    │  │ Connection State Handler          │   │
    │  │ - Track ws connection state       │   │
    │  │ - Update ApplicationState         │   │
    │  │ - Call callbacks                  │   │
    │  └───────────────────────────────────┘   │
    └─────────────────────────────────────────┘
                      │
         ┌────────────┴────────────┐
         │                         │
         │ EventBus.emit(          │ Callbacks
         │ "frame:received",       │ onDecodeStats,
         │ {frameId, data}         │ onFrameDecoded,
         │          )              │ onFrameReceive
         │                         │
         ▼                         ▼
    ┌─────────────────────────────────────────┐
    │      TEXTUREMANAGER.TS                   │
    │      (Texture Update System)             │
    │                                          │
    │  ┌───────────────────────────────────┐   │
    │  │ updateFromVideoFrame()            │   │
    │  │ - Receive VideoFrame event        │   │
    │  │ - Extract colorBitmap/depthRaw    │   │
    │  │ - Update colorTexture             │   │
    │  │ - Update depthTexture             │   │
    │  │ - Set needsUpdate = true          │   │
    │  └──────────┬────────────────────────┘   │
    │             │                            │
    │  ┌──────────▼────────────────────────┐   │
    │  │ Update Shader Uniforms            │   │
    │  │ - wsColorSampler.value = texture  │   │
    │  │ - wsDepthSampler.value = texture  │   │
    │  │ (updates fusion, debug, etc.)     │   │
    │  └───────────────────────────────────┘   │
    └─────────────────────────────────────────┘
                      │
                      │ Texture updates
                      ▼
    ┌──────────────────────────────────────────┐
    │      RENDERINGSYSTEM.TS                   │
    │      (Main Render Loop)                   │
    │                                          │
    │  ┌────────────────────────────────────┐  │
    │  │ render() method                    │  │
    │  │ - Called every frame               │  │
    │  │ - Check currentRenderMode          │  │
    │  │ - Dispatch to render function      │  │
    │  └────────────┬───────────────────────┘  │
    │              │                           │
    │  ┌───────────▼────────────────────────┐  │
    │  │ FUSION MODE (most common)          │  │
    │  │                                    │  │
    │  │ 1. Render local scene              │  │
    │  │    render(localScene, camera)      │  │
    │  │    → outputs to localRenderTarget  │  │
    │  │                                    │  │
    │  │ 2. Composite fusion shader         │  │
    │  │    - fusionMaterial.uniforms:      │  │
    │  │      • wsColorSampler (WebSocket)  │  │
    │  │      • wsDepthSampler (WebSocket)  │  │
    │  │      • localColorSampler           │  │
    │  │      • localDepthSampler           │  │
    │  │    - render(fusionScene, ortho)    │  │
    │  │    → outputs to screen             │  │
    │  └────────────────────────────────────┘  │
    └──────────────────────────────────────────┘
                      │
                      │ WebGL render
                      ▼
              [CANVAS / SCREEN]
```

### 4.2 프레임 수신 → 텍스처 업데이트 상세

```
decode-worker.ts
    │
    ├─ 1. WebSocket.onmessage() - 바이너리 프레임 수신
    ├─ 2. Parse header (frameId, formatType, width, height)
    ├─ 3. Decode (Canvas → ImageBitmap)
    └─ 4. postMessage({type: "frame", colorBitmap, depthRaw/depthBitmap})
              │
              ▼
WebSocketSystem.ts (Main Thread)
    │
    ├─ 5. worker.onmessage() - VideoFrame 인터페이스로 변환
    ├─ 6. eventBus.emit("frame:received", {frameId, data: VideoFrame})
    └─ 7. Callbacks 호출 (onDecodeStats, onFrameDecoded)
              │
              ▼
TextureManager.ts (Event Listener)
    │
    ├─ 8. updateFromVideoFrame(frame)
    │      ├─ colorBitmap → colorTexture.image
    │      ├─ depthRaw/depthBitmap → depthTexture.image
    │      └─ needsUpdate = true
    │
    └─ 9. Shader Uniforms 검증 및 업데이트
           fusionMaterial.uniforms.wsColorSampler.value = colorTexture
           fusionMaterial.uniforms.wsDepthSampler.value = depthTexture
```

### 4.3 카메라 → 서버 피드백 루프

```
User Input (mouse)
       ↓
OrbitControls (THREE.js)
       ↓
RenderingSystem.update()
       │
       ├─ hasCameraMoved() > threshold?
       │      YES ↓
       │  onCameraUpdate() callback
       │      ↓
       │  CameraController.getCameraFrame()
       │      ├─ view matrix (matrixWorld)
       │      ├─ projection matrix
       │      ├─ intrinsics (fx, fy, cx, cy)
       │      ├─ position, target, up
       │      └─ timeIndex (Dynamic Gaussian용)
       │      ↓
       └─ WebSocketSystem.sendCameraFrame()
              ↓
          worker.postMessage({type: "camera", frame})
              ↓
          Server renders new Gaussian frame
              ↓
          [cycle continues...]
```

---

## 5. 함수 호출 깊이 (Call Depth)

### 5.1 초기화 시퀀스

```
main.ts
 └─ initScene()                                          [Depth 1]
     ├─ THREE.js 객체 생성 (scene, camera, renderer)
     ├─ RenderTarget 생성
     └─ ShaderMaterial 생성
         └─ return {scene, camera, renderer, controls}

 └─ new Application(config)                              [Depth 1]
     └─ constructor()                                    [Depth 2]
         ├─ new EventBus()
         ├─ new ApplicationState()
         └─ new TimeController()

 └─ app.initializeWithExistingObjects(...)               [Depth 1]
     └─ RenderingContext 생성                            [Depth 2]
     └─ Systems 생성                                     [Depth 2]
         ├─ new WebSocketSystem()
         ├─ new TextureManager()
         ├─ new CameraController()
         └─ new RenderingSystem()
     └─ Promise.all([systems.initialize()])              [Depth 2]
         └─ WebSocketSystem.initialize()                 [Depth 3]
             └─ worker 생성 및 이벤트 핸들러 설정        [Depth 4]
         └─ TextureManager.initialize()                  [Depth 3]
             └─ createTextures()                         [Depth 4]
                 └─ renderingContext.registerTexture()   [Depth 5]
         └─ CameraController.initialize()                [Depth 3]
         └─ RenderingSystem.initialize()                 [Depth 3]

 └─ new UISystem(app, latencyTracker)                    [Depth 1]
     └─ constructor()                                    [Depth 2]
         ├─ new StatsPanel()                             [Depth 3]
         ├─ new ControlPanel({callbacks})                [Depth 3]
         ├─ new DebugPanel({callbacks})                  [Depth 3]
         ├─ new RecordingPanel()                         [Depth 3]
         ├─ new FPSTestPanel({callbacks})                [Depth 3]
         └─ new TimeControlUI(timeController)            [Depth 3]

 └─ app.start()                                          [Depth 1]
     └─ animate()                                        [Depth 2]
         └─ requestAnimationFrame(animate)               [Depth 3: recursive]
```

### 5.2 렌더 루프 호출 스택

```
animate() [Entry Point - Depth 0]
 │
 └─ renderingSystem.update(deltaTime)                    [Depth 1]
     ├─ controls.update()                                [Depth 2]
     ├─ hasCameraMoved()                                 [Depth 2]
     │   └─ camera.position.distanceToSquared()          [Depth 3]
     ├─ onUpdate callback                                [Depth 2]
     │   ├─ timeController.update(dt)                    [Depth 3]
     │   ├─ robot_animation()                            [Depth 3]
     │   │   └─ robot.setJointValue()                    [Depth 4]
     │   └─ uiSystem.updateFPSTestUI()                   [Depth 3]
     │       └─ fpsTestPanel.updateProgress()            [Depth 4]
     └─ onCameraUpdate callback (if moved)               [Depth 2]
         └─ app.sendCameraFrame()                        [Depth 3]
             └─ cameraController.getCameraFrame()        [Depth 4]
             └─ wsSystem.sendCameraFrame()               [Depth 4]
                 └─ worker.postMessage()                 [Depth 5]
 │
 └─ renderingSystem.render()                             [Depth 1]
     └─ switch(currentRenderMode)                        [Depth 2]
         └─ FUSION mode:
             ├─ renderer.setRenderTarget(local)          [Depth 3]
             ├─ renderer.render(localScene, camera)      [Depth 3]
             ├─ renderer.setRenderTarget(null)           [Depth 3]
             └─ renderer.render(fusionScene, ortho)      [Depth 3]
 │
 └─ requestAnimationFrame(animate)                       [Depth 1: recursive]
```

### 5.3 프레임 수신 호출 스택

```
worker.onmessage(event) [Entry Point - Depth 0]
 │
 └─ WebSocketSystem.handleWorkerMessage()                [Depth 1]
     ├─ Convert to VideoFrame                            [Depth 2]
     ├─ eventBus.emit("frame:received", data)            [Depth 2]
     │   └─ Application listener                         [Depth 3]
     │       └─ textureManager.updateFromVideoFrame()    [Depth 4]
     │           ├─ colorTexture.image = bitmap          [Depth 5]
     │           ├─ depthTexture update                  [Depth 5]
     │           └─ verifyMaterialReferences()           [Depth 5]
     │               └─ updateShaderMaterials()          [Depth 6]
     │
     └─ Callbacks                                        [Depth 2]
         ├─ onDecodeStats()                              [Depth 3]
         │   └─ uiSystem.updateDecodeFPS()               [Depth 4]
         │       └─ statsPanel.updateDecodeFPS()         [Depth 5]
         │
         ├─ onFrameDecoded()                             [Depth 3]
         │   └─ latencyTracker.recordDecodeComplete()    [Depth 4]
         │
         └─ onFrameReceive()                             [Depth 3]
             └─ latencyTracker.recordFrameReceive()      [Depth 4]
```

### 5.4 호출 깊이 요약

| 흐름 | 최대 깊이 | 설명 |
|------|----------|------|
| 초기화 | 5 | TextureManager → RenderingContext 등록 |
| 렌더 루프 | 5 | 카메라 업데이트 → Worker 메시지 |
| 프레임 수신 | 6 | Worker → TextureManager → ShaderMaterial |

---

## 6. 이벤트 시스템

### 6.1 EventBus 아키텍처

```typescript
EventBus
├─ TypedEventMap (AppEvents interface)
├─ Regular handlers (Set per event type)
├─ Once handlers (Set per event type)
│
├─ Methods:
│  ├─ on(event, handler) → unsubscribe function
│  ├─ once(event, handler) → auto-cleanup after first call
│  ├─ off(event, handler) → manual unsubscribe
│  ├─ emit(event, data) → synchronous fire
│  ├─ emitAsync(event, data) → Promise.all all handlers
│  ├─ clear() → remove all listeners
│  └─ listenerCount(event) → get handler count
│
└─ Type Safety:
   ├─ event: keyof AppEvents (only valid events)
   ├─ data: AppEvents[K] (correct payload type)
   └─ handler: (data) => void|Promise<void>
```

### 6.2 이벤트 목록

| 이벤트 | 발행자 | 구독자 | Payload |
|--------|--------|--------|---------|
| `connection:open` | WebSocketSystem | UI | `void` |
| `connection:close` | WebSocketSystem | UI | `CloseEvent` |
| `connection:error` | WebSocketSystem | UI | `Event` |
| `frame:sent` | WebSocketSystem | Timing | `{ frameId }` |
| `frame:received` | WebSocketSystem | TextureManager | `{ frameId, data: VideoFrame }` |
| `system:initialized` | All Systems | Application | `{ name }` |
| `system:error` | All Systems | Application | `{ name, error }` |
| `ui:mode-changed` | Application | UI | `{ mode: RenderMode }` |
| `camera:update` | RenderingSystem | Logging | `{ position, target }` |

### 6.3 이벤트 흐름 예시

```
Example 1: Frame Arrival & Texture Update
──────────────────────────────────────────
  Worker → postMessage("frame")
      ↓
  WebSocketSystem.worker.onmessage()
      ↓
  this.eventBus.emit("frame:received", {frameId, data: VideoFrame})
      ↓
  Application.setupEventListeners() registered "frame:received"
      ├─ listener: (data) => textureManager.updateFromVideoFrame(data)
      └─ triggers: complete texture update cascade


Example 2: Render Mode Change
──────────────────────────────
  User clicks mode button
      ↓
  ControlPanel.onRenderModeChange()
      ↓
  UISystem.handleRenderModeChange()
      ├─ RenderingSystem.setRenderMode(mode)
      ├─ RecordingPanel.setRenderMode(mode)
      └─ Special handling for FEED_FORWARD mode
      ↓
  Application.setRenderMode(mode)
      ↓
  this.eventBus.emit("ui:mode-changed", {mode})
```

### 6.4 상태 관리 (ApplicationState)

```typescript
ApplicationState
├─ key-value store with change notifications
├─ pub/sub listeners per key
│
├─ Initialization defaults:
│  ├─ connection:state = Disconnected
│  ├─ connection:url = ""
│  ├─ rendering:mode = Hybrid
│  ├─ rendering:width = 1280
│  ├─ rendering:height = 720
│  ├─ rendering:rescaleFactor = 0.8
│  ├─ frame:counter = 0
│  ├─ frame:timeIndex = 0.0
│  ├─ performance:fps = 0
│  ├─ debug:showStats = false
│  ├─ debug:showDepthMap = false
│  ├─ animation:playing = true
│  ├─ recording:active = false
│  ├─ texture:colorNeedsUpdate = false
│  └─ texture:depthNeedsUpdate = false
│
├─ Usage Pattern:
│  get(key): T | undefined
│  getOrDefault(key, default): T
│  set(key, value): notifies listeners only on change
│  subscribe(key, listener): returns unsubscribe function
│  unsubscribe(key, listener)
│  update(record): batch set
```

---

## 7. 렌더링 모드

### 7.1 모드 목록

| 모드 | Scene | 설명 |
|------|-------|------|
| `FUSION` | fusionScene | Gaussian + Local Mesh 합성 (기본) |
| `GAUSSIAN_ONLY` | gaussianOnlyScene | Gaussian만 표시 |
| `LOCAL_ONLY` | localOnlyScene | Local Mesh만 표시 |
| `DEPTH_FUSION` | depthFusionScene | Depth 기반 합성 |
| `FEED_FORWARD` | fusionScene | 프레임별 PLY 렌더링 |
| `DEBUG` | debugScene | 4분할 디버그 뷰 |

### 7.2 FUSION 모드 파이프라인

```
┌─ Local Scene ────┐     ┌─ WebSocket Gaussian ────┐
│ localRenderTarget │     │ wsColorSampler          │
│ localDepthTexture │     │ wsDepthSampler          │
└────────┬──────────┘     └────────┬────────────────┘
         └─────────┬───────────────┘
                   ↓
            Fusion Shader
         (Depth-based composite)
                   ↓
              [SCREEN]
```

### 7.3 GAUSSIAN_ONLY 모드

```
┌─ WebSocket Gaussian ────┐
│ wsColorSampler only     │
└────────┬────────────────┘
         ↓
   gaussianOnlyShader
         ↓
    [SCREEN]
```

### 7.4 FEED_FORWARD 모드

```
특수 모드: /ws/feedforward 엔드포인트 연결
서버가 미리 렌더링된 프레임 전송
카메라 업데이트 없음 (timeIndex로 애니메이션 제어)
```

---

## 8. 주요 타입 정의

### 8.1 CameraFrame (서버로 전송, 260 bytes)

```typescript
interface CameraFrame {
  view: Float32Array;        // 4×4 view matrix (matrixWorld)
  projection: Float32Array;  // 4×4 projection matrix
  intrinsics: Float32Array;  // 3×3 intrinsics (fx, fy, cx, cy)
  position: Float32Array;    // [x, y, z] world space
  target: Float32Array;      // [x, y, z] lookAt target
  up: Float32Array;          // [x, y, z] up vector
  frameId: number;
  timestamp: number;
  timeIndex: number;         // Dynamic Gaussian [0, 1]
}
```

### 8.2 VideoFrame (서버에서 수신)

```typescript
interface VideoFrame {
  frameId: number;
  formatType: FormatType;    // JPEG=0, H264=1, Raw=2

  // Color data (priority: colorBitmap > colorData)
  colorBitmap?: ImageBitmap;
  colorData?: Uint8Array;

  // Depth data (priority: depthRaw > depthBitmap > depthData)
  depthRaw?: Uint16Array;    // Float16 encoded (JPEG mode)
  depthBitmap?: ImageBitmap; // Grayscale (H264 mode)
  depthData?: Float32Array;  // Fallback

  width: number;
  height: number;

  timestamps: {
    client: number;
    server: number;
    renderStart: number;
    encodeEnd: number;
  };
}
```

### 8.3 System 인터페이스

```typescript
interface System {
  readonly name: string;
  initialize(context: SystemContext): Promise<void>;
  update(deltaTime: number): void;
  dispose(): void;
}

interface SystemContext {
  renderingContext: RenderingContext;
  eventBus: EventBus;
  state: ApplicationState;
  systems?: Map<string, System>;
}
```

### 8.4 Enumerations

```typescript
enum RenderMode {
  FUSION = "fusion",
  GAUSSIAN_ONLY = "gaussian-only",
  LOCAL_ONLY = "local-only",
  DEPTH_FUSION = "depth-fusion",
  FEED_FORWARD = "feed-forward"
}

enum ConnectionState {
  Disconnected = "disconnected",
  Connecting = "connecting",
  Connected = "connected",
  Reconnecting = "reconnecting",
  Error = "error"
}

enum FormatType {
  JPEG = 0,
  H264 = 1,
  Raw = 2
}

enum CameraMode {
  Orbit = "orbit",
  FirstPerson = "firstperson",
  ThirdPerson = "thirdperson"
}
```

---

## 9. 초기화 흐름

### 9.1 전체 초기화 시퀀스

```
PHASE 1: PRE-APPLICATION (main.ts)
═════════════════════════════════

1. Scene Setup (initScene())
   ├─ Create THREE.js Scene, Camera, Renderer
   ├─ Setup OrbitControls
   ├─ Create all render targets:
   │  ├─ localRenderTarget (for local 3D mesh)
   │  ├─ fusionScene/orthoCamera
   │  ├─ debugScene/debugCamera
   │  ├─ gaussianOnlyScene
   │  ├─ localOnlyScene
   │  └─ depthFusionScene
   ├─ Create shader materials with uniforms
   └─ Setup window.addEventListener('resize', ...)


PHASE 2: APPLICATION INSTANTIATION
═══════════════════════════════════

2. new Application(config)
   ├─ Creates EventBus
   ├─ Creates ApplicationState (with defaults)
   ├─ Creates TimeController (for dynamic gaussians)
   └─ Stores config (width, height, renderMode, cameraConfig)


PHASE 3: INITIALIZE WITH EXISTING OBJECTS
══════════════════════════════════════════

3. app.initializeWithExistingObjects(scene, camera, renderer, controls)

   3.1 Create RenderingContext wrapper
       └─ Wraps existing THREE.js objects

   3.2 Create System instances
       ├─ new WebSocketSystem()
       ├─ new TextureManager()
       ├─ new CameraController()
       └─ new RenderingSystem()

   3.3 Initialize all systems in PARALLEL (Promise.all)
       ├─ WebSocketSystem.initialize() → Create Worker
       ├─ TextureManager.initialize() → Create textures
       ├─ CameraController.initialize() → Setup camera
       └─ RenderingSystem.initialize() → Ready for config

   3.4 Setup event listeners
       └─ "frame:received" → textureManager.updateFromVideoFrame()


PHASE 4: UI SYSTEM CREATION
═══════════════════════════

4. new UISystem(app, latencyTracker)
   ├─ Create all UI panels
   └─ UISystem.initialize() → Mount to DOM


PHASE 5: CONFIGURATION
══════════════════════

5. Configure WebSocketSystem callbacks
   └─ Connect UI & LatencyTracker

6. Register shader materials with RenderingContext

7. Initialize TextureManager with shader materials

8. Configure RenderingSystem with scenes/cameras


PHASE 6: START
══════════════

9. app.start() → Begin render loop

10. wsSystem.connect() → Establish WebSocket connection
```

---

## 10. API Reference

### 10.1 Application

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

### 10.2 WebSocketSystem

| 메서드 | 설명 |
|--------|------|
| `initialize(context)` | Worker 생성, 핸들러 설정 |
| `connect(url, width, height)` | WebSocket 연결 시작 |
| `disconnect()` | WebSocket 연결 종료 |
| `reconnect(url, width, height)` | 재연결 |
| `sendCameraFrame(frame)` | 카메라 프레임 전송 |
| `changeEncoderType('jpeg'\|'h264')` | 인코더 변경 |
| `toggleJPEGFallback(enabled)` | JPEG 모드 전환 |
| `startFPSMeasurement()` | FPS 측정 시작 |
| `stopFPSMeasurement()` | FPS 측정 중지 |

### 10.3 WebSocketSystem Callbacks

```typescript
wsSystem.setCallbacks({
  onConnectionStateChange?: (state) => void;
  onDecodeStats?: (stats) => void;
  onFrameReceive?: (frameId, timestamps) => void;
  onFrameDecoded?: (frameId, time) => void;
  onClockSync?: (client, serverRx, serverTx) => void;
});
```

### 10.4 TextureManager

| 메서드 | 설명 |
|--------|------|
| `updateFromVideoFrame(frame)` | 비디오 프레임으로 텍스처 업데이트 |
| `resize(width, height)` | 텍스처 크기 변경 |
| `setJpegMode(enabled)` | JPEG/H264 모드 전환 |
| `getColorTexture()` | 컬러 텍스처 반환 |
| `getDepthTexture()` | Depth 텍스처 반환 |
| `initializeShaderMaterials()` | 셰이더 Material 초기화 |

### 10.5 CameraController

| 메서드 | 설명 |
|--------|------|
| `getCameraFrame(timeIndex)` | 서버 전송용 카메라 프레임 생성 |
| `setPosition(x, y, z)` | 카메라 위치 설정 |
| `setTarget(x, y, z)` | 카메라 타겟 설정 |
| `savePosition()` | localStorage에 저장 |
| `loadPosition()` | localStorage에서 불러오기 |
| `setControlsEnabled(enabled)` | 컨트롤 활성화/비활성화 |

### 10.6 RenderingSystem

| 메서드 | 설명 |
|--------|------|
| `configure(config)` | 렌더링 설정 구성 |
| `setRenderMode(mode)` | 렌더 모드 변경 |
| `render()` | 프레임 렌더링 |

---

## 요약

| 항목 | 설명 |
|------|------|
| **아키텍처 패턴** | Phase 1 하이브리드 (레거시 래핑 + 모듈화) |
| **상태 관리** | ApplicationState (중앙 집중, Pub/Sub) |
| **통신 방식** | EventBus + Callbacks (느슨한 결합) |
| **최대 호출 깊이** | 6단계 (프레임 수신 → 셰이더 업데이트) |
| **순환 의존성** | 없음 (단방향 + 이벤트 기반) |
| **Worker 활용** | decode-worker.ts (H.264/JPEG 디코딩 오프로드) |
| **렌더 모드** | 6가지 (FUSION, GAUSSIAN_ONLY, LOCAL_ONLY, DEPTH_FUSION, FEED_FORWARD, DEBUG) |

---

## 변경 이력

| 날짜 | 변경 | 설명 |
|------|------|------|
| 2025-01-14 | kebab-case 리팩토링 | 모든 파일명 kebab-case로 통일, 디렉토리 구조 정리 |
| 2025-01-14 | 아키텍처 문서 보완 | 의존성 그래프, 호출 깊이, 데이터 흐름 상세화 |
| 2025-01-13 | Phase 6 | Worker 통신 완전 통합 (main.ts → WebSocketSystem) |
| 2025-01-13 | Phase 5 | BasePanel 추상 클래스, 5개 패널 리팩토링 |
| 2025-01-13 | Phase 4 | 워커 메시지 핸들러 통합 |
| 2025-01-13 | Phase 1-3 | test/ 삭제, dead code 제거, 미사용 시스템 삭제 |
