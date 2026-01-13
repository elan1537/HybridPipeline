# Frontend Refactoring Plan

## 개요

Frontend 코드베이스 정리 및 리팩토링 계획서입니다.

**브랜치**: `refactor/frontend-cleanup`
**기준 브랜치**: `feature/collision-detect`
**작성일**: 2025-01-13

---

## 완료된 Phase

### Phase 1-3: Dead Code 및 미사용 시스템 제거 ✅

**커밋**: `eb3f48e`

| 삭제 항목 | 라인 수 |
|----------|--------|
| `physics/MeshGaussianCollision.ts` | -259 |
| `systems/DebugSystem.ts` | -233 |
| `systems/PhysicsSystem.ts` | -313 |
| `main.ts` 주석/중복 코드 | -155 |
| `Application.ts` Physics API | -119 |
| `debug-logger.ts` 미사용 메서드 | -34 |
| `types/index.ts` 미사용 타입 | -22 |
| **총 삭제** | **-1,132** |

**결과**: 번들 609 kB → 600 kB (-9 kB)

---

## Phase 4: decode-worker 메시지 핸들링 통합

### 현재 문제점

main.ts와 WebSocketSystem에서 워커 메시지를 **중복 처리**하고 있음:

```
main.ts (lines 301-402)          WebSocketSystem (lines 245-327)
├─ case 'pure-decode-stats'      ├─ case 'pure-decode-stats'
├─ case 'frame-receive'          ├─ case 'frame-receive'
├─ case 'frame'/'video-frame'    ├─ case 'frame'/'video-frame'
└─ case 'pong-received'          └─ case 'pong-received'
```

**브릿지 패턴** (main.ts lines 308-312):
```typescript
// 어색한 패턴: main.ts에서 WebSocketSystem으로 메시지 전달
if (app) {
    const wsSystem = app.getWebSocketSystem();
    if (wsSystem) {
        wsSystem.handleMessage(data);
    }
}
```

### 목표

1. 워커 메시지 핸들링을 WebSocketSystem으로 완전 통합
2. main.ts의 `worker.onmessage` 핸들러 단순화
3. 브릿지 패턴 제거

### 작업 내용

#### 4.1 WebSocketSystem 수정

**파일**: `frontend/src/systems/WebSocketSystem.ts`

1. **콜백 인터페이스 추가**:
```typescript
interface WebSocketSystemCallbacks {
    onConnectionStateChange?: (state: 'connected' | 'error' | 'closed') => void;
    onDecodeStats?: (stats: DecodeStats) => void;
    onFrameReceive?: (frameId: number, timestamps: any) => void;
    onClockSync?: (clientTime: number, serverReceiveTime: number, serverSendTime: number) => void;
}
```

2. **handleWorkerMessage 메서드 강화** (lines 245-327):
```typescript
handleWorkerMessage(data: any): void {
    switch (data.type) {
        case 'ws-ready':
            this.callbacks.onConnectionStateChange?.('connected');
            break;
        case 'ws-error':
            this.callbacks.onConnectionStateChange?.('error');
            break;
        case 'ws-close':
            this.callbacks.onConnectionStateChange?.('closed');
            break;
        case 'pure-decode-stats':
            this.callbacks.onDecodeStats?.(data);
            break;
        case 'frame-receive':
            this.callbacks.onFrameReceive?.(data.frameId, data.serverTimestamps);
            break;
        case 'frame':
        case 'video-frame':
            this.handleVideoFrame(data);
            break;
        case 'pong-received':
            this.callbacks.onClockSync?.(
                data.clientRequestTime,
                data.serverReceiveTime,
                data.serverSendTime
            );
            break;
    }
}
```

#### 4.2 main.ts 수정

**파일**: `frontend/src/main.ts`

1. **worker.onmessage 핸들러 단순화** (lines 301-402):
```typescript
// Before: 100+ lines
worker.onmessage = ({ data }) => {
    // 모든 메시지 타입별 처리...
}

// After: ~10 lines
worker.onmessage = ({ data }) => {
    if (data.type === 'error') {
        debug.error("decode-worker error: ", data.error);
        return;
    }

    // WebSocketSystem에 모든 처리 위임
    const wsSystem = app?.getWebSocketSystem();
    wsSystem?.handleWorkerMessage(data);
};
```

2. **WebSocketSystem 콜백 설정** (initScene 후):
```typescript
const wsSystem = app.getWebSocketSystem();
wsSystem?.setCallbacks({
    onConnectionStateChange: (state) => {
        uiSystem?.setConnectionState(state);
    },
    onDecodeStats: (stats) => {
        uiSystem?.updateDecodeFPS(stats.pureFPS, stats.avgDecodeTime);
        latencyTracker.recordPureDecodeFPS(stats.totalFrames, stats.avgDecodeTime);
    },
    onFrameReceive: (frameId, timestamps) => {
        latencyTracker.recordFrameReceive(frameId, timestamps);
    },
    onClockSync: (clientTime, serverReceive, serverSend) => {
        latencyTracker.recordClockSync(clientTime, serverReceive, serverSend);
    }
});
```

### 예상 효과

- **코드 감소**: ~100줄
- **중복 제거**: 메시지 핸들링 2곳 → 1곳
- **유지보수성**: 워커 메시지 처리 로직 중앙화

### 위험 요소

- 기존 동작과 완전히 동일한지 검증 필요
- 타이밍 이슈 가능성 (콜백 순서)
- UISystem, LatencyTracker 의존성 확인 필요

### 검증 방법

1. WebSocket 연결/재연결 동작 확인
2. FPS 통계 표시 정상 동작
3. 레이턴시 측정 정상 동작
4. 프레임 렌더링 정상 동작

---

## Phase 5: BasePanel 리팩토링

### 현재 문제점

5개 패널에서 **동일한 DOM 초기화 패턴이 반복**됨 (50+ 회):

```typescript
// ControlPanel, DebugPanel, StatsPanel, RecordingPanel, FPSTestPanel 모두 동일
private element: HTMLElement | null = null;
private initializeElements(): void {
    this.element = document.getElementById('element-id') as HTMLElement;
}
private setupListeners(): void {
    this.element?.addEventListener('click', () => this.handler());
}
```

### 목표

1. BasePanel 추상 클래스 생성
2. 공통 DOM 유틸리티 메서드 추출
3. 코드 중복 제거

### 작업 내용

#### 5.1 BasePanel 클래스 생성

**파일**: `frontend/src/ui/panels/BasePanel.ts` (신규)

```typescript
/**
 * BasePanel - Abstract base class for UI panels
 * Provides common DOM manipulation utilities
 */
export abstract class BasePanel {
    protected readonly prefix: string;
    protected elements: Map<string, HTMLElement> = new Map();

    constructor(prefix: string) {
        this.prefix = prefix;
    }

    /**
     * Get element by ID with optional type casting
     */
    protected getElement<T extends HTMLElement>(id: string): T | null {
        const cached = this.elements.get(id);
        if (cached) return cached as T;

        const element = document.getElementById(id) as T | null;
        if (element) {
            this.elements.set(id, element);
        }
        return element;
    }

    /**
     * Get element with prefix (e.g., 'control-panel-connect' for prefix='control-panel')
     */
    protected getPrefixedElement<T extends HTMLElement>(suffix: string): T | null {
        return this.getElement<T>(`${this.prefix}-${suffix}`);
    }

    /**
     * Safely update element text content
     */
    protected updateText(element: HTMLElement | null, value: string): void {
        if (element) {
            element.textContent = value;
        }
    }

    /**
     * Safely update element inner HTML
     */
    protected updateHTML(element: HTMLElement | null, html: string): void {
        if (element) {
            element.innerHTML = html;
        }
    }

    /**
     * Safely set element display style
     */
    protected setDisplay(element: HTMLElement | null, display: string): void {
        if (element) {
            element.style.display = display;
        }
    }

    /**
     * Safely show/hide element
     */
    protected setVisible(element: HTMLElement | null, visible: boolean): void {
        this.setDisplay(element, visible ? '' : 'none');
    }

    /**
     * Add event listener with null check
     */
    protected addListener<K extends keyof HTMLElementEventMap>(
        element: HTMLElement | null,
        event: K,
        handler: (e: HTMLElementEventMap[K]) => void
    ): void {
        element?.addEventListener(event, handler);
    }

    /**
     * Add click listener shorthand
     */
    protected onClick(element: HTMLElement | null, handler: () => void): void {
        this.addListener(element, 'click', handler);
    }

    /**
     * Add change listener shorthand
     */
    protected onChange(element: HTMLElement | null, handler: (e: Event) => void): void {
        this.addListener(element, 'change', handler);
    }

    /**
     * Dispose panel resources
     */
    dispose(): void {
        this.elements.clear();
    }
}
```

#### 5.2 ControlPanel 리팩토링 예시

**파일**: `frontend/src/ui/panels/ControlPanel.ts`

```typescript
// Before
export class ControlPanel {
    private connectBtn: HTMLButtonElement | null = null;
    private disconnectBtn: HTMLButtonElement | null = null;
    // ... 10+ more elements

    private initializeElements(): void {
        this.connectBtn = document.getElementById('control-panel-connect') as HTMLButtonElement;
        this.disconnectBtn = document.getElementById('control-panel-disconnect') as HTMLButtonElement;
        // ... 10+ more getElementById calls
    }

    private setupListeners(): void {
        this.connectBtn?.addEventListener('click', () => this.handleConnect());
        this.disconnectBtn?.addEventListener('click', () => this.handleDisconnect());
        // ... 10+ more event listeners
    }
}

// After
export class ControlPanel extends BasePanel {
    constructor(callbacks: ControlPanelCallbacks) {
        super('control-panel');
        this.callbacks = callbacks;
        this.initializeElements();
        this.setupListeners();
    }

    private initializeElements(): void {
        // Elements are lazily cached by BasePanel.getElement()
    }

    private setupListeners(): void {
        this.onClick(this.getPrefixedElement('connect'), () => this.handleConnect());
        this.onClick(this.getPrefixedElement('disconnect'), () => this.handleDisconnect());
        // ... cleaner listener setup
    }

    updateConnectionState(connected: boolean): void {
        this.setVisible(this.getPrefixedElement('connect'), !connected);
        this.setVisible(this.getPrefixedElement('disconnect'), connected);
    }
}
```

#### 5.3 Render Mode 리스너 통합

**파일**: `frontend/src/ui/panels/ControlPanel.ts` (lines 83-111)

```typescript
// Before: 5개의 거의 동일한 이벤트 리스너
this.fusionModeRadio?.addEventListener('change', () => {
    if (this.fusionModeRadio?.checked) {
        this.handleRenderModeChange(RenderMode.FUSION);
    }
});
this.gaussianModeRadio?.addEventListener('change', () => { /* ... */ });
// ... 3 more

// After: 루프 기반 설정
const renderModes = [
    { suffix: 'fusion-mode', mode: RenderMode.FUSION },
    { suffix: 'gaussian-mode', mode: RenderMode.GAUSSIAN_ONLY },
    { suffix: 'local-mode', mode: RenderMode.LOCAL_ONLY },
    { suffix: 'depth-fusion-mode', mode: RenderMode.DEPTH_FUSION },
    { suffix: 'feed-forward-mode', mode: RenderMode.FEED_FORWARD },
];

renderModes.forEach(({ suffix, mode }) => {
    const radio = this.getPrefixedElement<HTMLInputElement>(suffix);
    this.onChange(radio, () => {
        if (radio?.checked) this.handleRenderModeChange(mode);
    });
});
```

#### 5.4 다른 패널들 리팩토링

동일한 패턴으로 리팩토링:
- `DebugPanel.ts`
- `StatsPanel.ts`
- `RecordingPanel.ts`
- `FPSTestPanel.ts`

### 예상 효과

- **코드 감소**: ~300줄
- **중복 제거**: DOM 초기화/이벤트 핸들링 패턴 통일
- **유지보수성**: 새 패널 추가 시 BasePanel 상속만 하면 됨
- **타입 안정성**: Generic 메서드로 타입 캐스팅 중앙화

### 위험 요소

- 기존 동작과 완전히 동일한지 검증 필요
- Lazy initialization으로 인한 타이밍 이슈 가능성
- 모든 패널의 DOM ID 규칙 확인 필요

### 검증 방법

1. 모든 UI 패널 동작 테스트
   - ControlPanel: 연결/해제, 모드 전환, 인코더 변경
   - DebugPanel: 디버그 토글, 카메라 저장/불러오기
   - StatsPanel: 통계 표시 업데이트
   - RecordingPanel: 녹화 시작/정지
   - FPSTestPanel: FPS 측정 시작/정지/다운로드
2. 콘솔 에러 확인
3. 메모리 누수 확인 (dispose 호출 시 리소스 정리)

---

## 작업 순서

1. **Phase 4 먼저 진행** (decode-worker 통합)
   - WebSocketSystem 콜백 인터페이스 추가
   - main.ts 핸들러 단순화
   - 동작 검증

2. **Phase 5 진행** (BasePanel 리팩토링)
   - BasePanel 클래스 생성
   - 패널들 순차적으로 리팩토링 (ControlPanel → StatsPanel → ...)
   - 각 패널 리팩토링 후 동작 검증

3. **최종 검증 및 커밋**
   - 전체 빌드 확인
   - 런타임 테스트
   - 커밋 및 병합

---

## 예상 최종 결과

| 항목 | Phase 1-3 | Phase 4 | Phase 5 | 총합 |
|------|-----------|---------|---------|------|
| 코드 감소 | ~1,132줄 | ~100줄 | ~300줄 | **~1,532줄** |
| 번들 감소 | ~9 kB | ~2 kB | ~5 kB | **~16 kB** |

---

## 참고 파일

- **계획 파일**: `/home/wrl-ubuntu/.claude/plans/toasty-jumping-sphinx.md`
- **main.ts**: `frontend/src/main.ts`
- **WebSocketSystem**: `frontend/src/systems/WebSocketSystem.ts`
- **패널들**: `frontend/src/ui/panels/`
