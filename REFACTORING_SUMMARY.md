# Frontend 리팩토링 종합 정리

## 📅 작성일: 2025-10-27

---

## 🎯 프로젝트 목표 및 컨텍스트

### 최종 목표
- **VR/AR 원격 협업 환경**에서 **Mesh-Gaussian Collision Detection** 구현
- 사용 케이스: 손/컨트롤러가 Gaussian Splatting으로 렌더링된 실제 공간과 충돌
- 환경: 1280×720 이상 해상도, 도시 내 네트워크 (10-30ms latency)

### 핵심 요구사항
1. 벽 통과 방지
2. 물체 집기/놓기
3. 물리적 반발
4. 1인칭 메인, 3인칭 관전 모드 지원

---

## 🔍 기술적 분석 결과

### 1. 물리 엔진 검토 → 커스텀 구현 결정

**검토한 옵션**:
- Rapier (Rust/WASM, 고성능, 2MB)
- Cannon.js/cannon-es (경량, 유지보수 중)
- Ammo.js (Bullet Physics 포트, 무겁고 느림)

**결정: 커스텀 구현** ✅

**이유**:
- Gaussian Splatting은 전통적 mesh가 아님 (Depth map만 제공)
- 물리 엔진은 Plane/Box/Sphere만 지원
- Depth map 기반 충돌은 어차피 커스텀 필수
- 기본 물리(중력, 속도)는 Vector3 연산으로 충분
- 번들 크기 절약 (Rapier WASM 2MB 불필요)

---

## 🚨 현재 Frontend 문제점

### 코드 구조 문제
```
main.ts: 1947줄 (너무 비대)
전역 변수: 122개
관심사 혼재: WebSocket, Rendering, UI, Camera, Texture, FPS, Recording...
테스트 불가능한 구조
```

### 충돌 시스템 통합 시 예상 문제
- wsDepthTexture 63곳 참조
- 2000줄에서 버그 찾기 어려움
- VR/AR 확장 시 3000줄 예상
- Side effect 추적 불가능

---

## 💡 리팩토링 전략

### Minimal Refactoring (1일) 선택

**이유**:
- 즉시 구현 (3일) vs Minimal 리팩토링 (1일) + 구현 (3일) = 4일
- 투자 대비 효과 최대
- 테스트 가능한 구조
- 향후 확장성 확보

### 목표 구조
```
frontend/src/
├── core/
│   ├── RenderingContext.ts    # 렌더링 상태 캡슐화
│   └── Application.ts          # 메인 앱 클래스
├── systems/
│   ├── WebSocketSystem.ts     # WS 통신
│   ├── TextureManager.ts      # 텍스처 관리
│   └── CollisionSystem.ts     # 충돌 시스템 ⭐
├── physics/
│   ├── HandPhysics.ts         # 손/컨트롤러 물리
│   └── types.ts               # 공통 타입
├── debug/
│   └── CollisionDebugger.ts   # 디버깅 도구
└── main.ts                     # 100줄로 축소
```

---

## 🔄 Git Commit 체크포인트 전략

### 14개 체크포인트 (3 Phase)

#### Phase 1: Non-Breaking Preparation (CP 1-5)
```bash
CP1: "refactor: Add type definitions and interfaces for refactoring"
CP2: "refactor: Add RenderingContext wrapper class (read-only)"
CP3: "refactor: Add system classes without integration"
CP4: "refactor: Add parallel system initialization for testing"
CP5: "test: Verify new systems produce same results"
```
- **특징**: 기존 코드 영향 없음, 롤백 쉬움

#### Phase 2: Gradual Migration (CP 6-9)
```bash
CP6: "refactor: Migrate to new WebSocket system"
CP7: "refactor: Migrate texture management to TextureManager"
CP8: "refactor: Extract render loop to Application class"
CP9: "refactor: Remove legacy code and global variables" # ⚠️ 위험
```
- **특징**: Feature flag로 보호, CP9는 되돌리기 어려움

#### Phase 3: Collision System (CP 10-14)
```bash
CP10: "feat: Add CollisionSystem with depth map support"
CP11: "feat: Integrate collision system (disabled by default)"
CP12: "feat: Add collision debug visualization"
CP13: "feat: Enable collision system by default"
CP14: "feat: Add VR/AR controller support for collision"
```
- **특징**: 새 기능 추가, 독립적

**핵심 원칙**: 매 커밋마다 앱이 작동해야 함

---

## 🆚 Backend vs Frontend 리팩토링 비교

### Backend (이미 잘 구조화됨)
```python
# Factory Pattern + Dependency Injection
renderer_service = RendererService(
    scene_renderer=scene_renderer,  # DI
    encoder=encoder,                # DI
    buffer_type=buffer_type
)

# Abstract Base Classes
class BaseSceneRenderer(ABC):
    @abstractmethod
    def render(self, camera_data): pass

# Clear Separation
SceneRenderer: 렌더링만
Encoder: 인코딩만
Transport: 전송만
```

### Frontend (리팩토링 필요)
- 모놀리식 구조
- 전역 변수 난무
- 테스트 불가능

### 적용할 Backend 패턴
1. **Factory Pattern** ✅
2. **Dependency Injection** ✅
3. **Interface/Implementation 분리** ✅
4. **Single Responsibility** ✅

### Frontend 특화 추가
1. **Event Bus** (비동기 이벤트)
2. **Reactive State** (UI 상태)
3. **Component Lifecycle**
4. **Browser API Abstraction**

---

## ⚠️ 핵심 고려사항 (충돌 시스템)

### 해결된 우려사항 ✅
- **해상도**: 1280×720 충분 (픽셀당 ~1.4mm @ 1m)
- **3인칭 충돌**: NVS 특성상 일관성 유지
- **Screen space limitation**: 1인칭 중심이라 OK
- **Normal 정확도**: 물체 집기에는 불필요
- **Float16 정밀도**: 협업 거리(~10m)에서 충분

### 남은 과제 ⚠️

#### 1. Latency 누적 (60-80ms)
```typescript
// 해결책: Predictive compensation
const predictedPos = position + velocity * 0.08; // 80ms ahead
```

#### 2. Dynamic Gaussian (4DGS/3DGStream)
```typescript
// 문제: 움직이는 Gaussian scene
// 해결책: Conservative margin
const EPSILON = isDynamic ? 0.10 : 0.01; // 10cm vs 1cm
```

#### 3. Hand Tracking Jitter
```typescript
// 해결책: Temporal smoothing
const smoothedPos = lerp(prevPos, currentPos, 0.8);
```

#### 4. Occlusion Handling
```typescript
// 3인칭에서 가려진 플레이어
if (occluded) {
    usePlayerSelfReportedCollision();
}
```

---

## 📋 구현 우선순위

### MVP (Day 1)
#### 오전: Minimal Refactoring
1. RenderingContext 클래스 생성
2. WebSocketSystem 분리
3. TextureManager 분리
4. Application 클래스 생성
5. main.ts 100줄로 축소

#### 오후: Basic Collision
1. CollisionSystem 구현
   - worldToScreen projection
   - Depth sampling (1280×720)
   - Single point collision
   - Predictive compensation (80ms)
2. HandPhysics 구현
   - Hand/Controller tracking
   - Velocity calculation
   - Stop reaction

### Integration (Day 2)
1. Application 통합
2. Dynamic Gaussian 지원
   - Motion detection
   - Temporal buffer
   - Adaptive margin
3. Debug visualization
   - Depth overlay
   - Collision points
   - Performance metrics

### Polish (Day 3)
1. 테스트 씬 구축
2. 성능 최적화
3. VR/AR 입력 처리
4. 문서화

---

## 🎊 예상 결과

### 코드 품질 개선
| 항목 | Before | After |
|-----|--------|-------|
| main.ts | 1947줄 | 100줄 |
| 전역 변수 | 122개 | 0개 |
| 테스트 가능성 | ❌ | ✅ |
| 모듈화 | ❌ | ✅ |

### 구현 기능
- ✅ VR/AR 손/컨트롤러 충돌
- ✅ 80ms latency compensation
- ✅ Dynamic Gaussian 지원
- ✅ 1280×720 해상도 지원
- ✅ Stop/Bounce 반응
- ✅ 1인칭/3인칭 모드

### 성능 목표
- 충돌 체크: < 1ms/frame
- 60 FPS 유지
- 메모리: +10MB 이하
- 2 hands × 1 point = 2 checks/frame

---

## 📌 핵심 결정 사항

1. **물리 엔진 사용 안 함** (커스텀 구현)
   - Gaussian은 Depth map만 제공
   - 물리 엔진으로 처리 불가능

2. **Minimal Refactoring 선행** (1일 투자)
   - 즉시 가치 실현
   - 테스트 가능한 구조

3. **14개 commit 체크포인트**로 안전한 진행
   - 매 commit 작동 보장
   - 롤백 가능

4. **Backend 패턴 적용** + Frontend 특성 고려
   - Factory, DI, Interface
   - Event Bus, Reactive State 추가

5. **Conservative approach**로 Dynamic Gaussian 처리
   - Static: 1cm margin
   - Dynamic: 10cm margin

---

## 🚀 구현 시작

### 첫 번째 커밋 (CP1)
```bash
# Type definitions 생성
mkdir -p frontend/src/types
touch frontend/src/types/index.ts
git add .
git commit -m "refactor: Add type definitions and interfaces for refactoring"
```

### 태그 전략
```bash
git tag refactoring-phase1-complete  # CP5 이후
git tag refactoring-phase2-complete  # CP9 이후
git tag collision-system-complete    # CP14 이후
```

---

## 📚 참고 자료

- `PHYSICS_ENGINE_ANALYSIS.md` - 물리 엔진 분석
- `feature.collision.md` - 충돌 시스템 설계
- `CLAUDE.md` - 프로젝트 전체 상황
- `architecture.md` - 시스템 아키텍처

---

## 📝 메모

- Frontend 리팩토링은 Backend보다 복잡 (UI 상태, 비동기 이벤트)
- 롤백 전략 중요 (특히 CP9)
- Feature flag 활용으로 점진적 전환
- 테스트 커버리지 목표: 80% 이상

---

**작성자**: Claude
**최종 검토**: 2025-10-27
**상태**: 구현 준비 완료