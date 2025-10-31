# COLMAP Camera Integration - Troubleshooting Guide

## 문제 배경

Dynamic Gaussian Splatting (3DGStream) 렌더러에 COLMAP 카메라 데이터를 통합하는 과정에서 발생한 문제들과 해결 방법을 정리합니다.

**초기 증상**:
- Mock 카메라 (synthetic circular motion): 렌더링 성공 (단, 위아래 flip됨)
- COLMAP 카메라 (실제 학습 데이터): **완전 black 이미지**

---

## 문제 1: View Matrix Format 불일치

### 증상
```
Transport: Reconstructed eye [0, 0, 0]
Original eye: [0.029, -1.337, -0.215]
Difference: 1.35  ← 심각한 오차!
```

### 원인
3DGStream은 **transposed (column-major)** view matrix를 사용:
```
[[R00 R01 R02  0 ]   ← Rotation rows
 [R10 R11 R12  0 ]
 [R20 R21 R22  0 ]
 [tx  ty  tz   1 ]]   ← Translation in row 4 (특이함!)
```

표준 view matrix는 **row-major**:
```
[[R00 R01 R02  tx]   ← Translation in column 4
 [R10 R11 R12  ty]
 [R20 R21 R22  tz]
 [0   0   0    1 ]]
```

### 해결책
**아키텍처 결정: Transport는 표준 format 사용**
- Transport Service: 항상 표준 row-major view_matrix 전달
- StreamableGaussian Renderer: 받은 후 `.T` (transpose)로 변환
- 확장성: 새로운 렌더러 추가 시 Transport 수정 불필요

**구현**:
```python
# streamable_gaussian.py
view_matrix_standard = camera.view_matrix.cuda()  # Standard from Transport
view_matrix = view_matrix_standard.T  # Convert to 3DGStream format
```

---

## 문제 2: LookAt 재구성 시 Roll 정보 손실

### 증상
고정된 up vector로 LookAt 재구성 시:
```python
up = np.array([0., 1., 0.])  # 항상 Y-up 가정
```
→ 카메라 roll 정보 완전 손실

### 원인
COLMAP 카메라는 임의의 방향을 가질 수 있지만, 고정된 up vector는 특정 orientation만 표현 가능

### 해결책
**Frontend Protocol 확장: Up vector 추가**

Protocol (160 bytes):
```
[0:3]     eye (3 floats)
[3:6]     target (3 floats)
[6:15]    intrinsics (9 floats)
[15:16]   padding (1 float)
[16:19]   up (3 floats) ← NEW!
[19:32]   unused (13 floats)
```

**구현**:
```python
# test_client.py - Extract up from 3DGStream format
up = world_view_3dgs[1, :3]  # Row 1 = Up axis

# protocol_converter.py
up_x, up_y, up_z = vals[16], vals[17], vals[18]
if all zeros:
    up = [0, 1, 0]  # Fallback for Mock camera
else:
    up = normalize([up_x, up_y, up_z])  # Use provided up
```

---

## 문제 3: Double Y-flip 문제

### 증상
COLMAP 카메라를 Y-flip하여 전송했더니 rotation이 망가짐

### 원인
```
1. Client: -eye[1], -up[1] (pre-flip)
2. Transport: -vals[1] (다시 flip)
→ 결과: eye는 원위치, rotation 완전 손상
```

### 해결책
**Y-flip은 한 곳에서만 처리**
- COLMAP 카메라: Y-flip 하지 않음 (이미 올바른 coordinate)
- Mock 카메라: Y-flip 하지 않음 (기본 Y-up 사용)
- Transport: Frontend coordinate → Renderer coordinate 변환 담당

```python
# test_client.py (COLMAP) - NO pre-flip
camera_floats = [
    float(eye[0]), float(eye[1]), float(eye[2]),  # NO -eye[1]
    ...
]
```

---

## 문제 4: View Matrix 직접 전송 (최종 해결)

### 배경
eye/target/up → LookAt 재구성은 정보 손실 가능성 있음

### 해결책
**Unused protocol 영역 활용: View matrix 직접 전송**

```python
# test_client.py - Send view_matrix directly
view_matrix_flat = view_matrix_standard.flatten()  # 16 floats
camera_floats = [
    eye[0], eye[1], eye[2],      # For fallback
    target[0], target[1], target[2],
    *intrinsics_flat,
    0.0,  # padding
    *view_matrix_flat  # 16 floats (index 16-31)
]

# protocol_converter.py - Auto-detect
view_matrix_vals = vals[16:32]
if any(abs(v) > 0.01 for v in view_matrix_vals):
    # Use direct view_matrix
    view_matrix = np.array(view_matrix_vals).reshape(4, 4)
else:
    # Fallback: reconstruct from eye/target/up
    view_matrix = compute_lookat(eye, target, up)
```

**장점**:
- ✅ 정보 손실 없음 (bit-exact transmission)
- ✅ Backward compatible (Mock 카메라는 fallback 사용)
- ✅ COLMAP 카메라: 완벽한 카메라 행렬 전달

---

## 문제 5: 카메라 방향 문제 (정면이 아닌 view)

### 증상
test_streamable_gaussian.py로 렌더링 시 살짝 오른쪽 위를 바라봄

### 원인
**create_camera_frame_from_train_camera가 3DGStream format을 그대로 전달**

```python
# 잘못됨
view_matrix = train_camera.world_view_transform  # Transposed format
return CameraFrame(view_matrix=view_matrix)  # 잘못된 format!
```

CameraFrame은 표준 format을 기대하지만, transposed format을 받음 → 카메라 방향 왜곡

### 해결책
```python
# test_streamable_gaussian.py
def create_camera_frame_from_train_camera(train_camera, frame_id):
    view_matrix_3dgs = train_camera.world_view_transform  # Transposed
    view_matrix_standard = view_matrix_3dgs.T  # Convert to standard
    return CameraFrame(view_matrix=view_matrix_standard)  # Correct!
```

---

## 최종 아키텍처

### Data Flow

```
3DGStream Camera (transposed)
    ↓ .T
CameraFrame (standard row-major)
    ↓
Renderer receives standard
    ↓ .T
3DGStream format for rendering
```

### Protocol 명세 (확정)

**Frontend Protocol (160 bytes)**:
```
[0:6]     eye, target (6 floats)
[6:15]    intrinsics (9 floats)
[15:16]   padding (1 float)
[16:32]   view_matrix OR up vector (16 floats)
          - If non-zero matrix: use as direct view_matrix
          - If mostly zero: first 3 floats = up vector
[128:160] metadata (frame_id, timestamps, time_index)
```

**Renderer Protocol (168 bytes)**:
```
[0:64]    view_matrix (4×4, standard row-major)
[64:100]  intrinsics (3×3)
[100:116] width, height, near, far
[116:144] frame_id, timestamps, time_index
[144:168] reserved
```

### 코드 변경 요약

#### 1. test_client.py (COLMAP 모드)
```python
# Extract camera axes from 3DGStream transposed format
right = world_view_3dgs[0, :3]    # Row 0
up = world_view_3dgs[1, :3]       # Row 1
forward = -world_view_3dgs[2, :3] # -Row 2

# Send view_matrix directly (16 floats)
view_matrix_flat = view_matrix_standard.flatten()
camera_floats = [..., *view_matrix_flat]
```

#### 2. protocol_converter.py
```python
# Check if view_matrix provided
view_matrix_vals = vals[16:32]
if view_matrix_provided:
    view_matrix = np.array(view_matrix_vals).reshape(4, 4)
else:
    # Reconstruct from eye/target/up
    view_matrix = compute_lookat_with_up(eye, target, up)
```

#### 3. streamable_gaussian.py
```python
# Convert standard → 3DGStream format
view_matrix_standard = camera.view_matrix.cuda()
view_matrix = view_matrix_standard.T  # Transpose for 3DGStream
```

#### 4. test_streamable_gaussian.py
```python
# Convert 3DGStream → standard for CameraFrame
view_matrix_3dgs = train_camera.world_view_transform
view_matrix_standard = view_matrix_3dgs.T
return CameraFrame(view_matrix=view_matrix_standard)
```

---

## 검증 방법

### 1. Camera Center 일관성
모든 단계에서 camera center가 일치하는지 확인:
```
Client:    [ 0.029 -1.337 -0.215]
Transport: [ 0.029 -1.337 -0.215]  (차이 < 1e-6)
Renderer:  [ 0.029 -1.337 -0.215]  (차이 < 1e-6)
```

### 2. Rotation Matrix Orthogonality
```
det(R) = 1.0
R @ R^T = I (identity)
```

### 3. 렌더링 결과
- ✅ Black이 아닌 정상 이미지
- ✅ 정면 view (0번 카메라)
- ✅ 위아래 올바른 방향

---

## 교훈

### 1. 표준 규격의 중요성
- **Transport는 렌더러 독립적이어야 함**
- 각 렌더러가 자신의 format으로 변환 (Adapter Pattern)

### 2. Coordinate System 일관성
- Y-flip은 한 곳에서만 처리
- 명확한 책임 분리 (Transport vs Renderer)

### 3. 정보 손실 최소화
- LookAt 재구성보다 **직접 전송**이 안전
- Backward compatibility 유지

### 4. 철저한 검증
- 각 단계에서 camera center 확인
- Rotation matrix orthogonality 체크
- 실제 렌더링 결과로 최종 검증

---

## 참고 자료

- **3DGStream**: getWorld2View2(...).transpose(0, 1)
- **OpenGL/CV**: Standard row-major view matrix
- **Protocol**: Frontend(160) ↔ Renderer(168) binary format

## 관련 파일

- `backend/test/test_client.py`: COLMAP camera 전송
- `backend/transport/protocol_converter.py`: Protocol 변환
- `backend/renderer/scene_renderers/streamable_gaussian.py`: 3DGStream adapter
- `backend/test/test_streamable_gaussian.py`: Training/checkpoint 생성

---

**작성일**: 2025-10-29
**상태**: ✅ 해결 완료

---

# Dynamic Gaussian Frontend Integration - Troubleshooting Guide

본 섹션은 Dynamic Gaussian 렌더링을 Frontend에 통합하는 과정에서 발생한 문제와 해결책을 정리합니다.

**작성일**: 2025-10-30

---

## 문제 1: 프로토콜 용량 부족

### 증상
Frontend에서 View Matrix + Projection Matrix + Intrinsics를 모두 전송해야 하는데 기존 프로토콜이 부족함

### 원인
- 기존 프로토콜: 160 bytes (32 floats)
- 필요한 데이터: 41 floats (view:16 + projection:16 + intrinsics:9)

### 해결책
**프로토콜 확장: 160 → 224 bytes**

```
카메라 데이터: 192 bytes (48 floats)
  - view_matrix: 16 floats (indices 0-15)
  - projection_matrix: 16 floats (indices 16-31)
  - intrinsics: 9 floats (indices 32-40)
  - reserved: 7 floats (indices 41-47)
메타데이터: 32 bytes
  - frame_id: uint32 (offset 192)
  - client_timestamp: float64 (offset 200)
  - time_index: float32 (offset 208)
```

**수정 파일:**
- `frontend/src/types/index.ts`: CameraFrame 타입 수정
- `frontend/src/decode-worker.ts`: 224 bytes 프로토콜 구현
- `backend/transport/protocol_converter.py`: 224 bytes 파싱
- `backend/transport/adapters/websocket_adapter.py`: 프레임 크기 검증 업데이트

---

## 문제 2: 카메라 Intrinsics 왜곡

### 증상
Projection matrix에서 intrinsics를 역산하면 왜곡 발생 가능

### 원인
- `getCameraIntrinsics()`는 Gaussian 학습 시 사용된 파라미터를 재현하도록 설계됨
- Transport에서 재계산하면 pixel-space 정확도 손실
- Projection matrix는 normalized device coordinates를 사용하므로 해상도 정보 손실

### 해결책
**Frontend에서 계산한 intrinsics를 그대로 전송**

```typescript
// CameraController.getCameraFrame()
const intrinsics = new Float32Array(this.getCameraIntrinsics());
// getCameraIntrinsics()는 pixel-space 계산 수행:
// fx = (width / 2) * projectionMatrix[0]
// fy = (height / 2) * projectionMatrix[5]
```

**Transport는 검증 없이 직접 사용:**
```python
# protocol_converter.py
intrinsics_vals = camera_data[32:41]
intrinsics = np.array([...], dtype=np.float32).reshape(3, 3)
# NO VALIDATION - zero overhead
```

---

## 문제 3: Transport Layer 검증 오버헤드

### 증상
Transport에서 view_matrix의 orthonormality 검증이 60 FPS에서 성능 저하 유발

### 원인
```python
# 불필요한 검증
is_valid = np.allclose(R @ R.T, np.eye(3), atol=0.15)
```
- CameraController에서 보내는 데이터는 이미 신뢰할 수 있음
- 매 프레임 검증은 불필요한 오버헤드

### 해결책
**모든 검증 로직 제거**

```python
# AFTER
view_matrix = np.array(camera_data[0:16], dtype=np.float32).reshape(4, 4)
projection = np.array(camera_data[16:32], dtype=np.float32).reshape(4, 4)
intrinsics = np.array(camera_data[32:41], dtype=np.float32).reshape(3, 3)
# Direct use, no validation
```

**성능 개선:**
- Transport 변환: < 0.1ms (검증 제거 후)

---

## 문제 4: Dynamic Gaussian Frame Looping 실패

### 증상 1: 하드코딩된 total_frames
```python
total_frames = 300  # TODO: Get from config
frame_idx = int(camera.time_index * (total_frames - 1))
```

### 증상 2: 존재하지 않는 Frame 접근
```
[WARNING] PLY file not found: frame_000477/gaussian.ply
[RENDER] Render error at frame 477: No Gaussian state found
[WARNING] PLY file not found: frame_000393/gaussian.ply
```

### 원인
1. Frontend의 frame_id가 393, 394, 477... 계속 증가
2. 실제 파일은 frame_000001 ~ frame_000300만 존재
3. time_index 매핑이 파일 개수를 고려하지 않음

### 해결책

**1. available_frame_ids 리스트 생성**
```python
# streamable_gaussian.py - on_init()
glob_pattern = self.weight_path_pattern.replace("{frame_id:06d}", "*")
matching_files = sorted(glob.glob(glob_pattern))

self.available_frame_ids = []
for filepath in matching_files:
    match = re.search(r"frame_(\d+)", filepath)
    if match:
        frame_id = int(match.group(1))
        self.available_frame_ids.append(frame_id)

self.available_frame_ids.sort()
self.total_frames = len(self.available_frame_ids)

print(f"[INIT] Auto-detected {self.total_frames} frames")
print(f"[INIT] Frame ID range: {self.available_frame_ids[0]} ~ {self.available_frame_ids[-1]}")
```

**2. 모든 frame_idx를 % 파일개수로 순환**
```python
# streamable_gaussian.py - render()
# CRITICAL: 어떤 frame_idx가 오든 순환 보장
if self.available_frame_ids:
    idx = frame_idx % len(self.available_frame_ids)
    actual_frame_id = self.available_frame_ids[idx]
    frame_idx = actual_frame_id
```

**동작 예시 (300개 파일):**
| 입력 frame_idx | modulo (idx) | 실제 frame_id | 결과 |
|---------------|-------------|--------------|------|
| 0 | 0 | 1 | ✅ |
| 299 | 299 | 300 | ✅ |
| 300 | 0 | 1 | ✅ 순환 |
| 393 | 93 | 94 | ✅ 순환 |
| 477 | 177 | 178 | ✅ 순환 |
| 600 | 0 | 1 | ✅ 순환 |

---

## 문제 5: TimeController 초기 상태

### 증상
Dynamic Gaussian 렌더링인데 수동으로 Play 버튼을 클릭해야 재생 시작

### 원인
```typescript
private isPlaying: boolean = false;  // 초기 상태: 정지
```

### 해결책
**자동 재생 기본값으로 설정**

```typescript
export class TimeController {
  private isPlaying: boolean = true;  // Auto-play by default
  // ...
}
```

**UI 초기화도 수정:**
```typescript
// TimeControlUI.ts
this.playButton.textContent = timeController.isCurrentlyPlaying()
  ? "⏸ Pause"
  : "▶ Play";
```

---

## 문제 6: Frontend Time Loop 구현

### 요구사항
- time_index를 [0.0, 1.0] 범위로 정규화
- 자동으로 무한 루프 (1.0 도달 시 0.0으로 복귀)
- 속도 조절 가능 (0.01x ~ 5.0x)
- UI로 수동 제어 가능 (Play/Pause/Scrub)

### 해결책
**TimeController 구현**

```typescript
update(deltaTime: number): number {
  if (this.isPlaying && !this.manualOverride) {
    this.currentTime += deltaTime * this.playbackSpeed;
    this.currentTime = this.currentTime % 1.0;  // 자동 루프
  }
  return this.currentTime;
}
```

**Backend 매핑:**
```python
# time_index (0.0~1.0) → list_idx → actual_frame_id
idx = int(time_index * (len(self.available_frame_ids) - 1))
frame_id = self.available_frame_ids[idx]
```

**UI 컴포넌트:**
- `TimeControlUI.ts`: Play/Pause, Speed control
- `FrameScrubber.ts`: Timeline slider, Frame display
- `UISystem.ts`: UI 통합 및 update 호출

---

## 문제 7: UI 컴포넌트 미표시

### 증상
TimeControlUI와 FrameScrubber가 화면에 나타나지 않음

### 원인
1. HTML에 마운트 컨테이너 없음
2. CSS 파일 로드 안됨
3. UISystem에서 DOM 추가 누락

### 해결책

**1. HTML에 컨테이너 추가:**
```html
<!-- index.html -->
<link rel="stylesheet" href="./src/ui/styles.css" />
<!-- ... -->
<div id="time-controls"></div>
```

**2. UISystem에서 마운트:**
```typescript
// UISystem.ts - initialize()
const timeControlContainer = document.getElementById('time-controls');
if (timeControlContainer) {
  timeControlContainer.appendChild(this.timeControl.getElement());
  timeControlContainer.appendChild(this.frameScrubber.getElement());
}
```

**3. update() 호출:**
```typescript
// UISystem.ts - update()
update(_deltaTime: number): void {
  this.timeControl.update();
  this.frameScrubber.update();
}
```

---

## 최종 아키텍처

### CameraController: Single Source of Truth

**모든 카메라 데이터는 CameraController.getCameraFrame()을 통해서만 접근**

```typescript
getCameraFrame(timeIndex: number = 0): CameraFrame {
  const view = new Float32Array(this.camera.matrixWorldInverse.toArray());
  const projection = new Float32Array(this.camera.projectionMatrix.toArray());
  const intrinsics = new Float32Array(this.getCameraIntrinsics());

  return { view, projection, intrinsics, frameId, timestamp, timeIndex };
}
```

### Data Flow

```
Frontend (THREE.js)
  ↓ CameraController.getCameraFrame()
CameraFrame { view, projection, intrinsics }
  ↓ decode-worker (224 bytes)
WebSocket
  ↓
Transport (protocol_converter)
  ↓ No validation, direct reshape
CameraFrame (numpy arrays)
  ↓ Unix Socket
Renderer (streamable_gaussian)
  ↓ Frame index mapping
available_frame_ids[frame_idx % total_frames]
  ↓
3DGS Rendering
```

### Frame Index Mapping

```python
# Example: 300 frames (frame_000001 ~ frame_000300)
available_frame_ids = [1, 2, 3, ..., 300]

# Frontend → Backend
time_index=0.0   → idx=0   → frame_id=1
time_index=0.5   → idx=149 → frame_id=150
time_index=1.0   → idx=299 → frame_id=300
time_index=0.0   → idx=0   → frame_id=1 (loop)

# 순환 보장
frame_idx=393 → idx=93  → frame_id=94
frame_idx=477 → idx=177 → frame_id=178
```

---

## 요약: 핵심 변경사항

### Frontend (7개 파일)
1. ✅ `types/index.ts`: CameraFrame 타입 수정
2. ✅ `CameraController.ts`: view + projection + intrinsics 전송
3. ✅ `decode-worker.ts`: 224 bytes 프로토콜 구현
4. ✅ `Application.ts`: TimeController 추가 (auto-play)
5. ✅ `main.ts`: TimeController.update() 호출
6. ✅ `UISystem.ts`: TimeControlUI, FrameScrubber 통합
7. ✅ `index.html`: time-controls 컨테이너 + CSS

### Backend (2개 파일)
1. ✅ `protocol_converter.py`: 224 bytes 파싱, 검증 제거
2. ✅ `streamable_gaussian.py`: available_frame_ids + 순환

### UI 컴포넌트 (신규 4개)
1. ✅ `TimeControlUI.ts`: Play/Pause, Speed control
2. ✅ `FrameScrubber.ts`: Timeline slider
3. ✅ `ui/styles.css`: UI 스타일링
4. ✅ (UISystem 수정)

---

## 성능 및 안정성

### 성능
- Transport 변환: < 0.1ms (검증 제거)
- Frame looping: 100% 안정성
- 60 FPS 유지

### 안정성
- ✅ 존재하지 않는 frame 접근 방지
- ✅ time_index 범위 보장 [0.0, 1.0]
- ✅ 무한 루프 자동 순환
- ✅ UI 응답성 (60 FPS)

---

## 테스트 방법

### 1. Backend 시작
```bash
python -m renderer.main \
  --scene-type streamable \
  --model-path /path/to/scene.ply \
  --weight-path-pattern "/path/frame_{frame_id:06d}/gaussian.ply" \
  --checkpoints-dir /path/to/checkpoints \
  --encoder-type jpeg
```

### 2. Transport 시작
```bash
python -m transport.main --port 8765
```

### 3. Frontend 시작
```bash
cd frontend
npm run dev
```

### 4. 확인사항
- ✅ Console에 "[INIT] Auto-detected N frames" 출력
- ✅ "[INIT] Frame ID range: X ~ Y" 출력
- ✅ 하단 중앙에 Time Control UI 표시
- ✅ Play 버튼이 "⏸ Pause"로 표시 (자동 재생 중)
- ✅ Timeline slider가 자동으로 이동
- ✅ Frame이 0~N-1 범위로 순환
- ✅ 에러 없이 무한 루프

---

## 관련 파일

### Frontend
- `frontend/src/types/index.ts`
- `frontend/src/systems/CameraController.ts`
- `frontend/src/decode-worker.ts`
- `frontend/src/core/Application.ts`
- `frontend/src/main.ts`
- `frontend/src/ui/TimeControlUI.ts`
- `frontend/src/ui/FrameScrubber.ts`
- `frontend/src/systems/UISystem.ts`
- `frontend/src/ui/styles.css`
- `frontend/index.html`

### Backend
- `backend/transport/protocol_converter.py`
- `backend/transport/adapters/websocket_adapter.py`
- `backend/renderer/scene_renderers/streamable_gaussian.py`

---

## 교훈

### 1. Protocol Design
- 미래 확장성 고려 (reserved fields)
- Validation은 최소화 (performance)
- Single source of truth (CameraController)

### 2. Frame Management
- 실제 존재하는 파일만 추적 (available_frame_ids)
- Modulo 연산으로 순환 보장
- Auto-detection으로 설정 간소화

### 3. UI/UX
- Auto-play by default (dynamic content)
- Manual override 지원 (user control)
- 실시간 feedback (timeline, frame number)

### 4. Zero Overhead
- 검증은 개발 단계에서만
- Production에서는 trust + speed
- 병목 최소화

---

**작성일**: 2025-10-30
**상태**: ✅ 해결 완료, Production Ready

---

# Protocol v3 Migration - Coordinate System Issues

본 섹션은 matrixWorld 기반 렌더링에서 lookAt 기반 렌더링으로 전환하는 과정에서 발생한 문제와 임시 해결책을 정리합니다.

**작성일**: 2025-10-31
**상태**: ⚠️ 임시 해결 (근본 원인 해결 필요)

---

## 배경: Protocol v3로의 전환

### 기존 방식의 문제점
matrixWorld를 직접 gsplat에 전달하려 했으나, Three.js와 gsplat 간의 좌표계 변환이 복잡했습니다:

```python
# 경험적으로 도출된 변환 행렬 (복잡함)
X_fixed = torch.tensor([
    [-1.0,  0.0,         0.0,        0.0],
    [ 0.0,  0.98089421,  0.19454218,  0.99521220],  # Y-Z rotation
    [ 0.0,  0.19454220, -0.98089421,  0.09773874],
    [ 0.0,  0.0,         0.0,        1.0]
], device='cuda')

w2c = X_fixed @ torch.inverse(M_transposed)
```

### 새로운 방식: lookAt 기반
더 단순하고 명확한 접근:
- Frontend에서 `camera.position`, `controls.target`, `camera.up` 전송
- Backend에서 표준 lookAt 알고리즘으로 w2c 행렬 계산

---

## 문제 1: Protocol 업데이트 누락

### 증상
```
[WebSocket] Warning: Unknown message size 260
[CAMERA] Error receiving camera: Invalid camera frame size: 168 (expected 204)
[RENDER] Render error: Camera position, target, and up vectors are required
```

### 원인
Protocol v3로 업데이트했지만 여러 파일에서 누락:

1. **Frontend → Transport (260 bytes)**
   - ✅ `types/index.ts`: CameraFrame 인터페이스
   - ✅ `CameraController.ts`: position, target, up 추출
   - ✅ `decode-worker.ts`: 260 bytes 전송
   - ❌ **누락**: websocket_adapter.py가 224 bytes 기대

2. **Transport → Renderer (204 bytes)**
   - ✅ `protocol_converter.py`: 260 bytes 파싱
   - ❌ **누락**: unix_socket_adapter.py 주석만 업데이트
   - ❌ **누락**: protocol.py가 168 bytes로 pack/parse
   - ❌ **누락**: renderer_service.py가 168 bytes 읽기

3. **Renderer 내부**
   - ✅ `data_types.py`: position, target, up 필드
   - ❌ **누락**: camera_to_torch()가 필드 변환 안함
   - ❌ **누락**: static_gaussian.py가 numpy 가정

### 해결책

**1. WebSocket Adapter (260 bytes 인식)**
```python
# websocket_adapter.py
elif len(raw) == 260:  # Protocol v3
    camera = parse_frontend_camera(raw, ...)
```

**2. Protocol Converter (position, target, up 추출)**
```python
# protocol_converter.py
position = np.array(camera_data[41:44], dtype=np.float32)
target = np.array(camera_data[44:47], dtype=np.float32)
up = np.array(camera_data[47:50], dtype=np.float32)

return CameraFrame(..., position=position, target=target, up=up)
```

**3. Unix Socket Protocol (204 bytes)**
```python
# protocol.py
CAMERA_FRAME_SIZE = 204  # 168 + 36 (position:12 + target:12 + up:12)

def pack_camera_frame(camera):
    position_bytes = position.astype(np.float32).tobytes()  # 12 bytes
    target_bytes = target.astype(np.float32).tobytes()      # 12 bytes
    up_bytes = up.astype(np.float32).tobytes()              # 12 bytes

    return view_bytes + intrinsics_bytes + metadata_bytes + \
           position_bytes + target_bytes + up_bytes + reserved
```

**4. Renderer Service (204 bytes 읽기)**
```python
# renderer_service.py
remaining = await self.camera_reader.read(196)  # 204 - 8 = 196

if len(remaining) < 196:
    print(f"Incomplete camera frame: {len(data) + len(remaining)} bytes (expected 204)")
```

**5. camera_to_torch 업데이트**
```python
# renderer_service.py - camera_to_torch()
position=torch.from_numpy(camera.position).to(device) if camera.position is not None else None,
target=torch.from_numpy(camera.target).to(device) if camera.target is not None else None,
up=torch.from_numpy(camera.up).to(device) if camera.up is not None else None,
```

**6. static_gaussian.py (numpy/torch 호환)**
```python
# Handle both numpy and torch tensors
if isinstance(camera.position, torch.Tensor):
    cam_pos = camera.position.to(self.device)
else:
    cam_pos = torch.from_numpy(camera.position).to(self.device)
```

---

## 문제 2: Y축 대칭 문제 (⚠️ 임시 해결)

### 증상
lookAt 방식으로 렌더링했더니 **씬이 상하 반전**되어 표시됨

### 원인 분석

**Three.js vs gsplat 좌표계:**
- Three.js: Y-up, right-handed
- gsplat: 좌표계 불명확 (경험적으로 Y-down 추정)

**matrixWorld 방식 (이전):**
```python
# X_fixed 변환 행렬에 Y축 flip이 포함되어 있었음
X_fixed = torch.tensor([
    [-1.0,  0.0,         0.0,        0.0],  # X-axis flip
    [ 0.0,  0.98089421,  0.19454218,  ...],  # Y-Z rotation (~11.2°)
    ...
])
```

**lookAt 방식 (현재):**
```python
# 표준 lookAt 알고리즘만 사용
forward = target - cam_pos
right = cross(up_vector, forward)
up = cross(forward, right)
```
→ Y축 변환이 누락되어 상하 반전

### 임시 해결책: Frontend에서 Y축 Flip

**Frontend (CameraController.ts):**
```typescript
// TEMPORARY: Flip Y-axis for position and target
const position = new Float32Array([
  this.camera.position.x,
  -this.camera.position.y,  // Flip Y
  this.camera.position.z
]);
const target = new Float32Array([
  this.controls.target.x,
  -this.controls.target.y,  // Flip Y
  this.controls.target.z
]);
```

**Backend (static_gaussian.py):**
```python
# NOTE: Y-axis flip is now done in frontend (CameraController.ts)
# Frontend already sends Y-flipped coordinates

# 표준 lookAt 알고리즘 그대로 사용
forward = target - cam_pos  # Y-flipped 좌표 직접 사용
```

---

## ⚠️ 근본 원인 미해결 사항

### 알려진 문제점

1. **좌표계 불일치의 정확한 원인 불명**
   - Three.js와 gsplat의 정확한 좌표계 차이 미검증
   - X_fixed 변환 행렬이 왜 필요했는지 근본 원인 불명

2. **임시 해결책의 한계**
   - Frontend에서 Y축 flip = 시스템 전체가 "잘못된" 좌표계 사용
   - 다른 렌더러 추가 시 혼란 가능성
   - 물리 시뮬레이션, 충돌 감지 등에서 추가 변환 필요 가능

3. **matrixWorld를 버린 이유**
   - X_fixed 변환이 복잡하고 이해하기 어려움
   - lookAt이 더 직관적이라고 판단
   - 하지만 근본 문제는 해결 안됨

### 향후 해결 방안

**Option 1: 좌표계 정확히 파악**
```python
# gsplat의 정확한 좌표계 문서 확인
# Three.js → gsplat 변환 공식 도출
# 표준 변환 행렬 정의
```

**Option 2: matrixWorld 방식 재검토**
```python
# X_fixed 변환 행렬의 의미 파악
# 왜 X-axis flip + Y-Z rotation이 필요한지 이해
# 더 명확한 변환 공식 도출
```

**Option 3: gsplat 소스코드 분석**
```python
# gsplat rasterization의 좌표계 가정 확인
# w2c 행렬 입력 형식 정확히 파악
# 필요한 전처리 확인
```

---

## 데이터 플로우 (Protocol v3)

```
Frontend (Three.js, Y-up)
  ↓ Y-axis flip (임시)
  camera.position[1] = -camera.position[1]
  controls.target[1] = -controls.target[1]
  ↓ 260 bytes
WebSocket (8765)
  ↓ parse_frontend_camera()
Transport Service
  ↓ pack_camera_frame() → 204 bytes
Unix Socket (camera.sock)
  ↓ read(8) + read(196)
Renderer Service
  ↓ parse_camera_frame()
  ↓ camera_to_torch()
GsplatRenderer
  ↓ _compute_w2c_from_lookat() (표준 lookAt)
gsplat rasterization
  ✅ 올바른 렌더링 (하지만 임시 해결)
```

---

## 교훈

### 1. 프로토콜 변경 시 체크리스트
프로토콜 업데이트는 전체 파이프라인 검증 필요:
- [ ] Frontend: 데이터 생성 및 전송
- [ ] Transport: WebSocket 수신 크기 검증
- [ ] Transport: Protocol converter 파싱
- [ ] Transport: Unix Socket 전송 크기
- [ ] Renderer: Unix Socket 수신 크기
- [ ] Renderer: Protocol 파싱
- [ ] Renderer: 데이터 타입 변환
- [ ] Renderer: 실제 사용 (렌더러)

### 2. 좌표계 변환은 명확히
임시 해결책은 기술 부채:
- 근본 원인을 이해하고 해결
- 문서화 및 주석 필수
- 향후 유지보수를 위해 명확한 이유 기록

### 3. Python 캐시 관리
코드 변경 후 반드시:
```bash
find backend -type d -name "__pycache__" -exec rm -rf {} +
```

### 4. 점진적 검증
각 단계별 디버그 로그:
```python
print(f"[DEBUG] Packed camera frame size: {len(data)} bytes")
print(f"  position: {camera.position}")
print(f"  target: {camera.target}")
```

---

## 관련 파일

### Frontend
- `frontend/src/types/index.ts`: CameraFrame (260 bytes)
- `frontend/src/systems/CameraController.ts`: Y-axis flip (임시)
- `frontend/src/decode-worker.ts`: 260 bytes protocol

### Backend Transport
- `backend/transport/protocol_converter.py`: 260 → 204 변환
- `backend/transport/adapters/websocket_adapter.py`: 260 bytes
- `backend/transport/adapters/unix_socket_adapter.py`: 204 bytes

### Backend Renderer
- `backend/renderer/utils/protocol.py`: 204 bytes pack/parse
- `backend/renderer/renderer_service.py`: 204 bytes read, camera_to_torch
- `backend/renderer/scene_renderers/static_gaussian.py`: lookAt + numpy/torch 호환

---

## TODO: 향후 작업

### 우선순위 1: 좌표계 근본 원인 파악
- [ ] gsplat 문서/소스코드 분석
- [ ] Three.js matrixWorld 형식 재확인
- [ ] 정확한 변환 공식 도출
- [ ] Y-axis flip 제거

### 우선순위 2: matrixWorld 방식 재검토
- [ ] X_fixed 변환 행렬 의미 파악
- [ ] 더 명확한 구현 방식 제안
- [ ] lookAt vs matrixWorld 장단점 비교

### 우선순위 3: 테스트 강화
- [ ] 다양한 카메라 각도에서 검증
- [ ] 회전 행렬 orthogonality 체크
- [ ] 렌더링 결과 육안 검증

---

**작성일**: 2025-10-31
**상태**: ⚠️ 임시 해결 (Y-axis flip in frontend)
**근본 원인**: 미해결 (추후 작업 필요)

---

# Frontend UI Refactoring Issues

본 섹션은 Frontend를 Application + Systems 패턴으로 리팩토링한 후 발생한 문제들을 정리합니다.

**작성일**: 2025-10-31

---

## 문제 1: UISystem 통합 후 기능 미동작

### 증상
UISystem으로 통합한 후 Debug Option과 Render Mode 선택이 동작하지 않음

### 원인

1. **RenderMode enum 중복 정의**
   - `types/index.ts`: 구버전 (Mesh/Gaussian/Hybrid)
   - `ControlPanel.ts`: 신규 (fusion/gaussian-only/local-only/depth-fusion/feed-forward)
   - `RenderingSystem.ts`: 대문자 버전 (FUSION/GAUSSIAN_ONLY)

2. **RenderingSystem 업데이트 누락**
   - `UISystem.handleRenderModeChange()`에서 `RenderingSystem.setRenderMode()` 호출 없음
   - RenderMode 변경이 실제 렌더링에 반영 안됨

3. **DepthDebug 기능 미구현**
   - `UISystem.handleDepthDebugToggle()`에 TODO 주석만 존재
   - RenderingSystem에 depth debug 상태 관리 로직 없음

4. **FEED_FORWARD 모드 누락**
   - `RenderingSystem.ts`에 FEED_FORWARD case 없음

### 해결책

**1. RenderMode enum 통합 (types/index.ts)**
```typescript
export enum RenderMode {
  // Legacy modes
  Mesh = "mesh",
  Gaussian = "gaussian",
  Hybrid = "hybrid",

  // Active rendering modes
  FUSION = "fusion",
  GAUSSIAN_ONLY = "gaussian-only",
  LOCAL_ONLY = "local-only",
  DEPTH_FUSION = "depth-fusion",
  FEED_FORWARD = "feed-forward",
}
```

**2. UISystem에서 RenderingSystem 업데이트**
```typescript
private handleRenderModeChange(mode: RenderMode): void {
  const renderingSystem = this.app.getSystem('rendering');
  renderingSystem.setRenderMode(mode);  // 추가
  // ...
}
```

**3. DepthDebug 기능 구현**
```typescript
// RenderingSystem.ts
setDepthDebugEnabled(enabled: boolean): void {
  this.isDepthDebugEnabled = enabled;
}

render(): void {
  if (this.isDepthDebugEnabled) {
    this.renderer.render(this.config.debugScene, this.config.debugCamera);
    return;
  }
  // ...
}
```

**4. FEED_FORWARD 모드 추가**
```typescript
case RenderMode.FEED_FORWARD:
  this.renderer.render(
    this.config.gaussianOnlyScene,
    this.config.gaussianOnlyCamera
  );
  break;
```

---

## 문제 2: Render Mode별 UV 및 ColorSpace 불일치

### 증상
- Gaussian Only: 좌우반전되고 색상이 다름
- Local Only: 좌우반전됨
- Depth Debug: 전체가 반전되고 색상이 다름

### 원인

**Fusion의 실제 UV 변환:**
```glsl
// fusionFlipX=true + wsFlipX=true
currentUv.x = 1.0 - vUv.x;              // fusionFlipX
wsUv = vec2(currentUv.x, 1.0 - currentUv.y);
wsUv.x = 1.0 - wsUv.x;                  // wsFlipX
// 최종: vec2(vUv.x, 1.0 - vUv.y)  <- X축 두 번 플립되어 취소
```

**기존 코드:**
- Gaussian Only: `vec2(1.0 - vUv.x, 1.0 - vUv.y)` (X, Y 모두 플립)
- Local Only: 직접 3D 렌더링 (ColorSpace 불일치)
- Depth Debug: 플립 없이 직접 샘플링

### 해결책

**1. Gaussian Only UV 및 ColorSpace 통일**
```glsl
// gaussianOnlyFragmentShader
vec2 wsUv = vec2(vUv.x, 1.0 - vUv.y);  // Y축만 플립
gl_FragColor = linearToOutputTexel(wsColor);  // ColorSpace 변환
```

**2. Local Only를 Post-processing Quad로 변경**
```typescript
// localOnlyFragmentShader
vec2 localUv = vec2(1.0 - vUv.x, vUv.y);  // X축 플립
gl_FragColor = linearToOutputTexel(localColor);
```

```typescript
// RenderingSystem.ts - LOCAL_ONLY
this.renderer.setRenderTarget(this.config.localRenderTarget);
this.renderer.render(this.config.localScene, this.config.camera);
this.renderer.setRenderTarget(null);
this.renderer.render(this.config.localOnlyScene, this.config.localOnlyCamera);
```

**3. Depth Debug 각 뷰별 UV 설정**
```glsl
// debugColorShader.fs
if (gridPos.x < 1.0) {  // Local Color/Depth
  vec2 flippedUv = vec2(1.0 - localUv.x, localUv.y);  // X축 플립
  color = texture2D(localColorSampler, flippedUv);
  applyColorSpace = false;  // Local은 ColorSpace 변환 안함
}
else if (gridPos.x < 2.0) {  // Splat Color/Depth
  vec2 wsUv = vec2(localUv.x, 1.0 - localUv.y);  // Y축만 플립
  color = texture2D(wsColorSampler, wsUv);
  applyColorSpace = true;  // Gaussian은 ColorSpace 변환
}
else {  // Fusion Color/Depth
  vec2 localFlippedUv = vec2(1.0 - localUv.x, localUv.y);
  vec2 wsUv = vec2(localUv.x, 1.0 - localUv.y);
  // Fusion 로직...
  applyColorSpace = true;
}

// 조건부 ColorSpace 변환
if (applyColorSpace) {
  gl_FragColor = linearToOutputTexel(color);
} else {
  gl_FragColor = color;
}
```

---

## 최종 UV 매핑 규칙

### Local 텍스처
- **X축 플립 필요**: `vec2(1.0 - vUv.x, vUv.y)`
- **ColorSpace**: 변환 안함 (Three.js 기본)

### WebSocket (Gaussian) 텍스처
- **Y축만 플립**: `vec2(vUv.x, 1.0 - vUv.y)`
- **ColorSpace**: `linearToOutputTexel()` 적용

### Depth 시각화
- **ColorSpace**: 변환 안함 (grayscale)

---

## 관련 파일

### RenderMode 통합
- `frontend/src/types/index.ts`: 통합 enum
- `frontend/src/systems/RenderingSystem.ts`: FEED_FORWARD 추가
- `frontend/src/systems/UISystem.ts`: RenderingSystem 연동
- `frontend/src/ui/panels/ControlPanel.ts`: enum import 수정

### UV 및 ColorSpace
- `frontend/src/main.ts`: gaussianOnlyFragmentShader, localOnlyFragmentShader
- `frontend/src/shaders/debugColorShader.fs`: 6개 뷰별 UV 설정

---

**작성일**: 2025-10-31
**상태**: ✅ 해결 완료
