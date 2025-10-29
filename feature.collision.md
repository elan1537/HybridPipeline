# Mesh-Gaussian Collision Detection - Implementation

## 상태: ✅ 완료 (2025-10-29)

**Hybrid Rendering** 환경에서 Three.js Mesh 객체들이 Gaussian Splatting Scene과 물리적으로 충돌하도록 구현

---

## 구현 완료 사항

✅ Backend: Projection matrix에서 near/far 자동 추출
✅ Frontend: Log-normalized depth → Linear depth 변환
✅ TextureManager: Direct depth sampling (sampleLinearDepth)
✅ MeshGaussianCollision: Depth-based collision detection
✅ PhysicsSystem: Physics simulation + collision response
✅ Application: Physics API 통합

---

## 시스템 구조

### Hybrid Rendering Architecture

```
┌─────────────────────────────────────┐
│   Three.js Renderer                 │
│   - Mesh 객체 (캐릭터, 아이템 등)        │
│   - 동적 물체                         │
│   - Physics simulation              │
└─────────────────────────────────────┘
              ↓ (동일 Camera)
┌─────────────────────────────────────┐
│   Gaussian Renderer                 │
│   - 실제 스캔된 환경 (벽, 바닥 등)        │
│   - Depth map 제공                   │
│   - 정적 배경                         │
└─────────────────────────────────────┘
```

### 충돌 시나리오

```
Gaussian Scene (Depth Map으로 표현)
[Wall] ← depth = 5.0m
  ↑
 [M] ← Three.js Mesh (depth = 5.2m)
  ↑    (Mesh가 벽보다 뒤에 있음 → OK)


[Wall] ← depth = 5.0m
  ↑
 [M] ← Three.js Mesh (depth = 4.8m)
      (Mesh가 벽 앞에 있으려 함 → COLLISION!)
```

---

## 핵심 원리

### Depth-based Collision

**Mesh의 world position을 screen space로 투영하여 depth 비교**

```
1. Mesh position (world space) → Screen space (UV)
2. Depth map lookup (Gaussian depth at UV)
3. Mesh depth calculation (camera distance)
4. Comparison: meshDepth >= gaussianDepth → Collision!
```

### 부등식

```
d_mesh >= d_gaussian → 충돌

d_mesh: 카메라에서 Mesh까지의 거리
d_gaussian: 카메라에서 Gaussian scene까지의 거리 (depth map에서 읽음)
```

**의미**: Mesh가 Gaussian scene보다 뒤에 있거나 같은 위치면 충돌 (침투)

---

## 구현 설계

### 1. 기본 Collision Check

```typescript
class MeshGaussianCollision {
  private depthMap: Float32Array | null = null;
  private depthMapWidth: number = 0;
  private depthMapHeight: number = 0;
  private camera: THREE.Camera;

  // Depth map 업데이트 (매 프레임 수신)
  updateDepthMap(depthMap: Float32Array, width: number, height: number) {
    this.depthMap = depthMap;
    this.depthMapWidth = width;
    this.depthMapHeight = height;
  }

  // Mesh 충돌 체크
  checkCollision(
    mesh: THREE.Mesh,
    proposedPosition: THREE.Vector3
  ): CollisionResult {
    if (!this.depthMap) {
      return { collision: false };
    }

    // 1. World position → Screen space
    const screenUV = this.worldToScreen(proposedPosition);

    // 2. Screen UV 범위 체크
    if (screenUV.x < 0 || screenUV.x > 1 || screenUV.y < 0 || screenUV.y > 1) {
      return { collision: false }; // 화면 밖
    }

    // 3. Depth map lookup
    const x = Math.floor(screenUV.x * this.depthMapWidth);
    const y = Math.floor(screenUV.y * this.depthMapHeight);
    const gaussianDepth = this.depthMap[y * this.depthMapWidth + x];

    // 4. Mesh depth 계산
    const meshDepth = this.camera.position.distanceTo(proposedPosition);

    // 5. 충돌 판별
    const EPSILON = 0.01; // 1cm tolerance
    if (meshDepth >= gaussianDepth - EPSILON) {
      return {
        collision: true,
        gaussianDepth,
        meshDepth,
        penetrationDepth: meshDepth - gaussianDepth,
      };
    }

    return { collision: false };
  }

  // World space → Screen UV
  private worldToScreen(worldPos: THREE.Vector3): THREE.Vector2 {
    const projected = worldPos.clone().project(this.camera);

    // NDC [-1, 1] → UV [0, 1]
    return new THREE.Vector2(
      (projected.x + 1) / 2,
      (1 - projected.y) / 2 // Y 축 반전
    );
  }
}

interface CollisionResult {
  collision: boolean;
  gaussianDepth?: number;
  meshDepth?: number;
  penetrationDepth?: number;
}
```

### 2. Multi-Point Collision (Bounding Volume)

```typescript
// Mesh의 Bounding Sphere를 고려한 정확한 충돌 검사
checkCollisionAccurate(mesh: THREE.Mesh, proposedPosition: THREE.Vector3): CollisionResult {
    // Bounding sphere 계산
    mesh.geometry.computeBoundingSphere();
    const boundingSphere = mesh.geometry.boundingSphere!;
    const radius = boundingSphere.radius * mesh.scale.length(); // Scale 고려

    // 샘플 포인트들 (중심 + 6방향)
    const samplePoints = [
        proposedPosition.clone(),                                           // Center
        proposedPosition.clone().add(new THREE.Vector3(radius, 0, 0)),     // +X
        proposedPosition.clone().add(new THREE.Vector3(-radius, 0, 0)),    // -X
        proposedPosition.clone().add(new THREE.Vector3(0, radius, 0)),     // +Y
        proposedPosition.clone().add(new THREE.Vector3(0, -radius, 0)),    // -Y
        proposedPosition.clone().add(new THREE.Vector3(0, 0, radius)),     // +Z
        proposedPosition.clone().add(new THREE.Vector3(0, 0, -radius)),    // -Z
    ];

    let maxPenetration = -Infinity;
    let collisionDetected = false;

    for (const point of samplePoints) {
        const screenUV = this.worldToScreen(point);

        // 화면 밖은 스킵
        if (screenUV.x < 0 || screenUV.x > 1 || screenUV.y < 0 || screenUV.y > 1) {
            continue;
        }

        const x = Math.floor(screenUV.x * this.depthMapWidth);
        const y = Math.floor(screenUV.y * this.depthMapHeight);
        const gaussianDepth = this.depthMap![y * this.depthMapWidth + x];
        const meshDepth = this.camera.position.distanceTo(point);

        const penetration = meshDepth - gaussianDepth;
        if (penetration >= -0.01) { // 1cm tolerance
            collisionDetected = true;
            maxPenetration = Math.max(maxPenetration, penetration);
        }
    }

    if (collisionDetected) {
        return {
            collision: true,
            penetrationDepth: maxPenetration,
        };
    }

    return { collision: false };
}
```

### 3. Collision Response (물리 반응)

```typescript
class PhysicsMesh {
  mesh: THREE.Mesh;
  velocity: THREE.Vector3;
  acceleration: THREE.Vector3;
  collisionSystem: MeshGaussianCollision;

  constructor(mesh: THREE.Mesh, collisionSystem: MeshGaussianCollision) {
    this.mesh = mesh;
    this.velocity = new THREE.Vector3();
    this.acceleration = new THREE.Vector3(0, -9.8, 0); // 중력
    this.collisionSystem = collisionSystem;
  }

  update(deltaTime: number) {
    // 1. 물리 시뮬레이션
    this.velocity.add(this.acceleration.clone().multiplyScalar(deltaTime));

    // 2. 제안된 새 위치
    const currentPos = this.mesh.position.clone();
    const proposedPos = currentPos
      .clone()
      .add(this.velocity.clone().multiplyScalar(deltaTime));

    // 3. 충돌 검사
    const collision = this.collisionSystem.checkCollisionAccurate(
      this.mesh,
      proposedPos
    );

    if (collision.collision) {
      // 4. 충돌 반응
      this.handleCollision(currentPos, proposedPos, collision);
    } else {
      // 5. 충돌 없음 → 이동
      this.mesh.position.copy(proposedPos);
    }
  }

  private handleCollision(
    currentPos: THREE.Vector3,
    proposedPos: THREE.Vector3,
    collision: CollisionResult
  ) {
    // Option 1: Stop (멈춤)
    this.stopResponse(currentPos);

    // Option 2: Bounce (튕김)
    // this.bounceResponse(currentPos, collision);

    // Option 3: Slide (미끄러짐)
    // this.slideResponse(currentPos, proposedPos, collision);
  }

  // Response 1: Stop
  private stopResponse(currentPos: THREE.Vector3) {
    // 현재 위치 유지
    this.mesh.position.copy(currentPos);

    // 속도 제거 (땅에 닿으면 멈춤)
    this.velocity.set(0, 0, 0);
  }

  // Response 2: Bounce (반사)
  private bounceResponse(
    currentPos: THREE.Vector3,
    collision: CollisionResult
  ) {
    this.mesh.position.copy(currentPos);

    // Collision normal 추정
    const normal = this.estimateCollisionNormal(currentPos);

    // 속도 반사
    const restitution = 0.8; // 반발 계수
    this.velocity.reflect(normal).multiplyScalar(restitution);
  }

  // Response 3: Slide (미끄러짐)
  private slideResponse(
    currentPos: THREE.Vector3,
    proposedPos: THREE.Vector3,
    collision: CollisionResult
  ) {
    const normal = this.estimateCollisionNormal(currentPos);

    // 이동 벡터
    const movement = proposedPos.clone().sub(currentPos);

    // Normal 방향 성분 제거
    const normalComponent = normal.clone().multiplyScalar(movement.dot(normal));
    const slideMovement = movement.clone().sub(normalComponent);

    // Slide된 위치로 이동
    const slidePos = currentPos.clone().add(slideMovement);
    this.mesh.position.copy(slidePos);

    // 속도도 slide 방향으로 조정
    const velNormalComponent = normal
      .clone()
      .multiplyScalar(this.velocity.dot(normal));
    this.velocity.sub(velNormalComponent);
  }

  // Collision normal 추정 (Depth map gradient)
  private estimateCollisionNormal(position: THREE.Vector3): THREE.Vector3 {
    const screenUV = this.collisionSystem["worldToScreen"](position);
    const x = Math.floor(screenUV.x * this.collisionSystem["depthMapWidth"]);
    const y = Math.floor(screenUV.y * this.collisionSystem["depthMapHeight"]);

    const depthMap = this.collisionSystem["depthMap"]!;
    const width = this.collisionSystem["depthMapWidth"];

    // Sobel-like gradient
    const depthCenter = depthMap[y * width + x];
    const depthLeft = depthMap[y * width + Math.max(0, x - 1)];
    const depthRight = depthMap[y * width + Math.min(width - 1, x + 1)];
    const depthUp = depthMap[Math.max(0, y - 1) * width + x];
    const depthDown =
      depthMap[
        Math.min(this.collisionSystem["depthMapHeight"] - 1, y + 1) * width + x
      ];

    const dx = depthRight - depthLeft;
    const dy = depthDown - depthUp;

    // Screen space normal → World space normal (근사)
    const screenNormal = new THREE.Vector3(-dx, -dy, 1.0);
    screenNormal.normalize();

    // Camera space에서 world space로 변환
    const worldNormal = screenNormal.applyQuaternion(
      this.collisionSystem["camera"].quaternion
    );

    return worldNormal;
  }
}
```

---

## 통합 시스템

### Main Update Loop

```typescript
class HybridRenderingSystem {
  threeRenderer: THREE.WebGLRenderer;
  gaussianRenderer: GaussianRenderer;
  camera: THREE.Camera;

  physicsMeshes: PhysicsMesh[] = [];
  collisionSystem: MeshGaussianCollision;

  constructor() {
    this.collisionSystem = new MeshGaussianCollision(this.camera);
  }

  // 매 프레임
  update(deltaTime: number) {
    // 1. Gaussian rendering + Depth map 수신
    const { colorFrame, depthMap } = this.gaussianRenderer.render(this.camera);

    // 2. Collision system에 depth map 업데이트
    this.collisionSystem.updateDepthMap(
      depthMap,
      this.gaussianRenderer.width,
      this.gaussianRenderer.height
    );

    // 3. Three.js 물리 객체 업데이트 (충돌 포함)
    for (const physicsMesh of this.physicsMeshes) {
      physicsMesh.update(deltaTime);
    }

    // 4. Three.js 렌더링
    this.threeRenderer.render(this.scene, this.camera);

    // 5. Composite (Gaussian + Three.js)
    this.composite(colorFrame);
  }

  // Mesh 추가 (물리 활성화)
  addPhysicsMesh(mesh: THREE.Mesh) {
    const physicsMesh = new PhysicsMesh(mesh, this.collisionSystem);
    this.physicsMeshes.push(physicsMesh);
    this.scene.add(mesh);
  }
}
```

---

## 구현 계획

### Phase 1: 기본 Collision Detection

- [ ] `MeshGaussianCollision` 클래스 구현
- [ ] World → Screen projection 구현
- [ ] Single point collision check
- [ ] 기본 테스트 (구체를 바닥에 떨어뜨리기)

### Phase 2: Accurate Collision (Bounding Volume)

- [ ] Bounding sphere 기반 multi-point sampling
- [ ] Collision normal 추정 (depth gradient)
- [ ] 정확도 테스트

### Phase 3: Physics Response

- [ ] Stop response 구현
- [ ] Bounce response 구현 (반발 계수)
- [ ] Slide response 구현 (미끄러짐)
- [ ] 중력 + 충돌 통합 테스트

### Phase 4: 최적화

- [ ] Collision check 빈도 최적화 (필요한 mesh만)
- [ ] Depth map bilinear interpolation
- [ ] Spatial hashing (많은 mesh 처리 시)

### Phase 5: 고급 기능

- [ ] Continuous collision detection (빠른 물체)
- [ ] Mesh-Mesh collision (Three.js 간)
- [ ] Collision visualization (디버그 모드)

---

## 성능 목표

- **Collision Check (single mesh)**: < 0.5ms
- **Collision Check (10 meshes)**: < 5ms
- **Frame Rate 영향**: < 10% (목표: 60 FPS 유지)
- **Depth Map 해상도**: 640×480 (현재)

---

## 기술적 고려사항

### 1. Depth Map Latency

**문제**: Depth map은 이전 프레임의 결과 (네트워크 레이턴시)

**영향**:

- 빠르게 움직이는 물체의 경우 부정확할 수 있음
- 1-2 프레임 지연 (~16-33ms @ 60 FPS)

**해결책**:

1. **Conservative Collision**: EPSILON 값을 크게 (예: 5cm)
2. **Predictive Depth**: 카메라 움직임 기반 depth 예측 (미래 작업)
3. **Sub-stepping**: 물리 시뮬레이션을 더 작은 단위로 나눔

### 2. Depth Map 해상도

**현재**: 640×480

**트레이드오프**:

- 높은 해상도: 정확한 collision, 높은 bandwidth
- 낮은 해상도: 빠른 처리, 부정확한 collision

**대응**:

- 640×480은 대부분의 경우 충분
- 작은 물체 (<10cm)는 부정확할 수 있음
- Bilinear interpolation으로 보간

### 3. Screen Space Limitation

**문제**: 화면에 보이지 않는 영역은 depth 정보 없음

**영향**:

- 화면 밖 물체는 collision check 불가
- 카메라 뒤쪽 물체는 projection 불가

**해결책**:

1. **Fallback**: 화면 밖 물체는 collision 무시 또는 예측
2. **360° Depth Map**: 구형 depth map 사용 (미래 작업)
3. **Mesh-Mesh Collision**: Three.js 간 충돌로 보완

### 4. Gaussian Depth Accuracy

**3D Gaussian Splatting Depth 특성**:

- Alpha blending으로 계산된 depth
- 반투명 영역에서 부정확
- Edge 영역에서 noise 가능

**대응**:

- Median filtering (옵션)
- Conservative threshold
- Multi-point sampling으로 노이즈 완화

### 5. 회전하는 Mesh

**문제**: Bounding sphere는 회전 무시

**해결**:

- Oriented Bounding Box (OBB) 사용 (더 복잡)
- 또는 더 많은 샘플 포인트

---

## 사용 예시

### 예시 1: 바닥에 공 떨어뜨리기

```typescript
// 공 mesh 생성
const sphere = new THREE.Mesh(
  new THREE.SphereGeometry(0.1, 32, 32),
  new THREE.MeshStandardMaterial({ color: 0xff0000 })
);
sphere.position.set(0, 2, 0); // 2m 높이

// 물리 활성화
renderingSystem.addPhysicsMesh(sphere);

// → 중력으로 낙하
// → Gaussian 바닥과 충돌
// → 튕기거나 멈춤
```

### 예시 2: 캐릭터가 Gaussian 환경 걷기

```typescript
// 캐릭터 mesh
const character = new THREE.Mesh(...);
character.position.set(0, 0, 0);

const physicsMesh = new PhysicsMesh(character, collisionSystem);

// User input으로 이동
function onKeyPress(key: string) {
    const movement = new THREE.Vector3();
    if (key === 'w') movement.z = -0.1;
    if (key === 's') movement.z = 0.1;
    if (key === 'a') movement.x = -0.1;
    if (key === 'd') movement.x = 0.1;

    const proposedPos = character.position.clone().add(movement);

    // Collision check
    const collision = collisionSystem.checkCollision(character, proposedPos);

    if (!collision.collision) {
        character.position.copy(proposedPos);
    } else {
        // 벽에 막힘 → slide
        const adjustedPos = physicsMesh.slideResponse(...);
        character.position.copy(adjustedPos);
    }
}
```

---

## 참고 사항

### 관련 파일

**Frontend**:

- `frontend/src/main.ts`: Three.js scene 관리
- `frontend/src/decode-worker.ts`: Depth map decoding
- `frontend/src/collision-system.ts`: 새로 구현 필요 ⭐
- `frontend/src/physics-mesh.ts`: 새로 구현 필요 ⭐

**Backend**:

- `backend/renderer/encoders/jpeg.py`: Depth encoding
- `backend/renderer/utils/depth_utils.py`: Depth utilities

### 관련 프로토콜

- Video Payload: 56 bytes header + color + depth
- Depth format: float16, linear depth

### 테스트 시나리오

1. **낙하 테스트**: 공을 떨어뜨려 바닥과 충돌
2. **벽 충돌**: 물체를 벽으로 던지기
3. **경사면**: 물체가 경사면을 따라 미끄러짐
4. **튕김**: 물체가 바닥에서 튕김

---

## 다음 단계

1. **Prototype**: Single point collision (1일)
2. **Bounding Volume**: Multi-point sampling (1일)
3. **Physics Response**: Stop/Bounce/Slide (1-2일)
4. **통합 테스트**: 실제 Gaussian scene + Three.js mesh (1일)

**예상 소요 시간**: 4-5일

---

## 업데이트 로그

- **2025-10-23**: 문서 생성, Mesh-Gaussian Collision 설계 완료

---

## 현재 시스템 상태 (2025-01-29 Updated)

### ✅ Depth Map Pipeline 완료

**Backend → Frontend 실시간 전송 완료**:
- Gaussian splatting depth map 생성
- JPEG/H264 encoding으로 전송
- TextureManager에서 depth texture 관리
- 해상도: 1024×576, 60 FPS
- Latency: ~50ms

**Collision 구현 준비 완료**:
```typescript
// 현재 사용 가능한 데이터
const depthTexture = app.getTextureManager().getDepthTexture();
const camera = app.getCameraController().getCamera();
const scene = localScene; // THREE.Scene
```

### 🔥 decode-worker 리팩토링과 독립적

**Collision System**:
- Frontend의 기존 depth texture 사용
- decode-worker는 depth 제공만
- **서로 영향 없음**

**구현 순서**:
1. ⭐ **Physics/Collision 먼저** (2-3일)
2. decode-worker 리팩토링 나중에

### 📋 Immediate Next Steps

**Phase 1: Basic Collision (이번 주)**
```
frontend/src/physics/
├── MeshGaussianCollision.ts
├── CollisionResponse.ts
└── types.ts

frontend/src/systems/
└── PhysicsSystem.ts
```

**Test**: 공을 Gaussian 바닥에 떨어뜨려서 충돌 확인

**Feed-forward Renderer**: 미래 작업 (collision과 독립적)

---

## ✅ 실제 구현 내역 (2025-10-29)

### 파일 구조

```
backend/transport/
├── protocol_converter.py       # ✅ projection matrix에서 near/far 추출

frontend/src/
├── physics/
│   ├── MeshGaussianCollision.ts  # ✅ Collision detection
│   └── depth-utils.ts            # (미사용 - TextureManager에 통합)
│
├── systems/
│   ├── TextureManager.ts        # ✅ sampleLinearDepth() 추가
│   └── PhysicsSystem.ts         # ✅ Physics simulation + response
│
├── core/
│   └── Application.ts           # ✅ Physics API 통합
│
└── main.ts                      # ✅ 사용 예제
```

### 1. Backend: Near/Far 동기화

**파일**: `backend/transport/protocol_converter.py`

```python
# Frontend projection matrix (16 floats)에서 near/far 추출
projection_vals = vals[16:32]  # indices 16-31
A = projection_vals[10]  # -(far + near) / (far - near)
B = projection_vals[14]  # -2 * far * near / (far - near)

# Solve for near and far
near = B / (-2.0 * (A + 1.0))
far = B / (-2.0 * (A - 1.0))
```

Frontend가 projection matrix를 보내면 Backend가 자동으로 near/far를 계산합니다.

### 2. Frontend: Depth Sampling

**파일**: `frontend/src/systems/TextureManager.ts`

```typescript
/**
 * Sample linear depth value at pixel coordinates
 *
 * Backend encodes: d_norm = (log(depth/near) / log(far/near)) * 0.80823
 * Inverse: depth = near * exp((d_norm / 0.80823) * log(far/near))
 */
sampleLinearDepth(x: number, y: number, near: number, far: number): number | null {
  const data = this.depthTexture.image.data;
  const pixelIndex = Math.floor(y) * this.width + Math.floor(x);

  let normalized: number;
  if (this.isJpegMode) {
    // JPEG: Uint16Array (Float16)
    normalized = (data as Uint16Array)[pixelIndex] / 65535.0;
  } else {
    // H264: Uint8Array
    normalized = (data as Uint8Array)[pixelIndex] / 255.0;
  }

  // Log-normalized → Linear
  const scale = 0.80823;
  const logRatio = (normalized / scale) * Math.log(far / near);
  return near * Math.exp(logRatio);
}
```

Direct texture data 접근으로 추가 버퍼 없이 depth 샘플링이 가능합니다.

### 3. Collision Detection

**파일**: `frontend/src/physics/MeshGaussianCollision.ts`

```typescript
class MeshGaussianCollision {
  // Single point collision
  checkCollision(worldPos: THREE.Vector3): CollisionResult {
    const screenUV = this.worldToScreen(worldPos);
    const x = Math.floor(screenUV.x * width);
    const y = Math.floor(screenUV.y * height);

    const gaussianDepth = this.textureManager.sampleLinearDepth(x, y, near, far);
    const meshDepth = this.camera.position.distanceTo(worldPos);

    if (meshDepth >= gaussianDepth - epsilon) {
      return { collision: true, penetrationDepth: meshDepth - gaussianDepth };
    }
    return { collision: false };
  }

  // Mesh bounding box collision (8 corners)
  checkMeshCollision(mesh: THREE.Mesh, proposedPosition: THREE.Vector3) {
    // Sample 8 corners of bounding box in world space
    // Return worst collision
  }
}
```

### 4. Physics System

**파일**: `frontend/src/systems/PhysicsSystem.ts`

```typescript
class PhysicsSystem implements System {
  update(deltaTime: number): void {
    for (const physicsMesh of this.physicsMeshes) {
      // 1. Apply physics
      physicsMesh.velocity.add(physicsMesh.acceleration * deltaTime);
      const proposedPos = currentPos + velocity * deltaTime;

      // 2. Check collision
      const collision = this.collisionDetector.checkMeshCollision(mesh, proposedPos);

      // 3. Handle collision
      if (collision.collision) {
        this.handleCollision(physicsMesh, collision); // stop/bounce/slide
      } else {
        physicsMesh.mesh.position.copy(proposedPos);
      }
    }
  }
}
```

### 5. Application API

**파일**: `frontend/src/core/Application.ts`

```typescript
class Application {
  // Add physics mesh
  addPhysicsMesh(mesh: THREE.Mesh, options?: {
    velocity?: THREE.Vector3;
    acceleration?: THREE.Vector3;
    mass?: number;
    restitution?: number;
    friction?: number;
  }): PhysicsMesh | null

  // Remove physics mesh
  removePhysicsMesh(mesh: THREE.Mesh): boolean

  // Set collision response: "stop" | "bounce" | "slide"
  setPhysicsResponseType(type: string): void

  // Set gravity
  setGravity(gravity: THREE.Vector3): void

  // Set collision threshold (meters)
  setCollisionEpsilon(epsilon: number): void
}
```

### 6. 사용 예제

**파일**: `frontend/src/main.ts`

```typescript
// 1. Create a falling sphere
const sphere = new THREE.Mesh(
  new THREE.SphereGeometry(0.2, 32, 32),
  new THREE.MeshStandardMaterial({ color: 0xff0000 })
);
sphere.position.set(0, 3, -2);
localScene.add(sphere);

// 2. Add to physics
app.addPhysicsMesh(sphere, {
  velocity: new THREE.Vector3(0, 0, 0),
  acceleration: new THREE.Vector3(0, -9.8, 0), // Gravity
  mass: 1.0,
  restitution: 0.7, // 70% bounce
  friction: 0.3     // 30% friction
});

// 3. Set collision response
app.setPhysicsResponseType("bounce");

// 4. Adjust threshold
app.setCollisionEpsilon(0.05); // 5cm
```

**테스트 활성화**: `main.ts:968` 라인의 `false`를 `true`로 변경

---

## 핵심 개선사항

1. **Near/Far 자동 동기화**: Frontend projection matrix → Backend 자동 추출
2. **Direct Texture 접근**: `depthTexture.image.data` 직접 사용
3. **Inline 변환**: Log→Linear 변환을 TextureManager에 inline 구현
4. **System 통합**: SystemContext.systems로 시스템 간 의존성 관리
5. **Clean API**: Application level에서 간단한 메서드로 접근

---

## 테스트 방법

1. Backend 시작 (Gaussian renderer + transport)
2. Frontend 시작
3. `main.ts:968` 라인: `false` → `true` 변경
4. 브라우저 새로고침
5. 빨간 공이 떨어지면서 Gaussian 표면과 충돌하는지 확인

---

## 다음 단계

- [ ] Collision normal 계산 개선 (depth gradient → surface normal)
- [ ] Multi-mesh collision optimization (spatial partitioning)
- [ ] Character controller 구현
- [ ] Feed-forward renderer 통합

