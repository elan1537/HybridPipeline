# Physics Engine 적용 여부 분석

## 조사 일자: 2025-10-23

---

## 1. Three.js 물리 엔진 옵션 (2025년 기준)

### 1.1 Rapier (추천 ⭐)

**특징**:
- Rust로 작성, WASM으로 배포
- 고성능 (SIMD 최적화)
- 활발히 유지보수 중
- 현대적 아키텍처

**성능**:
- Ammo.js 대비 4배 빠른 constraint solving
- AoSoA SIMD 최적화

**Three.js 통합**:
```bash
npm install @dimforge/rapier3d-compat
```

**커뮤니티 평가**: "2025년 현재 새 프로젝트에 가장 추천"

### 1.2 Cannon.js / cannon-es

**특징**:
- 경량, 사용하기 쉬움
- cannon-es는 유지보수되는 포크 (원본은 중단됨)

**Three.js 통합**:
```bash
npm install cannon-es
```

**제한사항**: 복잡한 mesh 충돌 지원 제한

### 1.3 Ammo.js

**특징**:
- Bullet Physics의 JavaScript 포트
- 많은 기능, 산업 표준

**단점**:
- 성능이 Rapier보다 낮음
- 무거움

---

## 2. 충돌 감지 방법론

### 2.1 물리 엔진 기반

**지원하는 충돌 형태**:
- ✅ Plane, Box, Sphere, Cylinder
- ✅ Convex Hull (단순한 형태)
- ❌ **Arbitrary Mesh** (복잡한 형태)
- ❌ **Depth Map 기반** (특수한 경우)

**장점**:
- 검증된 알고리즘
- 복잡한 물리 시뮬레이션 (관절, 제약조건 등)
- 최적화된 성능
- Mesh-Mesh 충돌 자동 처리

**단점**:
- 특수한 형태 (Gaussian Splatting) 지원 불가
- 오버헤드 (단순한 경우)

### 2.2 커스텀 구현

**방법들**:
- AABB (Axis-Aligned Bounding Box) - 가장 빠름
- OBB (Oriented Bounding Box) - 회전 고려
- BVH (Bounding Volume Hierarchy) - 80% 충돌 체크 감소
- **Depth Map 기반** - 본 프로젝트의 경우 ⭐

**장점**:
- 완전한 제어
- 특수한 경우 최적화 가능
- 경량 (필요한 기능만)

**단점**:
- 직접 구현 필요
- 버그 가능성
- 복잡한 물리는 구현 어려움

---

## 3. 본 프로젝트의 특수성 분석

### 3.1 Hybrid Rendering 아키텍처

```
┌─────────────────────────────────────┐
│   Three.js Layer                    │
│   - Mesh 객체 (캐릭터, 아이템)       │  ← 동적, 물리 필요
│   - 일반적인 3D 객체                 │
└─────────────────────────────────────┘
              ↓ 충돌 필요!
┌─────────────────────────────────────┐
│   Gaussian Splatting Layer          │
│   - 실제 스캔된 환경                 │  ← 정적, Depth Map만 제공
│   - Depth Map으로만 표현됨           │
└─────────────────────────────────────┘
```

### 3.2 핵심 문제: Gaussian Scene과의 충돌

**Gaussian Splatting의 특성**:
- ❌ Traditional mesh가 아님
- ❌ Vertex, Face 정보 없음
- ✅ **Depth Map만 제공** (640×480, Float16)
- ✅ Screen space 기반

**물리 엔진의 한계**:
```javascript
// ❌ 물리 엔진으로 불가능
const gaussianBody = new RAPIER.RigidBody(depthMap); // 지원 안함!

// ✅ 커스텀 구현 필요
const collision = checkDepthMapCollision(meshPosition, depthMap);
```

**결론**: **Gaussian-Mesh 충돌은 커스텀 구현 필수**

### 3.3 필요한 물리 기능 분류

| 기능 | Gaussian과 충돌 | Mesh-Mesh 충돌 | 물리 엔진 필요? |
|------|-----------------|----------------|----------------|
| Depth Map 기반 충돌 | ✅ | ❌ | ❌ (커스텀) |
| Mesh-Mesh 충돌 | ❌ | ✅ | ✅ (선택적) |
| 중력 | ✅ | ✅ | ❌ (간단) |
| 속도/가속도 | ✅ | ✅ | ❌ (간단) |
| 충돌 반응 (Stop/Bounce/Slide) | ✅ | ✅ | ⚠️ (커스텀 가능) |
| 관절/제약조건 | ❌ | ❌ | ✅ (미래) |

---

## 4. 권장 아키텍처

### 4.1 Hybrid Approach (추천 ⭐)

```typescript
// 1. Gaussian-Mesh 충돌: 커스텀 구현 (필수)
class MeshGaussianCollision {
    checkCollision(mesh, depthMap) {
        // Depth map 기반 충돌 감지
    }
}

// 2. 기본 물리: 간단한 커스텀 구현
class SimplePhysics {
    velocity: Vector3;
    acceleration: Vector3;

    update(deltaTime) {
        velocity += acceleration * deltaTime;
        position += velocity * deltaTime;
    }
}

// 3. Mesh-Mesh 충돌: 물리 엔진 (선택적, 미래)
// 필요 시 Rapier 추가
```

### 4.2 구현 단계

**Phase 1: MVP (커스텀 구현)**
- ✅ Depth map 기반 충돌 감지
- ✅ 간단한 물리 (중력, 속도)
- ✅ 기본 충돌 반응 (Stop, Bounce)
- **예상 소요**: 2-3일
- **복잡도**: 중간
- **의존성**: 없음 (순수 Three.js)

**Phase 2: 고급 기능 (선택적)**
- ⚠️ Mesh-Mesh 충돌 (Rapier 도입)
- ⚠️ 복잡한 물리 시뮬레이션
- **예상 소요**: 추가 2-3일
- **복잡도**: 높음
- **의존성**: Rapier (~2MB WASM)

---

## 5. 최종 결정

### 5.1 Phase 1: 물리 엔진 **사용 안함** (커스텀 구현)

**이유**:
1. **Gaussian 충돌은 커스텀 필수** - 물리 엔진으로 불가능
2. **기본 물리는 간단** - 중력, 속도 정도는 직접 구현 가능
3. **번들 크기** - Rapier WASM 2MB 추가 불필요 (현재 단계)
4. **복잡도** - 물리 엔진 학습 시간 > 커스텀 구현 시간

**구현 범위**:
```typescript
// 1. Depth map 기반 충돌 (MeshGaussianCollision)
- World → Screen projection
- Depth sampling & comparison
- Multi-point collision (Bounding Sphere)
- Normal estimation

// 2. 간단한 물리 (PhysicsMesh)
- Velocity, Acceleration (Vector3)
- Gravity (constant)
- Simple integration (Euler method)

// 3. 충돌 반응
- Stop (velocity = 0)
- Bounce (velocity.reflect(normal))
- Slide (velocity -= normal * dot(velocity, normal))
```

### 5.2 Phase 2 (미래): Rapier 도입 검토

**조건**:
- Mesh-Mesh 충돌이 많아질 때
- 복잡한 물리 시뮬레이션 필요 시
- 로봇 관절, 제약조건 등

**통합 방법**:
```typescript
// Gaussian 충돌은 여전히 커스텀
const gaussianCollision = new MeshGaussianCollision(camera);

// Mesh-Mesh 충돌만 Rapier 사용
import RAPIER from '@dimforge/rapier3d-compat';
const world = new RAPIER.World({ x: 0, y: -9.81, z: 0 });

// Hybrid physics update
function update(deltaTime) {
    // 1. Rapier physics step (Mesh-Mesh)
    world.step();

    // 2. Custom Gaussian collision (Mesh-Gaussian)
    for (const mesh of meshes) {
        const collision = gaussianCollision.check(mesh);
        if (collision) handleCollision(mesh, collision);
    }
}
```

---

## 6. 벤치마크 (예상)

| 구현 방식 | 번들 크기 | 초기 구현 시간 | 성능 (60 FPS) | 유지보수 |
|-----------|-----------|----------------|---------------|----------|
| 커스텀 only | +5KB | 2-3일 | ✅ 충분 | 간단 |
| Rapier only | +2MB | 1-2일 | ✅ 최적 | 쉬움 |
| Hybrid (커스텀 + Rapier) | +2MB | 4-5일 | ✅ 최적 | 복잡 |

**현재 단계**: **커스텀 only** (MVP)

---

## 7. 참고 자료

### Three.js 물리 엔진
- [Rapier 공식 문서](https://rapier.rs/)
- [Three.js Journey - Physics](https://threejs-journey.com/lessons/physics)
- [cannon-es 문서](https://pmndrs.github.io/cannon-es/docs/)

### 커스텀 충돌 감지
- [MDN - 3D Collision Detection](https://developer.mozilla.org/en-US/docs/Games/Techniques/3D_collision_detection)
- [Three.js Tutorials - OBB Collision](https://sbcode.net/threejs/obb/)

### Depth Map 기반
- 본 프로젝트의 특수 케이스, 참고 자료 없음
- feature.collision.md 참조

---

## 결론

**Phase 1 (현재)**: 물리 엔진 사용 안함, 커스텀 구현
- Gaussian-Mesh 충돌: Depth map 기반 커스텀
- 기본 물리: 간단한 Vector3 연산
- 충돌 반응: 직접 구현

**Phase 2 (미래)**: Rapier 도입 검토
- Mesh-Mesh 충돌 필요 시
- 복잡한 물리 시뮬레이션 필요 시

**시작하기**: `collision-system.ts`, `physics-mesh.ts` 구현
