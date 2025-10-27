# E2E Integration Test Guide

## Phase 1 (MVP): Backend 설정 고정 통합 테스트

이 가이드는 Backend와 Frontend를 연결하여 전체 시스템을 테스트하는 방법을 설명합니다.

---

## 🎯 목표

- Backend (Renderer + Transport) ↔ Frontend (Browser) 연결 검증
- WebSocket 통신 확인
- 실시간 렌더링 동작 확인

---

## ⚙️ 사전 준비

### 1. Docker 컨테이너 실행 확인

```bash
docker ps --format "table {{.Names}}\t{{.Image}}\t{{.Status}}"
```

**필수 컨테이너**:
- `compute-3dgstream` (3dgstream:latest) - Renderer 실행
- `transport-service` (python:3.11.13) - Transport 실행

**컨테이너가 없으면**:
```bash
cd /home/wrl-ubuntu/workspace/HybridPipeline
docker-compose up -d
```

### 2. Scene 파일 확인

사용 가능한 scene 파일 목록:
```bash
find data -name "*.ply" | head -5
```

**예시**:
- `data/livinglab-scene/sparse_pc.ply`
- `data/flame_steak/point_cloud/iteration_14000/point_cloud.ply`

---

## 🚀 실행 방법

### Step 1: Backend Services 시작

```bash
cd /home/wrl-ubuntu/workspace/HybridPipeline/backend
./run_e2e_test.sh
```

**환경변수로 설정 변경** (선택사항):
```bash
# H.264 인코더 (기본값)
ENCODER_TYPE=h264 ./run_e2e_test.sh

# JPEG 인코더
ENCODER_TYPE=jpeg ./run_e2e_test.sh

# 다른 Scene 사용
SCENE_PATH=/workspace/data/your-scene.ply ./run_e2e_test.sh

# 해상도 변경
WIDTH=1920 HEIGHT=1080 ./run_e2e_test.sh
```

**예상 출력**:
```
=======================================================================
HybridPipeline E2E Integration Test - Docker Execution
=======================================================================

Configuration:
  Renderer Container:  compute-3dgstream
  Transport Container: transport-service
  Scene:               /workspace/data/livinglab-scene/sparse_pc.ply
  Encoder:             h264
  Resolution:          1280x720

...

[1/2] Starting Renderer Service in compute-3dgstream...
      ✓ Renderer ready (camera socket created)

[2/2] Starting Transport Service in transport-service...
      WebSocket:  0.0.0.0:8765

Backend Services Running in Docker Containers
Press Ctrl+C to stop
-----------------------------------------------------------------------
[WebSocket] Starting server on 0.0.0.0:8765...
[WebSocket] Server started, waiting for connections...
```

### Step 2: Frontend 개발 서버 시작

**새 터미널 열기**:

```bash
cd /home/wrl-ubuntu/workspace/HybridPipeline/frontend

# 최초 1회만 (의존성 설치)
npm install

# 개발 서버 시작
npm run dev
```

**예상 출력**:
```
VITE v6.2.2  ready in 500 ms

➜  Local:   https://localhost:8001/
➜  Network: https://192.168.x.x:8001/
➜  press h + enter to show help
```

### Step 3: 브라우저에서 접속

1. **브라우저 열기**: https://localhost:8001
2. **인증서 경고 무시** (개발 환경):
   - Chrome: "고급" → "안전하지 않음(localhost)(으)로 이동"
   - Firefox: "고급" → "위험을 감수하고 계속"
3. **UI 확인**:
   - 왼쪽 상단에 FPS/레이턴시 정보 표시
   - "WS State: Connected" 확인

---

## ✅ 검증 체크리스트

### Backend 로그 확인

**Terminal 1 (Backend)**:
```
[WebSocket] Client connected: ('127.0.0.1', xxxxx), path=/ws/h264
[WebSocket] Handshake: resolution 1280x720
[WebSocket] Received camera frame 1 (time_index=0.000)
[Renderer] Rendering frame 1...
[Transport] Sending video frame 1 to frontend
```

### Frontend 브라우저 확인

**브라우저 콘솔 (F12)**:
```javascript
[DEBUG] [Main] Initializing scene
[DEBUG WORKER] Received message: {type: 'init', ...}
[DEBUG] [WebSocket] Connection from ...
[DEBUG] Frame 1 received: Color image 1280×720
```

**화면 UI**:
```
Decode FPS: 60.00
Render FPS: 60.00
WS State: Connected

Latency (ms)
Total: 45.2
Network: 12.3
Server: 28.5
Client: 4.4
```

### 실시간 렌더링 확인

- [ ] 3D Scene이 화면에 표시됨
- [ ] 마우스로 카메라 회전 가능 (OrbitControls)
- [ ] FPS가 30fps 이상 유지
- [ ] 레이턴시가 100ms 이하

---

## 🐛 문제 해결

### 1. WebSocket 연결 실패

**증상**: `WS State: Error` 또는 `WS State: Closed`

**원인**:
- Backend Transport가 실행되지 않음
- 포트 8765가 이미 사용 중

**해결**:
```bash
# Transport 프로세스 확인
docker exec transport-service ps aux | grep transport

# 포트 사용 확인
netstat -tlnp | grep 8765

# 프로세스 재시작
docker exec transport-service pkill -f transport.main
./run_e2e_test.sh
```

### 2. Renderer 초기화 실패

**증상**: Backend 로그에 "Failed to load scene" 에러

**원인**:
- Scene 파일 경로 오류
- 컨테이너 볼륨 마운트 문제

**해결**:
```bash
# Scene 파일 존재 확인 (컨테이너 내부)
docker exec compute-3dgstream ls -l /workspace/data/livinglab-scene/sparse_pc.ply

# 없으면 호스트에서 확인
ls -l data/livinglab-scene/sparse_pc.ply

# 볼륨 마운트 확인
docker inspect compute-3dgstream | grep -A 10 Mounts
```

### 3. 프레임 수신되지 않음

**증상**: Backend는 정상이지만 Frontend에 화면 표시 안 됨

**원인**:
- 인코더 타입 불일치
- Depth 데이터 포맷 오류

**해결**:
```bash
# Backend 로그에서 인코더 확인
# "Encoder Type: h264" 또는 "Encoder Type: jpeg"

# Frontend 브라우저 콘솔 확인
# Depth array 크기 확인

# Debug 출력 활성화
# Frontend: UI에서 "Show Console Debug" 체크
```

### 4. HTTPS 인증서 문제

**증상**: 브라우저에서 https://localhost:8001 접속 불가

**원인**:
- Self-signed certificate 생성 실패

**해결**:
```bash
# package.json에서 basic-ssl 플러그인 확인
grep "basic-ssl" frontend/package.json

# 없으면 설치
cd frontend
npm install @vitejs/plugin-basic-ssl --save

# 개발 서버 재시작
npm run dev
```

---

## 📊 성능 측정

### FPS 테스트 (60초)

1. Frontend UI에서 **"Start FPS Test (60s)"** 클릭
2. 60초 대기
3. 결과 확인:
   - Pure Decode FPS
   - Frame Processing FPS
   - Render FPS
   - Average Latency
4. **"Download Results"** 클릭하여 결과 저장

### 수동 성능 확인

```bash
# Backend 로그 확인 (프레임 처리 시간)
docker logs compute-3dgstream --tail 100 | grep "Rendering frame"

# Transport 로그 확인 (네트워크 전송)
docker logs transport-service --tail 100 | grep "Sending video"
```

---

## 🔧 고급 설정

### 멀티 해상도 테스트

```bash
# 720p
WIDTH=1280 HEIGHT=720 ./run_e2e_test.sh

# 1080p
WIDTH=1920 HEIGHT=1080 ./run_e2e_test.sh

# 4K (GPU 메모리 주의)
WIDTH=3840 HEIGHT=2160 ./run_e2e_test.sh
```

### Debug 출력 저장

Backend 스크립트 실행 시 자동으로 저장됨:
- `backend/renderer/output/` - Renderer 렌더링 결과
- `backend/transport/input/` - Transport 수신 데이터

확인:
```bash
# Renderer 출력 확인
docker exec compute-3dgstream ls -l /workspace/backend/renderer/output/

# Transport 입력 확인
docker exec transport-service ls -l /workspace/backend/transport/input/
```

---

## 📝 제약사항 (Phase 1 MVP)

1. **인코더 타입 고정**:
   - Frontend UI에서 JPEG/H264 전환해도 Backend 설정 따름
   - Backend 시작 시 `ENCODER_TYPE` 환경변수로만 변경 가능

2. **단일 클라이언트**:
   - 여러 브라우저 동시 연결 시 마지막 연결만 유효
   - Production에서는 세션 관리 필요

3. **해상도 고정**:
   - Frontend window resize 시 재연결되지만 Backend 해상도는 고정
   - 동적 해상도 변경은 Phase 2에서 지원 예정

---

## 🚀 다음 단계 (Phase 2)

1. **경로 기반 라우팅**:
   - `/ws/h264`, `/ws/jpeg` 경로별로 다른 Renderer 연결
   - Frontend 인코더 선택 지원

2. **멀티 클라이언트 지원**:
   - 클라이언트별 세션 관리
   - 독립적인 카메라 제어

3. **동적 해상도 협상**:
   - Frontend 요청에 따라 Renderer 해상도 변경
   - Adaptive streaming

---

## 📞 문제 보고

테스트 중 문제 발생 시:
1. Backend 로그 수집: `docker logs compute-3dgstream > renderer.log`
2. Transport 로그 수집: `docker logs transport-service > transport.log`
3. Frontend 브라우저 콘솔 스크린샷
4. Issue에 로그 첨부

---

**작성일**: 2025-10-22
**버전**: Phase 1 (MVP)
**문서 위치**: `/home/wrl-ubuntu/workspace/HybridPipeline/E2E_TEST_GUIDE.md`
