import * as THREE from 'three'
import { OrbitControls } from 'three/addons'
import { object_setup } from './scene-setup';
import { SceneState } from './state/scene-state';
import { LatencyTracker, LatencyStats } from './latency-tracker';

import fusionVertexShader from './shaders/fusionVertexShader.vs?raw';
import fusionColorFragmentShader from './shaders/fusionColorShader.fs?raw';
import debugVertexShader from './shaders/debugVertexShader.vs?raw';
import debugFragmentShader from './shaders/debugColorShader.fs?raw';

const rtWidth = 1920 / 2;
const rtHeight = 1080 / 2;

let lastCameraUpdateTime = 0
const cameraUpdateInterval = 1 / 60


let camera: THREE.PerspectiveCamera
let renderer: THREE.WebGLRenderer
let controls: OrbitControls

let robot;
let localScene: THREE.Scene;
let localRenderTarget: THREE.WebGLRenderTarget;
let localDepthTexture: THREE.DepthTexture;

let fusionQuad: THREE.Mesh;
let fusionMaterial: THREE.ShaderMaterial;

let fusionScene: THREE.Scene;
let orthoCamera: THREE.OrthographicCamera;

let wsColorTexture: THREE.Texture
let wsDepthTexture: THREE.DataTexture

let debugScreen: THREE.Mesh;
let debugMaterial: THREE.ShaderMaterial;
let debugScene: THREE.Scene;
let debugCamera: THREE.OrthographicCamera;

const clock = new THREE.Clock();
// 레이턴시 추적기
const latencyTracker = new LatencyTracker();

const fpsDiv = document.getElementById('decode-fps') as HTMLDivElement;
const renderFpsDiv = document.getElementById('render-fps') as HTMLDivElement;
const jpegFallbackCheckbox = document.getElementById('jpeg-fallback-checkbox') as HTMLInputElement;
const wsConnectButton = document.getElementById('ws-connect-button') as HTMLInputElement;
const wsDisconnectButton = document.getElementById('ws-disconnect-button') as HTMLInputElement;
const wsStateConsoleText = document.getElementById('ws-state-console-text') as HTMLDivElement;

// 레이턴시 표시 UI 요소들
const latencyDiv = document.getElementById('latency-display') as HTMLDivElement;
const totalLatencyDiv = document.getElementById('total-latency') as HTMLDivElement;
const networkLatencyDiv = document.getElementById('network-latency') as HTMLDivElement;
const serverLatencyDiv = document.getElementById('server-latency') as HTMLDivElement;
const decodeLatencyDiv = document.getElementById('decode-latency') as HTMLDivElement;
const clockOffsetDiv = document.getElementById('clock-offset') as HTMLDivElement;

wsConnectButton.addEventListener('click', () => wsConnectButtonClick())
wsDisconnectButton.addEventListener('click', () => wsDisconnectButtonClick())

function recreateDepthTexture(isJpegMode: boolean) {
    wsDepthTexture.dispose()

    if (isJpegMode) {
        // JPEG mode: Float16 data in Uint16Array format
        wsDepthTexture = new THREE.DataTexture(new Uint16Array(rtWidth * rtHeight), rtWidth, rtHeight, THREE.RedFormat, THREE.HalfFloatType);
    } else {
        // H264 mode: 8-bit RGB ImageBitmap data
        wsDepthTexture = new THREE.DataTexture(new Uint8Array(rtWidth * rtHeight), rtWidth, rtHeight, THREE.RedFormat, THREE.UnsignedByteType);
    }

    // Update shader uniforms with new texture
    fusionMaterial.uniforms.wsDepthSampler.value = wsDepthTexture;
    debugMaterial.uniforms.wsDepthSampler.value = wsDepthTexture;

    console.log(`Recreated depth texture for ${isJpegMode ? 'JPEG' : 'H264'} mode`);
}
jpegFallbackCheckbox.addEventListener('click', () => jpegFallbackButtonClick())

let near = 0.3;
let far = 100
let fov = 80

let renderStart = 0;
let renderCnt = 0;

let canvas: HTMLCanvasElement;

interface CameraBuffer {
    position: Float32Array;
    target: Float32Array;
    intrinsics: Float32Array;
    projection: Float32Array;
}

let worker = new Worker(new URL("./decode-worker.ts", import.meta.url), { type: "module" })
let workerReady = false;

if (jpegFallbackCheckbox.checked) {
    worker.postMessage({
        type: 'init',
        width: rtWidth,
        height: rtHeight,
        wsURL: 'wss://' + location.host + '/ws/jpeg',
    })
} else {
    worker.postMessage({
        type: 'init',
        width: rtWidth,
        height: rtHeight,
        wsURL: 'wss://' + location.host + '/ws/h264'
    })
}

worker.onmessage = ({ data }) => {
    if (data.type === "ws-ready") {
        workerReady = true;
        wsStateConsoleText.textContent = "WS State: Connected"
        return;
    }

    if (data.type === "ws-error") {
        workerReady = false;
        wsStateConsoleText.textContent = "WS State: Error"
        return;
    }

    if (data.type === "ws-close") {
        workerReady = false;
        wsStateConsoleText.textContent = "WS State: Closed"
        return;
    }

    if (data.type === 'frame-receive') {
        // 서버로부터 프레임 수신 시점 기록
        latencyTracker.recordFrameReceive(data.frameId, data.serverTimestamps);
        return;
    }

    if (data.type === 'frame') {
        wsColorTexture.image = data.image;
        wsColorTexture.colorSpace = THREE.LinearSRGBColorSpace;

        wsDepthTexture.image.data = data.depth;
        wsColorTexture.needsUpdate = true;
        wsDepthTexture.needsUpdate = true;

        // 디코딩 완료 시점 기록
        if (data.frameId && data.decodeCompleteTime) {
            latencyTracker.recordDecodeComplete(data.frameId);
            
            // 다음 렌더 프레임에서 렌더링 완료를 기록
            requestAnimationFrame(() => {
                const stats = latencyTracker.recordRenderComplete(data.frameId);
                if (stats) {
                    // 개별 프레임 레이턴시 로깅 (디버깅용)
                    // console.log(`Frame ${data.frameId}: ${stats.totalLatency.toFixed(1)}ms total`);
                }
            });
        }
    }

    if (data.type === 'pong-received') {
        // 클럭 동기화 데이터 기록
        latencyTracker.recordClockSync(
            data.clientRequestTime,
            data.serverReceiveTime,
            data.serverSendTime
        );
    }

    if (data.type === 'fps') {
        fpsDiv.textContent = `Decode FPS: ${data.decode.toFixed(2)}`
    }

    if (data.type === 'error') {
        console.error("decode-worker error: ", data.error)
    }
}

function wsConnectButtonClick() {
    worker.postMessage({
        type: 'change',
        wsURL: 'wss://' + location.host + '/ws/h264'
    })
}

function wsDisconnectButtonClick() {
    console.log("wsDisconnectButtonClick")
    worker.postMessage({
        type: 'ws-close'
    })

    wsColorTexture.dispose();
    wsDepthTexture.dispose();

    wsColorTexture.needsUpdate = true;
    wsDepthTexture.needsUpdate = true;
}

function jpegFallbackButtonClick() {
    const isJpegMode = jpegFallbackCheckbox.checked;
    console.log(`Switching to ${isJpegMode ? 'JPEG' : 'H264'} mode`)

    // Recreate depth texture for the new mode
    recreateDepthTexture(isJpegMode);

    // Switch WebSocket connection
    const wsURL = isJpegMode ? 'wss://' + location.host + '/ws/jpeg' : 'wss://' + location.host + '/ws/h264';
    worker.postMessage({
        type: 'change',
        wsURL: wsURL
    })
}

async function initScene() {
    console.log("initScene")
    camera = new THREE.PerspectiveCamera(fov, rtWidth / rtHeight, near, far);

    camera.position.copy(
        // new THREE.Vector3().fromArray([-2.9227930527270307, 0.7843796894035835, -1.1898402543170186])
        // new THREE.Vector3().fromArray([0.7843796894035835, 1.1898402543170186, 2.9227930527270307,])
        new THREE.Vector3().fromArray([1, 1.2, 2.9])
    );
    camera.lookAt(
        // new THREE.Vector3().fromArray([-0.7849031700643463, 0.5938976614459955, 0.5316901796392622])
        new THREE.Vector3().fromArray([0.5, 0.5, 0.5])
    );

    localScene = new THREE.Scene();
    SceneState.scene = localScene;

    renderer = new THREE.WebGLRenderer({
        antialias: true,
        depth: true,
        logarithmicDepthBuffer: true,
    });
    renderer.setPixelRatio(1);
    renderer.setSize(rtWidth, rtHeight);
    renderer.outputColorSpace = THREE.SRGBColorSpace;

    document.body.appendChild(renderer.domElement);

    controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.1;
    controls.autoRotate = true;
    controls.autoRotateSpeed = 0.5

    canvas = renderer.domElement as HTMLCanvasElement

    canvas.style.touchAction = 'none'
    canvas.style.cursor = 'grab'

    // robot_setup();
    object_setup();

    localDepthTexture = new THREE.DepthTexture(rtWidth, rtHeight);
    localDepthTexture.type = THREE.FloatType;

    localRenderTarget = new THREE.WebGLRenderTarget(rtWidth, rtHeight, {
        depthBuffer: true,
        stencilBuffer: false,
        depthTexture: localDepthTexture,
        samples: 4,
        colorSpace: THREE.LinearSRGBColorSpace,
    });

    wsColorTexture = new THREE.Texture()
    wsColorTexture.minFilter = THREE.LinearFilter;
    wsColorTexture.magFilter = THREE.LinearFilter;
    wsColorTexture.colorSpace = THREE.LinearSRGBColorSpace;

    const initialIsJpegMode = jpegFallbackCheckbox.checked;
    if (initialIsJpegMode) {
        wsDepthTexture = new THREE.DataTexture(new Uint16Array(rtWidth * rtHeight), rtWidth, rtHeight, THREE.RedFormat, THREE.HalfFloatType);
    } else {
        wsDepthTexture = new THREE.DataTexture(new Uint8Array(rtWidth * rtHeight), rtWidth, rtHeight, THREE.RedFormat, THREE.UnsignedByteType);
    }

    debugMaterial = new THREE.ShaderMaterial({
        uniforms: {
            localColorSampler: { value: localRenderTarget.texture },
            localDepthSampler: { value: localDepthTexture },
            wsColorSampler: { value: wsColorTexture },
            wsDepthSampler: { value: wsDepthTexture },
            width: { value: rtWidth },
            height: { value: rtHeight }
        },
        vertexShader: debugVertexShader,
        fragmentShader: debugFragmentShader,
        depthTest: false,
        depthWrite: false,
        transparent: true,
    })

    fusionMaterial = new THREE.ShaderMaterial({
        uniforms: {
            localColorSampler: { value: localRenderTarget.texture },
            localDepthSampler: { value: localDepthTexture },
            wsColorSampler: { value: wsColorTexture },
            wsDepthSampler: { value: wsDepthTexture },
            flipX: { value: true }, // X축 flip 활성화
            contrast: { value: 0.82 }, // 대비 조정 (1.0보다 작게)
            brightness: { value: 1 }, // 밝기 조정
        },
        vertexShader: fusionVertexShader,
        fragmentShader: fusionColorFragmentShader,
        depthTest: false,
        depthWrite: false,
        // transparent: true,
    })

    fusionQuad = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), fusionMaterial);
    fusionScene = new THREE.Scene();
    fusionScene.add(fusionQuad);
    orthoCamera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);

    debugScreen = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), debugMaterial);
    debugScene = new THREE.Scene();
    debugScene.add(debugScreen);
    debugCamera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
}


function sendCameraSnapshot(tag?: string) {
    if (!workerReady) return;

    // 프레임 ID 생성 및 레이턴시 추적 시작
    const frameId = latencyTracker.generateFrameId();
    latencyTracker.recordCameraSend(frameId);

    const projectionMatrix = camera.projectionMatrix.clone().toArray()
    const intrinsics = getCameraIntrinsics(camera, rtWidth, rtHeight);
    const camBuf: CameraBuffer = {
        position: new Float32Array(camera.position.toArray()),
        target: new Float32Array(controls.target.toArray()),
        intrinsics: new Float32Array(intrinsics),
        projection: new Float32Array(projectionMatrix)
    };

    worker.postMessage({
        type: 'camera',
        position: camBuf.position.buffer,
        target: camBuf.target.buffer,
        intrinsics: camBuf.intrinsics.buffer,
        projection: camBuf.projection.buffer,
        frameId,
        tag // optional debug
    }, [
        camBuf.position.buffer,
        camBuf.target.buffer,
        camBuf.intrinsics.buffer,
        camBuf.projection.buffer
    ]);

    // 주기적으로 ping 전송 (클럭 동기화)
    if (latencyTracker.shouldSendPing()) {
        const pingTime = performance.now();
        worker.postMessage({
            type: 'ping',
            clientTime: pingTime
        });
    }
}

function renderLoop() {
    requestAnimationFrame(renderLoop)
    controls.update(clock.getDelta());

    const now = performance.now()
    if (now - renderStart > 1000) {
        const fps = renderCnt / ((now - renderStart) / 1000)
        renderFpsDiv.textContent = `Render FPS: ${fps.toFixed(2)}`
        renderCnt = 0
        renderStart = now
    }

    if (now - lastCameraUpdateTime > cameraUpdateInterval) {
        sendCameraSnapshot('render');
        lastCameraUpdateTime = now;
    }
    // robot_animation();

    renderCnt++;

    // Local Scene 렌더링
    renderer.setRenderTarget(localRenderTarget)
    renderer.render(localScene, camera);

    renderer.setRenderTarget(null)
    // renderer.render(debugScene, debugCamera);
    renderer.render(fusionScene, orthoCamera);

    // 렌더링 완료 시점 기록 및 레이턴시 통계 업데이트
    updateLatencyStats();
}


initScene().then(() => {
    renderStart = performance.now()
    renderLoop()
})



function getCameraIntrinsics(camera: THREE.PerspectiveCamera, renderWidth: number, renderHeight: number) {
    const fov = camera.fov;
    const aspect = camera.aspect;
    const fovRad = (fov * Math.PI) / 180;

    // const fy = renderHeight / (2 * Math.tan(fovRad / 2));
    // const fx = fy * aspect;

    const projmat = camera.projectionMatrix;
    const fx = (renderWidth / 2) * projmat.elements[0];
    const fy = (renderHeight / 2) * projmat.elements[5];
    const cx = renderWidth / 2;
    const cy = renderHeight / 2;

    return [fx, 0, cx, 0, fy, cy, 0, 0, 1]
}

let lastStatsUpdate = 0;
const statsUpdateInterval = 500; // 0.5초마다 통계 업데이트

function updateLatencyStats() {
    const now = performance.now();
    if (now - lastStatsUpdate < statsUpdateInterval) return;
    
    lastStatsUpdate = now;
    
    const stats = latencyTracker.getRecentStats(50);
    
    // UI 업데이트 (요소가 존재하는 경우에만)
    if (totalLatencyDiv && stats.avg.totalLatency !== undefined) {
        totalLatencyDiv.textContent = `Total: ${stats.avg.totalLatency.toFixed(1)}ms (${stats.p95.totalLatency?.toFixed(1)}ms p95)`;
    }
    
    if (networkLatencyDiv && stats.avg.networkUploadTime !== undefined && stats.avg.networkDownloadTime !== undefined) {
        const totalNetwork = stats.avg.networkUploadTime + stats.avg.networkDownloadTime;
        networkLatencyDiv.textContent = `Network: ${totalNetwork.toFixed(1)}ms`;
    }
    
    if (serverLatencyDiv && stats.avg.serverProcessingTime !== undefined) {
        serverLatencyDiv.textContent = `Server: ${stats.avg.serverProcessingTime.toFixed(1)}ms`;
    }
    
    if (decodeLatencyDiv && stats.avg.clientDecodeTime !== undefined) {
        const totalClient = stats.avg.clientDecodeTime + (stats.avg.clientRenderTime || 0);
        decodeLatencyDiv.textContent = `Client: ${totalClient.toFixed(1)}ms`;
    }
    
    // 클럭 오프셋 표시
    if (clockOffsetDiv) {
        const offset = latencyTracker.getClockOffset();
        clockOffsetDiv.textContent = `Clock offset: ${offset.toFixed(1)}ms`;
    }

    // 콘솔에 상세 정보 출력 (개발용)
    if (stats.avg.totalLatency !== undefined) {
        console.log(`Latency Stats - Total: ${stats.avg.totalLatency.toFixed(1)}ms, ` +
                   `Network: ${((stats.avg.networkUploadTime || 0) + (stats.avg.networkDownloadTime || 0)).toFixed(1)}ms, ` +
                   `Server: ${(stats.avg.serverProcessingTime || 0).toFixed(1)}ms, ` +
                   `Decode: ${(stats.avg.clientDecodeTime || 0).toFixed(1)}ms, ` +
                   `Render: ${(stats.avg.clientRenderTime || 0).toFixed(1)}ms, ` +
                   `Offset: ${latencyTracker.getClockOffset().toFixed(1)}ms`);
    }
}