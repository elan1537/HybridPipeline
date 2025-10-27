import * as THREE from 'three'
import { OrbitControls } from 'three/addons'
import { robot_setup, object_setup, robot_animation } from './scene-setup';
import { SceneState } from './state/scene-state';
import { LatencyTracker, LatencyStats, FPSMeasurementResult } from './latency-tracker';
import { uiController } from './ui-controller';
import { debug } from './debug-logger';

// CP4: New modular architecture (runs in parallel with legacy code)
import { Application } from './core/Application';
import { RenderMode as NewRenderMode } from './types';

// Feature flags
const USE_NEW_ARCHITECTURE = true; // CP4: Enable new architecture
const USE_NEW_TEXTURE_MANAGER = true; // CP7: Use TextureManager textures for rendering

import fusionVertexShader from './shaders/fusionVertexShader.vs?raw';
import fusionColorFragmentShader from './shaders/fusionColorShader.fs?raw';
import debugVertexShader from './shaders/debugVertexShader.vs?raw';
import debugFragmentShader from './shaders/debugColorShader.fs?raw';
import depthFusionFragmentShader from './shaders/depthFusionShader.fs?raw';

// Simple shader for displaying WebSocket color texture only (based on fusionColorShader)
const gaussianOnlyFragmentShader = `
  varying vec2 vUv;
  uniform sampler2D wsColorSampler;
  
  void main() {
    // Same UV flipping as in fusionColorShader
    vec2 wsUv = vec2(1.0 - vUv.x, 1.0 - vUv.y);
    vec4 wsColor = texture2D(wsColorSampler, wsUv);
    gl_FragColor = vec4(wsColor.rgb, 1.0);
  }
`;

// Window-based resolution management
let rescaleFactor = 0.8;
let rtWidth = Math.floor(window.innerWidth * rescaleFactor);
let rtHeight = Math.floor(window.innerHeight * rescaleFactor);

let lastCameraUpdateTime = 0
const cameraUpdateInterval = 1 / 60

// Adaptive camera update variables
let lastCameraPosition = new THREE.Vector3();
let lastCameraTarget = new THREE.Vector3();
const cameraPositionThreshold = 0.001; // Minimum movement to trigger update
const cameraTargetThreshold = 0.001;
// Pre-calculate squared thresholds to avoid sqrt in render loop
const cameraPositionThresholdSquared = cameraPositionThreshold * cameraPositionThreshold;
const cameraTargetThresholdSquared = cameraTargetThreshold * cameraTargetThreshold;


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

let gaussianOnlyQuad: THREE.Mesh;
let gaussianOnlyMaterial: THREE.ShaderMaterial;
let gaussianOnlyScene: THREE.Scene;
let gaussianOnlyCamera: THREE.OrthographicCamera;

let depthFusionQuad: THREE.Mesh;
let depthFusionMaterial: THREE.ShaderMaterial;
let depthFusionScene: THREE.Scene;
let depthFusionCamera: THREE.OrthographicCamera;

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
const totalLatencyDiv = document.getElementById('total-latency') as HTMLDivElement;
const networkLatencyDiv = document.getElementById('network-latency') as HTMLDivElement;
const serverLatencyDiv = document.getElementById('server-latency') as HTMLDivElement;
const decodeLatencyDiv = document.getElementById('decode-latency') as HTMLDivElement;
const clockOffsetDiv = document.getElementById('clock-offset') as HTMLDivElement;

// UI 요소들
const depthDebugCheckbox = document.getElementById('depth-debug-checkbox') as HTMLInputElement;
const consoleDebugCheckbox = document.getElementById('console-debug-checkbox') as HTMLInputElement;
const cameraDebugCheckbox = document.getElementById('camera-debug-checkbox') as HTMLInputElement;

// FPS 측정 도구 UI 요소들
const fpsTestButton = document.getElementById('fps-measurement-button') as HTMLInputElement;
const fpsTestProgress = document.getElementById('fps-measurement-progress') as HTMLDivElement;
const fpsTestCurrent = document.getElementById('fps-measurement-current') as HTMLDivElement;
const fpsTestResult = document.getElementById('fps-measurement-result') as HTMLDivElement;
const fpsResultDownload = document.getElementById('fps-result-download') as HTMLInputElement;

// 화면 녹화 UI 요소들
const recordingButton = document.getElementById('recording-button') as HTMLInputElement;
const recordingStatus = document.getElementById('recording-status') as HTMLDivElement;
const recordingTime = document.getElementById('recording-time') as HTMLDivElement;
const recordingMode = document.getElementById('recording-mode') as HTMLDivElement;
const recordingSize = document.getElementById('recording-size') as HTMLDivElement;
const recordingDownload = document.getElementById('recording-download') as HTMLInputElement;
const recordingCompatibility = document.getElementById('recording-compatibility') as HTMLDivElement;

// 카메라 정보 UI 요소들
const cameraInfoSection = document.getElementById('camera-info-section') as HTMLDivElement;
const cameraPositionDiv = document.getElementById('camera-position') as HTMLDivElement;
const cameraTargetDiv = document.getElementById('camera-target') as HTMLDivElement;
const saveCameraButton = document.getElementById('save-camera-button') as HTMLInputElement;
const loadCameraButton = document.getElementById('load-camera-button') as HTMLInputElement;

// Manual camera control UI elements
const cameraPosXInput = document.getElementById('camera-pos-x') as HTMLInputElement;
const cameraPosYInput = document.getElementById('camera-pos-y') as HTMLInputElement;
const cameraPosZInput = document.getElementById('camera-pos-z') as HTMLInputElement;
const cameraTarXInput = document.getElementById('camera-tar-x') as HTMLInputElement;
const cameraTarYInput = document.getElementById('camera-tar-y') as HTMLInputElement;
const cameraTarZInput = document.getElementById('camera-tar-z') as HTMLInputElement;
const applyCameraButton = document.getElementById('apply-camera-button') as HTMLInputElement;

// Window size display UI element
const windowSizeDisplay = document.getElementById('window-size-display') as HTMLDivElement;

// Render mode radio buttons
const fusionModeRadio = document.getElementById('fusion-mode') as HTMLInputElement;
const gaussianOnlyModeRadio = document.getElementById('gaussian-only-mode') as HTMLInputElement;
const localOnlyModeRadio = document.getElementById('local-only-mode') as HTMLInputElement;
const depthFusionModeRadio = document.getElementById('depth-fusion-mode') as HTMLInputElement;
const feedForwardModeRadio = document.getElementById('feed-forward-mode') as HTMLInputElement;

// Render mode constants
enum RenderMode {
    FUSION = 'fusion',
    GAUSSIAN_ONLY = 'gaussian',
    LOCAL_ONLY = 'local',
    DEPTH_FUSION = 'depth-fusion',
    FEED_FORWARD = 'feed-forward'
}

let currentRenderMode: RenderMode = RenderMode.FUSION;
let currentTimeIndex: number = 0.0;
let frameCounter: number = 0; // Integer frame counter for 4DGS (0-299)
let isPlaying = true;

// 화면 녹화 관련 변수들
let mediaRecorder: MediaRecorder | null = null;
let recordedChunks: Blob[] = [];
let recordingStartTime = 0;
let recordingTimer: number | null = null;
let recordingStream: MediaStream | null = null;
let isRecordingSupported = false;
let recordingBlob: Blob | null = null;

// Texture update tracking
let wsColorTextureNeedsUpdate = false;
let wsDepthTextureNeedsUpdate = false;

// Performance tracking (only enabled in development)
const ENABLE_PERFORMANCE_TRACKING = false; // Set to true for debugging
let textureUpdateCount = 0;
let cameraUpdateCount = 0;
let renderTargetSwitchCount = 0;
let lastPerformanceLogTime = 0;
const performanceLogInterval = 10000; // Log every 10 seconds

// 독립적인 프레임 처리 성능 측정
let frameProcessingCount = 0;
let frameProcessingStart = performance.now();
let totalFrameProcessingTime = 0;
let frameProcessingHistory: number[] = [];
const maxFrameProcessingHistory = 100;
let currentFrameStartTime = 0;

// FPS 측정 중 샘플링 데이터 (60초 측정용)
let fpsMeasurementSamples: number[] = [];
let lastFpsMeasurementSampleTime = 0;
let mainThreadFpsMeasurementActive = false;

wsConnectButton.addEventListener('click', () => wsConnectButtonClick())
wsDisconnectButton.addEventListener('click', () => wsDisconnectButtonClick())

// Debug console toggle event listener
consoleDebugCheckbox.addEventListener('change', () => {
    const isEnabled = consoleDebugCheckbox.checked;
    debug.setDebugEnabled(isEnabled);
    debug.logMain(`Console debug logging ${isEnabled ? 'enabled' : 'disabled'}`);

    // Worker에게도 debug 상태 전송
    worker.postMessage({
        type: 'debug-toggle',
        enabled: isEnabled
    });
});


// Cookie utility functions
function setCookie(name: string, value: string, days: number = 30) {
    const expires = new Date();
    expires.setTime(expires.getTime() + (days * 24 * 60 * 60 * 1000));
    document.cookie = `${name}=${value};expires=${expires.toUTCString()};path=/`;
}

function getCookie(name: string): string | null {
    const nameEQ = name + "=";
    const ca = document.cookie.split(';');
    for (let i = 0; i < ca.length; i++) {
        let c = ca[i];
        while (c.charAt(0) === ' ') c = c.substring(1, c.length);
        if (c.indexOf(nameEQ) === 0) return c.substring(nameEQ.length, c.length);
    }
    return null;
}

// Camera position save/load functions
function saveCameraPosition() {
    const cameraData = {
        position: {
            x: camera.position.x,
            y: camera.position.y,
            z: camera.position.z
        },
        target: {
            x: controls.target.x,
            y: controls.target.y,
            z: controls.target.z
        }
    };

    setCookie('hybridpipeline_camera', JSON.stringify(cameraData));
    debug.logMain(`Camera position saved: ${JSON.stringify(cameraData)}`);
}

function loadCameraPosition(): boolean {
    const cookieData = getCookie('hybridpipeline_camera');
    if (!cookieData) {
        debug.logMain('No saved camera position found');
        return false;
    }

    try {
        const cameraData = JSON.parse(cookieData);

        // 카메라 위치 설정
        camera.position.set(cameraData.position.x, cameraData.position.y, cameraData.position.z);
        controls.target.set(cameraData.target.x, cameraData.target.y, cameraData.target.z);

        // OrbitControls 업데이트
        controls.update();

        debug.logMain(`Camera position loaded: ${JSON.stringify(cameraData)}`);
        return true;
    } catch (error) {
        debug.error('Failed to load camera position:', error);
        return false;
    }
}


// Render mode event handlers
fusionModeRadio.addEventListener('change', () => {
    if (fusionModeRadio.checked) {
        currentRenderMode = RenderMode.FUSION;
        debug.logMain('Switched to Fusion Mode');
    }
});

gaussianOnlyModeRadio.addEventListener('change', () => {
    if (gaussianOnlyModeRadio.checked) {
        currentRenderMode = RenderMode.GAUSSIAN_ONLY;
        debug.logMain('Switched to Gaussian Splatting Only Mode');
    }
});

localOnlyModeRadio.addEventListener('change', () => {
    if (localOnlyModeRadio.checked) {
        currentRenderMode = RenderMode.LOCAL_ONLY;
        debug.logMain('Switched to Local Rendering Only Mode');
    }
});

depthFusionModeRadio.addEventListener('change', () => {
    if (depthFusionModeRadio.checked) {
        currentRenderMode = RenderMode.DEPTH_FUSION;
        debug.logMain('Switched to Depth Fusion Mode');
    }
});

feedForwardModeRadio.addEventListener('change', () => {
    if (feedForwardModeRadio.checked) {
        currentRenderMode = RenderMode.FEED_FORWARD;
        debug.logMain('Switched to Feed Forward Mode');

        worker.postMessage({
            type: "ws-close",
        })

        setTimeout(() => {
            const wsURL = 'wss://' + location.host + '/ws/feedforward'

            worker.postMessage({
                type: 'change',
                wsURL: wsURL,
                width: rtWidth,
                height: rtHeight
            });

            debug.logMain(`[reconnectWithNewResolution] Reconnected with ${rtWidth}×${rtHeight}`);
        }, 100);
    }
});

function recreateDepthTexture(isJpegMode: boolean) {
    if (wsDepthTexture) {
        debug.logMain(`[recreateDepthTexture] Disposing old depth texture: ${wsDepthTexture.image.width}×${wsDepthTexture.image.height}`);
        wsDepthTexture.dispose();
    }

    debug.logMain(`[recreateDepthTexture] Creating new texture for ${isJpegMode ? 'JPEG' : 'H264'} mode: ${rtWidth}×${rtHeight}`);

    if (isJpegMode) {
        // JPEG mode: Float16 data in Uint16Array format
        const depthArray = new Uint16Array(rtWidth * rtHeight);
        wsDepthTexture = new THREE.DataTexture(depthArray, rtWidth, rtHeight, THREE.RedFormat, THREE.HalfFloatType);
        debug.logMain(`[recreateDepthTexture] JPEG mode: Created Uint16Array of size ${depthArray.length}`);
    } else {
        // H264 mode: Uint8Array grayscale data
        const depthArray = new Uint8Array(rtWidth * rtHeight);
        wsDepthTexture = new THREE.DataTexture(depthArray, rtWidth, rtHeight, THREE.RedFormat, THREE.UnsignedByteType);
        debug.logMain(`[recreateDepthTexture] H264 mode: Created Uint8Array of size ${depthArray.length}`);
    }

    // Update shader uniforms with new texture
    if (fusionMaterial) {
        fusionMaterial.uniforms.wsDepthSampler.value = wsDepthTexture;
    }
    if (debugMaterial) {
        debugMaterial.uniforms.wsDepthSampler.value = wsDepthTexture;
    }
    if (depthFusionMaterial) {
        depthFusionMaterial.uniforms.wsDepthSampler.value = wsDepthTexture;
    }

    debug.logMain(`[recreateDepthTexture] Depth texture recreated: ${wsDepthTexture.image.width}×${wsDepthTexture.image.height}`);
}
jpegFallbackCheckbox.addEventListener('click', () => jpegFallbackButtonClick())
depthDebugCheckbox.addEventListener('click', () => depthDebugButtonClick())
fpsTestButton.addEventListener('click', () => fpsTestButtonClick())
fpsResultDownload.addEventListener('click', () => downloadFPSResults())

// Camera save/load button event listeners
saveCameraButton.addEventListener('click', () => saveCameraPosition())
loadCameraButton.addEventListener('click', () => loadCameraPosition())

// Manual camera input functions
function updateCameraInputFields() {
    if (!cameraDebugCheckbox.checked) return;

    cameraPosXInput.value = camera.position.x.toFixed(3);
    cameraPosYInput.value = camera.position.y.toFixed(3);
    cameraPosZInput.value = camera.position.z.toFixed(3);
    cameraTarXInput.value = controls.target.x.toFixed(3);
    cameraTarYInput.value = controls.target.y.toFixed(3);
    cameraTarZInput.value = controls.target.z.toFixed(3);
}

function applyCameraFromInputs() {
    const posX = parseFloat(cameraPosXInput.value) || 0;
    const posY = parseFloat(cameraPosYInput.value) || 0;
    const posZ = parseFloat(cameraPosZInput.value) || 0;
    const tarX = parseFloat(cameraTarXInput.value) || 0;
    const tarY = parseFloat(cameraTarYInput.value) || 0;
    const tarZ = parseFloat(cameraTarZInput.value) || 0;

    camera.position.set(posX, posY, posZ);
    controls.target.set(tarX, tarY, tarZ);
    controls.update();

    debug.logMain(`Camera applied: pos(${posX.toFixed(3)}, ${posY.toFixed(3)}, ${posZ.toFixed(3)}), target(${tarX.toFixed(3)}, ${tarY.toFixed(3)}, ${tarZ.toFixed(3)})`);
}

// Apply camera button event listener
applyCameraButton.addEventListener('click', () => applyCameraFromInputs());

// Update input fields when camera moves (via manual input changes)
[cameraPosXInput, cameraPosYInput, cameraPosZInput, cameraTarXInput, cameraTarYInput, cameraTarZInput].forEach(input => {
    input.addEventListener('input', () => {
        // Auto-apply changes after a short delay
        clearTimeout(input.dataset.timeout as any);
        input.dataset.timeout = setTimeout(() => applyCameraFromInputs(), 500) as any;
    });

    // Apply immediately on Enter key
    input.addEventListener('keydown', (event) => {
        if (event.key === 'Enter') {
            clearTimeout(input.dataset.timeout as any);
            applyCameraFromInputs();
        }
    });
});

// 화면 녹화 이벤트 리스너
recordingButton.addEventListener('click', () => recordingButtonClick())
recordingDownload.addEventListener('click', () => downloadRecording())



// 모든 렌더 타겟과 텍스처 재생성 (해상도 변경 시)
function recreateRenderTargets() {
    debug.logMain(`[recreateRenderTargets] Recreating for resolution: ${rtWidth}×${rtHeight}`);

    // 기존 렌더 타겟들 정리
    if (localRenderTarget) {
        localRenderTarget.dispose();
    }
    if (localDepthTexture) {
        localDepthTexture.dispose();
    }

    // 새 해상도로 렌더 타겟 재생성
    localDepthTexture = new THREE.DepthTexture(rtWidth, rtHeight);
    localDepthTexture.type = THREE.FloatType;

    localRenderTarget = new THREE.WebGLRenderTarget(rtWidth, rtHeight, {
        depthBuffer: true,
        stencilBuffer: false,
        depthTexture: localDepthTexture,
        samples: 4,
    });

    // WebSocket 텍스처들 재생성
    recreateDepthTexture(jpegFallbackCheckbox.checked);

    // 모든 셰이더 유니폼 업데이트
    updateShaderUniforms();

    debug.logMain(`[recreateRenderTargets] Completed for ${rtWidth}×${rtHeight}`);
}

// 모든 셰이더 유니폼 업데이트
function updateShaderUniforms() {
    // Fusion material 업데이트
    if (fusionMaterial) {
        fusionMaterial.uniforms.localColorSampler.value = localRenderTarget.texture;
        fusionMaterial.uniforms.localDepthSampler.value = localDepthTexture;
        fusionMaterial.uniforms.wsDepthSampler.value = wsDepthTexture;
    }

    // Debug material 업데이트
    if (debugMaterial) {
        debugMaterial.uniforms.localColorSampler.value = localRenderTarget.texture;
        debugMaterial.uniforms.localDepthSampler.value = localDepthTexture;
        debugMaterial.uniforms.wsDepthSampler.value = wsDepthTexture;
        debugMaterial.uniforms.width.value = rtWidth;
        debugMaterial.uniforms.height.value = rtHeight;
    }

    // Depth fusion material 업데이트
    if (depthFusionMaterial) {
        depthFusionMaterial.uniforms.localColorSampler.value = localRenderTarget.texture;
        depthFusionMaterial.uniforms.localDepthSampler.value = localDepthTexture;
        depthFusionMaterial.uniforms.wsDepthSampler.value = wsDepthTexture;
    }
}

// 카메라 종단비를 윈도우 크기에 맞게 업데이트
function updateCameraAspectRatio() {
    if (!camera) return;

    const windowAspect = window.innerWidth / window.innerHeight;
    camera.aspect = windowAspect;
    camera.updateProjectionMatrix();

    debug.logMain(`[updateCameraAspectRatio] Updated to ${windowAspect.toFixed(3)} (${rtWidth}×${rtHeight})`);

    // Update UI displays
    updateSizeDisplays();
}

// UI 크기 정보 업데이트
function updateSizeDisplays() {
    if (windowSizeDisplay) {
        windowSizeDisplay.textContent = `Window: ${window.innerWidth}×${window.innerHeight} (RT: ${rtWidth}×${rtHeight})`;
    }
}

// 새로운 해상도로 WebSocket 재연결
function reconnectWithNewResolution() {
    debug.logMain('[reconnectWithNewResolution] Reconnecting with new resolution...');

    // 기존 연결 종료
    worker.postMessage({ type: 'ws-close' });

    // CP4.5: New architecture disconnection
    if (USE_NEW_ARCHITECTURE && app) {
        app.disconnectWebSocket();
    }

    // 잠시 대기 후 새 해상도로 재연결
    setTimeout(() => {
        const wsURL = jpegFallbackCheckbox.checked ?
            'wss://' + location.host + '/ws/jpeg' :
            'wss://' + location.host + '/ws/h264';

        // Legacy worker reconnection
        worker.postMessage({
            type: 'change',
            wsURL: wsURL,
            width: rtWidth,
            height: rtHeight
        });

        // CP4.5: New architecture reconnection
        if (USE_NEW_ARCHITECTURE && app) {
            debug.logMain(`[CP4.5] Reconnecting new architecture with ${rtWidth}×${rtHeight}`);
            const wsSystem = app.getWebSocketSystem();
            if (wsSystem) {
                wsSystem.reconnect(wsURL, rtWidth, rtHeight);
            }
        }

        debug.logMain(`[reconnectWithNewResolution] Reconnected with ${rtWidth}×${rtHeight}`);
    }, 100);
}

// 기존 로컬 렌더 타겟 재생성 (호환성 유지)
function recreateLocalRenderTarget() {
    if (localRenderTarget) {
        localRenderTarget.dispose();
    }
    if (localDepthTexture) {
        localDepthTexture.dispose();
    }

    localDepthTexture = new THREE.DepthTexture(rtWidth, rtHeight);
    localDepthTexture.type = THREE.FloatType;

    localRenderTarget = new THREE.WebGLRenderTarget(rtWidth, rtHeight, {
        depthBuffer: true,
        stencilBuffer: false,
        depthTexture: localDepthTexture,
        samples: 4,
    });

    // 셰이더 유니폼 업데이트
    if (fusionMaterial) {
        fusionMaterial.uniforms.localColorSampler.value = localRenderTarget.texture;
        fusionMaterial.uniforms.localDepthSampler.value = localDepthTexture;
    }
    if (debugMaterial) {
        debugMaterial.uniforms.localColorSampler.value = localRenderTarget.texture;
        debugMaterial.uniforms.localDepthSampler.value = localDepthTexture;
    }
    if (depthFusionMaterial) {
        depthFusionMaterial.uniforms.localColorSampler.value = localRenderTarget.texture;
        depthFusionMaterial.uniforms.localDepthSampler.value = localDepthTexture;
    }
}



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

// Worker error handler
worker.onerror = (error) => {
    console.error('[Worker ERROR]', error);
    debug.error('[Worker] Error:', error.message);
};

// 워커 초기화는 initScene에서 수행하도록 변경

worker.onmessage = ({ data }) => {
    // Debug: Log all worker messages
    if (data.type !== 'frame-receive' && data.type !== 'pure-decode-stats') {
        console.log('[main.ts] Worker message:', data.type);
    }

    // CP4.5: Forward messages to new architecture
    if (USE_NEW_ARCHITECTURE && app) {
        const wsSystem = app.getWebSocketSystem();
        if (wsSystem) {
            wsSystem.handleMessage(data);
        }
    }

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

    if (data.type === 'pure-decode-stats') {
        // 순수 디코딩 성능 통계 처리
        debug.logMain(`[Pure Decode] FPS: ${data.pureFPS.toFixed(2)}, ` +
            `Avg Time: ${data.avgDecodeTime.toFixed(2)}ms, ` +
            `Range: ${data.minDecodeTime.toFixed(2)}-${data.maxDecodeTime.toFixed(2)}ms, ` +
            `Recent Avg: ${data.recentAvg.toFixed(2)}ms`);

        // UI 업데이트 (기존 decode FPS 대신 순수 decode FPS 표시)
        fpsDiv.textContent = `Decode FPS: ${data.pureFPS.toFixed(2)} (Pure: ${data.avgDecodeTime.toFixed(1)}ms avg)`;

        // FPS 측정 중이면 샘플 데이터로 Latency Tracker 업데이트
        if (data.fpsMeasurementData && mainThreadFpsMeasurementActive) {
            latencyTracker.recordPureDecodeFPSSample(data.fpsMeasurementData.totalCount, data.fpsMeasurementData.avgTime);
            debug.logFPS(`Recording decode sample: ${data.fpsMeasurementData.totalCount} frames, ${data.fpsMeasurementData.avgTime.toFixed(2)}ms avg`);
        } else {
            // 평상시에는 1초 단위 데이터 사용 (기존 방식 유지)
            latencyTracker.recordPureDecodeFPS(data.totalFrames, data.avgDecodeTime);
        }
        return;
    }

    if (data.type === 'frame-receive') {
        // 서버로부터 프레임 수신 시점 기록
        latencyTracker.recordFrameReceive(data.frameId, data.serverTimestamps);
        return;
    }

    if (data.type === 'frame') {
        wsColorTexture.image = data.image;
        wsColorTexture.colorSpace = THREE.SRGBColorSpace;

        // Depth 데이터 상세 로깅
        const expectedSize = rtWidth * rtHeight;

        debug.logMain(`Received frame ${data.frameId}: Color image ${data.image.width}×${data.image.height}`);

        if (data.depth instanceof Uint8Array) {
            // H264 모드 - depth는 Uint8Array (grayscale)
            debug.logMain(`H264 depth array length: ${data.depth.length}, Expected: ${expectedSize} (${rtWidth}×${rtHeight})`);

            if (data.depth.length !== expectedSize) {
                debug.warn(`H264 depth array size mismatch! Got ${data.depth.length}, expected ${expectedSize}`);
                debug.error(`wsDepthTexture size: ${wsDepthTexture.image.width}×${wsDepthTexture.image.height}`);

                // H264 모드도 JPEG와 동일하게 emergency recreation 수행
                debug.logMain(`Recreating H264 depth texture to match data size...`);

                // 실제 받은 데이터로부터 해상도 추정
                const actualPixels = data.depth.length;
                let actualWidth: number, actualHeight: number;

                // 표준 해상도들 중에서 픽셀 수가 일치하는 것 찾기
                const standardResolutions = [
                    { w: 854, h: 480 },   // 480p = 409,920
                    { w: 1280, h: 720 },  // 720p = 921,600  
                    { w: 1920, h: 1080 }  // 1080p = 2,073,600
                ];

                const matchedRes = standardResolutions.find(res => res.w * res.h === actualPixels);

                if (matchedRes) {
                    actualWidth = matchedRes.w;
                    actualHeight = matchedRes.h;
                    debug.logMain(`Matched standard resolution: ${actualWidth}×${actualHeight}`);
                } else {
                    // 표준 해상도가 아닌 경우 정사각형으로 가정
                    actualWidth = Math.sqrt(actualPixels);
                    actualHeight = actualWidth;

                    if (actualWidth !== Math.floor(actualWidth)) {
                        debug.error(`Cannot determine resolution for ${actualPixels} pixels`);
                        return; // 처리할 수 없는 경우 스킵
                    }

                    actualWidth = Math.floor(actualWidth);
                    actualHeight = actualWidth;
                    debug.logMain(`Using square resolution: ${actualWidth}×${actualHeight}`);
                }

                // 기존 텍스처 정리 및 새 텍스처 생성
                wsDepthTexture.dispose();
                wsDepthTexture = new THREE.DataTexture(data.depth, actualWidth, actualHeight, THREE.RedFormat, THREE.UnsignedByteType);

                // 셰이더 유니폼 업데이트
                if (fusionMaterial) {
                    fusionMaterial.uniforms.wsDepthSampler.value = wsDepthTexture;
                }
                if (debugMaterial) {
                    debugMaterial.uniforms.wsDepthSampler.value = wsDepthTexture;
                }

                debug.logMain(`Emergency recreated H264 depth texture: ${actualWidth}×${actualHeight}`);

            } else {
                wsDepthTexture.image.data = data.depth;
            }
        } else if (data.depth instanceof Uint16Array) {
            // JPEG 모드 - depth는 Uint16Array  
            debug.logMain(`[Main] Depth array length: ${data.depth.length}, Expected: ${expectedSize}`);
            debug.logMain(`[Main] Current stream resolution: ${rtWidth}×${rtHeight}`);

            if (data.depth.length !== expectedSize) {
                debug.error(`[Main] JPEG depth array size mismatch! Got ${data.depth.length}, expected ${expectedSize}`);
                debug.error(`[Main] wsDepthTexture size: ${wsDepthTexture.image.width}×${wsDepthTexture.image.height}`);

                // 크기가 맞지 않으면 텍스처를 다시 생성
                debug.logMain(`[Main] Recreating JPEG depth texture to match data size...`);

                // JPEG 모드도 H264와 동일한 해상도 추정 로직 사용
                const actualPixels = data.depth.length;
                let actualWidth: number, actualHeight: number;

                // 표준 해상도들 중에서 픽셀 수가 일치하는 것 찾기
                const standardResolutions = [
                    { w: 854, h: 480 },   // 480p = 409,920
                    { w: 1280, h: 720 },  // 720p = 921,600  
                    { w: 1920, h: 1080 }  // 1080p = 2,073,600
                ];

                const matchedRes = standardResolutions.find(res => res.w * res.h === actualPixels);

                if (matchedRes) {
                    actualWidth = matchedRes.w;
                    actualHeight = matchedRes.h;
                    debug.logMain(`[Main] JPEG matched standard resolution: ${actualWidth}×${actualHeight}`);
                } else {
                    // 표준 해상도가 아닌 경우 정사각형으로 가정
                    actualWidth = Math.sqrt(actualPixels);
                    actualHeight = actualWidth;

                    if (actualWidth !== Math.floor(actualWidth)) {
                        debug.error(`[Main] Cannot determine JPEG resolution for ${actualPixels} pixels`);
                        // 크기가 맞지 않아도 기존 데이터 사용
                        wsDepthTexture.image.data = data.depth;
                        return;
                    }

                    actualWidth = Math.floor(actualWidth);
                    actualHeight = actualWidth;
                    debug.logMain(`[Main] JPEG using square resolution: ${actualWidth}×${actualHeight}`);
                }

                // 기존 텍스처 정리 및 새 텍스처 생성
                wsDepthTexture.dispose();
                wsDepthTexture = new THREE.DataTexture(data.depth, actualWidth, actualHeight, THREE.RedFormat, THREE.HalfFloatType);

                // 셰이더 유니폼 업데이트
                if (fusionMaterial) {
                    fusionMaterial.uniforms.wsDepthSampler.value = wsDepthTexture;
                }
                if (debugMaterial) {
                    debugMaterial.uniforms.wsDepthSampler.value = wsDepthTexture;
                }

                debug.logMain(`[Main] Emergency recreated JPEG depth texture: ${actualWidth}×${actualHeight}`);
            } else {
                wsDepthTexture.image.data = data.depth;
            }
        } else if (data.depth instanceof ImageBitmap) {
            // H264 모드에서 ImageBitmap으로 전달된 depth 데이터 처리
            debug.logMain(`[Main] H264 depth as ImageBitmap: ${data.depth.width}×${data.depth.height}`);

            // ImageBitmap을 Uint8Array로 변환
            const canvas = new OffscreenCanvas(data.depth.width, data.depth.height);
            const ctx = canvas.getContext('2d')!;
            ctx.drawImage(data.depth, 0, 0);

            const imageData = ctx.getImageData(0, 0, data.depth.width, data.depth.height);
            const pixels = imageData.data; // RGBA data

            // RGBA에서 grayscale로 변환 (R채널만 사용)
            const grayscaleData = new Uint8Array(data.depth.width * data.depth.height);
            for (let i = 0; i < grayscaleData.length; i++) {
                grayscaleData[i] = pixels[i * 4]; // R channel
            }

            debug.logMain(`[Main] Converted ImageBitmap to Uint8Array: ${grayscaleData.length} pixels`);

            // 기존 텍스처와 크기가 다르면 새로 생성
            if (wsDepthTexture.image.width !== data.depth.width || wsDepthTexture.image.height !== data.depth.height) {
                debug.logMain(`[Main] Recreating depth texture for ImageBitmap: ${data.depth.width}×${data.depth.height}`);
                wsDepthTexture.dispose();
                wsDepthTexture = new THREE.DataTexture(grayscaleData, data.depth.width, data.depth.height, THREE.RedFormat, THREE.UnsignedByteType);

                // 셰이더 유니폼 업데이트
                if (fusionMaterial) {
                    fusionMaterial.uniforms.wsDepthSampler.value = wsDepthTexture;
                }
                if (debugMaterial) {
                    debugMaterial.uniforms.wsDepthSampler.value = wsDepthTexture;
                }
            } else {
                wsDepthTexture.image.data = grayscaleData;
            }
        } else {
            debug.error(`[Main] Unknown depth data type:`, typeof data.depth, data.depth.constructor.name);
            debug.error(`[Main] Expected Uint8Array (H264), Uint16Array (JPEG), or ImageBitmap (H264), got:`, data.depth);
        }

        // Mark textures for update (will be applied in render loop)
        wsColorTextureNeedsUpdate = true;
        wsDepthTextureNeedsUpdate = true;

        // 디코딩 완료 시점 기록
        if (data.frameId && data.decodeCompleteTime) {
            latencyTracker.recordDecodeComplete(data.frameId);

            // 다음 렌더 프레임에서 렌더링 완료를 기록
            requestAnimationFrame(() => {
                const stats = latencyTracker.recordRenderComplete(data.frameId);
                if (stats) {
                    // 개별 프레임 레이턴시 로깅 (디버깅용)
                    // debug.logMain(`Frame ${data.frameId}: ${stats.totalLatency.toFixed(1)}ms total`);
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
        debug.error("decode-worker error: ", data.error)
    }
}

function wsConnectButtonClick() {
    const wsURL = 'wss://' + location.host + '/ws/h264';

    // Legacy worker connection
    worker.postMessage({
        type: 'change',
        wsURL: wsURL
    })

    // CP4.5: New architecture connection
    if (USE_NEW_ARCHITECTURE && app) {
        debug.logMain('[CP4.5] Connecting new architecture WebSocket...');
        const wsSystem = app.getWebSocketSystem();
        if (wsSystem) {
            wsSystem.reconnect(wsURL, rtWidth, rtHeight);
        }
    }
}

function wsDisconnectButtonClick() {
    debug.logMain("wsDisconnectButtonClick")

    // Legacy worker disconnection
    worker.postMessage({
        type: 'ws-close'
    })

    wsColorTexture.dispose();
    wsDepthTexture.dispose();

    // Reset texture update flags since textures are disposed
    wsColorTextureNeedsUpdate = false;
    wsDepthTextureNeedsUpdate = false;

    // CP4.5: New architecture disconnection
    if (USE_NEW_ARCHITECTURE && app) {
        debug.logMain('[CP4.5] Disconnecting new architecture WebSocket...');
        app.disconnectWebSocket();
    }
}

function jpegFallbackButtonClick() {
    const isJpegMode = jpegFallbackCheckbox.checked;
    debug.logMain(`Switching to ${isJpegMode ? 'JPEG' : 'H264'} mode`)

    if (USE_NEW_ARCHITECTURE && app) {
        // New architecture: Use WebSocketSystem methods
        debug.logMain(`[CP4.5] Switching to ${isJpegMode ? 'JPEG' : 'H264'} mode in new architecture`);

        const wsSystem = app.getWebSocketSystem();
        if (wsSystem) {
            // 1. Backend에 encoder 변경 요청
            const encoderType = isJpegMode ? 'jpeg' : 'h264';
            console.log(`[main.ts] Requesting Backend encoder change to ${encoderType}`);
            wsSystem.changeEncoderType(encoderType);

            // 2. Frontend decoder 업데이트 - Worker에 JPEG fallback 플래그 전달
            wsSystem.toggleJPEGFallback(isJpegMode);
        }

        // 3. TextureManager 업데이트
        const texManager = app.getTextureManager();
        if (texManager) {
            texManager.setJpegMode(isJpegMode);

            // CP7: Update shader uniforms with new depth texture
            const newDepthTexture = texManager.getDepthTexture();
            console.log(`[jpegFallback] Getting depth texture from TextureManager:`, newDepthTexture);

            if (newDepthTexture) {
                console.log(`[jpegFallback] Updating shader materials with new depth texture`);
                if (fusionMaterial) {
                    fusionMaterial.uniforms.wsDepthSampler.value = newDepthTexture;
                    console.log(`[jpegFallback] ✅ Updated fusionMaterial depth texture`);
                } else {
                    console.warn(`[jpegFallback] fusionMaterial is null!`);
                }
                if (debugMaterial) {
                    debugMaterial.uniforms.wsDepthSampler.value = newDepthTexture;
                    console.log(`[jpegFallback] ✅ Updated debugMaterial depth texture`);
                }
                if (depthFusionMaterial) {
                    depthFusionMaterial.uniforms.wsDepthSampler.value = newDepthTexture;
                    console.log(`[jpegFallback] ✅ Updated depthFusionMaterial depth texture`);
                }
            } else {
                console.error(`[jpegFallback] newDepthTexture is null!`);
            }
        }

    } else {
        // Legacy: Direct worker message
        worker.postMessage({
            type: 'toggle-jpeg-fallback',
            enabled: isJpegMode
        });

        // Legacy: Recreate depth texture
        recreateDepthTexture(isJpegMode);
    }
}

function depthDebugButtonClick() {
    const isChecked = depthDebugCheckbox.checked;
    debug.logMain(`Depth Debug: ${isChecked ? 'Enabled' : 'Disabled'}`);
}

function fpsTestButtonClick() {
    if (latencyTracker.isFPSMeasurementActive()) {
        // 측정 중지
        stopFPSMeasurement();
    } else {
        // 측정 시작
        startFPSMeasurement();
    }
}

function startFPSMeasurement() {
    // Worker에 FPS 측정 시작 알림
    worker.postMessage({ type: 'fps-measurement-start' });

    // Main thread FPS 측정 시작 - 샘플링 방식 초기화
    mainThreadFpsMeasurementActive = true;
    fpsMeasurementSamples = [];
    lastFpsMeasurementSampleTime = performance.now();

    // Latency Tracker 측정 시작
    latencyTracker.startFPSMeasurement();

    // UI 업데이트
    fpsTestButton.value = "Stop FPS Test";
    fpsTestProgress.style.display = "block";
    fpsTestCurrent.style.display = "block";
    fpsTestResult.style.display = "none";
    fpsResultDownload.style.display = "none";

    debug.logFPS("Started 60-second measurement - Worker and main thread active");
}

function stopFPSMeasurement() {
    // Worker에 FPS 측정 중지 알림
    worker.postMessage({ type: 'fps-measurement-stop' });

    // Main thread FPS 측정 중지
    mainThreadFpsMeasurementActive = false;

    // Latency Tracker에서 결과 가져오기
    const result = latencyTracker.stopFPSMeasurement();
    if (result) {
        displayFPSResult(result);
        debug.logFPS("Measurement completed successfully", result);
    } else {
        debug.error("Failed to get FPS measurement result");
        // Fallback UI 업데이트
        fpsTestResult.innerHTML = `
            <div style="color: #ff6666; font-weight: bold;">Measurement Failed</div>
            <div style="color: #aaaaaa; font-size: 11px;">Unable to collect sufficient data. Please try again.</div>
        `;
        fpsTestResult.style.display = "block";
    }

    // UI 상태 복원
    fpsTestButton.value = "Start FPS Test (60s)";
    fpsTestProgress.style.display = "none";
    fpsTestCurrent.style.display = "none";

    debug.logFPS("Measurement stopped - Worker and main thread deactivated");
}

function displayFPSResult(result: FPSMeasurementResult) {
    debug.logFPS('Displaying FPS result:', result);

    // 데이터 검증
    const hasValidData = result && result.measurementDurationMs > 0;
    const duration = hasValidData ? (result.measurementDurationMs / 1000).toFixed(1) : '0.0';

    // 각 메트릭의 유효성 검사
    const pureDecodeFPS = (result.pureDecodeFPS && isFinite(result.pureDecodeFPS)) ? result.pureDecodeFPS : 0;
    const frameProcessingFPS = (result.frameProcessingFPS && isFinite(result.frameProcessingFPS)) ? result.frameProcessingFPS : 0;
    const renderFPS = (result.renderFPS && isFinite(result.renderFPS)) ? result.renderFPS : 0;
    const legacyDecodeFPS = (result.decodeFPS && isFinite(result.decodeFPS)) ? result.decodeFPS : 0;

    const avgDecodeTime = (result.avgDecodeTime && isFinite(result.avgDecodeTime)) ? result.avgDecodeTime : 0;
    const avgProcessingTime = (result.avgProcessingTime && isFinite(result.avgProcessingTime)) ? result.avgProcessingTime : 0;
    const avgRenderTime = (result.avgRenderTime && isFinite(result.avgRenderTime)) ? result.avgRenderTime : 0;

    const averageLatency = (result.averageLatency && isFinite(result.averageLatency)) ? result.averageLatency : 0;
    const minLatency = (result.minLatency && isFinite(result.minLatency)) ? result.minLatency : 0;
    const maxLatency = (result.maxLatency && isFinite(result.maxLatency)) ? result.maxLatency : 0;
    const totalFrames = result.totalFrames || 0;

    // 경고 메시지 생성
    let warningHtml = '';
    const warnings = [];

    if (pureDecodeFPS === 0) warnings.push('No decode data collected');
    if (frameProcessingFPS === 0) warnings.push('No frame processing data collected');
    if (renderFPS === 0) warnings.push('No render data collected');
    if (totalFrames < 10) warnings.push(`Low frame count (${totalFrames})`);
    if (!hasValidData) warnings.push('Invalid measurement duration');

    if (warnings.length > 0) {
        warningHtml = `
            <div style="margin-bottom: 4px; padding: 4px; background: rgba(255,170,0,0.1); border-left: 2px solid #ffaa00; font-size: 11px;">
                <div style="font-weight: bold; color: #ffaa00; margin-bottom: 2px;">⚠️ Data Quality Warnings:</div>
                ${warnings.map(warning => `<div style="color: #cccccc; font-size: 10px;">• ${warning}</div>`).join('')}
                <div style="color: #aaaaaa; font-size: 10px; margin-top: 2px;">Results may be incomplete. Try reconnecting and retesting.</div>
            </div>
        `;
    }

    // 병목 구간 HTML 생성
    let bottleneckHtml = '';
    if (result.bottlenecks && result.bottlenecks.length > 0) {
        bottleneckHtml = '<div style="margin-top: 4px; padding-top: 4px; border-top: 1px solid #555;">';
        bottleneckHtml += '<div style="font-weight: bold; color: #ff6666; margin-bottom: 2px;">Performance Issues:</div>';

        result.bottlenecks.forEach(bottleneck => {
            const severityColor = bottleneck.severity === 'critical' ? '#ff4444' : '#ffaa00';
            bottleneckHtml += `
                <div style="margin-bottom: 3px; padding: 2px 4px; background: rgba(255,68,68,0.1); border-left: 2px solid ${severityColor}; font-size: 11px;">
                    <div style="font-weight: bold; color: ${severityColor};">${bottleneck.stage} (${bottleneck.avgTime.toFixed(1)}ms)</div>
                    <div style="color: #cccccc; font-size: 10px;">${bottleneck.suggestion}</div>
                </div>
            `;
        });

        bottleneckHtml += '</div>';
    }

    fpsTestResult.innerHTML = `
        <div style="margin-bottom: 2px; font-weight: bold;">Performance Test Complete (${duration}s):</div>
        
        ${warningHtml}
        
        <div style="margin-bottom: 4px; padding: 2px 0;">
            <div style="color: #00ff00; font-size: 12px; margin-bottom: 1px;">📊 FPS Metrics:</div>
            <div style="margin-left: 8px; font-size: 11px;">
                <div>Pure Decode: ${pureDecodeFPS.toFixed(2)} fps (${avgDecodeTime.toFixed(1)}ms avg)</div>
                <div>Frame Processing: ${frameProcessingFPS.toFixed(2)} fps (${avgProcessingTime.toFixed(1)}ms avg)</div>
                <div>Render: ${renderFPS.toFixed(2)} fps (${avgRenderTime.toFixed(1)}ms avg)</div>
                <div>Legacy Decode: ${legacyDecodeFPS.toFixed(2)} fps</div>
            </div>
        </div>
        
        <div style="margin-bottom: 4px; padding: 2px 0;">
            <div style="color: #00ff00; font-size: 12px; margin-bottom: 1px;">⚡ Latency:</div>
            <div style="margin-left: 8px; font-size: 11px;">
                <div>Average: ${averageLatency.toFixed(1)}ms</div>
                <div>Range: ${minLatency.toFixed(1)}ms - ${maxLatency.toFixed(1)}ms</div>
                <div>Total Frames: ${totalFrames}</div>
            </div>
        </div>
        
        ${bottleneckHtml}
    `;

    fpsTestResult.style.display = "block";

    // 마지막 결과 저장 및 다운로드 버튼 표시 (유효한 데이터가 있을 때만)
    if (hasValidData && (pureDecodeFPS > 0 || frameProcessingFPS > 0 || renderFPS > 0)) {
        lastFPSResult = result;
        fpsResultDownload.style.display = "block";
        debug.logFPS('Result displayed successfully with download option');
    } else {
        debug.warn('FPS result displayed but insufficient data for download');
    }

    debug.logFPS("Enhanced test result:", result);

    // 병목 구간이 있으면 별도 로그 출력
    if (result.bottlenecks && result.bottlenecks.length > 0) {
        debug.logFPS("🚨 Performance Bottlenecks Detected:");
        result.bottlenecks.forEach(bottleneck => {
            debug.logFPS(`${bottleneck.severity.toUpperCase()}: ${bottleneck.stage} - ${bottleneck.avgTime.toFixed(1)}ms (${bottleneck.percentage.toFixed(1)}%)`);
            debug.logFPS(`   Suggestion: ${bottleneck.suggestion}`);
        });
    }
}

async function initScene() {
    debug.logMain("Initializing scene")

    // 카메라 aspect ratio를 윈도우 크기에 맞춤
    const windowAspect = window.innerWidth / window.innerHeight;
    camera = new THREE.PerspectiveCamera(fov, windowAspect, near, far);
    debug.logMain(`[initScene] Camera aspect ratio: ${windowAspect.toFixed(3)} (${rtWidth}×${rtHeight})`);

    camera.position.copy(
        // new THREE.Vector3().fromArray([-3.15, -0.6, -4])
        new THREE.Vector3().fromArray([-3.6, 0.5, -3.6])
    );
    // camera.lookAt(
    //     // new THREE.Vector3().fromArray([-0.77, 0.43, 0.95])
    //     new THREE.Vector3().fromArray([0.0, 0.0, 0.0])
    // );

    localScene = new THREE.Scene();
    SceneState.scene = localScene;

    renderer = new THREE.WebGLRenderer({
        antialias: true,
        depth: true,
        logarithmicDepthBuffer: true,
    });
    renderer.setPixelRatio(1);
    renderer.setSize(window.innerWidth, window.innerHeight);

    document.body.appendChild(renderer.domElement);


    controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.1;
    controls.autoRotate = true;
    controls.autoRotateSpeed = 0.001

    // controls.target = new THREE.Vector3().fromArray([-0.77, 0.43, 0.95]);
    controls.target = new THREE.Vector3().fromArray([0.0, 0.0, 0.0]);
    // controls.target = new THREE.Vector3().fromArray([-0.92, -0.3, -1.2,]);

    // Initialize adaptive camera tracking
    lastCameraPosition.copy(camera.position);
    lastCameraTarget.copy(controls.target);

    // Auto-load saved camera position if available
    loadCameraPosition();

    // Initialize UI displays
    updateSizeDisplays();

    canvas = renderer.domElement as HTMLCanvasElement

    canvas.style.touchAction = 'none'
    canvas.style.cursor = 'grab'

    // robot_setup();
    object_setup();


    // 워커 초기화 (선택된 해상도로)
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

    debug.logMain(`[initScene] Initialized WebSocket with resolution: ${rtWidth}×${rtHeight}`);

    // 윈도우 리사이즈 이벤트 리스너 추가
    window.addEventListener('resize', () => {
        const newWidth = Math.floor(window.innerWidth * rescaleFactor);
        const newHeight = Math.floor(window.innerHeight * rescaleFactor);

        debug.logMain(`[Window Resize] New size: ${newWidth}×${newHeight} (Old: ${rtWidth}×${rtHeight})`);

        // Update render target dimensions
        rtWidth = newWidth;
        rtHeight = newHeight;

        // Update camera aspect ratio to match new window size
        camera.aspect = rtWidth / rtHeight;
        camera.updateProjectionMatrix();

        // 렌더러 크기 업데이트
        renderer.setSize(rtWidth, rtHeight);

        // Recreate render targets with new size
        recreateRenderTargets();

        // Reconnect WebSocket with new resolution
        reconnectWithNewResolution();

        debug.logMain(`[Window Resize] Updated to ${rtWidth}×${rtHeight}, aspect: ${camera.aspect.toFixed(3)}`);

        // Update UI displays
        updateSizeDisplays();
    });

    localDepthTexture = new THREE.DepthTexture(rtWidth, rtHeight);
    localDepthTexture.type = THREE.FloatType;

    localRenderTarget = new THREE.WebGLRenderTarget(rtWidth, rtHeight, {
        depthBuffer: true,
        stencilBuffer: false,
        depthTexture: localDepthTexture,
        samples: 4,
    });

    wsColorTexture = new THREE.Texture()
    wsColorTexture.minFilter = THREE.LinearFilter;
    wsColorTexture.magFilter = THREE.LinearFilter;
    wsColorTexture.colorSpace = THREE.SRGBColorSpace;

    const initialIsJpegMode = jpegFallbackCheckbox.checked;
    if (initialIsJpegMode) {
        // JPEG mode: Float16 data in Uint16Array format
        wsDepthTexture = new THREE.DataTexture(new Uint16Array(rtWidth * rtHeight), rtWidth, rtHeight, THREE.RedFormat, THREE.HalfFloatType);
    } else {
        // H264 mode: Uint8Array grayscale data
        wsDepthTexture = new THREE.DataTexture(new Uint8Array(rtWidth * rtHeight), rtWidth, rtHeight, THREE.RedFormat, THREE.UnsignedByteType);
    }

    debugMaterial = new THREE.ShaderMaterial({
        uniforms: {
            localColorSampler: { value: localRenderTarget.texture },
            localDepthSampler: { value: localDepthTexture },
            wsColorSampler: { value: wsColorTexture },
            wsDepthSampler: { value: wsDepthTexture },
            width: { value: rtWidth },
            height: { value: rtHeight },
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
            wsFlipX: { value: true }, // X축 flip 활성화
            fusionFlipX: { value: true }, // X축 flip 활성화
            contrast: { value: 1.0 }, // 대비 조정 (1.0보다 작게)
            brightness: { value: 1.0 }, // 밝기 조정
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

    // Gaussian-only scene setup
    gaussianOnlyMaterial = new THREE.ShaderMaterial({
        uniforms: {
            wsColorSampler: { value: wsColorTexture }
        },
        vertexShader: fusionVertexShader,
        fragmentShader: gaussianOnlyFragmentShader,
        depthTest: false,
        depthWrite: false,
    });

    gaussianOnlyQuad = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), gaussianOnlyMaterial);
    gaussianOnlyScene = new THREE.Scene();
    gaussianOnlyScene.add(gaussianOnlyQuad);
    gaussianOnlyCamera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);

    // Depth fusion scene setup
    depthFusionMaterial = new THREE.ShaderMaterial({
        uniforms: {
            localColorSampler: { value: localRenderTarget.texture },
            localDepthSampler: { value: localDepthTexture },
            wsColorSampler: { value: wsColorTexture },
            wsDepthSampler: { value: wsDepthTexture },
            wsFlipX: { value: true },
            fusionFlipX: { value: true },
        },
        vertexShader: fusionVertexShader,
        fragmentShader: depthFusionFragmentShader,
        depthTest: false,
        depthWrite: false,
    });

    depthFusionQuad = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), depthFusionMaterial);
    depthFusionScene = new THREE.Scene();
    depthFusionScene.add(depthFusionQuad);
    depthFusionCamera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);

    // 녹화 기능 브라우저 호환성 검사
    isRecordingSupported = checkRecordingSupport();
    if (!isRecordingSupported) {
        recordingCompatibility.style.display = 'block';
        recordingButton.disabled = true;
        debug.warn('[Recording] Screen recording not supported in this browser');
    } else {
        debug.logMain('[Recording] Screen recording is supported');
    }
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

    // Debug: timeIndex 값 확인 (주기적으로만 출력)
    if (frameId % 60 === 0) {
        debug.logMain(`[sendCameraSnapshot] frameCounter=${frameCounter}, timeIndex=${currentTimeIndex.toFixed(4)}`);
    }

    worker.postMessage({
        type: 'camera',
        position: camBuf.position.buffer,
        target: camBuf.target.buffer,
        intrinsics: camBuf.intrinsics.buffer,
        projection: camBuf.projection.buffer,
        frameId,
        timeIndex: currentTimeIndex,
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
    const deltaTime = clock.getDelta();
    controls.update(deltaTime);

    // CP8: Update new architecture systems
    if (USE_NEW_ARCHITECTURE && app) {
        app.getState().set('frame:counter', frameCounter);
        app.getState().set('frame:timeIndex', currentTimeIndex);
    }

    // console.log(fov, camera.aspect, near, far)

    const now = performance.now()
    const elapsed = now - renderStart;

    if (isPlaying) {
        frameCounter = (frameCounter + 1) % 300;  // 0~299 loop
        currentTimeIndex = frameCounter / 299.0;  // Normalize to 0.0~1.0
    }

    // 최소 1초 경과 후 FPS 계산 (너무 짧은 간격으로 인한 오류 방지)
    if (elapsed > 1000) {
        const duration = elapsed / 1000;
        const fps = renderCnt / duration;

        // FPS 값 검증 및 표시
        if (fps > 0 && fps <= 240 && isFinite(fps)) {
            renderFpsDiv.textContent = `Render FPS: ${fps.toFixed(2)}`;
        } else {
            debug.warn(`Render FPS: Invalid value ${fps.toFixed(2)}, keeping previous display`);
        }

        renderCnt = 0;
        renderStart = now;
    }

    // Adaptive camera update: only send if camera has moved significantly
    if (now - lastCameraUpdateTime > cameraUpdateInterval) {
        const positionDeltaSquared = camera.position.distanceToSquared(lastCameraPosition);
        const targetDeltaSquared = controls.target.distanceToSquared(lastCameraTarget);

        // if (positionDeltaSquared > cameraPositionThresholdSquared || targetDeltaSquared > cameraTargetThresholdSquared) {
        sendCameraSnapshot('render');

        // CP8: Send camera frame via new architecture
        if (USE_NEW_ARCHITECTURE && app) {
            app.sendCameraFrame();
        }

        lastCameraPosition.copy(camera.position);
        lastCameraTarget.copy(controls.target);
        lastCameraUpdateTime = now;
        if (ENABLE_PERFORMANCE_TRACKING) cameraUpdateCount++;
        // }
    }
    robot_animation();

    renderCnt++;

    // Apply texture updates only when needed - this is the actual GPU upload point
    const hasTextureUpdates = wsColorTextureNeedsUpdate || wsDepthTextureNeedsUpdate;
    if (hasTextureUpdates) {
        // 실제 프레임 처리 시작 시점 기록 (GPU 텍스처 업로드 직전)
        recordFrameProcessingStart();
    }

    if (wsColorTextureNeedsUpdate) {
        wsColorTexture.needsUpdate = true;
        wsColorTextureNeedsUpdate = false;
        if (ENABLE_PERFORMANCE_TRACKING) textureUpdateCount++;
    }
    if (wsDepthTextureNeedsUpdate) {
        wsDepthTexture.needsUpdate = true;
        wsDepthTextureNeedsUpdate = false;
        if (ENABLE_PERFORMANCE_TRACKING) textureUpdateCount++;
    }

    if (hasTextureUpdates) {
        // 실제 프레임 처리 완료 시점 기록 (GPU 텍스처 업로드 완료 후)
        recordFrameProcessingComplete();
    }

    // Optimized rendering based on current mode
    if (depthDebugCheckbox.checked) {
        // Depth debug mode: render local scene to target, then debug scene
        renderer.setRenderTarget(localRenderTarget);
        if (ENABLE_PERFORMANCE_TRACKING) renderTargetSwitchCount++;
        renderer.render(localScene, camera);
        renderer.setRenderTarget(null);
        if (ENABLE_PERFORMANCE_TRACKING) renderTargetSwitchCount++;
        renderer.render(debugScene, debugCamera);
    } else {
        switch (currentRenderMode) {
            case RenderMode.FUSION:
                // Fusion mode: render local to target, then fusion scene
                renderer.setRenderTarget(localRenderTarget);
                if (ENABLE_PERFORMANCE_TRACKING) renderTargetSwitchCount++;
                renderer.render(localScene, camera);
                renderer.setRenderTarget(null);
                if (ENABLE_PERFORMANCE_TRACKING) renderTargetSwitchCount++;
                renderer.render(fusionScene, orthoCamera);
                break;
            case RenderMode.GAUSSIAN_ONLY:
                // Gaussian only: skip local rendering, only render gaussian scene
                renderer.render(gaussianOnlyScene, gaussianOnlyCamera);
                break;
            case RenderMode.LOCAL_ONLY:
                // Local only: render local scene directly to screen
                renderer.render(localScene, camera);
                break;
            case RenderMode.DEPTH_FUSION:
                // Depth fusion mode: render local to target, then depth fusion scene
                renderer.setRenderTarget(localRenderTarget);
                if (ENABLE_PERFORMANCE_TRACKING) renderTargetSwitchCount++;
                renderer.render(localScene, camera);
                renderer.setRenderTarget(null);
                if (ENABLE_PERFORMANCE_TRACKING) renderTargetSwitchCount++;
                renderer.render(depthFusionScene, depthFusionCamera);
                break;
        }
    }

    // Performance logging (only in debug mode)
    if (ENABLE_PERFORMANCE_TRACKING && now - lastPerformanceLogTime > performanceLogInterval) {
        const fps = renderCnt / ((now - renderStart) / 1000);
        debug.logMain(`[Performance] FPS: ${fps.toFixed(1)} | Texture Updates: ${textureUpdateCount} | Camera Updates: ${cameraUpdateCount} | Render Target Switches: ${renderTargetSwitchCount} | Mode: ${currentRenderMode}`);

        // Reset counters
        textureUpdateCount = 0;
        cameraUpdateCount = 0;
        renderTargetSwitchCount = 0;
        lastPerformanceLogTime = now;
    }

    // 렌더링 완료 시점 기록 및 레이턴시 통계 업데이트
    updateLatencyStats();

    // FPS 측정 중이면 진행률 업데이트
    updateFPSTestUI();

    // Camera debug info 업데이트
    updateCameraDebugInfo();
}


// CP4: New architecture instance (Phase 1: runs in parallel with legacy)
let app: Application | null = null;

initScene().then(async () => {
    renderStart = performance.now()
    renderLoop()

    // UI 컨트롤러 활성화
    debug.logMain('UI Controller initialized:', uiController.isVisible())

    // CP4: Initialize new architecture in parallel (non-breaking)
    if (USE_NEW_ARCHITECTURE) {
        try {
            debug.logMain('[CP4] Initializing new architecture...');

            app = new Application({
                canvas: renderer.domElement,
                wsUrl: '', // Will be set when connecting
                width: rtWidth,
                height: rtHeight,
                renderMode: NewRenderMode.Hybrid,
                debugMode: true,
            });

            // Phase 1: Wrap existing objects (non-breaking) with existing worker
            await app.initializeWithExistingObjects(
                localScene,
                camera,
                renderer,
                controls,
                worker // CP4.5: Pass existing worker
            );

            debug.logMain('[CP4] New architecture initialized successfully');
            debug.logMain('[CP4] Systems are now running in parallel with legacy code');

            // CP7: Connect TextureManager textures to rendering
            if (USE_NEW_TEXTURE_MANAGER && app) {
                const texManager = app.getTextureManager();
                if (texManager) {
                    const newColorTexture = texManager.getColorTexture();
                    const newDepthTexture = texManager.getDepthTexture();

                    if (newColorTexture && newDepthTexture) {
                        // Update all shader materials to use new textures
                        if (fusionMaterial) {
                            fusionMaterial.uniforms.wsColorSampler.value = newColorTexture;
                            fusionMaterial.uniforms.wsDepthSampler.value = newDepthTexture;
                            debug.logMain('[CP7] FusionMaterial updated with new textures');
                        }

                        if (debugMaterial) {
                            debugMaterial.uniforms.wsColorSampler.value = newColorTexture;
                            debugMaterial.uniforms.wsDepthSampler.value = newDepthTexture;
                            debug.logMain('[CP7] DebugMaterial updated with new textures');
                        }

                        if (gaussianOnlyMaterial) {
                            gaussianOnlyMaterial.uniforms.wsColorSampler.value = newColorTexture;
                            debug.logMain('[CP7] GaussianOnlyMaterial updated with new texture');
                        }

                        if (depthFusionMaterial) {
                            depthFusionMaterial.uniforms.wsColorSampler.value = newColorTexture;
                            depthFusionMaterial.uniforms.wsDepthSampler.value = newDepthTexture;
                            debug.logMain('[CP7] DepthFusionMaterial updated with new textures');
                        }

                        debug.logMain('[CP7] All materials now using TextureManager textures');
                    }
                }
            }
        } catch (error) {
            console.error('[CP4] Failed to initialize new architecture:', error);
        }
    }
})



function getCameraIntrinsics(camera: THREE.PerspectiveCamera, renderWidth: number, renderHeight: number) {
    const projmat = camera.projectionMatrix;

    // 실제 해상도에 맞는 focal length 계산
    // projection matrix에서 직접 계산하여 해상도별로 적절한 스케일링 적용
    const fx = (renderWidth / 2) * projmat.elements[0];
    const fy = (renderHeight / 2) * projmat.elements[5];

    // Principal point는 해상도의 중심점
    const cx = renderWidth / 2;
    const cy = renderHeight / 2;

    debug.logMain(`[getCameraIntrinsics] Resolution: ${renderWidth}×${renderHeight}, Dynamic fx/fy: fx=${fx.toFixed(2)}, fy=${fy.toFixed(2)}`);

    return [fx, 0, cx, 0, fy, cy, 0, 0, 1]
}

let lastStatsUpdate = 0;
const statsUpdateInterval = 500; // 0.5초마다 통계 업데이트

// 마지막 FPS 측정 결과 저장 (다운로드용)
let lastFPSResult: FPSMeasurementResult | null = null;

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
        debug.logLatency(`Stats - Total: ${stats.avg.totalLatency.toFixed(1)}ms, ` +
            `Network: ${((stats.avg.networkUploadTime || 0) + (stats.avg.networkDownloadTime || 0)).toFixed(1)}ms, ` +
            `Server: ${(stats.avg.serverProcessingTime || 0).toFixed(1)}ms, ` +
            `Decode: ${(stats.avg.clientDecodeTime || 0).toFixed(1)}ms, ` +
            `Render: ${(stats.avg.clientRenderTime || 0).toFixed(1)}ms, ` +
            `Offset: ${latencyTracker.getClockOffset().toFixed(1)}ms`);
    }
}

function updateFPSTestUI() {
    if (!latencyTracker.isFPSMeasurementActive()) return;

    const progress = latencyTracker.getFPSMeasurementProgress();
    const currentStats = latencyTracker.getCurrentFPSTestStats();

    if (progress) {
        const remainingSeconds = Math.ceil(progress.remainingMs / 1000);
        const progressPercent = (progress.progress * 100).toFixed(1);
        fpsTestProgress.textContent = `Progress: ${progressPercent}% (${remainingSeconds}s left)`;
    }

    if (currentStats) {
        fpsTestCurrent.textContent = `Current: Decode ${currentStats.decodeFPS.toFixed(1)}fps, Render ${currentStats.renderFPS.toFixed(1)}fps`;
    }

    // 측정이 완료되었으면 자동으로 결과 표시
    if (progress && progress.remainingMs <= 0) {
        debug.logFPS("Auto-completion triggered by progress timer");
        stopFPSMeasurement();
    }
}

function updateCameraDebugInfo() {
    if (!cameraDebugCheckbox.checked) return;

    const position = camera.position;
    const target = controls.target;

    cameraPositionDiv.textContent = `Position: (${position.x.toFixed(3)}, ${position.y.toFixed(3)}, ${position.z.toFixed(3)})`;
    cameraTargetDiv.textContent = `Target: (${target.x.toFixed(3)}, ${target.y.toFixed(3)}, ${target.z.toFixed(3)})`;

    // Update input fields as well
    updateCameraInputFields();
}

function downloadFPSResults() {
    if (!lastFPSResult) {
        debug.warn("No FPS test results available for download");
        return;
    }

    const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
    const filename = `fps-test-results-${timestamp}.txt`;

    // 현재 설정 정보 수집
    const currentResolution = document.querySelector('#resolution-select') as HTMLSelectElement;
    const selectedResolution = currentResolution ? currentResolution.value : 'unknown';
    const jpegMode = jpegFallbackCheckbox.checked;
    const renderMode = currentRenderMode;

    // txt 내용 생성
    const content = generateFPSReportText(lastFPSResult, {
        resolution: selectedResolution,
        jpegMode,
        renderMode,
        timestamp: new Date().toISOString()
    });

    // 파일 다운로드
    const blob = new Blob([content], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);

    debug.logFPS(`Downloaded test results: ${filename}`);
}

function generateFPSReportText(result: FPSMeasurementResult, config: {
    resolution: string;
    jpegMode: boolean;
    renderMode: string;
    timestamp: string;
}): string {
    const duration = (result.measurementDurationMs / 1000).toFixed(1);

    // 병목 구간 텍스트 생성
    let bottleneckText = '';
    if (result.bottlenecks && result.bottlenecks.length > 0) {
        bottleneckText = '\nPerformance Bottlenecks Detected:\n';
        bottleneckText += '----------------------------------------\n';
        result.bottlenecks.forEach((bottleneck, index) => {
            bottleneckText += `${index + 1}. ${bottleneck.stage} [${bottleneck.severity.toUpperCase()}]\n`;
            bottleneckText += `   - Average Time: ${bottleneck.avgTime.toFixed(2)} ms\n`;
            bottleneckText += `   - Threshold: ${bottleneck.threshold.toFixed(2)} ms\n`;
            bottleneckText += `   - Percentage of Total: ${bottleneck.percentage.toFixed(1)}%\n`;
            bottleneckText += `   - Recommendation: ${bottleneck.suggestion}\n\n`;
        });
    }

    return `StreamSplat Enhanced Performance Test Results
=====================================================

Test Configuration:
- Timestamp: ${config.timestamp}
- Resolution: ${config.resolution}
- Encoding: ${config.jpegMode ? 'JPEG' : 'H.264'}
- Render Mode: ${config.renderMode}
- Test Duration: ${duration} seconds

Enhanced Performance Metrics:
- Pure Decode FPS: ${result.pureDecodeFPS.toFixed(2)} fps (${result.avgDecodeTime.toFixed(2)} ms avg per frame)
- Frame Processing FPS: ${result.frameProcessingFPS.toFixed(2)} fps (${result.avgProcessingTime.toFixed(2)} ms avg per frame)
- Render FPS: ${result.renderFPS.toFixed(2)} fps (${result.avgRenderTime.toFixed(2)} ms avg per frame)
- Legacy Decode FPS: ${result.decodeFPS.toFixed(2)} fps (main thread measurement)
- Total Frames Processed: ${result.totalFrames}

Performance Analysis:
- Most accurate FPS metric: Pure Decode FPS (${result.pureDecodeFPS.toFixed(2)} fps)
- Decode processing is ${result.pureDecodeFPS > result.decodeFPS ? 'faster' : 'slower'} than legacy measurement suggests
- Frame processing overhead: ${result.avgProcessingTime.toFixed(2)} ms per frame
- Rendering overhead: ${result.avgRenderTime.toFixed(2)} ms per frame

Latency Statistics:
- Average Total Latency: ${result.averageLatency.toFixed(2)} ms
- Minimum Latency: ${result.minLatency.toFixed(2)} ms  
- Maximum Latency: ${result.maxLatency.toFixed(2)} ms
- Latency Range: ${(result.maxLatency - result.minLatency).toFixed(2)} ms
${bottleneckText}
System Information:
- User Agent: ${navigator.userAgent}
- Window Size: ${window.innerWidth}×${window.innerHeight}
- Pixel Ratio: ${window.devicePixelRatio}
- Hardware Decoding: ${config.jpegMode ? 'Not Used (JPEG Mode)' : 'Available (H.264 Mode)'}

Performance Insights:
- Your decode performance: ${result.pureDecodeFPS.toFixed(0)} fps is ${result.pureDecodeFPS >= 60 ? 'excellent' : result.pureDecodeFPS >= 30 ? 'good' : 'below optimal'} for real-time streaming
- Frame processing latency: ${result.avgProcessingTime.toFixed(1)} ms is ${result.avgProcessingTime <= 3 ? 'excellent' : result.avgProcessingTime <= 8 ? 'acceptable' : 'high'}
- Overall system performance: ${result.averageLatency <= 50 ? 'Excellent' : result.averageLatency <= 100 ? 'Good' : 'Needs optimization'} (${result.averageLatency.toFixed(0)} ms total latency)

=====================================================
Generated by StreamSplat Enhanced Performance Testing Tool
`;
}

// 화면 녹화 기능들
function checkRecordingSupport(): boolean {
    return !!(window.MediaRecorder &&
        HTMLCanvasElement.prototype.captureStream);
}

function setupRecording() {
    try {
        if (!canvas) {
            debug.error('[Recording] Canvas not available');
            return false;
        }

        // Canvas에서 MediaStream 생성 (60fps)
        recordingStream = canvas.captureStream(60);
        if (!recordingStream) {
            debug.error('[Recording] Failed to capture stream from canvas');
            return false;
        }

        debug.logMain('[Recording] Canvas stream created successfully');

        // MediaRecorder 생성
        const options: MediaRecorderOptions = {
            mimeType: 'video/webm;codecs=vp9',
            videoBitsPerSecond: 8000000, // 8 Mbps
        };

        // VP9가 지원되지 않으면 VP8 시도
        if (!MediaRecorder.isTypeSupported(options.mimeType!)) {
            options.mimeType = 'video/webm;codecs=vp8';
            debug.logMain('[Recording] VP9 not supported, using VP8');
        }

        // VP8도 지원되지 않으면 기본 webm
        if (!MediaRecorder.isTypeSupported(options.mimeType!)) {
            options.mimeType = 'video/webm';
            debug.logMain('[Recording] VP8 not supported, using default webm');
        }

        mediaRecorder = new MediaRecorder(recordingStream, options);

        // 녹화 이벤트 핸들러
        mediaRecorder.ondataavailable = (event) => {
            if (event.data && event.data.size > 0) {
                recordedChunks.push(event.data);
                debug.logMain(`[Recording] Data chunk received: ${event.data.size} bytes`);
            }
        };

        mediaRecorder.onstop = () => {
            debug.logMain('[Recording] Recording stopped');
            recordingBlob = new Blob(recordedChunks, { type: 'video/webm' });
            debug.logMain(`[Recording] Final video blob size: ${recordingBlob.size} bytes`);

            // 다운로드 버튼 활성화
            recordingDownload.style.display = 'block';
            recordingDownload.value = `Download Recording (${(recordingBlob.size / 1024 / 1024).toFixed(1)}MB)`;
        };

        mediaRecorder.onerror = (event) => {
            debug.error('[Recording] MediaRecorder error:', event);
        };

        return true;

    } catch (error) {
        debug.error('[Recording] Setup failed:', error);
        return false;
    }
}

function recordingButtonClick() {
    if (!isRecordingSupported) {
        recordingCompatibility.style.display = 'block';
        recordingCompatibility.textContent = 'Screen recording not supported in this browser';
        return;
    }

    if (mediaRecorder?.state === 'recording') {
        // 녹화 중지
        stopRecording();
    } else {
        // 녹화 시작
        startRecording();
    }
}

function startRecording() {
    try {
        // MediaRecorder 설정
        if (!setupRecording()) {
            debug.error('[Recording] Failed to setup recording');
            return;
        }

        // 녹화 데이터 초기화
        recordedChunks = [];
        recordingBlob = null;
        recordingStartTime = performance.now();

        // MediaRecorder 시작
        mediaRecorder!.start(1000); // 1초마다 데이터 청크 생성

        // UI 업데이트
        recordingButton.value = 'Stop Recording';
        recordingStatus.style.display = 'block';
        recordingTime.style.display = 'block';
        recordingMode.style.display = 'block';
        recordingSize.style.display = 'block';
        recordingDownload.style.display = 'none';

        recordingStatus.textContent = 'Status: Recording...';
        recordingMode.textContent = `Mode: ${currentRenderMode}`;

        // 타이머 시작
        recordingTimer = setInterval(updateRecordingUI, 100);

        debug.logMain(`[Recording] Started recording in ${currentRenderMode} mode`);

    } catch (error) {
        debug.error('[Recording] Failed to start recording:', error);
        recordingStatus.textContent = 'Status: Failed to start recording';
    }
}

function stopRecording() {
    try {
        if (mediaRecorder && mediaRecorder.state === 'recording') {
            mediaRecorder.stop();
        }

        // 타이머 정리
        if (recordingTimer) {
            clearInterval(recordingTimer);
            recordingTimer = null;
        }

        // 스트림 정리
        if (recordingStream) {
            recordingStream.getTracks().forEach(track => track.stop());
            recordingStream = null;
        }

        // UI 업데이트
        recordingButton.value = 'Start Recording';
        recordingStatus.textContent = 'Status: Stopped';

        debug.logMain('[Recording] Recording stopped successfully');

    } catch (error) {
        debug.error('[Recording] Error stopping recording:', error);
    }
}

function updateRecordingUI() {
    if (recordingStartTime > 0) {
        const elapsedMs = performance.now() - recordingStartTime;
        const elapsedSeconds = Math.floor(elapsedMs / 1000);
        const minutes = Math.floor(elapsedSeconds / 60);
        const seconds = elapsedSeconds % 60;

        recordingTime.textContent = `Duration: ${minutes}:${seconds.toString().padStart(2, '0')}`;

        // 현재까지 녹화된 데이터 크기 표시
        if (recordedChunks.length > 0) {
            const totalSize = recordedChunks.reduce((total, chunk) => total + chunk.size, 0);
            recordingSize.textContent = `Size: ${(totalSize / 1024 / 1024).toFixed(1)}MB`;
        }
    }
}

function downloadRecording() {
    if (!recordingBlob) {
        debug.warn('[Recording] No recording available for download');
        return;
    }

    const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
    const filename = `recording-${currentRenderMode}-${timestamp}.webm`;

    const url = URL.createObjectURL(recordingBlob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);

    debug.logMain(`[Recording] Downloaded: ${filename}`);
}

// 독립적인 프레임 처리 성능 측정 함수들
function recordFrameProcessingStart() {
    currentFrameStartTime = performance.now();
}

function recordFrameProcessingComplete() {
    if (currentFrameStartTime === 0) return;

    const processingTime = performance.now() - currentFrameStartTime;
    frameProcessingCount++;
    totalFrameProcessingTime += processingTime;

    // FPS 측정 중이면 1초마다 FPS 샘플 수집
    if (mainThreadFpsMeasurementActive) {
        const now = performance.now();
        if (now - lastFpsMeasurementSampleTime >= 1000) {
            // 1초간의 프레임 처리 FPS 계산
            const duration = (now - lastFpsMeasurementSampleTime) / 1000;
            const fps = frameProcessingCount / duration;

            // FPS 값 검증 및 샘플 추가
            if (fps > 0 && fps <= 240 && isFinite(fps)) {
                fpsMeasurementSamples.push(fps);
                debug.logFPS(`Sample collected: ${fps.toFixed(2)} fps`);
            } else {
                debug.warn(`Invalid FPS sample rejected: ${fps.toFixed(2)} fps`);
            }

            lastFpsMeasurementSampleTime = now;
        }
    }

    // 히스토리에 추가
    frameProcessingHistory.push(processingTime);
    if (frameProcessingHistory.length > maxFrameProcessingHistory) {
        frameProcessingHistory.shift();
    }

    currentFrameStartTime = 0;

    // 1초마다 프레임 처리 성능 통계 업데이트
    const now = performance.now();
    const elapsed = now - frameProcessingStart;

    if (elapsed > 1000) {
        const duration = elapsed / 1000;
        const frameProcessingFPS = frameProcessingCount / duration;
        const avgProcessingTime = frameProcessingCount > 0 ? totalFrameProcessingTime / frameProcessingCount : 0;

        // 최근 프레임들의 처리 시간 분석
        const recentTimes = frameProcessingHistory.slice(-Math.min(60, frameProcessingHistory.length));
        const recentAvg = recentTimes.reduce((a, b) => a + b, 0) / recentTimes.length;
        const recentMin = Math.min(...recentTimes);
        const recentMax = Math.max(...recentTimes);

        // FPS 값 검증 후 GPU 텍스처 업데이트 기반 프레임 처리 FPS 표시
        if (frameProcessingFPS > 0 && frameProcessingFPS <= 240 && isFinite(frameProcessingFPS)) {
            renderFpsDiv.textContent = `GPU Processing FPS: ${frameProcessingFPS.toFixed(2)} (${avgProcessingTime.toFixed(1)}ms avg)`;
        } else {
            debug.warn(`GPU Processing FPS: Invalid value ${frameProcessingFPS.toFixed(2)}, keeping previous display`);
        }

        debug.logGPU(`Frame Processing - FPS: ${frameProcessingFPS.toFixed(2)}, ` +
            `Avg: ${avgProcessingTime.toFixed(2)}ms, ` +
            `Recent: ${recentAvg.toFixed(2)}ms (${recentMin.toFixed(2)}-${recentMax.toFixed(2)}ms), ` +
            `FPS measurement ${mainThreadFpsMeasurementActive ? 'ACTIVE' : 'inactive'} (${fpsMeasurementSamples.length} samples)`);

        // FPS 측정 중이면 샘플 데이터로 Latency Tracker 업데이트
        if (mainThreadFpsMeasurementActive && fpsMeasurementSamples.length > 0) {
            const avgFPS = fpsMeasurementSamples.reduce((a, b) => a + b, 0) / fpsMeasurementSamples.length;
            const avgTime = avgFPS > 0 ? 1000 / avgFPS : 0; // FPS를 ms로 변환
            latencyTracker.recordFrameProcessingFPSSample(fpsMeasurementSamples.length, avgTime);
            debug.logFPS(`Recording GPU processing sample: ${fpsMeasurementSamples.length} samples, ${avgFPS.toFixed(2)} fps avg`);
        } else if (!mainThreadFpsMeasurementActive) {
            // 평상시에는 1초 단위 데이터 사용 (기존 방식 유지)
            latencyTracker.recordFrameProcessingFPS(frameProcessingCount, avgProcessingTime);
        }

        // 1초 단위 카운터만 리셋 (FPS 측정 누적 데이터는 유지)
        frameProcessingCount = 0;
        frameProcessingStart = now;
        totalFrameProcessingTime = 0;
    }
}