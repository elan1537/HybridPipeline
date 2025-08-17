import * as THREE from 'three'
import { OrbitControls } from 'three/addons'
import { object_setup } from './scene-setup';
import { SceneState } from './state/scene-state';
import { LatencyTracker, LatencyStats } from './latency-tracker';
import { uiController } from './ui-controller';

import fusionVertexShader from './shaders/fusionVertexShader.vs?raw';
import fusionColorFragmentShader from './shaders/fusionColorShader.fs?raw';
import debugVertexShader from './shaders/debugVertexShader.vs?raw';
import debugFragmentShader from './shaders/debugColorShader.fs?raw';

// Shader for displaying WebSocket color texture with proper aspect ratio
const gaussianOnlyFragmentShader = `
  varying vec2 vUv;
  uniform sampler2D wsColorSampler;
  uniform float streamAspect;  // Stream resolution aspect ratio
  uniform float windowAspect;  // Window aspect ratio
  
  void main() {
    vec2 uv = vUv;
    
    // Aspect ratio correction to prevent stretching
    if (streamAspect > windowAspect) {
      // Stream is wider than window - fit width, center height
      float scale = windowAspect / streamAspect;
      uv.y = (uv.y - 0.5) * scale + 0.5;
    } else {
      // Stream is taller than window - fit height, center width  
      float scale = streamAspect / windowAspect;
      uv.x = (uv.x - 0.5) * scale + 0.5;
    }
    
    // Check if UV is within valid range
    if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) {
      gl_FragColor = vec4(0.0, 0.0, 0.0, 1.0); // Black border
      return;
    }
    
    vec2 flippedUv = vec2(1.0 - uv.x, 1.0 - uv.y);
    vec4 wsColor = texture2D(wsColorSampler, flippedUv);
    gl_FragColor = vec4(wsColor.rgb, 1.0);
  }
`;

// 해상도는 이제 resolutionManager에서 관리
let rtWidth = 1280;  // 기본값 (720p)
let rtHeight = 720;

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

// Render mode radio buttons
const fusionModeRadio = document.getElementById('fusion-mode') as HTMLInputElement;
const gaussianOnlyModeRadio = document.getElementById('gaussian-only-mode') as HTMLInputElement;
const localOnlyModeRadio = document.getElementById('local-only-mode') as HTMLInputElement;

// Render mode constants
enum RenderMode {
    FUSION = 'fusion',
    GAUSSIAN_ONLY = 'gaussian',
    LOCAL_ONLY = 'local'
}

let currentRenderMode: RenderMode = RenderMode.FUSION;

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

wsConnectButton.addEventListener('click', () => wsConnectButtonClick())
wsDisconnectButton.addEventListener('click', () => wsDisconnectButtonClick())


// Render mode event handlers
fusionModeRadio.addEventListener('change', () => {
    if (fusionModeRadio.checked) {
        currentRenderMode = RenderMode.FUSION;
        console.log('Switched to Fusion Mode');
    }
});

gaussianOnlyModeRadio.addEventListener('change', () => {
    if (gaussianOnlyModeRadio.checked) {
        currentRenderMode = RenderMode.GAUSSIAN_ONLY;
        console.log('Switched to Gaussian Splatting Only Mode');
    }
});

localOnlyModeRadio.addEventListener('change', () => {
    if (localOnlyModeRadio.checked) {
        currentRenderMode = RenderMode.LOCAL_ONLY;
        console.log('Switched to Local Rendering Only Mode');
    }
});

function recreateDepthTexture(isJpegMode: boolean) {
    if (wsDepthTexture) {
        console.log(`[recreateDepthTexture] Disposing old depth texture: ${wsDepthTexture.image.width}×${wsDepthTexture.image.height}`);
        wsDepthTexture.dispose();
    }

    console.log(`[recreateDepthTexture] Creating new texture for ${isJpegMode ? 'JPEG' : 'H264'} mode: ${rtWidth}×${rtHeight}`);

    if (isJpegMode) {
        // JPEG mode: Float16 data in Uint16Array format
        const depthArray = new Uint16Array(rtWidth * rtHeight);
        wsDepthTexture = new THREE.DataTexture(depthArray, rtWidth, rtHeight, THREE.RedFormat, THREE.HalfFloatType);
        console.log(`[recreateDepthTexture] JPEG mode: Created Uint16Array of size ${depthArray.length}`);
    } else {
        // H264 mode: Uint8Array grayscale data
        const depthArray = new Uint8Array(rtWidth * rtHeight);
        wsDepthTexture = new THREE.DataTexture(depthArray, rtWidth, rtHeight, THREE.RedFormat, THREE.UnsignedByteType);
        console.log(`[recreateDepthTexture] H264 mode: Created Uint8Array of size ${depthArray.length}`);
    }

    // Update shader uniforms with new texture
    if (fusionMaterial) {
        fusionMaterial.uniforms.wsDepthSampler.value = wsDepthTexture;
    }
    if (debugMaterial) {
        debugMaterial.uniforms.wsDepthSampler.value = wsDepthTexture;
    }

    console.log(`[recreateDepthTexture] Depth texture recreated: ${wsDepthTexture.image.width}×${wsDepthTexture.image.height}`);
}
jpegFallbackCheckbox.addEventListener('click', () => jpegFallbackButtonClick())
depthDebugCheckbox.addEventListener('click', () => depthDebugButtonClick())



// 로컬 렌더 타겟 재생성
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
        colorSpace: THREE.LinearSRGBColorSpace,
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

// 워커 초기화는 initScene에서 수행하도록 변경

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

        // Depth 데이터 상세 로깅
        const expectedSize = rtWidth * rtHeight;

        console.log(`[Main] Received frame ${data.frameId}:`);
        console.log(`[Main] Color image: ${data.image.width}×${data.image.height}`);

        if (data.depth instanceof Uint8Array) {
            // H264 모드 - depth는 Uint8Array (grayscale)
            console.log(`[Main] H264 depth array length: ${data.depth.length}, Expected: ${expectedSize}`);
            console.log(`[Main] Current stream resolution: ${rtWidth}×${rtHeight}`);

            if (data.depth.length !== expectedSize) {
                console.warn(`[Main] H264 depth array size mismatch! Got ${data.depth.length}, expected ${expectedSize}`);
                console.error(`[Main] wsDepthTexture size: ${wsDepthTexture.image.width}×${wsDepthTexture.image.height}`);

                // H264 모드도 JPEG와 동일하게 emergency recreation 수행
                console.log(`[Main] Recreating H264 depth texture to match data size...`);

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
                    console.log(`[Main] Matched standard resolution: ${actualWidth}×${actualHeight}`);
                } else {
                    // 표준 해상도가 아닌 경우 정사각형으로 가정
                    actualWidth = Math.sqrt(actualPixels);
                    actualHeight = actualWidth;

                    if (actualWidth !== Math.floor(actualWidth)) {
                        console.error(`[Main] Cannot determine resolution for ${actualPixels} pixels`);
                        return; // 처리할 수 없는 경우 스킵
                    }

                    actualWidth = Math.floor(actualWidth);
                    actualHeight = actualWidth;
                    console.log(`[Main] Using square resolution: ${actualWidth}×${actualHeight}`);
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

                console.log(`[Main] Emergency recreated H264 depth texture: ${actualWidth}×${actualHeight}`);

            } else {
                wsDepthTexture.image.data = data.depth;
            }
        } else if (data.depth instanceof Uint16Array) {
            // JPEG 모드 - depth는 Uint16Array  
            console.log(`[Main] Depth array length: ${data.depth.length}, Expected: ${expectedSize}`);
            console.log(`[Main] Current stream resolution: ${rtWidth}×${rtHeight}`);

            if (data.depth.length !== expectedSize) {
                console.error(`[Main] JPEG depth array size mismatch! Got ${data.depth.length}, expected ${expectedSize}`);
                console.error(`[Main] wsDepthTexture size: ${wsDepthTexture.image.width}×${wsDepthTexture.image.height}`);

                // 크기가 맞지 않으면 텍스처를 다시 생성
                console.log(`[Main] Recreating JPEG depth texture to match data size...`);

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
                    console.log(`[Main] JPEG matched standard resolution: ${actualWidth}×${actualHeight}`);
                } else {
                    // 표준 해상도가 아닌 경우 정사각형으로 가정
                    actualWidth = Math.sqrt(actualPixels);
                    actualHeight = actualWidth;

                    if (actualWidth !== Math.floor(actualWidth)) {
                        console.error(`[Main] Cannot determine JPEG resolution for ${actualPixels} pixels`);
                        // 크기가 맞지 않아도 기존 데이터 사용
                        wsDepthTexture.image.data = data.depth;
                        return;
                    }

                    actualWidth = Math.floor(actualWidth);
                    actualHeight = actualWidth;
                    console.log(`[Main] JPEG using square resolution: ${actualWidth}×${actualHeight}`);
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

                console.log(`[Main] Emergency recreated JPEG depth texture: ${actualWidth}×${actualHeight}`);
            } else {
                wsDepthTexture.image.data = data.depth;
            }
        } else {
            console.error(`[Main] Unknown depth data type:`, typeof data.depth, data.depth.constructor.name);
            console.error(`[Main] Expected Uint8Array (H264) or Uint16Array (JPEG), got:`, data.depth);
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

    // Reset texture update flags since textures are disposed
    wsColorTextureNeedsUpdate = false;
    wsDepthTextureNeedsUpdate = false;
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

function depthDebugButtonClick() {
    const isChecked = depthDebugCheckbox.checked;
    console.log(`Depth Debug: ${isChecked ? 'Enabled' : 'Disabled'}`);
}

async function initScene() {
    console.log("initScene")

    // 카메라 aspect ratio를 윈도우 크기에 맞춤 (가우시안 씬이 전체 창에 맞게 표시)
    camera = new THREE.PerspectiveCamera(fov, window.innerWidth / window.innerHeight, near, far);

    camera.position.copy(
        new THREE.Vector3().fromArray([0.9, 1.11, 2.22])
    );
    camera.lookAt(
        new THREE.Vector3().fromArray([-0.77, 0.43, 0.95])
    );

    localScene = new THREE.Scene();
    SceneState.scene = localScene;

    renderer = new THREE.WebGLRenderer({
        antialias: true,
        depth: true,
        logarithmicDepthBuffer: true,
    });
    renderer.setPixelRatio(1);
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.outputColorSpace = THREE.SRGBColorSpace;

    document.body.appendChild(renderer.domElement);


    controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.1;
    controls.autoRotate = false;
    controls.autoRotateSpeed = 0.5

    controls.target = new THREE.Vector3().fromArray([-0.77, 0.43, 0.95]);

    // Initialize adaptive camera tracking
    lastCameraPosition.copy(camera.position);
    lastCameraTarget.copy(controls.target);

    canvas = renderer.domElement as HTMLCanvasElement

    canvas.style.touchAction = 'none'
    canvas.style.cursor = 'grab'

    // object_setup();
    object_setup();


    // 워커 초기화
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

    // 윈도우 리사이즈 이벤트 리스너 추가
    window.addEventListener('resize', () => {
        console.log(`[Window Resize] New size: ${window.innerWidth}×${window.innerHeight}`);

        // 카메라 비율을 윈도우 크기에 맞춤 (가우시안 씬이 전체 창에 맞게 표시)
        camera.aspect = window.innerWidth / window.innerHeight;
        camera.updateProjectionMatrix();

        // 렌더러 크기 업데이트
        renderer.setSize(window.innerWidth, window.innerHeight);


        console.log(`[Window Resize] Camera aspect updated to window ratio: ${camera.aspect.toFixed(3)}`);
    });

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

    // Adaptive camera update: only send if camera has moved significantly
    if (now - lastCameraUpdateTime > cameraUpdateInterval) {
        const positionDeltaSquared = camera.position.distanceToSquared(lastCameraPosition);
        const targetDeltaSquared = controls.target.distanceToSquared(lastCameraTarget);

        if (positionDeltaSquared > cameraPositionThresholdSquared || targetDeltaSquared > cameraTargetThresholdSquared) {
            sendCameraSnapshot('render');
            lastCameraPosition.copy(camera.position);
            lastCameraTarget.copy(controls.target);
            lastCameraUpdateTime = now;
            if (ENABLE_PERFORMANCE_TRACKING) cameraUpdateCount++;
        }
    }
    // robot_animation();

    renderCnt++;

    // Apply texture updates only when needed
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
        }
    }

    // Performance logging (only in debug mode)
    if (ENABLE_PERFORMANCE_TRACKING && now - lastPerformanceLogTime > performanceLogInterval) {
        const fps = renderCnt / ((now - renderStart) / 1000);
        console.log(`[Performance] FPS: ${fps.toFixed(1)} | Texture Updates: ${textureUpdateCount} | Camera Updates: ${cameraUpdateCount} | Render Target Switches: ${renderTargetSwitchCount} | Mode: ${currentRenderMode}`);

        // Reset counters
        textureUpdateCount = 0;
        cameraUpdateCount = 0;
        renderTargetSwitchCount = 0;
        lastPerformanceLogTime = now;
    }

    // 렌더링 완료 시점 기록 및 레이턴시 통계 업데이트
    updateLatencyStats();
}


initScene().then(() => {
    renderStart = performance.now()
    renderLoop()

    // UI 컨트롤러 활성화
    // uiController는 이미 import 시 초기화됨
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

    console.log(`[getCameraIntrinsics] Resolution: ${renderWidth}×${renderHeight}, Dynamic fx/fy: fx=${fx.toFixed(2)}, fy=${fy.toFixed(2)}`);

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