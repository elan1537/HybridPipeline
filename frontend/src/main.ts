import * as THREE from 'three'
import { OrbitControls } from 'three/addons'
import { robot_setup, object_setup, robot_animation } from './scene-setup';
import { SceneState } from './state/scene-state';
import { LatencyTracker, LatencyStats } from './latency-tracker';
import { uiController } from './ui-controller';
import { debug } from './debug-logger';
import { CameraStateManager } from './ui/managers/CameraStateManager';

// CP4: New modular architecture (runs in parallel with legacy code)
import { Application } from './core/Application';
import { RenderMode as NewRenderMode } from './types';
import { UISystem } from './systems/UISystem';

// Feature flags removed - new architecture is now the default

import fusionVertexShader from './shaders/fusionVertexShader.vs?raw';
import fusionColorFragmentShader from './shaders/fusionColorShader.fs?raw';
import debugVertexShader from './shaders/debugVertexShader.vs?raw';
import debugFragmentShader from './shaders/debugColorShader.fs?raw';
import depthFusionFragmentShader from './shaders/depthFusionShader.fs?raw';

// Simple shader for displaying WebSocket color texture only (based on fusionColorShader)
// Uses same UV convention and colorspace as Fusion for consistency
const gaussianOnlyFragmentShader = `
  varying vec2 vUv;
  uniform sampler2D wsColorSampler;

  void main() {
    // Same final UV as fusionColorShader (after both flips cancel out X-axis)
    // fusionFlipX=true + wsFlipX=true results in: vec2(vUv.x, 1.0 - vUv.y)
    vec2 wsUv = vec2(vUv.x, 1.0 - vUv.y);
    vec4 wsColor = texture2D(wsColorSampler, wsUv);

    // Use same colorspace conversion as Fusion
    gl_FragColor = linearToOutputTexel(wsColor);
  }
`;

// Simple shader for displaying local color texture only (for LOCAL_ONLY mode)
// Uses same UV flip and colorspace as Fusion for consistency
const localOnlyFragmentShader = `
  varying vec2 vUv;
  uniform sampler2D localColorSampler;

  void main() {
    // Same X-axis flip as Fusion (when fusionFlipX=true: currentUv.x = 1.0 - vUv.x)
    vec2 localUv = vec2(1.0 - vUv.x, vUv.y);
    vec4 localColor = texture2D(localColorSampler, localUv);

    // Use same colorspace conversion as Fusion
    gl_FragColor = linearToOutputTexel(localColor);
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

// wsColorTexture removed - TextureManager manages this
// wsDepthTexture removed - TextureManager manages this

let debugScreen: THREE.Mesh;
let debugMaterial: THREE.ShaderMaterial;
let debugScene: THREE.Scene;
let debugCamera: THREE.OrthographicCamera;

let gaussianOnlyQuad: THREE.Mesh;
let gaussianOnlyMaterial: THREE.ShaderMaterial;
let gaussianOnlyScene: THREE.Scene;
let gaussianOnlyCamera: THREE.OrthographicCamera;

let localOnlyQuad: THREE.Mesh;
let localOnlyMaterial: THREE.ShaderMaterial;
let localOnlyScene: THREE.Scene;
let localOnlyCamera: THREE.OrthographicCamera;

let depthFusionQuad: THREE.Mesh;
let depthFusionMaterial: THREE.ShaderMaterial;
let depthFusionScene: THREE.Scene;
let depthFusionCamera: THREE.OrthographicCamera;

const clock = new THREE.Clock();
// 레이턴시 추적기
const latencyTracker = new LatencyTracker();

// Legacy UI variables removed - now managed by UISystem

// Render mode constants removed - now using RenderMode from types/index.ts

// Legacy variables removed:
// - currentTimeIndex, frameCounter, isPlaying: now managed by Application.timeController
// - Recording variables: now managed by RecordingPanel
// - Texture update tracking: now handled by TextureManager
// - Performance tracking: removed
// - Frame processing and FPS: now handled by LatencyTracker

// Legacy event listeners and functions removed - now managed by UISystem
// Legacy depth texture management removed - now managed by TextureManager

// Legacy camera and recording event listeners removed - now managed by UISystem

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

    // TextureManager handles depth texture recreation automatically via resize()
    // No need to manually recreate depth texture here

    // 모든 셰이더 유니폼 업데이트 (local textures only)
    updateShaderUniforms();

    debug.logMain(`[recreateRenderTargets] Completed for ${rtWidth}×${rtHeight}`);
}

// 모든 셰이더 유니폼 업데이트 (local render target textures only)
// WebSocket textures (wsColorTexture, wsDepthTexture) are managed by TextureManager
function updateShaderUniforms() {
    // Fusion material 업데이트
    if (fusionMaterial) {
        fusionMaterial.uniforms.localColorSampler.value = localRenderTarget.texture;
        fusionMaterial.uniforms.localDepthSampler.value = localDepthTexture;
    }

    // Debug material 업데이트
    if (debugMaterial) {
        debugMaterial.uniforms.localColorSampler.value = localRenderTarget.texture;
        debugMaterial.uniforms.localDepthSampler.value = localDepthTexture;
        debugMaterial.uniforms.width.value = rtWidth;
        debugMaterial.uniforms.height.value = rtHeight;
    }

    // Local-only material 업데이트
    if (localOnlyMaterial) {
        localOnlyMaterial.uniforms.localColorSampler.value = localRenderTarget.texture;
    }

    // Depth fusion material 업데이트
    if (depthFusionMaterial) {
        depthFusionMaterial.uniforms.localColorSampler.value = localRenderTarget.texture;
        depthFusionMaterial.uniforms.localDepthSampler.value = localDepthTexture;
    }
}

// updateCameraAspectRatio() removed - camera aspect ratio is managed automatically by RenderingContext
// updateSizeDisplays() removed - UI displays are managed by UISystem

// 새로운 해상도로 WebSocket 재연결
function reconnectWithNewResolution() {
    debug.logMain('[reconnectWithNewResolution] Reconnecting with new resolution...');

    // 기존 연결 종료
    worker.postMessage({ type: 'ws-close' });

    // Disconnect WebSocket
    if (app) {
        app.disconnectWebSocket();
    }

    // 잠시 대기 후 새 해상도로 재연결
    setTimeout(() => {
        const isJpegMode = uiSystem?.control.isJpegMode() ?? false;
        const wsURL = isJpegMode ?
            'wss://' + location.host + '/ws/jpeg' :
            'wss://' + location.host + '/ws/h264';

        // Legacy worker reconnection
        worker.postMessage({
            type: 'change',
            wsURL: wsURL,
            width: rtWidth,
            height: rtHeight
        });

        // Reconnect new architecture
        if (app) {
            debug.logMain(`[Reconnect] New architecture with ${rtWidth}×${rtHeight}`);
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
    if (localOnlyMaterial) {
        localOnlyMaterial.uniforms.localColorSampler.value = localRenderTarget.texture;
    }
    if (depthFusionMaterial) {
        depthFusionMaterial.uniforms.localColorSampler.value = localRenderTarget.texture;
        depthFusionMaterial.uniforms.localDepthSampler.value = localDepthTexture;
    }
}



// Camera configuration - centralized
const cameraConfig = {
    fov: 80,
    near: 0.1,
    far: 100,
    position: new THREE.Vector3(-3.6, 0.5, -3.6),
    target: new THREE.Vector3(0, 0, 0)
};

// renderStart, renderCnt removed - FPS tracking now in LatencyTracker

let canvas: HTMLCanvasElement;

// CameraBuffer interface removed - not used

let worker = new Worker(new URL("./decode-worker.ts", import.meta.url), { type: "module" })
// workerReady removed - WebSocketSystem handles connection state

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

    // Forward messages to WebSocketSystem
    if (app) {
        const wsSystem = app.getWebSocketSystem();
        if (wsSystem) {
            wsSystem.handleMessage(data);
        }
    }

    if (data.type === "ws-ready") {
        // workerReady state tracking removed - WebSocketSystem handles this
        if (uiSystem) {
            uiSystem.setConnectionState('connected');
        }
        return;
    }

    if (data.type === "ws-error") {
        // workerReady removed - state handled by WebSocketSystem
        if (uiSystem) {
            uiSystem.setConnectionState('error');
        }
        return;
    }

    if (data.type === "ws-close") {
        // workerReady removed - state handled by WebSocketSystem
        if (uiSystem) {
            uiSystem.setConnectionState('closed');
        }
        return;
    }

    if (data.type === 'pure-decode-stats') {
        // 순수 디코딩 성능 통계 처리
        debug.logMain(`[Pure Decode] FPS: ${data.pureFPS.toFixed(2)}, ` +
            `Avg Time: ${data.avgDecodeTime.toFixed(2)}ms, ` +
            `Range: ${data.minDecodeTime.toFixed(2)}-${data.maxDecodeTime.toFixed(2)}ms, ` +
            `Recent Avg: ${data.recentAvg.toFixed(2)}ms`);

        // UI 업데이트 (기존 decode FPS 대신 순수 decode FPS 표시)
        if (uiSystem) {
            uiSystem.updateDecodeFPS(data.pureFPS, data.avgDecodeTime);
        }

        // FPS 측정 중이면 샘플 데이터로 Latency Tracker 업데이트
        if (data.fpsMeasurementData && latencyTracker.isFPSMeasurementActive()) {
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

    // Legacy 'frame' message handling removed - now handled by WebSocketSystem + TextureManager
    // All texture updates are managed by TextureManager.updateFromVideoFrame()
    if (data.type === 'frame' || data.type === 'video-frame') {
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
        // Legacy FPS display removed - now handled by 'pure-decode-stats' message type
    }

    if (data.type === 'error') {
        debug.error("decode-worker error: ", data.error)
    }
}

// Legacy button handler functions removed - now managed by UISystem

// Legacy FPS test functions removed - now handled by UISystem and FPSTestPanel

async function initScene() {
    debug.logMain("Initializing scene")

    // 카메라 aspect ratio를 윈도우 크기에 맞춤
    const windowAspect = window.innerWidth / window.innerHeight;
    camera = new THREE.PerspectiveCamera(
        cameraConfig.fov,
        windowAspect,
        cameraConfig.near,
        cameraConfig.far
    );
    debug.logMain(`[initScene] Camera aspect ratio: ${windowAspect.toFixed(3)} (${rtWidth}×${rtHeight})`);
    debug.logMain(`[initScene] Camera config: fov=${cameraConfig.fov}, near=${cameraConfig.near}, far=${cameraConfig.far}`);

    // Set camera position from config
    camera.position.copy(cameraConfig.position);;
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

    // Set control target from config
    controls.target.copy(cameraConfig.target);

    // Initialize adaptive camera tracking
    lastCameraPosition.copy(camera.position);
    lastCameraTarget.copy(controls.target);

    // Auto-load saved camera position if available
    CameraStateManager.load(camera, controls);

    // UI displays initialization removed - now handled by UISystem

    canvas = renderer.domElement as HTMLCanvasElement

    canvas.style.touchAction = 'none'
    canvas.style.cursor = 'grab'

    // robot_setup();
    object_setup();


    // 워커 초기화 (선택된 해상도로)
    // Note: JPEG mode is determined by UISystem, but at this point UISystem is not yet initialized
    // So we use default H.264 mode here, and it will be changed later if needed
    const initialIsJpegMode = false; // Will be set correctly after UISystem initialization
    if (initialIsJpegMode) {
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

        // UI displays update removed - UISystem handles this automatically
    });

    localDepthTexture = new THREE.DepthTexture(rtWidth, rtHeight);
    localDepthTexture.type = THREE.FloatType;

    localRenderTarget = new THREE.WebGLRenderTarget(rtWidth, rtHeight, {
        depthBuffer: true,
        stencilBuffer: false,
        depthTexture: localDepthTexture,
        samples: 4,
    });

    // wsColorTexture and wsDepthTexture initialization removed
    // TextureManager creates and manages both textures
    // Shader materials will have wsColorSampler and wsDepthSampler set by TextureManager after initialization

    debugMaterial = new THREE.ShaderMaterial({
        uniforms: {
            localColorSampler: { value: localRenderTarget.texture },
            localDepthSampler: { value: localDepthTexture },
            wsColorSampler: { value: null }, // Will be set by TextureManager
            wsDepthSampler: { value: null }, // Will be set by TextureManager
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
            wsColorSampler: { value: null }, // Will be set by TextureManager
            wsDepthSampler: { value: null }, // Will be set by TextureManager
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
            wsColorSampler: { value: null } // Will be set by TextureManager
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

    // Local-only scene setup (displays local color texture with same colorspace as Fusion)
    localOnlyMaterial = new THREE.ShaderMaterial({
        uniforms: {
            localColorSampler: { value: localRenderTarget.texture }
        },
        vertexShader: fusionVertexShader,
        fragmentShader: localOnlyFragmentShader,
        depthTest: false,
        depthWrite: false,
    });

    localOnlyQuad = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), localOnlyMaterial);
    localOnlyScene = new THREE.Scene();
    localOnlyScene.add(localOnlyQuad);
    localOnlyCamera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);

    // Depth fusion scene setup
    depthFusionMaterial = new THREE.ShaderMaterial({
        uniforms: {
            localColorSampler: { value: localRenderTarget.texture },
            localDepthSampler: { value: localDepthTexture },
            wsColorSampler: { value: null }, // Will be set by TextureManager
            wsDepthSampler: { value: null }, // Will be set by TextureManager
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

    // Recording compatibility check removed - now handled by RecordingPanel
}


// sendCameraSnapshot removed - now handled by Application.sendCameraFrame() via CameraController

// Application instance
let app: Application | null = null;
let uiSystem: UISystem | null = null;

initScene().then(async () => {
    // renderStart removed - FPS tracking now in LatencyTracker

    // UI 컨트롤러 활성화
    debug.logMain('UI Controller initialized:', uiController.isVisible())

    // Initialize Application
    try {
        debug.logMain('[Init] Initializing Application...');

        app = new Application({
            canvas: renderer.domElement,
            wsUrl: '', // Will be set when connecting
            width: rtWidth,
            height: rtHeight,
            renderMode: NewRenderMode.Hybrid,
            debugMode: true,
            cameraConfig: cameraConfig, // Pass camera configuration
        });

        // Wrap existing objects with existing worker
        await app.initializeWithExistingObjects(
            localScene,
            camera,
            renderer,
            controls,
            worker
        );

        debug.logMain('[Init] Application initialized successfully');

        // Create and register UISystem
        debug.logMain('[Init] Creating UISystem...');
        uiSystem = new UISystem(app, latencyTracker);

        // Initialize UISystem
        const context = {
            renderingContext: app.getRenderingContext(),
            eventBus: app.getEventBus(),
            state: app.getState(),
        };
        await uiSystem.initialize(context);

        debug.logMain('[Init] UISystem initialized successfully');

        // Register shader materials with RenderingContext
        if (app) {
            const renderingContext = app.getRenderingContext();
            if (renderingContext) {
                if (fusionMaterial) {
                    renderingContext.registerMaterial('fusion', fusionMaterial);
                    debug.logMain('[Init] Registered fusion material');
                }
                if (debugMaterial) {
                    renderingContext.registerMaterial('debug', debugMaterial);
                    debug.logMain('[Init] Registered debug material');
                }
                if (depthFusionMaterial) {
                    renderingContext.registerMaterial('depthFusion', depthFusionMaterial);
                    debug.logMain('[Init] Registered depthFusion material');
                }
                if (gaussianOnlyMaterial) {
                    renderingContext.registerMaterial('gaussianOnly', gaussianOnlyMaterial);
                    debug.logMain('[Init] Registered gaussianOnly material');
                }
                debug.logMain('[Init] All shader materials registered with RenderingContext');
            }

            // Initialize shader materials with textures
            // TextureManager now handles all texture assignment automatically
            const texManager = app.getTextureManager();
            if (texManager) {
                texManager.initializeShaderMaterials();
                debug.logMain('[Init] TextureManager initialized shader materials');
            }
        }

        // Configure RenderingSystem with scenes and cameras
        const renderingSystem = app.getRenderingSystem();
        if (renderingSystem) {
            renderingSystem.configure({
                localScene,
                fusionScene,
                debugScene,
                gaussianOnlyScene,
                localOnlyScene,
                depthFusionScene,
                camera,
                orthoCamera,
                debugCamera,
                gaussianOnlyCamera,
                localOnlyCamera,
                depthFusionCamera,
                localRenderTarget,
                controls,
                clock,
                // Camera update callback - sends data to server
                onCameraUpdate: () => {
                    console.log('[main.ts] onCameraUpdate callback called, app=', !!app);
                    // sendCameraSnapshot removed - app.sendCameraFrame() handles this via CameraController
                    if (app) {
                        console.log('[main.ts] Calling app.sendCameraFrame()');
                        app.sendCameraFrame();
                    } else {
                        console.error('[main.ts] app is null/undefined!');
                    }
                },
                // Per-frame update callback
                onUpdate: () => {
                    // Update time controller for dynamic gaussians
                    if (app) {
                        const deltaTime = clock.getDelta();
                        app.timeController.update(deltaTime);
                    }

                    robot_animation();
                    updateLatencyStats();
                    uiSystem!.updateFPSTestUI();
                    // Camera debug info now updated via DebugPanel.updateCameraInfo()
                },
            });
            debug.logMain('[Init] RenderingSystem configured');
        }

        // Configure PhysicsSystem (optional - for testing collision detection)
        const physicsSystem = app.getPhysicsSystem();
        if (physicsSystem && true) { // Set to true to enable physics test
            debug.logMain('[Init] Configuring PhysicsSystem...');

            // Example: Create a test sphere that falls and collides with Gaussian scene
            const testSphere = new THREE.Mesh(
                new THREE.SphereGeometry(0.5, 32, 32),
                new THREE.MeshStandardMaterial({
                    color: 0xff0000,
                    metalness: 0.5,
                    roughness: 0.5
                })
            );
            testSphere.position.set(0, 0.3, -3.2); // Start high above ground
            testSphere.castShadow = true;
            localScene.add(testSphere);

            // Add to physics with gravity
            app.addPhysicsMesh(testSphere, {
                velocity: new THREE.Vector3(0, 0, 0),
                acceleration: new THREE.Vector3(0, -9.8, 0), // Gravity
                mass: 1.0,
                restitution: 0.7, // 70% bounce
                friction: 0.3     // 30% friction
            });

            // Set collision response type
            app.setPhysicsResponseType("stop"); // Options: "stop", "bounce", "slide"

            // Adjust collision threshold (5cm tolerance)
            app.setCollisionEpsilon(0.05);

            debug.logMain('[Init] PhysicsSystem configured with test sphere');
        }

        // Start the render loop
        app.start();
        debug.logMain('[Init] Render loop started');

    } catch (error) {
        console.error('[Init] Failed to initialize:', error);
    }
})



// getCameraIntrinsics removed - now handled by CameraController.getCameraIntrinsics()

// Legacy stats update throttling removed - now handled by UISystem if needed

function updateLatencyStats() {
    if (!uiSystem) return;

    const stats = latencyTracker.getRecentStats(50);
    const clockOffset = latencyTracker.getClockOffset();

    // Delegate to UISystem
    uiSystem.updateLatencyStats(stats, clockOffset);
}

// updateFPSTestUI and updateCameraDebugInfo removed - now handled by UISystem

// downloadFPSResults and generateFPSReportText removed - now handled by FPSTestPanel

// Legacy recording and frame processing functions removed - now handled by RecordingPanel
// Note: Frame processing metrics are now tracked differently via LatencyTracker

// ============================================================================
// Camera Configuration Helpers (for console debugging)
// ============================================================================

/**
 * Update camera configuration at runtime
 * Usage in console:
 *   updateCameraConfig({ near: 0.1, far: 200 })
 *   updateCameraConfig({ fov: 90 })
 */
(window as any).updateCameraConfig = function (params: any) {
    if (!app) {
        console.error('[CameraConfig] Application not initialized');
        return;
    }

    console.log('[CameraConfig] Updating camera config:', params);
    app.updateCameraConfig(params);

    // Log current config
    const camera = app.getCameraController()?.getCamera();
    if (camera) {
        console.log('[CameraConfig] Current camera params:', {
            fov: camera.fov,
            near: camera.near,
            far: camera.far,
            aspect: camera.aspect
        });
    }
};

/**
 * Get current camera configuration
 * Usage in console:
 *   getCameraConfig()
 */
(window as any).getCameraConfig = function () {
    if (!app) {
        console.error('[CameraConfig] Application not initialized');
        return null;
    }

    const config = app.getCameraController()?.getConfig();
    console.log('[CameraConfig] Current config:', config);
    return config;
};

/**
 * Reset camera to initial config
 * Usage in console:
 *   resetCameraConfig()
 */
(window as any).resetCameraConfig = function () {
    if (!app) {
        console.error('[CameraConfig] Application not initialized');
        return;
    }

    console.log('[CameraConfig] Resetting to initial config:', cameraConfig);
    app.updateCameraConfig({
        fov: cameraConfig.fov,
        near: cameraConfig.near,
        far: cameraConfig.far
    });
};

console.log('[main.ts] Camera config helpers available:');
console.log('  - updateCameraConfig({ near, far, fov })');
console.log('  - getCameraConfig()');
console.log('  - resetCameraConfig()');