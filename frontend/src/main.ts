import * as THREE from 'three'
import { OrbitControls } from 'three/addons'
import { robot_setup, object_setup, robot_animation } from './scene-setup';
import { SceneState } from './state/scene-state';
import { LatencyTracker, LatencyStats } from './latency-tracker';
import { uiController } from './ui-controller';
import { debug } from './debug-logger';
import { CameraStateManager } from './ui/managers/CameraStateManager';

import { Application } from './core/Application';
import { RenderMode as NewRenderMode } from './types';
import { UISystem } from './systems/UISystem';

import fusionVertexShader from './shaders/fusionVertexShader.vs?raw';
import fusionColorFragmentShader from './shaders/fusionColorShader.fs?raw';
import debugVertexShader from './shaders/debugVertexShader.vs?raw';
import debugFragmentShader from './shaders/debugColorShader.fs?raw';
import depthFusionFragmentShader from './shaders/depthFusionShader.fs?raw';
import gaussianOnlyFragmentShader from './shaders/gaussianOnlyShader.fs?raw';
import localOnlyFragmentShader from './shaders/localOnlyShader.fs?raw';

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

// 새로운 해상도로 WebSocket 재연결
// Phase 6: All WebSocket communication goes through WebSocketSystem
function reconnectWithNewResolution() {
    debug.logMain('[reconnectWithNewResolution] Reconnecting with new resolution...');

    if (!app) {
        debug.warn('[reconnectWithNewResolution] Application not initialized');
        return;
    }

    const wsSystem = app.getWebSocketSystem();
    if (!wsSystem) {
        debug.warn('[reconnectWithNewResolution] WebSocketSystem not available');
        return;
    }

    const isJpegMode = uiSystem?.control.isJpegMode() ?? false;
    const wsURL = isJpegMode ?
        'wss://' + location.host + '/ws/jpeg' :
        'wss://' + location.host + '/ws/h264';

    debug.logMain(`[reconnectWithNewResolution] Reconnecting with ${rtWidth}×${rtHeight}`);
    wsSystem.reconnect(wsURL, rtWidth, rtHeight);
}

// Camera configuration - centralized
const cameraConfig = {
    fov: 80,
    near: 0.1,
    far: 100,
    position: new THREE.Vector3(-3.6, 0.5, -3.6),
    target: new THREE.Vector3(0, 0, 0)
};

let canvas: HTMLCanvasElement;
// Phase 6: Worker is created internally by WebSocketSystem - no longer needed in main.ts

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

    camera.position.copy(cameraConfig.position);

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

    canvas = renderer.domElement as HTMLCanvasElement

    canvas.style.touchAction = 'none'
    canvas.style.cursor = 'grab'

    // robot_setup();
    object_setup();

    // Phase 6: Initial WebSocket connection is deferred to after Application initialization
    // Connection will be established via WebSocketSystem.connect() in initScene().then()

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
    });

    localDepthTexture = new THREE.DepthTexture(rtWidth, rtHeight);
    localDepthTexture.type = THREE.FloatType;

    localRenderTarget = new THREE.WebGLRenderTarget(rtWidth, rtHeight, {
        depthBuffer: true,
        stencilBuffer: false,
        depthTexture: localDepthTexture,
        samples: 4,
    });

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
}

// Application instance
let app: Application | null = null;
let uiSystem: UISystem | null = null;

initScene().then(async () => {
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

        // Wrap existing objects - worker is created internally by WebSocketSystem
        await app.initializeWithExistingObjects(
            localScene,
            camera,
            renderer,
            controls
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

        // Configure WebSocketSystem callbacks for UI updates and latency tracking
        const wsSystem = app.getWebSocketSystem();
        if (wsSystem) {
            wsSystem.setCallbacks({
                onConnectionStateChange: (state) => {
                    if (uiSystem) {
                        uiSystem.setConnectionState(state);
                    }
                },
                onDecodeStats: (stats) => {
                    debug.logMain(`[Pure Decode] FPS: ${stats.pureFPS.toFixed(2)}, ` +
                        `Avg Time: ${stats.avgDecodeTime.toFixed(2)}ms, ` +
                        `Range: ${stats.minDecodeTime.toFixed(2)}-${stats.maxDecodeTime.toFixed(2)}ms, ` +
                        `Recent Avg: ${stats.recentAvg.toFixed(2)}ms`);

                    if (uiSystem) {
                        uiSystem.updateDecodeFPS(stats.pureFPS, stats.avgDecodeTime);
                    }

                    // FPS measurement mode
                    if (stats.fpsMeasurementData && latencyTracker.isFPSMeasurementActive()) {
                        latencyTracker.recordPureDecodeFPSSample(
                            stats.fpsMeasurementData.totalCount,
                            stats.fpsMeasurementData.avgTime
                        );
                        debug.logFPS(`Recording decode sample: ${stats.fpsMeasurementData.totalCount} frames, ${stats.fpsMeasurementData.avgTime.toFixed(2)}ms avg`);
                    } else {
                        latencyTracker.recordPureDecodeFPS(stats.totalFrames, stats.avgDecodeTime);
                    }
                },
                onFrameReceive: (frameId, serverTimestamps) => {
                    latencyTracker.recordFrameReceive(frameId, serverTimestamps);
                },
                onFrameDecoded: (frameId, _decodeCompleteTime) => {
                    latencyTracker.recordDecodeComplete(frameId);
                    requestAnimationFrame(() => {
                        latencyTracker.recordRenderComplete(frameId);
                    });
                },
                onClockSync: (clientRequestTime, serverReceiveTime, serverSendTime) => {
                    latencyTracker.recordClockSync(clientRequestTime, serverReceiveTime, serverSendTime);
                },
            });
            debug.logMain('[Init] WebSocketSystem callbacks configured');

            // Phase 6: Initial WebSocket connection (moved from initScene)
            // Default to H.264 mode; UISystem will reconnect if JPEG mode is selected later
            const initialWsURL = 'wss://' + location.host + '/ws/h264';
            wsSystem.connect(initialWsURL, rtWidth, rtHeight);
            debug.logMain(`[Init] Initial WebSocket connection started with ${rtWidth}×${rtHeight}`);
        }

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
                    if (app) {
                        app.sendCameraFrame();
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
                },
            });
            debug.logMain('[Init] RenderingSystem configured');
        }

        // Start the render loop
        app.start();
        debug.logMain('[Init] Render loop started');

    } catch (error) {
        console.error('[Init] Failed to initialize:', error);
    }
})

function updateLatencyStats() {
    if (!uiSystem) return;

    const stats = latencyTracker.getRecentStats(50);
    const clockOffset = latencyTracker.getClockOffset();

    // Delegate to UISystem
    uiSystem.updateLatencyStats(stats, clockOffset);
}

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