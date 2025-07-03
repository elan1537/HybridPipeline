// main.ts
import * as THREE from 'three';
import {
    OrbitControls,
    TransformControls,
    GLTFLoader,
    ColladaLoader,
    FirstPersonControls,
} from 'three/addons';


import { object_setup, robot_setup, robot_animation } from './scene-setup';
import { SceneState } from './state/scene-state';
import { getCameraIntrinsics } from './state/render-state';

// --- 디코딩 경로 선택 플래그 ---
export const USE_H264_DECODER = true; // true: H.264 VideoDecoder 사용, false: 기존 JPEG 워커 사용
export const USE_RAW_RGB = false; // true: RAW RGB 데이터 사용, false: 기존 JPEG/H.264 사용

// 1분 평균 클라이언트 측 시간 측정을 위한 유틸리티 객체
const perfMonitor = {
    timings: {
        total_onmessage: 0.0,
        header_parse: 0.0,
        color_processing: 0.0,    // JPEG 경로 측정용
        h264_decode_call: 0.0,      // H.264 경로 측정용
        raw_rgb_processing: 0.0,    // RAW RGB 경로 측정용
        depth_processing: 0.0,
        fusion_tex_update: 0.0,
    },
    frameCount: 0,
    lastPrintTime: 0,

    init() {
        this.lastPrintTime = performance.now();
    },

    reset() {
        this.frameCount = 0;
        for (const key in this.timings) {
            this.timings[key] = 0.0;
        }
        this.lastPrintTime = performance.now();
    },

    record(stats: { [key: string]: number }) {
        for (const key in stats) {
            if (key in this.timings) {
                this.timings[key] += stats[key];
            }
        }
        this.frameCount++;
    },

    logStatsIfReady(periodInMs = 1000) { // 1분
        const now = performance.now();
        if (now - this.lastPrintTime > periodInMs) {
            if (this.frameCount > 0) {
                console.log(`--- Client Performance Stats (avg over ${this.frameCount} frames) ---`);
                for (const key in this.timings) {
                    const avgTime = this.timings[key] / this.frameCount;
                    console.log(`  ${key}: ${avgTime.toFixed(3)} ms`);
                }
                console.log('--------------------------------------------------');
            }
            this.reset();
        }
    }
};
perfMonitor.init();

let camera: THREE.PerspectiveCamera;;
let renderer: THREE.WebGLRenderer;
let controls: OrbitControls | FirstPersonControls;
let transform: TransformControls;

let wsColorTexture: THREE.Texture;
let wsDepthTexture: THREE.DataTexture;
let wsColorVisQuad: THREE.Mesh;
let wsDepthVisQuad: THREE.Mesh;
let wsColorNode: THREE.Texture;
let wsDepthNode: THREE.Texture;

// --- ImageBitmap based pipeline (after YUV→RGB conversion) ---
let imageBitmapOnTexture: ImageBitmap | null = null;

// Local Renderer 텍스처 관련 변수
let localRenderTarget: THREE.WebGLRenderTarget;
let localDepthTexture: THREE.DepthTexture;
let localColorVisQuad: THREE.Mesh;
let localDepthVisQuad: THREE.Mesh;
let localColorNode: THREE.Texture;
let localDepthNode: THREE.Texture;

let device: GPUDevice;

// Fusion 패스 관련 변수
let fusionQuad: THREE.Mesh;
let fusionMaterial: THREE.ShaderMaterial;
let fusedDepthVisQuad: THREE.Mesh;
let fusedDepthVisMaterial: THREE.ShaderMaterial;

// let imageDecoder; // TODO: ImageDecoder 타입 정의 필요

let stats;
let loggingEnabled = true;
let quad: THREE.Mesh;

let gl;
let ext

const queryPool: WebGLQuery[] = [];
const pendingQueries: WebGLQuery[] = []; // FIFO of in‑flight queries

let fusionScene: THREE.Scene; // fusionQuad를 담을 씬
let orthoCamera: THREE.OrthographicCamera; // fusionQuad 렌더링용 카메라

const near = 0.1;
const far = 50;

let rtWidth = window.innerWidth;
let rtHeight = window.innerHeight;
const dpr = window.devicePixelRatio;

// 파일 상단 또는 적절한 스코프에 추가
const CLIENT_ASSUMED_SCALE_Y = 1.0; // 서버에서 가정한 값과 동일하게

let ws: WebSocket;
let gs_index_time = 0;
let animationStartTime: number;
let readbackRenderTarget: THREE.RenderTarget;
let materialFX: THREE.MeshBasicMaterial;

// main.ts 상단 또는 적절한 스코프에 추가
let latestCapturedDepthPixels: Float32Array | null = null;
let capturedDepthWidth: number = 0;
let capturedDepthHeight: number = 0;

let prev_time = Date.now();

let buf: ArrayBuffer;
let floatView: Float32Array;
let legacyWay = true;
const clock = new THREE.Clock(); // Clock 추가

// Vertex Shader (공통으로 사용)
const fusionVertexShader = `
  varying vec2 vUv;
  void main() {
    vUv = uv;
    gl_Position = vec4(position.xy, 0.0, 1.0); // 클립 공간으로 직접 출력
  }
`;

// Fragment Shader for Color Fusion
const fusionColorFragmentShader = `
  uniform sampler2D localColorSampler;
  uniform sampler2D localDepthSampler;
  uniform sampler2D wsColorSampler;
  uniform sampler2D wsDepthSampler;
  varying vec2 vUv;

  void main() {
    vec2 wsColorUv = vec2(1.0 - vUv.x, 1.0 - vUv.y);
    vec2 wsDepthUv = vec2(1.0 - vUv.x, 1.0 - vUv.y);

    vec4 colorLocal = texture2D(localColorSampler, vUv);
    float depthLocal = texture2D(localDepthSampler, vUv).r; // [0, 1] 범위로 가정 (0=near, 1=far)

    vec4 colorWSFull = texture2D(wsColorSampler, wsColorUv);
    float depthWS_ndc = texture2D(wsDepthSampler, wsDepthUv).r; // [-1, 1] NDC 범위 (서버에서 전송)
    
    float depthWS = (depthWS_ndc + 1.0) * 0.5; // [0, 1] 범위로 변환

    vec4 colorWS = colorWSFull;

    bool localValid = (depthLocal == depthLocal) && (depthLocal < 0.999); // far plane에 매우 가까운 값 제외
    bool wsValid = (depthWS == depthWS) && (depthWS < 0.999); // far plane에 매우 가까운 값 제외

    bool useLocal = localValid && (!wsValid || (depthLocal <= depthWS)); // 더 작은 뎁스 값(가까운 쪽) 우선

    vec4 backgroundColor = vec4(0.0, 0.0, 0.0, 0.0); // 투명 배경
    vec4 finalColor = useLocal ? colorLocal : colorWS;
    vec4 outputColor = (localValid || wsValid) ? finalColor : backgroundColor;

    gl_FragColor = linearToOutputTexel(outputColor);
  }
`;

// Fragment Shader for Fused Depth Visualization (fusionDepthMaterialFn 대체)
const fusionDepthVisFragmentShader = `
  uniform sampler2D localDepthSampler;
  uniform sampler2D wsDepthSampler;
  varying vec2 vUv;

  void main() {
    vec2 wsDepthUv = vec2(1.0 - vUv.x, 1.0 - vUv.y); // 원래 TSL 로직

    float depthLocal = texture2D(localDepthSampler, vUv).r; // [0, 1] 범위
    float depthWS_ndc = texture2D(wsDepthSampler, wsDepthUv).r; // [-1, 1] NDC 범위

    float depthWS = (depthWS_ndc + 1.0) * 0.5; // [0, 1] 범위로 변환

    bool localValid = (depthLocal == depthLocal) && (depthLocal < 0.999);
    bool wsValid = (depthWS == depthWS) && (depthWS < 0.999);

    bool useLocal = localValid && (!wsValid || (depthLocal <= depthWS));

    float finalDepthValue = useLocal ? depthLocal : depthWS; 
    float outputDepth = (localValid || wsValid) ? finalDepthValue : 1.0;

    float visualizedDepth = clamp(outputDepth, 0.0, 1.0);
    gl_FragColor = vec4(visualizedDepth, visualizedDepth, visualizedDepth, 1.0);
  }
`;

// WebSocket 메시지 처리 워커 인스턴스 생성
const wsProcessorWorker = new Worker(new URL('./websocket-processor.worker.ts', import.meta.url), { type: 'module' });

// JPEG 디코더 워커 인스턴스 생성 (기존 코드와의 호환성을 위해 유지)
const jpegWorker = new Worker(new URL('./jpeg-decoder.worker.ts', import.meta.url), { type: 'module' });

function updateFusionMaterialTextures() {
    if (!fusionMaterial || !fusedDepthVisMaterial) return;

    if (fusionMaterial && fusionMaterial instanceof THREE.ShaderMaterial) {
        fusionMaterial.uniforms.localColorSampler.value = localRenderTarget.texture;
        fusionMaterial.uniforms.localDepthSampler.value = localDepthTexture;
        fusionMaterial.uniforms.wsColorSampler.value = wsColorTexture;
        fusionMaterial.uniforms.wsDepthSampler.value = wsDepthTexture;
        fusionMaterial.needsUpdate = true; // 유니폼 변경 시 필요할 수 있음
    }

    if (fusedDepthVisMaterial && fusedDepthVisMaterial instanceof THREE.ShaderMaterial) {
        fusedDepthVisMaterial.uniforms.localDepthSampler.value = localDepthTexture;
        fusedDepthVisMaterial.uniforms.wsDepthSampler.value = wsDepthTexture;
        fusedDepthVisMaterial.needsUpdate = true; // 유니폼 변경 시 필요할 수 있음
    }
}

// Improved frame synchronization to reduce flickering
function updateWsColorTextureFromImageBitmap(imageBitmap: ImageBitmap) {
    // 1. release old bitmap tied to the last rendered texture
    if (imageBitmapOnTexture) {
        imageBitmapOnTexture.close();
        imageBitmapOnTexture = null;
    }

    // 2. no new bitmap → nothing to do
    if (!imageBitmap) return;

    const bmp = imageBitmap;

    // 3. (re)use or recreate texture with improved settings
    const reusable =
        wsColorTexture &&
        wsColorTexture.image &&
        wsColorTexture.image.width === bmp.width &&
        wsColorTexture.image.height === bmp.height;

    if (reusable) {
        wsColorTexture.image = bmp;
        wsColorTexture.needsUpdate = true;
    } else {
        if (wsColorTexture) {
            if (wsColorTexture.image instanceof ImageBitmap)
                wsColorTexture.image.close();
            wsColorTexture.dispose();
        }
        wsColorTexture = new THREE.Texture(bmp);
        wsColorTexture.colorSpace = THREE.SRGBColorSpace;
        wsColorTexture.generateMipmaps = false;
        wsColorTexture.magFilter = THREE.LinearFilter; // 노이즈 감소를 위한 필터링
        wsColorTexture.minFilter = THREE.LinearFilter;
        wsColorTexture.needsUpdate = true;
    }

    // 4. track bitmap currently used by GPU so we can dispose next frame
    imageBitmapOnTexture = bmp;
}

// JPEG 디코딩을 위한 헬퍼 함수
async function decodeJPEG(jpegBlob: Blob): Promise<ImageBitmap> {
    jpegWorker.postMessage(jpegBlob);
    return new Promise<ImageBitmap>((resolve, reject) => {
        const workerListener = (messageEvent: MessageEvent) => {
            jpegWorker.removeEventListener('message', workerListener);
            if (messageEvent.data.success) {
                resolve(messageEvent.data.imageBitmap);
            } else {
                reject(new Error(messageEvent.data.error || 'JPEG decoding failed in worker'));
            }
        };
        jpegWorker.addEventListener('message', workerListener);
    });
}

// WS 컬러 텍스처 업데이트를 위한 헬퍼 함수
function updateWsColorTexture(imageBitmap: ImageBitmap) {
    updateWsColorTextureFromImageBitmap(imageBitmap);
}

// RAW RGB 텍스처 업데이트를 위한 헬퍼 함수
function updateWsColorTextureFromRawRGB(rgbArrayBuffer: ArrayBuffer) {
    const rgbData = new Uint8Array(rgbArrayBuffer);
    const expectedServerWidth = rtWidth;
    const expectedServerHeight = Math.round(rtHeight * (1.0 / CLIENT_ASSUMED_SCALE_Y));
    const expectedRGBSize = expectedServerWidth * expectedServerHeight * 3; // RGB 3채널
    const expectedRGBASize = expectedServerWidth * expectedServerHeight * 4; // RGBA 4채널

    // 서버에서 실제로 전송하는 크기에 따라 처리
    // if (rgbData.length === expectedRGBSize) {
    //   console.log(`[DEBUG] Processing RGB 3-channel data: ${rgbData.length} bytes`);
    // } else if (rgbData.length === expectedRGBASize) {
    //   console.log(`[DEBUG] Processing RGBA 4-channel data: ${rgbData.length} bytes`);
    // } else {
    //   console.error(`RAW RGB size mismatch: expected ${expectedRGBSize} (RGB) or ${expectedRGBASize} (RGBA), got ${rgbData.length}`);
    //   return;
    // }

    // 기존 텍스처가 있다면 메모리에서 완전 해제
    if (wsColorTexture) {
        if (wsColorTexture.image instanceof ImageBitmap) {
            wsColorTexture.image.close();
        }
        wsColorTexture.dispose();
    }

    // 새 텍스처 생성 (RAW RGBA 데이터)
    const textureFormat = rgbData.length === expectedRGBASize ? THREE.RGBAFormat : THREE.RGBFormat;
    wsColorTexture = new THREE.DataTexture(
        rgbData,
        expectedServerWidth,
        expectedServerHeight,
        textureFormat,
        THREE.UnsignedByteType
    );
    wsColorTexture.colorSpace = THREE.SRGBColorSpace;
    wsColorTexture.generateMipmaps = false;
    wsColorTexture.needsUpdate = true;
}

// WS 뎁스 텍스처 업데이트를 위한 헬퍼 함수
function updateWsDepthTexture(depthArrayBuffer: ArrayBuffer) {
    const nonlinearDepth = new Uint16Array(depthArrayBuffer);
    const serverCorrectionY = 1.0 / CLIENT_ASSUMED_SCALE_Y;
    const expectedServerHeight = Math.round(rtHeight * serverCorrectionY);
    const expectedServerWidth = rtWidth;

    // 가드: 텍스처가 이미 존재하고 크기가 같다면, 데이터만 업데이트하고 종료 (가장 흔한 케이스)
    const isTextureReusable = wsDepthTexture &&
        wsDepthTexture.image.width === expectedServerWidth &&
        wsDepthTexture.image.height === expectedServerHeight &&
        wsDepthTexture.image.data.byteLength === nonlinearDepth.byteLength;

    if (isTextureReusable) {
        (wsDepthTexture.image.data as Uint16Array).set(nonlinearDepth);
        wsDepthTexture.needsUpdate = true;
        return; // 여기서 함수 종료
    }

    // 위 조건들을 통과했다면, 텍스처를 무조건 새로 생성해야 함 (최초 생성 또는 사이즈 변경)

    // 기존 텍스처가 있다면 메모리에서 완전 해제
    if (wsDepthTexture) {
        wsDepthTexture.dispose();
    }

    // 새 텍스처 생성
    wsDepthTexture = new THREE.DataTexture(
        nonlinearDepth,
        expectedServerWidth,
        expectedServerHeight,
        THREE.RedFormat,
        THREE.HalfFloatType
    );
    wsDepthTexture.magFilter = THREE.LinearFilter;
    wsDepthTexture.minFilter = THREE.LinearFilter;
    wsDepthTexture.needsUpdate = true;
}

// 데이터 파싱 함수는 Worker로 이동했으므로 제거

// 함수 실행 시간을 측정하고 기록하는 헬퍼 함수
async function timeExecution(
    fn: () => Promise<void> | void,
    timings: { [key: string]: number },
    key: string
) {
    const start = performance.now();
    await fn();
    timings[key] = performance.now() - start;
}

// H.264 디코딩 함수들은 Worker로 이동했으므로 제거

async function initWebSocket() {
    ws = new WebSocket('wss://' + location.host + '/ws');
    console.log('ws: ', ws);
    console.log('Attempting WebSocket connection...');

    ws.binaryType = 'arraybuffer';
    ws.onopen = () => {
        console.log('Websocket connection opened');

        const buf = new ArrayBuffer(4);
        new DataView(buf).setUint16(0, rtWidth, true);
        new DataView(buf).setUint16(2, rtHeight, true);
        ws.send(buf);


        if (USE_H264_DECODER && !(self as any).USE_H264_FALLBACK_DISABLED) {
            console.log(`[DEBUG] Initializing H.264 decoder...`);
            initializeVideoDecoder();
        }

        // Notify worker that the WebSocket is now open so it can pre‑allocate decoders
        wsProcessorWorker.postMessage({ type: 'ws_open' });
    };

    // Worker에 설정 전달
    wsProcessorWorker.postMessage({
        type: 'config',
        config: {
            USE_H264_DECODER,
            USE_RAW_RGB,
            CLIENT_ASSUMED_SCALE_Y,
            rtWidth,
            rtHeight
        }
    });

    // Worker로부터 처리된 프레임 수신
    wsProcessorWorker.onmessage = async (event) => {
        const { type, processedFrame, error, stats } = event.data;

        if (type === 'error') {
            console.error('Worker error:', error);
            return;
        }

        // if (type === 'performance_stats') {
        //   console.log('--- Worker Performance Stats Received ---');
        //   console.log(`Frame count: ${stats.frameCount}`);
        //   for (const key in stats.timings) {
        //     const avgTime = stats.timings[key] / stats.frameCount;
        //     console.log(`  ${key}: ${avgTime.toFixed(3)} ms`);
        //   }
        //   console.log('----------------------------------------');
        //   return;
        // }

        if (type === 'processed_frame') {
            // console.log(`[DEBUG] Received processed frame from worker`);
            const t_onmessage_start = performance.now();

            // 디버깅: depthNDC 상태 확인
            console.log(`[DEBUG] processedFrame.depthNDC:`, processedFrame.depthNDC ? 'defined' : 'undefined');
            console.log(`[DEBUG] processedFrame.depthData:`, processedFrame.depthData ? `${processedFrame.depthData.byteLength} bytes` : 'undefined');

            // 1. 컬러 데이터 처리
            if (processedFrame.colorImageBitmap) {
                // console.log(`[DEBUG] Processing color ImageBitmap`);
                updateWsColorTexture(processedFrame.colorImageBitmap);
            } else if (processedFrame.colorRawRGB) {
                // console.log(`[DEBUG] Processing raw RGB data`);
                updateWsColorTextureFromRawRGB(processedFrame.colorRawRGB);
            }

            // 2. 뎁스 데이터 처리
            try {
                if (processedFrame.depthNDC) {
                    // HEVC 디코딩된 NDC 데이터 사용
                    console.log(`[DEBUG] Using decoded depthNDC data`);
                    updateWsDepthTexture(processedFrame.depthNDC);
                } else {
                    // 원본 depthData 사용 (기존 로직)
                    console.log(`[DEBUG] Using original depthData, falling back to legacy processing`);
                    updateWsDepthTexture(processedFrame.depthData);
                }
            } catch (e) {
                console.error('Depth processing error:', e);
            }

            // 3. 퓨전 텍스처 업데이트
            updateFusionMaterialTextures();

            // 4. 성능 통계 기록 (Worker에서 받은 타이밍 사용)
            const frameTimings = {
                ...processedFrame.frameTimings,
                fusion_tex_update: performance.now() - t_onmessage_start
            };
            frameTimings.total_onmessage = performance.now() - t_onmessage_start;
            perfMonitor.record(frameTimings);
        }
    };

    ws.onmessage = async event => {
        // console.log(`[DEBUG] Received WebSocket message, size: ${event.data.byteLength} bytes`);

        // Worker로 프레임 데이터 전송
        wsProcessorWorker.postMessage({
            type: 'frame',
            data: event.data
        }, [event.data]); // ArrayBuffer를 transfer로 전송
    };

    ws.onerror = e => console.error('WebSocket Error:', e);
    ws.onclose = e => {
        console.log('Websocket closed:', e.code, e.reason);
    };
}


async function main() {
    animationStartTime = Date.now(); // animationStartTime을 Date.now()로 초기화

    camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, near, far);

    if (!legacyWay) {
        let worldScene = new THREE.Scene();
        let scene = new THREE.Group();

        worldScene.add(scene);
        worldScene.background = new THREE.Color(0x112233);

        const initialViewMat = new THREE.Matrix4().set(
            -0.9681401252746582,
            0.01763979159295559,
            0.24978689849376678,
            0.0,
            0.01678032986819744,
            0.9998436570167542,
            -0.0055700428783893585,
            0.0,
            0.24984610080718994,
            0.0012010756181553006,
            0.9682847857475281,
            0.0,
            -4.818760395050049,
            -0.011148083955049515,
            -5.045684576034546,
            1.0
        );

        const targetMatrix = new THREE.Matrix4().set(1, 0, 0, -3, 0, 1, 0, 0, 0, 0, 1, -22.5, 0, 0, 0, 1);

        camera.applyMatrix4(initialViewMat);
        scene.applyMatrix4(initialViewMat.multiply(targetMatrix));

        // 싱글톤으로 저장
        SceneState.worldScene = worldScene;
        SceneState.scene = scene;
    } else {
        let scene = new THREE.Scene();
        // camera.position.set(2.91, 0.82, -1.93);
        // camera.position.set(-4.38, 2.11, -4.13);
        camera.position.set(2, 1, 2)

        // 싱글톤으로 저장
        SceneState.scene = scene;

        // camera.position.set(-3.9133210842648842, 0.4901867402292127, -4.823079192768786)
        // camera.lookAt(0.24063812947078528, 0.18225546204825407, -0.2190283792363025);
    }

    object_setup();
    // robot_setup();

    renderer = new THREE.WebGLRenderer({
        antialias: true,
        depth: true,
    });
    renderer.setPixelRatio(dpr);
    renderer.setSize(window.innerWidth, window.innerHeight);
    // renderer.getContext().pixelStorei(renderer.getContext().UNPACK_ALIGNMENT, 1)

    gl = renderer.getContext()
    ext = gl.getExtension('EXT_disjoint_timer_query_webgl2');


    document.body.appendChild(renderer.domElement);

    if (legacyWay) {
        controls = new OrbitControls(camera, renderer.domElement);
        controls.enableDamping = true;
        controls.dampingFactor = 0.1;
        controls.autoRotate = true;
        controls.autoRotateSpeed = 1
        // controls.target.set(0.24063812947078528, 0.18225546204825407, -0.2190283792363025);

        controls.update();
    } else {
        controls = new FirstPersonControls(camera, renderer.domElement);
        controls.lookSpeed = 0.01; // 마우스 감도 조절 (기본값: 0.005). 값을 높이면 더 민감해집니다.
        controls.movementSpeed = 1; // 이동 속도 조절 (기본값: 1)
        controls.autoForward = false;
        controls.activeLook = true;
        console.log(controls.mouseDragOn);

        if (controls instanceof FirstPersonControls) {
            controls.activeLook = false; // 초기에는 마우스 이동으로 화면 회전 비활성화

            renderer.domElement.addEventListener('mousedown', () => {
                if (controls instanceof FirstPersonControls) {
                    controls.activeLook = true;
                }
            });

            renderer.domElement.addEventListener('mouseup', () => {
                if (controls instanceof FirstPersonControls) {
                    controls.activeLook = false;
                }
            });

            renderer.domElement.addEventListener('mouseleave', () => {
                if (controls instanceof FirstPersonControls && controls.activeLook) {
                    controls.activeLook = false;
                }
            });
        }
    }

    localDepthTexture = new THREE.DepthTexture(rtWidth * dpr, rtHeight * dpr);
    localDepthTexture.type = THREE.FloatType;

    localRenderTarget = new THREE.WebGLRenderTarget(rtWidth * dpr, rtHeight * dpr, {
        depthBuffer: true,
        stencilBuffer: false,
        depthTexture: localDepthTexture,
        samples: 4,
    });

    materialFX = new THREE.MeshBasicMaterial();
    materialFX.map = localDepthTexture;
    quad = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), materialFX);

    const localVisMaterialColor = new THREE.MeshBasicMaterial();
    localVisMaterialColor.map = localRenderTarget.texture;
    localColorVisQuad = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), localVisMaterialColor);

    const localVisMaterialDepth = new THREE.MeshBasicMaterial();
    localVisMaterialDepth.map = localDepthTexture;
    localDepthVisQuad = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), localVisMaterialDepth);

    const wsVisMaterialColor = new THREE.MeshBasicMaterial();
    wsVisMaterialColor.map = new THREE.Texture();
    wsColorVisQuad = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), wsVisMaterialColor);

    const wsVisMaterialDepth = new THREE.MeshBasicMaterial();
    wsVisMaterialDepth.map = new THREE.Texture();
    wsDepthVisQuad = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), wsVisMaterialDepth);

    fusionMaterial = new THREE.ShaderMaterial({
        uniforms: {
            localColorSampler: { value: null },
            localDepthSampler: { value: null },
            wsColorSampler: { value: null },
            wsDepthSampler: { value: null },
        },
        vertexShader: fusionVertexShader,
        fragmentShader: fusionColorFragmentShader,
        depthTest: false,
        depthWrite: false,
        transparent: true,
    });

    fusedDepthVisMaterial = new THREE.ShaderMaterial({
        uniforms: {
            localDepthSampler: { value: null },
            wsDepthSampler: { value: null },
        },
        vertexShader: fusionVertexShader,
        fragmentShader: fusionDepthVisFragmentShader,
        depthTest: false,
        depthWrite: false,
    });

    fusionQuad = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), fusionMaterial);
    fusedDepthVisQuad = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), fusedDepthVisMaterial);

    fusionScene = new THREE.Scene();
    fusionScene.add(fusionQuad);
    orthoCamera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);

    updateFusionMaterialTextures();

    window.addEventListener('resize', onWindowResize);

    animate();
}

function onWindowResize() {
    const width = window.innerWidth;
    const height = window.innerHeight;

    rtWidth = width;
    rtHeight = height;

    camera.aspect = width / height;
    camera.updateProjectionMatrix();

    renderer.setSize(width, height);
    localRenderTarget.setSize(width * dpr, height * dpr);

    // localDepthTexture는 WebGLRenderTarget에 의해 내부적으로 관리되므로
    // 수동으로 크기를 재조정할 필요가 없습니다.

    // 해상도 변경 시 Worker에 설정 업데이트
    wsProcessorWorker.postMessage({
        type: 'config',
        config: {
            USE_H264_DECODER,
            USE_RAW_RGB,
            CLIENT_ASSUMED_SCALE_Y,
            rtWidth,
            rtHeight
        }
    });

    if (ws && ws.readyState === WebSocket.OPEN) {
        const buf = new ArrayBuffer(4);
        new DataView(buf).setUint16(0, width, true);
        new DataView(buf).setUint16(2, height, true);
        console.log("first frame send")
        ws.send(buf);
    }
}


let fpsSum = 0; // Σ FPS
let fpsCount = 0; // number of samples accumulated
const AVG_WINDOW = 120; // how many samples per average report

let secondStart = performance.now();
let secondFrames = 0;

function getGPUFPS() {
    if (pendingQueries.length) {
        const oldest = pendingQueries[0];
        const ready = gl.getQueryParameter(oldest, gl.QUERY_RESULT_AVAILABLE);
        const disjoint = gl.getParameter(ext.GPU_DISJOINT_EXT);

        if (ready && !disjoint) {
            const ns = gl.getQueryParameter(oldest, gl.QUERY_RESULT) as number;
            const fpsInstant = 1e9 / ns;
            secondFrames += 1;
            const now = performance.now();
            if (now - secondStart >= 1000) {
                const secFps = (secondFrames * 1000) / (now - secondStart);
                // console.info(`GPU-FPS-1s: ${secFps.toFixed(1)} fps | Client Display: 120Hz expected`);
                secondStart = now;
                secondFrames = 0;
            }
            // console.info('GPU-FPS', fpsInstant.toFixed(1));

            // ----- accumulate per‑frame time for average -----
            fpsSum += ns; // store *time*, not fps
            fpsCount += 1;

            if (fpsCount >= AVG_WINDOW) {
                const avgNs = fpsSum / fpsCount;
                const avgFps = 1e9 / avgNs;

                fpsSum = 0;
                fpsCount = 0;
            }

            // recycle query object
            queryPool.push(oldest);
            pendingQueries.shift(); // remove from FIFO
        }
    }
}

const TARGET_FPS = 120;
const FRAME_INTERVAL = 1000 / TARGET_FPS;
let lastFrameTime = performance.now()
let lastMeshRenderTime = performance.now()
const MESH_RENDER_INTERVAL = 1000 / TARGET_FPS;

function animate() {
    if (!legacyWay && !SceneState.worldScene) return;
    if (!SceneState.scene) return;

    requestAnimationFrame(animate);

    const now = performance.now();
    const elapsed = now - lastFrameTime;

    if (elapsed < FRAME_INTERVAL) return;
    lastFrameTime = now

    const intrinsics = getCameraIntrinsics(camera, rtWidth, rtHeight);

    const clientSendUnixTimestamp = performance.timeOrigin + performance.now();

    if (legacyWay) {
        const num_legacy_floats = 3 + 3 + 9 + 1 + 16; // 32 floats
        // ArrayBuffer 크기: (기존 float 개수 * 4 바이트) + (타임스탬프 double * 8 바이트)
        buf = new ArrayBuffer(num_legacy_floats * Float32Array.BYTES_PER_ELEMENT + Float64Array.BYTES_PER_ELEMENT);
        floatView = new Float32Array(buf, 0, num_legacy_floats); // 기존 데이터 영역
        const timestampView = new Float64Array(buf, num_legacy_floats * Float32Array.BYTES_PER_ELEMENT); // 타임스탬프 영역

        floatView[0] = camera.position.x;
        floatView[1] = camera.position.y;
        floatView[2] = camera.position.z;

        floatView[3] = (controls as OrbitControls).target.x;
        floatView[4] = (controls as OrbitControls).target.y;
        floatView[5] = (controls as OrbitControls).target.z;

        floatView.set(intrinsics, 6);
        floatView[15] = 0.0;
        floatView.set(camera.projectionMatrix.clone().toArray(), 16);

        timestampView[0] = clientSendUnixTimestamp; // 버퍼 끝에 타임스탬프 추가

    } else {
        const num_modern_floats = 16 + 9 + 1 + 16 + 3; // 45 floats
        buf = new ArrayBuffer(num_modern_floats * Float32Array.BYTES_PER_ELEMENT + Float64Array.BYTES_PER_ELEMENT);
        floatView = new Float32Array(buf, 0, num_modern_floats);
        const timestampView = new Float64Array(buf, num_modern_floats * Float32Array.BYTES_PER_ELEMENT);

        floatView.set(camera.matrixWorld.clone().toArray(), 0);
        floatView.set(intrinsics, 16);
        floatView[25] = gs_index_time;
        floatView.set(camera.projectionMatrix.clone().toArray(), 26);

        floatView[42] = camera.position.x;
        floatView[43] = camera.position.y;
        floatView[44] = camera.position.z;

        timestampView[0] = clientSendUnixTimestamp; // 버퍼 끝에 타임스탬프 추가
    }

    const animation_loop_elapsed_time = Date.now() - animationStartTime;
    gs_index_time = (animation_loop_elapsed_time / 10000) % 1.0;

    if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(buf);
    }

    prev_time = Date.now();

    const delta = clock.getDelta()
    const q = queryPool.pop() ?? gl.createQuery();
    gl.beginQuery(ext.TIME_ELAPSED_EXT, q);

    const t_start = performance.now();

    // 1. 비동기 데이터로부터 텍스처를 동기적으로 업데이트 (프레임 동기화 개선)
    // H.264 디코딩은 이제 Worker에서 처리되므로 이 부분은 제거
    // TODO: JPEG 경로도 동일한 동기화 패턴으로 리팩토링 필요 시 여기에 추가
    // 2. 텍스처 업데이트가 끝난 후, 퓨전 재질의 유니폼을 업데이트
    updateFusionMaterialTextures();

    // object_animations(); // mesh가 초기화되지 않았으므로 주석 처리
    robot_animation();

    controls.update(delta);

    const nowMesh = performance.now();
    const shouldRenderMesh = (nowMesh - lastMeshRenderTime) >= MESH_RENDER_INTERVAL;

    if (shouldRenderMesh) {
        lastMeshRenderTime = nowMesh;

        renderer.setRenderTarget(localRenderTarget);
        if (!legacyWay) {
            if (SceneState.worldScene) {
                renderer.render(SceneState.worldScene, camera);
            }
        } else {
            if (SceneState.scene) {
                renderer.render(SceneState.scene, camera);
            }
        }
        renderer.setRenderTarget(null);
    }

    renderer.setRenderTarget(null);
    renderer.clear();

    renderer.setViewport(0, 0, rtWidth, rtHeight);
    if (fusionMaterial && fusionMaterial.uniforms.localColorSampler.value && fusionMaterial.uniforms.wsColorSampler.value) {
        renderer.render(fusionScene, orthoCamera); // 미리 정의된 scene과 orthographic 카메라 사용
    }

    const t_end = performance.now();
    // console.log(`[Legacy] Rendering time: ${t_end - t_start}ms`);

    if (stats) stats.end();
    gl.endQuery(ext.TIME_ELAPSED_EXT);
    pendingQueries.push(q); // add to FIFO

    // perfMonitor.logStatsIfReady();
}

await initWebSocket();
await main();
