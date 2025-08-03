import * as THREE from 'three'
import { OrbitControls } from 'three/addons'
import { object_setup } from './scene-setup';
import { SceneState } from './state/scene-state';

import fusionVertexShader from './shaders/fusionVertexShader.vs?raw';
import fusionColorFragmentShader from './shaders/fusionColorShader.fs?raw';
import debugVertexShader from './shaders/debugVertexShader.vs?raw';
import debugFragmentShader from './shaders/debugColorShader.fs?raw';

const rtWidth = 1920;
const rtHeight = 1080;
let lastCameraUpdateTime = 0
const cameraUpdateInterval = 16


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

const depthCanvas = new OffscreenCanvas(rtWidth, rtHeight);

const clock = new THREE.Clock();
const fpsDiv = document.getElementById('decode-fps') as HTMLDivElement;
const renderFpsDiv = document.getElementById('render-fps') as HTMLDivElement;

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

worker.postMessage({
    type: 'init',
    canvas: depthCanvas,
    width: rtWidth,
    height: rtHeight,
    wsURL: 'wss://' + location.host + '/ws'
}, [depthCanvas])

worker.onmessage = ({ data }) => {
    if (data.type === "ws-ready") {
        workerReady = true;
        return;
    }

    if (data.type === 'frame') {
        wsColorTexture.image = data.image;
        wsColorTexture.colorSpace = THREE.LinearSRGBColorSpace;
        wsDepthTexture.image.data = data.depth

        wsColorTexture.needsUpdate = true;
        wsDepthTexture.needsUpdate = true;
    }

    if (data.type === 'fps') {
        fpsDiv.textContent = `Decode FPS: ${data.decode.toFixed(2)}`
    }
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
    wsColorTexture.colorSpace = THREE.SRGBColorSpace;

    wsDepthTexture = new THREE.DataTexture(new Uint8Array(rtWidth * rtHeight), rtWidth, rtHeight, THREE.RedFormat, THREE.UnsignedByteType);

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
        tag // optional debug
    }, [
        camBuf.position.buffer,
        camBuf.target.buffer,
        camBuf.intrinsics.buffer,
        camBuf.projection.buffer
    ]);
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