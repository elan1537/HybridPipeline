// main.ts
import * as THREE from 'three/webgpu';
import {
  texture,
  vec4,
  uv,
  vec2,
  sub,
  Fn,
  float,
  vec3,
  uniform,
  positionLocal,
  cameraProjectionMatrix,
  cameraNear,
  cameraFar,
  mul,
  clamp,
  min,
  add,
  max,
  cameraViewMatrix,
  modelWorldMatrix,
} from 'three/tsl';
import { OrbitControls, TransformControls, GLTFLoader, ColladaLoader } from 'three/addons';

import URDFLoader from 'urdf-loader';

// import Stats from 'stats.js';

const projection3DGSWay = Fn(() => {
  const fx = float(cameraIntrinsicsUniform.x); // Custom Uniform
  const fy = float(cameraIntrinsicsUniform.y); // Custom Uniform
  const cx = float(cameraPrincipalPointUniform.x); // Custom Uniform
  const cy = float(cameraPrincipalPointUniform.y); // Custom Uniform
  const width = float(renderTargetSizeUniform.x); // Custom Uniform (Logical Width){}
  const height = float(renderTargetSizeUniform.y); // Custom Uniform (Logical Height)

  const viewPos = cameraViewMatrix.mul(modelWorldMatrix.mul(vec4(positionLocal, 1.0)));

  const z_camera_dist = max(1e-6, viewPos.z.negate());

  const pixel_pos = vec2(
    fx.mul(viewPos.x).div(z_camera_dist).add(cx),
    fy.mul(viewPos.y).div(z_camera_dist).add(cy)
  );

  const ndc_x = pixel_pos.x.div(width).mul(2.0).sub(1.0);
  const ndc_y = pixel_pos.y.div(height).mul(2.0).sub(1.0);
  const ndc_z = clamp(
    mul(cameraFar, sub(z_camera_dist, cameraNear)).div(
      mul(z_camera_dist, sub(cameraFar, cameraNear))
    ),
    0.0,
    1.0
  );

  return vec4(ndc_x, ndc_y, ndc_z, 1.0);
});

const customVertexMaterial = new THREE.MeshStandardNodeMaterial();
customVertexMaterial.vertexNode = projection3DGSWay();

let camera: THREE.PerspectiveCamera;
let scene: THREE.Scene;
let renderer: THREE.WebGPURenderer;
let controls: OrbitControls;
let transform: TransformControls;

let wsColorTexture: THREE.Texture;
let wsDepthTexture: THREE.DataTexture;
let wsColorVisQuad: THREE.QuadMesh;
let wsDepthVisQuad: THREE.QuadMesh;
let wsColorNode: THREE.TextureNode;
let wsDepthNode: THREE.TextureNode;

// Local Renderer 텍스처 관련 변수
let localRenderTarget: THREE.RenderTarget;
let localDepthTexture: THREE.DepthTexture;
let localColorVisQuad: THREE.QuadMesh;
let localDepthVisQuad: THREE.QuadMesh;
let localColorNode: THREE.TextureNode;
let localDepthNode: THREE.TextureNode;

// Fusion 패스 관련 변수
let fusionQuad: THREE.QuadMesh;
let fusionMaterial: THREE.MeshBasicNodeMaterial;
let fusedDepthVisQuad: THREE.QuadMesh;
let fusedDepthVisMaterial: THREE.MeshBasicNodeMaterial;

let stats;
let robot;
let mesh: THREE.Mesh;
let mesh2: THREE.Mesh;
let loggingEnabled = true;

const near = 0.1;
const far = 100;

let rtWidth = window.innerWidth;
let rtHeight = window.innerHeight;
const dpr = window.devicePixelRatio;

// 파일 상단 또는 적절한 스코프에 추가
const CLIENT_ASSUMED_SCALE_Y = 1.0; // 서버에서 가정한 값과 동일하게

let ws: WebSocket;

const cameraIntrinsicsUniform = uniform(vec3(0.0, 0.0, 0.0));
const cameraPrincipalPointUniform = uniform(vec2(0.0, 0.0));

const renderTargetSizeUniform = uniform(vec2(rtWidth * dpr, rtHeight * dpr));

let timing: { [key: string]: number[] } = {
  parse: [],
  decode_jpeg: [],
  parse_depth: [],
  update_fusion_material_textures: [],
  total: [],
};

let frame_warmup = 100;
let frame_collect = 10;
let frame_idx = 0;

// --- 새로운 함수: 커스텀 프로젝션 Grid 생성 ---
function createCustomProjectedGrid(size, divisions) {
  const step = size / divisions;
  const halfSize = size / 2;

  const vertices: number[] = []; // 타입 명시

  const color = new THREE.Color(0x888888);

  for (let i = 0; i <= divisions; i++) {
    const xy = -halfSize + i * step;
    vertices.push(xy, 0, -halfSize);
    vertices.push(xy, 0, halfSize);
    vertices.push(-halfSize, 0, xy);
    vertices.push(halfSize, 0, xy);
  }

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.Float32BufferAttribute(vertices, 3));

  // Grid를 위한 Custom TSL Material 생성 (NodeMaterial 사용)
  const gridMaterial = new THREE.MeshBasicNodeMaterial(); // NodeMaterial 임포트 확인

  // 재사용 가능한 3DGS 프로젝션 함수 사용
  gridMaterial.vertexNode = projection3DGSWay();

  // Grid 색상 설정 (단색)
  gridMaterial.colorNode = vec3(color.r, color.g, color.b); // vec3 임포트 확인

  const grid = new THREE.LineSegments(geometry, gridMaterial);
  grid.matrixAutoUpdate = false;

  return grid;
}

const fusionMaterialFn = Fn(() => {
  const uvCoords = uv();

  const colorLocal = localColorNode.sample(uvCoords);

  /* note: WebSocket frame comes flipped in X so we mirror the lookup      */
  const colorWSFull = wsColorNode.sample(vec2(sub(1.0, uvCoords.x), sub(1.0, uvCoords.y)));
  const depthLocal = localDepthNode.sample(uvCoords).r;
  const depthWS = wsDepthNode.sample(vec2(sub(1.0, uvCoords.x), uvCoords.y)).r;

  const alphaWS = colorWSFull.a; // use alpha to invalidate soft edge
  const colorWS = colorWSFull; // keep full RGBA for output

  /* validity masks ------------------------------------------------------ */
  const localValid = depthLocal.equal(depthLocal).and(depthLocal.lessThan(1.0)); // not‑NaN & < 1
  const wsValid = depthWS.equal(depthWS).and(depthWS.lessThan(1.0)).and(alphaWS.greaterThan(0.3)); // α must be solid enough

  /* choose which pipeline wins per‑pixel -------------------------------- */
  const useLocal = localValid.and(wsValid.not().or(depthLocal.lessThanEqual(depthWS)));

  const backgroundColor = vec4(0.0, 0.0, 0.0, 0.0);
  const finalColor = useLocal.select(colorLocal, colorWS);
  const outputColor = localValid.or(wsValid).select(finalColor, backgroundColor);

  return outputColor;
});

const fusionDepthMaterialFn = Fn(() => {
  const uvCoords = uv();

  const depthLocal = localDepthNode.sample(uvCoords).r;
  const depthWS = wsDepthNode.sample(vec2(sub(1.0, uvCoords.x), uvCoords.y)).r;

  const localValid = depthLocal.equal(depthLocal).and(depthLocal.lessThan(1.0)); // not‑NaN & < 1
  const wsValid = depthWS.equal(depthWS).and(depthWS.lessThan(1.0)); // not‑NaN & < 1
  const useLocal = localValid.and(wsValid.not().or(depthLocal.lessThanEqual(depthWS)));
  const finalDepthValue = useLocal.select(depthLocal, depthWS);
  const outputDepth = localValid.or(wsValid).select(finalDepthValue, float(1.0));

  return outputDepth;
});

function updateFusionMaterialTextures() {
  if (!fusionMaterial) return;

  localColorNode = texture(localRenderTarget?.texture || new THREE.Texture());
  localDepthNode = texture(
    localDepthTexture || new THREE.DepthTexture(window.innerWidth, window.innerHeight)
  );
  wsColorNode = texture(wsColorTexture || new THREE.Texture());
  wsDepthNode = texture(wsDepthTexture || new THREE.Texture());

  if (fusionMaterial) {
    fusionMaterial.colorNode = fusionMaterialFn();
    fusionMaterial.needsUpdate = true;
  }

  if (fusedDepthVisMaterial) {
    const rawFusedDepth = fusionDepthMaterialFn();
    const visualizedDepth = clamp(rawFusedDepth, 0.0, 1.0);
    fusedDepthVisMaterial.colorNode = vec4(visualizedDepth.xxx, 1.0);
    fusedDepthVisMaterial.needsUpdate = true;
  }
}

async function initWebSocket() {
  ws = new WebSocket('ws://localhost:8765');
  console.log('Attempting WebSocket connection...');

  ws.binaryType = 'arraybuffer';
  ws.onopen = () => {
    console.log('Websocket connection opened');

    const buf = new ArrayBuffer(4);
    new DataView(buf).setUint16(0, rtWidth, true);
    new DataView(buf).setUint16(2, rtHeight, true);
    ws.send(buf);
  };

  ws.onmessage = async event => {
    const t_start = performance.now();
    const data = event.data;
    if (!(data instanceof ArrayBuffer)) return; // 타입 체크

    const dv = new DataView(data);
    if (data.byteLength < 8) return; // 헤더 길이 체크
    const rgbLen = dv.getUint32(0, true);
    const depthLen = dv.getUint32(4, true);
    const rgbStart = 8;
    const depthStart = 8 + rgbLen;

    if (data.byteLength < depthStart + depthLen) return;
    const t_parse_data = performance.now();

    try {
      const jpegBlob = new Blob([data.slice(rgbStart, rgbStart + rgbLen)], { type: 'image/jpeg' });
      const imageBitmap = await createImageBitmap(jpegBlob);
      if (!wsColorTexture) {
        // 최초 생성
        wsColorTexture = new THREE.Texture(imageBitmap);
        wsColorTexture.colorSpace = THREE.SRGBColorSpace;
      } else {
        // --- 기존 텍스처 업데이트 ---
        // 현재 텍스처 이미지의 크기와 새로 받은 이미지 비트맵 크기 비교
        if (
          wsColorTexture.image.width !== imageBitmap.width ||
          wsColorTexture.image.height !== imageBitmap.height
        ) {
          // 크기가 변경되었으므로 텍스처 재생성
          console.log('Recreating wsColorTexture due to size change.'); // 디버깅 로그
          if (wsColorTexture.image instanceof ImageBitmap) {
            wsColorTexture.image.close(); // 이전 ImageBitmap 리소스 해제
          }
          wsColorTexture.dispose(); // 이전 텍스처 GPU 리소스 해제

          wsColorTexture = new THREE.Texture(imageBitmap); // 새 ImageBitmap으로 텍스처 생성
          wsColorTexture.colorSpace = THREE.SRGBColorSpace; // 색 공간 다시 설정
        } else {
          // 크기가 동일하면 imageBitmap만 교체
          if (wsColorTexture.image instanceof ImageBitmap) {
            wsColorTexture.image.close(); // 이전 ImageBitmap 리소스 해제
          }
          wsColorTexture.image = imageBitmap;
        }
      }
      wsColorTexture.needsUpdate = true;
    } catch (e) {
      console.error('Color processing error:', e);
    }

    const t_decode_jpeg = performance.now();

    try {
      const depthArrayBuffer = data.slice(depthStart, depthStart + depthLen);
      const nonlinearDepth = new Float16Array(depthArrayBuffer);

      // --- 서버가 렌더링했을 높이 계산 ---
      const serverCorrectionY = 1.0 / CLIENT_ASSUMED_SCALE_Y;
      // 서버에서 계산했을 높이 (정수로 변환)
      const expectedServerHeight = Math.round(rtHeight * serverCorrectionY);
      // 너비는 클라이언트와 동일하다고 가정
      const expectedServerWidth = rtWidth;
      // ---------------------------------

      const t_parse_depth = performance.now();

      if (!wsDepthTexture) {
        // 최초 생성
        wsDepthTexture = new THREE.DataTexture(
          nonlinearDepth, // Float16Array 사용
          expectedServerWidth, // 서버가 렌더링한 너비 사용
          expectedServerHeight, // 서버가 렌더링한 높이 사용
          THREE.RedFormat,
          THREE.HalfFloatType
        );
        wsDepthTexture.magFilter = THREE.LinearFilter;
        wsDepthTexture.minFilter = THREE.LinearFilter;
      } else {
        // --- 기존 텍스처 업데이트 ---
        // 크기 비교 시에도 예상 서버 크기 사용
        if (
          wsDepthTexture.image.width !== expectedServerWidth ||
          wsDepthTexture.image.height !== expectedServerHeight ||
          wsDepthTexture.image.data.byteLength !== nonlinearDepth.byteLength
        ) {
          console.log('Recreating wsDepthTexture due to size change (expected server size).');
          wsDepthTexture.dispose();

          wsDepthTexture = new THREE.DataTexture(
            nonlinearDepth, // 새 Float16Array 데이터
            expectedServerWidth, // 새 서버 너비
            expectedServerHeight, // 새 서버 높이
            THREE.RedFormat,
            THREE.HalfFloatType
          );
          wsDepthTexture.magFilter = THREE.LinearFilter;
          wsDepthTexture.minFilter = THREE.LinearFilter;
        } else {
          // 크기가 동일한 경우 (예상 서버 크기 기준), .set() 호출
          (wsDepthTexture.image.data as Float16Array).set(nonlinearDepth);
          // 참고: 이 경우 width/height 속성 업데이트는 불필요 (이미 동일함)
        }
      }
      wsDepthTexture.needsUpdate = true;
    } catch (e) {
      console.error('Depth processing error:', e);
    }

    const t_update_fusion_material_textures = performance.now();
    updateFusionMaterialTextures();
    const t_end = performance.now();

    if (frame_idx >= frame_warmup && frame_idx <= frame_warmup + frame_collect) {
      timing.total.push(t_end - t_start);
      timing.parse.push(t_parse_data - t_start);
      timing.decode_jpeg.push(t_decode_jpeg - t_parse_data);
      timing.parse_depth.push(t_update_fusion_material_textures - t_decode_jpeg);
      timing.total.push(t_end - t_update_fusion_material_textures);
    }

    frame_idx += 1;
  };

  ws.onerror = e => console.error('WebSocket Error:', e);
  ws.onclose = e => {
    console.log('Websocket closed:', e.code, e.reason);
    // console.log('Stage              | Avg     | Std     | Min     | Max     | Median  | P95');
    // console.log('--------------------------------------------------------------------------------');

    // for (const stage in timing) {
    //   if (timing.hasOwnProperty(stage)) {
    //     const values = timing[stage];
    //     console.log(values);
    //     const stagePadded = stage.padEnd(18, ' ');
    //     const avg = values[0].toFixed(4).padStart(7, ' ');
    //     const std = values[1].toFixed(4).padStart(7, ' ');
    //     const min = values[2].toFixed(4).padStart(7, ' ');
    //     const max = values[3].toFixed(4).padStart(7, ' ');
    //     const median = values[4].toFixed(4).padStart(7, ' ');
    //     const p95 = values[5].toFixed(4).padStart(7, ' ');
    //     console.log(`${stagePadded}| ${avg} | ${std} | ${min} | ${max} | ${median} | ${p95}`);
    //   }
    // }
  };
}

let capturedDepthData: { data: Uint8Array | null; width: number; height: number } = {
  data: null,
  width: 0,
  height: 0,
};

async function main() {
  // stats = new Stats();
  // stats.showPanel(0);
  // document.body.appendChild(stats.dom);
  // stats.dom.style.display = loggingEnabled ? 'block' : 'none';

  // // 다운로드 버튼 추가
  // const downloadButton = document.createElement('button');
  // downloadButton.id = 'downloadFrame';
  // downloadButton.textContent = '프레임 다운로드';
  // downloadButton.style.position = 'absolute';
  // downloadButton.style.top = '10px';
  // downloadButton.style.left = '10px';
  // downloadButton.style.zIndex = '100';
  // document.body.appendChild(downloadButton);

  camera = new THREE.PerspectiveCamera(80, window.innerWidth / window.innerHeight, near, far);
  scene = new THREE.Scene();
  scene.background = new THREE.Color(0x112233);

  camera.position.set(-2.5, 1, 2.5);
  camera.lookAt(0, 0, 0);

  const geometry = new THREE.TorusKnotGeometry(1, 0.3, 128, 32);
  // const geometry = new THREE.CylinderGeometry(1, 1, 2, 32);

  const material1 = new THREE.MeshStandardMaterial({
    color: 0x00ff00,
    roughness: 0.5,
  });

  const material2 = new THREE.MeshStandardMaterial({
    color: 0x0000ff,
    roughness: 0.5,
  });

  const mesh1mat = customVertexMaterial.clone();
  mesh1mat.colorNode = vec3(0.0, 1.0, 0.0);
  mesh1mat.roughnessNode = float(0.5);

  const mesh2mat = customVertexMaterial.clone();
  mesh2mat.colorNode = vec3(0.0, 0.0, 1.0);
  mesh2mat.roughnessNode = float(0.5);

  mesh = new THREE.Mesh(geometry, material1);
  mesh.scale.set(0.3, 0.3, 0.3);
  mesh.position.set(0, 0.3, 0);
  scene.add(mesh);

  mesh2 = new THREE.Mesh(new THREE.SphereGeometry(0.5), material2);
  mesh2.scale.set(0.3, 0.3, 0.3);
  mesh2.position.set(1.0, 0, 1.0);
  scene.add(mesh2);

  // const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
  // scene.add(ambientLight);
  // const dirLight = new THREE.DirectionalLight(0xffffff, 5);
  // dirLight.position.set(-5, 4, -2);
  // scene.add(dirLight);

  // const dirLight2 = new THREE.DirectionalLight(0xffffff, 5);
  // dirLight2.position.set(0, 4, 0);
  // scene.add(dirLight2);

  const hemisphereLight = new THREE.HemisphereLight(0xffffff, 0x080808, 15);
  scene.add(hemisphereLight);

  const axisRadius = 0.05;
  const axisLength = 5;
  const segments = 8;
  // const xAxisGeo = new THREE.CylinderGeometry(axisRadius, axisRadius, axisLength, segments);
  // const xAxisMat = new THREE.MeshBasicNodeMaterial({ color: 0xff0000 });
  // xAxisMat.vertexNode = projection3DGSWay();
  // const xAxisMesh = new THREE.Mesh(xAxisGeo, xAxisMat);
  // xAxisMesh.rotation.z = -Math.PI / 2;
  // xAxisMesh.position.x = axisLength / 2;
  // scene.add(xAxisMesh);

  // const yAxisGeo = new THREE.CylinderGeometry(axisRadius, axisRadius, axisLength, segments);
  // const yAxisMat = new THREE.MeshBasicNodeMaterial({ color: 0x00ff00 });
  // yAxisMat.vertexNode = projection3DGSWay();
  // const yAxisMesh = new THREE.Mesh(yAxisGeo, yAxisMat);
  // yAxisMesh.position.y = axisLength / 2;
  // scene.add(yAxisMesh);

  // const zAxisGeo = new THREE.CylinderGeometry(axisRadius, axisRadius, axisLength, segments);
  // const zAxisMat = new THREE.MeshBasicNodeMaterial({ color: 0x0000ff });
  // zAxisMat.vertexNode = projection3DGSWay();
  // const zAxisMesh = new THREE.Mesh(zAxisGeo, zAxisMat);
  // zAxisMesh.rotation.x = Math.PI / 2;
  // zAxisMesh.position.z = axisLength / 2;
  // scene.add(zAxisMesh);

  renderer = new THREE.WebGPURenderer({ antialias: true });
  renderer.setPixelRatio(dpr);
  renderer.setSize(window.innerWidth, window.innerHeight);

  document.body.appendChild(renderer.domElement);

  controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.1;

  // controls.autoRotate = true;
  // controls.autoRotateSpeed = 0.5;

  localDepthTexture = new THREE.DepthTexture(rtWidth * dpr, rtHeight * dpr);
  localDepthTexture.type = THREE.FloatType;
  localRenderTarget = new THREE.RenderTarget(rtWidth * dpr, rtHeight * dpr, {
    depthBuffer: true,
    stencilBuffer: false,
    depthTexture: localDepthTexture,
    samples: 4,
  });

  const localVisMaterialColor = new THREE.MeshBasicNodeMaterial();
  localVisMaterialColor.colorNode = texture(localRenderTarget.texture);
  localColorVisQuad = new THREE.QuadMesh(localVisMaterialColor);

  const localVisMaterialDepth = new THREE.MeshBasicNodeMaterial();
  localVisMaterialDepth.colorNode = vec4(texture(localDepthTexture).xxx, 1.0);
  localDepthVisQuad = new THREE.QuadMesh(localVisMaterialDepth);

  const wsVisMaterialColor = new THREE.MeshBasicNodeMaterial();
  wsVisMaterialColor.colorNode = texture(new THREE.Texture());
  wsColorVisQuad = new THREE.QuadMesh(wsVisMaterialColor);

  const wsVisMaterialDepth = new THREE.MeshBasicNodeMaterial();
  wsVisMaterialDepth.colorNode = vec4(texture(new THREE.Texture()).xxx, 1.0);

  wsDepthVisQuad = new THREE.QuadMesh(wsVisMaterialDepth);
  fusionMaterial = new THREE.MeshBasicNodeMaterial();
  fusionMaterial.depthTest = false;
  fusionMaterial.depthWrite = false;
  fusionQuad = new THREE.QuadMesh(fusionMaterial);

  fusedDepthVisMaterial = new THREE.MeshBasicNodeMaterial();
  fusedDepthVisMaterial.depthTest = false;
  fusedDepthVisMaterial.depthWrite = false;
  fusedDepthVisQuad = new THREE.QuadMesh(fusedDepthVisMaterial);

  updateFusionMaterialTextures();

  window.addEventListener('resize', onWindowResize);

  // // --- 커스텀 프로젝션 Grid 생성 및 추가 ---
  // const customGridSize = 100;
  // const customGridDivisions = 100;
  // const customGrid = createCustomProjectedGrid(customGridSize, customGridDivisions);
  // customGrid.position.set(0, 0.5, 0);
  // scene.add(customGrid); // 생성된 커스텀 Grid 추가

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

  localDepthTexture.image.width = width * dpr;
  localDepthTexture.image.height = height * dpr;

  // WebSocket 크기 변경 알림 (선택 사항)
  if (ws && ws.readyState === WebSocket.OPEN) {
    const buf = new ArrayBuffer(4);
    new DataView(buf).setUint16(0, width, true);
    new DataView(buf).setUint16(2, height, true);
    ws.send(buf);
  }
}

function getCameraIntrinsics(camera: THREE.PerspectiveCamera, renderWidth, renderHeight) {
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

  const intrinsics = [fx, 0, cx, 0, fy, cy, 0, 0, 1];

  return intrinsics;
}

async function animate() {
  const time = Date.now() * 0.0003; // 공통 시간 변수

  const t_frame_start = performance.now();

  const cam_int = getCameraIntrinsics(camera, rtWidth, rtHeight);

  cameraIntrinsicsUniform.value.x = cam_int[0];
  cameraIntrinsicsUniform.value.y = cam_int[4];
  cameraPrincipalPointUniform.value.x = cam_int[2];
  cameraPrincipalPointUniform.value.y = cam_int[5];
  renderTargetSizeUniform.value.x = rtWidth;
  renderTargetSizeUniform.value.y = rtHeight;

  // if (robot) {
  //   robot.setJointValue('shoulder_lift_joint', (-12.5 * Math.PI) / 180);
  //   robot.setJointValue('elbow_joint', (-55 * Math.PI) / 180);
  //   robot.setJointValue('wrist_1_joint', (-90 * Math.PI) / 180);
  //   robot.setJointValue('wrist_2_joint', (90 * Math.PI) / 180);
  // }

  // if (robot && robot.joints && typeof robot.setJointValue === 'function') {
  //   const time = Date.now() * 0.0003; // 공통 시간 변수

  //   // shoulder_pan_joint 애니메이션
  //   const shoulderPanJoint = 'shoulder_pan_joint';
  //   if (robot.joints[shoulderPanJoint]) {
  //     const panAngle = Math.sin(time) * (Math.PI / 2);
  //     robot.setJointValue(shoulderPanJoint, panAngle);
  //   }

  //   // shoulder_lift_joint 애니메이션
  //   const shoulderLiftJoint = 'shoulder_lift_joint';
  //   if (robot.joints[shoulderLiftJoint]) {
  //     // limit: {lower: -2.356194490192345, upper: 2.356194490192345} => 약 -135도 ~ 135도
  //     // 좀 더 작은 범위로 움직이도록 설정
  //     const liftAngle = Math.sin(time * 0.7 + Math.PI / 2) * (Math.PI / 2) - Math.PI / 4; //  -75도 ~ +15도 범위 (예시)
  //     robot.setJointValue(shoulderLiftJoint, liftAngle);
  //   }

  //   // elbow_joint 애니메이션
  //   const elbowJoint = 'elbow_joint';
  //   if (robot.joints[elbowJoint]) {
  //     // limit: {lower: -2.6179938779914944, upper: 2.6179938779914944} => 약 -150도 ~ 150도
  //     const elbowAngle = Math.cos(time * 0.9) * (Math.PI / 2); // -90도에서 +90도
  //     robot.setJointValue(elbowJoint, elbowAngle);
  //   }

  //   // wrist_1_joint 애니메이션
  //   const wrist1Joint = 'wrist_1_joint';
  //   if (robot.joints[wrist1Joint]) {
  //     // limit: {lower: -6.283185307179586, upper: 6.283185307179586} => 약 -360도 ~ 360도 (연속 회전 가능)
  //     const wrist1Angle = Math.sin(time * 1.1) * Math.PI; // -180도에서 +180도
  //     robot.setJointValue(wrist1Joint, wrist1Angle);
  //   }

  //   // wrist_2_joint 애니메이션
  //   const wrist2Joint = 'wrist_2_joint';
  //   if (robot.joints[wrist2Joint]) {
  //     // limit: {lower: -6.283185307179586, upper: 6.283185307179586}
  //     const wrist2Angle = Math.cos(time * 1.3 + Math.PI / 3) * (Math.PI / 1.5);
  //     robot.setJointValue(wrist2Joint, wrist2Angle);
  //   }

  //   // wrist_3_joint 애니메이션
  //   const wrist3Joint = 'wrist_3_joint';
  //   if (robot.joints[wrist3Joint]) {
  //     // limit: {lower: -6.283185307179586, upper: 6.283185307179586}
  //     const wrist3Angle = Math.sin(time * 1.5 + Math.PI / 1.5) * Math.PI;
  //     robot.setJointValue(wrist3Joint, wrist3Angle);
  //   }
  // }

  // mesh.rotation.y = time * 10;
  // // mesh.scale.x = Math.sin(time) * 0.5;
  // // mesh.scale.y = Math.cos(time) * 0.5;
  // // mesh.scale.z = Math.sin(time) * 0.5;
  // mesh.position.y = -(Math.sin(time) * 1.0) + 0.3;

  // mesh2.position.x = 2.5 * Math.sin(time) * 0.5;
  // mesh2.position.z = 2.5 * Math.cos(time) * 0.5;
  // if (stats) stats.begin();

  controls.update();

  renderer.setRenderTarget(localRenderTarget);
  await renderer.clearAsync();
  await renderer.renderAsync(scene, camera);

  const buf = new ArrayBuffer(15 * Float32Array.BYTES_PER_ELEMENT);
  const floatView = new Float32Array(buf);

  floatView[0] = camera.position.x;
  floatView[1] = camera.position.y;
  floatView[2] = camera.position.z;

  floatView[3] = controls.target.x;
  floatView[4] = controls.target.y;
  floatView[5] = controls.target.z;

  const intrinsics = getCameraIntrinsics(camera, rtWidth, rtHeight);
  floatView.set(intrinsics, 6);

  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(buf);
  }

  renderer.setRenderTarget(null);
  await renderer.clearAsync();

  renderer.setViewport(0, 0, rtWidth, rtHeight);
  await fusionQuad.renderAsync(renderer);

  // if (stats) stats.end(); // Stats 종료

  const downloadButton = document.getElementById('downloadFrame');
  if (downloadButton) {
    downloadButton.onclick = async () => {
      const colorCanvas = renderer.domElement;
      const colorImage = colorCanvas.toDataURL('image/png', 1.0);
      const colorLink = document.createElement('a');
      colorLink.href = colorImage;
      colorLink.download = `frame_color_${Date.now()}.png`; // 컬러 이미지 파일명
      colorLink.click(); // 컬러 이미지 다운로드 실행
    };
  }

  requestAnimationFrame(animate);
}

function scene_setup() {
  const manager = new THREE.LoadingManager();
  const loader = new URDFLoader(manager);
  loader.loadAsync('../ur_description/urdf/ur5.urdf').then(result => {
    robot = result;
  });

  manager.onLoad = () => {
    robot.rotation.x = -Math.PI / 2;
    robot.rotation.z = Math.PI / 2;
    robot.updateMatrixWorld(true);
    robot.position.set(0, 0.005, 0.02);
    robot.scale.set(3.7, 3.7, 3.7);
    const processMaterial = (originalMaterial: THREE.Material): THREE.Material => {
      let targetMaterial: THREE.MeshStandardNodeMaterial;

      if (originalMaterial && (originalMaterial as THREE.MeshStandardNodeMaterial).isNodeMaterial) {
        // 이미 NodeMaterial 계열인 경우
        targetMaterial = originalMaterial as THREE.MeshStandardNodeMaterial;
      } else if (originalMaterial) {
        // NodeMaterial이 아닌 경우, 새로 생성하고 속성 복사
        targetMaterial = new THREE.MeshStandardNodeMaterial();

        // 공통 속성 복사 (THREE.Material에서 상속받는 속성들)
        if ('opacity' in originalMaterial) targetMaterial.opacity = originalMaterial.opacity;
        if ('transparent' in originalMaterial)
          targetMaterial.transparent = originalMaterial.transparent;
        if ('blending' in originalMaterial) targetMaterial.blending = originalMaterial.blending;
        if ('blendSrc' in originalMaterial) targetMaterial.blendSrc = originalMaterial.blendSrc;
        if ('blendDst' in originalMaterial) targetMaterial.blendDst = originalMaterial.blendDst;
        if ('blendEquation' in originalMaterial)
          targetMaterial.blendEquation = originalMaterial.blendEquation;
        if ('depthTest' in originalMaterial) targetMaterial.depthTest = originalMaterial.depthTest;
        if ('depthWrite' in originalMaterial)
          targetMaterial.depthWrite = originalMaterial.depthWrite;
        if ('polygonOffset' in originalMaterial)
          targetMaterial.polygonOffset = originalMaterial.polygonOffset;
        if ('polygonOffsetFactor' in originalMaterial)
          targetMaterial.polygonOffsetFactor = originalMaterial.polygonOffsetFactor;
        if ('polygonOffsetUnits' in originalMaterial)
          targetMaterial.polygonOffsetUnits = originalMaterial.polygonOffsetUnits;
        if ('alphaTest' in originalMaterial) targetMaterial.alphaTest = originalMaterial.alphaTest;
        // 'premultipliedAlpha' 속성은 MeshStandardNodeMaterial에 직접적으로 없을 수 있으므로 주의
        if ('visible' in originalMaterial) targetMaterial.visible = originalMaterial.visible;
        if ('side' in originalMaterial) targetMaterial.side = originalMaterial.side;
        if ('colorWrite' in originalMaterial)
          targetMaterial.colorWrite = originalMaterial.colorWrite;
        if ('toneMapped' in originalMaterial)
          targetMaterial.toneMapped = originalMaterial.toneMapped;

        // MeshStandardMaterial 또는 유사 재질(MeshPhongMaterial 등)의 속성 복사
        // .color 는 NodeMaterial의 colorNode로 직접 매핑되지 않으므로, colorNode를 설정하거나,
        // MeshStandardNodeMaterial의 .color 속성에 복사합니다.
        if ('color' in originalMaterial && (originalMaterial as any).color instanceof THREE.Color) {
          targetMaterial.color.copy((originalMaterial as any).color);
        }
        if ('map' in originalMaterial && (originalMaterial as any).map instanceof THREE.Texture) {
          targetMaterial.map = (originalMaterial as any).map;
        }
        // MeshStandardMaterial 특화 속성
        if ('roughness' in originalMaterial) targetMaterial.roughness = 0.5;
        if ('metalness' in originalMaterial) targetMaterial.metalness = 0.5;
        if (
          'aoMap' in originalMaterial &&
          (originalMaterial as any).aoMap instanceof THREE.Texture
        ) {
          targetMaterial.aoMap = (originalMaterial as any).aoMap;
        }
        if ('aoMapIntensity' in originalMaterial)
          targetMaterial.aoMapIntensity = (originalMaterial as any).aoMapIntensity;
        if (
          'emissive' in originalMaterial &&
          (originalMaterial as any).emissive instanceof THREE.Color
        ) {
          targetMaterial.emissive.copy((originalMaterial as any).emissive);
        }
        if (
          'emissiveMap' in originalMaterial &&
          (originalMaterial as any).emissiveMap instanceof THREE.Texture
        ) {
          targetMaterial.emissiveMap = (originalMaterial as any).emissiveMap;
        }
        if ('emissiveIntensity' in originalMaterial)
          targetMaterial.emissiveIntensity = (originalMaterial as any).emissiveIntensity;
        if (
          'bumpMap' in originalMaterial &&
          (originalMaterial as any).bumpMap instanceof THREE.Texture
        ) {
          targetMaterial.bumpMap = (originalMaterial as any).bumpMap;
        }
        if ('bumpScale' in originalMaterial)
          targetMaterial.bumpScale = (originalMaterial as any).bumpScale;
        if (
          'normalMap' in originalMaterial &&
          (originalMaterial as any).normalMap instanceof THREE.Texture
        ) {
          targetMaterial.normalMap = (originalMaterial as any).normalMap;
        }
        if ('normalMapType' in originalMaterial)
          targetMaterial.normalMapType = (originalMaterial as any).normalMapType;
        if (
          'normalScale' in originalMaterial &&
          (originalMaterial as any).normalScale instanceof THREE.Vector2
        ) {
          targetMaterial.normalScale.copy((originalMaterial as any).normalScale);
        }
        if (
          'displacementMap' in originalMaterial &&
          (originalMaterial as any).displacementMap instanceof THREE.Texture
        ) {
          targetMaterial.displacementMap = (originalMaterial as any).displacementMap;
        }
        if ('displacementScale' in originalMaterial)
          targetMaterial.displacementScale = (originalMaterial as any).displacementScale;
        if ('displacementBias' in originalMaterial)
          targetMaterial.displacementBias = (originalMaterial as any).displacementBias;
        if (
          'roughnessMap' in originalMaterial &&
          (originalMaterial as any).roughnessMap instanceof THREE.Texture
        ) {
          targetMaterial.roughnessMap = (originalMaterial as any).roughnessMap;
        }
        if (
          'metalnessMap' in originalMaterial &&
          (originalMaterial as any).metalnessMap instanceof THREE.Texture
        ) {
          targetMaterial.metalnessMap = (originalMaterial as any).metalnessMap;
        }
        if (
          'alphaMap' in originalMaterial &&
          (originalMaterial as any).alphaMap instanceof THREE.Texture
        ) {
          targetMaterial.alphaMap = (originalMaterial as any).alphaMap;
        }
        if (
          'envMap' in originalMaterial &&
          (originalMaterial as any).envMap instanceof THREE.Texture
        ) {
          targetMaterial.envMap = (originalMaterial as any).envMap;
        }
        if ('envMapIntensity' in originalMaterial)
          targetMaterial.envMapIntensity = (originalMaterial as any).envMapIntensity;
        if ('wireframe' in originalMaterial)
          targetMaterial.wireframe = (originalMaterial as any).wireframe;

        // 만약 원래 재질의 색상을 colorNode로 사용하고 싶다면 아래 주석을 해제하고 위 color 복사 부분을 주석처리
        // if ('color' in originalMaterial && (originalMaterial as any).color instanceof THREE.Color) {
        //   const oldColor = (originalMaterial as any).color;
        //   targetMaterial.colorNode = vec3(oldColor.r, oldColor.g, oldColor.b);
        // }
      } else {
        // 재질이 없는 경우, 기본 MeshStandardNodeMaterial 생성
        targetMaterial = new THREE.MeshStandardNodeMaterial();
      }

      targetMaterial.vertexNode = projection3DGSWay();
      targetMaterial.needsUpdate = true;
      return targetMaterial;
    };

    robot.traverse(child => {
      child.castShadow = true;
      if (child.material) {
        if (Array.isArray(child.material)) {
          child.material = child.material
            .map(mat => processMaterial(mat))
            .filter(mat => mat !== null) as THREE.Material[];
        } else {
          child.material = processMaterial(child.material as THREE.Material);
        }
      }
    });
    scene.add(robot);
  };
}

await initWebSocket();
await main();
// scene_setup();
