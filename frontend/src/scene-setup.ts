import * as THREE from 'three';
import URDFLoader from 'urdf-loader';
import { SceneState } from './state/scene-state';

function object_setup() {
    const geometry = new THREE.TorusKnotGeometry(1, 0.3, 128, 32);

    const material1 = new THREE.MeshStandardMaterial({
        color: 0x00ff00,
        roughness: 0.5,
    });

    const material2 = new THREE.MeshStandardMaterial({
        color: 0x0000ff,
        roughness: 0.5,
    });

    if (!SceneState.scene) {
        console.warn('SceneState.scene is null. object_setup()는 SceneState.scene 초기화 이후에 호출되어야 합니다.');
        return;
    }
    SceneState.mesh = new THREE.Mesh(geometry, material1);
    SceneState.mesh.scale.set(0.2, 0.2, 0.2);
    SceneState.mesh.position.set(-0.5, 0.8, 1.0);
    SceneState.mesh.rotation.y = Math.PI / 2;
    // mesh.rotation.z = -10 * Math.PI / 180;
    SceneState.scene.add(SceneState.mesh);

    const blueMaterial = new THREE.MeshStandardMaterial({
        color: 0x00ff00,
        roughness: 0.5,
    });

    // const redMaterial = new THREE.MeshStandardMaterial({
    //   color: 0xff0000,
    //   roughness: 0.5,
    // });

    // for (let i = 0; i < 20; i++) {
    //   const tmp = new THREE.Mesh(new THREE.SphereGeometry(0.5), redMaterial);
    //   tmp.scale.set(0.05, 0.05, 0.05);
    //   tmp.position.set(i - 10, 0, 0.0);
    //   scene.add(tmp);
    // }

    // for (let i = 0; i < 20; i++) {
    //   const tmp = new THREE.Mesh(new THREE.SphereGeometry(0.5), blueMaterial);
    //   tmp.scale.set(0.05, 0.05, 0.05);
    //   tmp.position.set(0, 0, i - 10);
    //   scene.add(tmp);
    // }

    SceneState.mesh2 = new THREE.Mesh(new THREE.SphereGeometry(0.5), material2);
    SceneState.mesh2.scale.set(0.3, 0.3, 0.3);
    SceneState.mesh2.position.set(0, 0, 0);
    // scene.add(mesh2);

    // const mesh3 = new THREE.Mesh(geometry, material2);
    // mesh3.scale.set(0.5, 0.5, 0.5);
    // mesh3.position.set(10, 0.3, 0);
    // scene.add(mesh3);

    // const mesh4 = new THREE.Mesh(geometry, material2);
    // mesh4.scale.set(0.5, 0.5, 0.5);
    // mesh4.position.set(0, 0.3, 16);
    // scene.add(mesh4);

    // const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
    // scene.add(ambientLight);

    // const dirLight = new THREE.DirectionalLight(0xffffff, 5);
    // dirLight.position.set(-2, 4, 5);
    // scene.add(dirLight);

    // const dirLight2 = new THREE.DirectionalLight(0xffffff, 5);
    // dirLight2.position.set(0, 4, 0);
    // scene.add(dirLight2);

    const hemisphereLight = new THREE.HemisphereLight(0xffffff, 0x080808, 17);
    hemisphereLight.position.set(0, 1, -0.9);
    SceneState.scene.add(hemisphereLight);

    // const axisRadius = 0.05;
    // const axisLength = 5;
    // const segments = 8;

    // const xAxisGeo = new THREE.CylinderGeometry(axisRadius, axisRadius, axisLength, segments);
    // const xAxisMat = new THREE.MeshBasicNodeMaterial({ color: 0xff0000 });
    // const xAxisMesh = new THREE.Mesh(xAxisGeo, xAxisMat);
    // xAxisMesh.rotation.z = -Math.PI / 2;
    // xAxisMesh.position.x = axisLength / 2;
    // scene.add(xAxisMesh);

    // const yAxisGeo = new THREE.CylinderGeometry(axisRadius, axisRadius, axisLength, segments);
    // const yAxisMat = new THREE.MeshBasicNodeMaterial({ color: 0x00ff00 });
    // const yAxisMesh = new THREE.Mesh(yAxisGeo, yAxisMat);
    // yAxisMesh.position.y = axisLength / 2;
    // scene.add(yAxisMesh);

    // const zAxisGeo = new THREE.CylinderGeometry(axisRadius, axisRadius, axisLength, segments);
    // const zAxisMat = new THREE.MeshBasicNodeMaterial({ color: 0x0000ff });
    // const zAxisMesh = new THREE.Mesh(zAxisGeo, zAxisMat);
    // zAxisMesh.rotation.x = Math.PI / 2;
    // zAxisMesh.position.z = axisLength / 2;
    // scene.add(zAxisMesh);

    const grid = new THREE.GridHelper(100, 100, 0x888888, 0x888888);
    grid.position.set(0, 0, 0);
    // scene.add(grid);
}

function robot_setup() {
    const manager = new THREE.LoadingManager();
    const loader = new URDFLoader(manager);
    loader.loadAsync('../ur_description/urdf/ur5.urdf').then(result => {
        SceneState.robot = result;
    });

    manager.onLoad = () => {
        if (!SceneState.robot) return;

        SceneState.robot.rotation.x = -Math.PI / 2;
        SceneState.robot.rotation.z = Math.PI / 2;
        SceneState.robot.updateMatrixWorld(true);
        SceneState.robot.position.set(0, 0, 0);
        SceneState.robot.scale.set(3.7, 3.7, 3.7);

        SceneState.robot.traverse(child => {
            child.castShadow = true;
        });

        if (SceneState.scene) {
            SceneState.scene.add(SceneState.robot);
        }
    };
}


function object_animation() {
    // mesh, mesh2가 아직 생성되지 않았다면 애니메이션을 진행하지 않습니다.
    if (!SceneState.mesh || !SceneState.mesh2) return;

    const time = Date.now() * 0.0003; // 공통 시간 변수
    SceneState.mesh.rotation.y = time * 10;
    SceneState.mesh.scale.x = Math.sin(time) * 0.5;
    SceneState.mesh.scale.y = Math.cos(time) * 0.5;
    SceneState.mesh.scale.z = Math.sin(time) * 0.5;
    SceneState.mesh.position.y = -(Math.sin(time) * 1.0) + 0.3;

    SceneState.mesh2.position.x = 2.5 * Math.sin(time) * 0.5;
    SceneState.mesh2.position.z = 2.5 * Math.cos(time) * 0.5;
}


function robot_animation() {
    if (SceneState.robot) {
        SceneState.robot.setJointValue('shoulder_lift_joint', (-12.5 * Math.PI) / 180);
        SceneState.robot.setJointValue('elbow_joint', (-55 * Math.PI) / 180);
        SceneState.robot.setJointValue('wrist_1_joint', (-90 * Math.PI) / 180);
        SceneState.robot.setJointValue('wrist_2_joint', (90 * Math.PI) / 180);
    }

    if (SceneState.robot && SceneState.robot.joints && typeof SceneState.robot.setJointValue === 'function') {
        const time = Date.now() * 0.0003; // 공통 시간 변수

        // shoulder_pan_joint 애니메이션
        const shoulderPanJoint = 'shoulder_pan_joint';
        if (SceneState.robot.joints[shoulderPanJoint]) {
            // const panAngle = Math.sin(time) * (Math.PI / 2);
            const panAngle = -5.5 * Math.PI / 180
            SceneState.robot.setJointValue(shoulderPanJoint, panAngle);
        }

        // shoulder_lift_joint 애니메이션
        const shoulderLiftJoint = 'shoulder_lift_joint';
        if (SceneState.robot.joints[shoulderLiftJoint]) {
            // limit: {lower: -2.356194490192345, upper: 2.356194490192345} => 약 -135도 ~ 135도
            // 좀 더 작은 범위로 움직이도록 설정
            const liftAngle = Math.sin(time * 0.7 + Math.PI / 2) * (Math.PI / 2) - Math.PI / 4; //  -75도 ~ +15도 범위 (예시)
            // const liftAngle = -70 * Math.PI / 180
            SceneState.robot.setJointValue(shoulderLiftJoint, liftAngle);
        }

        // elbow_joint 애니메이션
        const elbowJoint = 'elbow_joint';
        if (SceneState.robot.joints[elbowJoint]) {
            // limit: {lower: -2.6179938779914944, upper: 2.6179938779914944} => 약 -150도 ~ 150도
            const elbowAngle = Math.cos(time * 0.9) * (Math.PI / 2); // -90도에서 +90도
            // const elbowAngle = 80 * Math.PI / 180
            SceneState.robot.setJointValue(elbowJoint, elbowAngle);
        }

        // wrist_1_joint 애니메이션
        const wrist1Joint = 'wrist_1_joint';
        if (SceneState.robot.joints[wrist1Joint]) {
            // limit: {lower: -6.283185307179586, upper: 6.283185307179586} => 약 -360도 ~ 360도 (연속 회전 가능)
            const wrist1Angle = Math.sin(time * 1.1) * Math.PI; // -180도에서 +180도
            // const wrist1Angle = 165 * Math.PI / 180
            SceneState.robot.setJointValue(wrist1Joint, wrist1Angle);
        }

        // wrist_2_joint 애니메이션
        const wrist2Joint = 'wrist_2_joint';
        if (SceneState.robot.joints[wrist2Joint]) {
            // limit: {lower: -6.283185307179586, upper: 6.283185307179586}
            const wrist2Angle = Math.cos(time * 1.3 + Math.PI / 3) * (Math.PI / 1.5);
            // const wrist2Angle = -90 * Math.PI / 180
            SceneState.robot.setJointValue(wrist2Joint, wrist2Angle);
        }

        // wrist_3_joint 애니메이션
        const wrist3Joint = 'wrist_3_joint';
        if (SceneState.robot.joints[wrist3Joint]) {
            // limit: {lower: -6.283185307179586, upper: 6.283185307179586}
            // const wrist3Angle = Math.sin(time * 1.5 + Math.PI / 1.5) * Math.PI;
            const wrist3Angle = 0
            SceneState.robot.setJointValue(wrist3Joint, wrist3Angle);
        }
    }
}

export { object_setup, robot_setup, robot_animation, object_animation };