import * as THREE from 'three';
import type { URDFRobot } from 'urdf-loader';

export interface SceneStateType {
    worldScene: THREE.Scene | THREE.Group | null
    scene: THREE.Scene | THREE.Group | null
    mesh: THREE.Mesh | null
    mesh2: THREE.Mesh | null
    robot: URDFRobot | null
}

export const SceneState: SceneStateType = {
    worldScene: null,
    scene: null,
    mesh: null,
    mesh2: null,
    robot: null,
}