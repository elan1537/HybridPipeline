/**
 * RenderingContext - Encapsulates THREE.js rendering state
 * CP2: Read-only wrapper class that wraps existing THREE.js objects
 * This class doesn't modify existing code, just provides a cleaner interface
 */

import * as THREE from "three";
import { OrbitControls } from "three/addons";
import { RenderConfig } from "../types";

/**
 * RenderingContext wraps and manages all THREE.js rendering objects
 * Phase 1: Read-only wrapper (doesn't modify existing code)
 * Phase 2: Will become the authoritative source for rendering state
 */
export class RenderingContext {
  // Core rendering objects
  readonly scene: THREE.Scene;
  readonly camera: THREE.PerspectiveCamera;
  readonly renderer: THREE.WebGLRenderer;
  readonly canvas: HTMLCanvasElement;

  // Controls
  readonly controls: OrbitControls | null;

  // Render targets and textures
  private renderTargets: Map<string, THREE.WebGLRenderTarget>;
  private textures: Map<string, THREE.Texture>;
  private scenes: Map<string, THREE.Scene>;
  private cameras: Map<string, THREE.Camera>;
  private meshes: Map<string, THREE.Mesh>;
  private materials: Map<string, THREE.Material>;

  /**
   * Phase 1 Constructor: Wraps existing objects without creating new ones
   * This allows us to use RenderingContext alongside legacy code
   */
  constructor(
    scene: THREE.Scene,
    camera: THREE.PerspectiveCamera,
    renderer: THREE.WebGLRenderer,
    controls?: OrbitControls
  ) {
    this.scene = scene;
    this.camera = camera;
    this.renderer = renderer;
    this.canvas = renderer.domElement as HTMLCanvasElement;
    this.controls = controls || null;

    // Initialize maps
    this.renderTargets = new Map();
    this.textures = new Map();
    this.scenes = new Map();
    this.cameras = new Map();
    this.meshes = new Map();
    this.materials = new Map();

    // Register the main scene
    this.scenes.set("main", scene);
    this.cameras.set("main", camera);
  }

  /**
   * Phase 2 Constructor (Future): Creates all objects internally
   * This will be used when we fully migrate away from global variables
   */
  static create(config: RenderConfig, container: HTMLElement): RenderingContext {
    // Create camera
    const aspect = config.width / config.height;
    const camera = new THREE.PerspectiveCamera(75, aspect, 0.1, 1000);

    // Create scene
    const scene = new THREE.Scene();

    // Create renderer
    const renderer = new THREE.WebGLRenderer({
      antialias: config.antialias,
      logarithmicDepthBuffer: true,
    });
    renderer.setPixelRatio(config.pixelRatio || window.devicePixelRatio);
    renderer.setSize(config.width, config.height);

    container.appendChild(renderer.domElement);

    // Create controls
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.1;

    return new RenderingContext(scene, camera, renderer, controls);
  }

  // ========================================================================
  // Render Target Management
  // ========================================================================

  registerRenderTarget(name: string, renderTarget: THREE.WebGLRenderTarget): void {
    this.renderTargets.set(name, renderTarget);
  }

  getRenderTarget(name: string): THREE.WebGLRenderTarget | undefined {
    return this.renderTargets.get(name);
  }

  removeRenderTarget(name: string): void {
    const rt = this.renderTargets.get(name);
    if (rt) {
      rt.dispose();
      this.renderTargets.delete(name);
    }
  }

  // ========================================================================
  // Texture Management
  // ========================================================================

  registerTexture(name: string, texture: THREE.Texture): void {
    this.textures.set(name, texture);
  }

  getTexture(name: string): THREE.Texture | undefined {
    return this.textures.get(name);
  }

  removeTexture(name: string): void {
    const texture = this.textures.get(name);
    if (texture) {
      texture.dispose();
      this.textures.delete(name);
    }
  }

  // ========================================================================
  // Scene Management
  // ========================================================================

  registerScene(name: string, scene: THREE.Scene): void {
    this.scenes.set(name, scene);
  }

  getScene(name: string): THREE.Scene | undefined {
    return this.scenes.get(name);
  }

  removeScene(name: string): void {
    this.scenes.delete(name);
  }

  // ========================================================================
  // Camera Management
  // ========================================================================

  registerCamera(name: string, camera: THREE.Camera): void {
    this.cameras.set(name, camera);
  }

  getCamera(name: string): THREE.Camera | undefined {
    return this.cameras.get(name);
  }

  removeCamera(name: string): void {
    this.cameras.delete(name);
  }

  // ========================================================================
  // Mesh Management
  // ========================================================================

  registerMesh(name: string, mesh: THREE.Mesh): void {
    this.meshes.set(name, mesh);
  }

  getMesh(name: string): THREE.Mesh | undefined {
    return this.meshes.get(name);
  }

  removeMesh(name: string): void {
    const mesh = this.meshes.get(name);
    if (mesh) {
      mesh.geometry.dispose();
      if (mesh.material instanceof THREE.Material) {
        mesh.material.dispose();
      } else if (Array.isArray(mesh.material)) {
        mesh.material.forEach((m) => m.dispose());
      }
      this.meshes.delete(name);
    }
  }

  // ========================================================================
  // Material Management
  // ========================================================================

  registerMaterial(name: string, material: THREE.Material): void {
    this.materials.set(name, material);
  }

  getMaterial(name: string): THREE.Material | undefined {
    return this.materials.get(name);
  }

  removeMaterial(name: string): void {
    const material = this.materials.get(name);
    if (material) {
      material.dispose();
      this.materials.delete(name);
    }
  }

  // ========================================================================
  // Utility Methods
  // ========================================================================

  /**
   * Resize the renderer and update camera
   */
  resize(width: number, height: number): void {
    this.renderer.setSize(width, height);
    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();
  }

  /**
   * Get current viewport size
   */
  getSize(): { width: number; height: number } {
    const size = new THREE.Vector2();
    this.renderer.getSize(size);
    return { width: size.x, height: size.y };
  }

  /**
   * Clear all resources
   */
  clear(): void {
    // Dispose all render targets
    this.renderTargets.forEach((rt) => rt.dispose());
    this.renderTargets.clear();

    // Dispose all textures
    this.textures.forEach((tex) => tex.dispose());
    this.textures.clear();

    // Dispose all meshes
    this.meshes.forEach((mesh) => {
      mesh.geometry.dispose();
      if (mesh.material instanceof THREE.Material) {
        mesh.material.dispose();
      } else if (Array.isArray(mesh.material)) {
        mesh.material.forEach((m) => m.dispose());
      }
    });
    this.meshes.clear();

    // Dispose all materials
    this.materials.forEach((mat) => mat.dispose());
    this.materials.clear();

    // Clear collections
    this.scenes.clear();
    this.cameras.clear();
  }

  /**
   * Dispose of the rendering context
   */
  dispose(): void {
    this.clear();
    this.renderer.dispose();
    if (this.controls) {
      this.controls.dispose();
    }
  }
}
