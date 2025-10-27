/**
 * CameraController - Manages camera and controls
 * CP3: Centralized camera management
 * Uses the same projection logic as legacy code to avoid depth mapping distortion
 */

import * as THREE from "three";
import { OrbitControls } from "three/addons";
import { System, SystemContext, CameraMode, CameraFrame } from "../types";

export class CameraController implements System {
  readonly name = "camera";

  private context: SystemContext | null = null;
  private camera: THREE.PerspectiveCamera | null = null;
  private controls: OrbitControls | null = null;
  private mode: CameraMode = CameraMode.Orbit;

  // Adaptive camera update tracking
  private lastPosition: THREE.Vector3 = new THREE.Vector3();
  private lastTarget: THREE.Vector3 = new THREE.Vector3();
  private positionThresholdSquared: number = 0.001 * 0.001;
  private targetThresholdSquared: number = 0.001 * 0.001;

  // Frame counter
  private frameId: number = 0;

  // Render resolution (for intrinsics calculation)
  private renderWidth: number = 1280;
  private renderHeight: number = 720;

  async initialize(context: SystemContext): Promise<void> {
    this.context = context;

    // Get camera and controls from rendering context
    this.camera = context.renderingContext.camera;
    this.controls = context.renderingContext.controls;

    // Get render resolution from state
    this.renderWidth = context.state.getOrDefault("rendering:width", 1280);
    this.renderHeight = context.state.getOrDefault("rendering:height", 720);

    // Subscribe to resolution changes
    context.state.subscribe("rendering:width", (width: number) => {
      this.renderWidth = width;
    });

    context.state.subscribe("rendering:height", (height: number) => {
      this.renderHeight = height;
    });

    if (this.camera) {
      this.lastPosition.copy(this.camera.position);
    }

    if (this.controls) {
      this.lastTarget.copy(this.controls.target);
    }

    console.log("[CameraController] Initialized");
  }

  update(deltaTime: number): void {
    if (!this.camera || !this.controls) {
      return;
    }

    // Update controls (damping, etc.)
    this.controls.update();

    // Check if camera moved significantly
    const moved = this.hasCameraMoved();
    if (moved) {
      this.lastPosition.copy(this.camera.position);
      this.lastTarget.copy(this.controls.target);

      if (this.context) {
        this.context.eventBus.emit("system:initialized", {
          name: "camera-moved",
        });
      }
    }
  }

  dispose(): void {
    if (this.controls) {
      this.controls.dispose();
      this.controls = null;
    }
  }

  // ========================================================================
  // Camera Movement Detection
  // ========================================================================

  private hasCameraMoved(): boolean {
    if (!this.camera || !this.controls) {
      return false;
    }

    const positionDelta = this.camera.position.distanceToSquared(this.lastPosition);
    const targetDelta = this.controls.target.distanceToSquared(this.lastTarget);

    return (
      positionDelta > this.positionThresholdSquared ||
      targetDelta > this.targetThresholdSquared
    );
  }

  // ========================================================================
  // Camera Intrinsics (Legacy Logic)
  // ========================================================================

  /**
   * Calculate camera intrinsics matrix (3x3 matrix as 9-element array)
   * This follows the exact logic from the legacy code to avoid depth mapping distortion
   *
   * IMPORTANT: This must match the backend's expectation for proper Gaussian-depth alignment
   */
  private getCameraIntrinsics(): number[] {
    if (!this.camera) {
      return [1, 0, 0, 0, 1, 0, 0, 0, 1];
    }

    const projmat = this.camera.projectionMatrix;

    // Calculate focal length from projection matrix
    // This scales properly with resolution
    const fx = (this.renderWidth / 2) * projmat.elements[0];
    const fy = (this.renderHeight / 2) * projmat.elements[5];

    // Principal point is at the center of the resolution
    const cx = this.renderWidth / 2;
    const cy = this.renderHeight / 2;

    // Return 3x3 intrinsics matrix in row-major order:
    // [fx,  0, cx]
    // [ 0, fy, cy]
    // [ 0,  0,  1]
    return [fx, 0, cx, 0, fy, cy, 0, 0, 1];
  }

  // ========================================================================
  // Public API
  // ========================================================================

  /**
   * Set camera mode
   */
  setMode(mode: CameraMode): void {
    this.mode = mode;

    if (this.controls) {
      switch (mode) {
        case CameraMode.Orbit:
          this.controls.enabled = true;
          break;
        case CameraMode.FirstPerson:
        case CameraMode.ThirdPerson:
          this.controls.enabled = false;
          break;
      }
    }
  }

  /**
   * Set camera position
   */
  setPosition(x: number, y: number, z: number): void {
    if (this.camera) {
      this.camera.position.set(x, y, z);
      this.lastPosition.copy(this.camera.position);
    }
  }

  /**
   * Set camera target
   */
  setTarget(x: number, y: number, z: number): void {
    if (this.controls) {
      this.controls.target.set(x, y, z);
      this.lastTarget.copy(this.controls.target);
    }
  }

  /**
   * Get camera position
   */
  getPosition(): THREE.Vector3 {
    return this.camera ? this.camera.position.clone() : new THREE.Vector3();
  }

  /**
   * Get camera target
   */
  getTarget(): THREE.Vector3 {
    return this.controls ? this.controls.target.clone() : new THREE.Vector3();
  }

  /**
   * Get view matrix
   */
  getViewMatrix(): THREE.Matrix4 {
    if (!this.camera) {
      return new THREE.Matrix4();
    }
    return this.camera.matrixWorldInverse.clone();
  }

  /**
   * Get projection matrix
   */
  getProjectionMatrix(): THREE.Matrix4 {
    if (!this.camera) {
      return new THREE.Matrix4();
    }
    return this.camera.projectionMatrix.clone();
  }

  /**
   * Get camera frame for sending to server
   * Uses the same intrinsics calculation as legacy code
   */
  getCameraFrame(timeIndex: number = 0): CameraFrame {
    if (!this.camera || !this.controls) {
      return {
        eye: new Float32Array(3),
        target: new Float32Array(3),
        intrinsics: new Float32Array(9),
        frameId: this.frameId++,
        timestamp: performance.now(),
        timeIndex,
      };
    }

    const eye = new Float32Array([
      this.camera.position.x,
      this.camera.position.y,
      this.camera.position.z,
    ]);

    const target = new Float32Array([
      this.controls.target.x,
      this.controls.target.y,
      this.controls.target.z,
    ]);

    // Use the legacy intrinsics calculation method
    const intrinsics = new Float32Array(this.getCameraIntrinsics());

    return {
      eye,
      target,
      intrinsics,
      frameId: this.frameId++,
      timestamp: performance.now(),
      timeIndex,
    };
  }

  /**
   * Save camera position to local storage
   */
  savePosition(): void {
    if (!this.camera || !this.controls) {
      return;
    }

    const cameraState = {
      position: this.camera.position.toArray(),
      target: this.controls.target.toArray(),
      fov: this.camera.fov,
    };

    localStorage.setItem("cameraPosition", JSON.stringify(cameraState));
    console.log("[CameraController] Camera position saved");
  }

  /**
   * Load camera position from local storage
   */
  loadPosition(): boolean {
    const saved = localStorage.getItem("cameraPosition");
    if (!saved || !this.camera || !this.controls) {
      return false;
    }

    try {
      const cameraState = JSON.parse(saved);
      this.camera.position.fromArray(cameraState.position);
      this.controls.target.fromArray(cameraState.target);
      this.camera.fov = cameraState.fov;
      this.camera.updateProjectionMatrix();

      this.lastPosition.copy(this.camera.position);
      this.lastTarget.copy(this.controls.target);

      console.log("[CameraController] Camera position loaded");
      return true;
    } catch (error) {
      console.error("[CameraController] Failed to load camera position:", error);
      return false;
    }
  }

  /**
   * Enable/disable controls
   */
  setControlsEnabled(enabled: boolean): void {
    if (this.controls) {
      this.controls.enabled = enabled;
    }
  }

  /**
   * Enable/disable auto-rotate
   */
  setAutoRotate(enabled: boolean, speed: number = 0.001): void {
    if (this.controls) {
      this.controls.autoRotate = enabled;
      this.controls.autoRotateSpeed = speed;
    }
  }

  /**
   * Update render resolution (for intrinsics calculation)
   */
  setRenderResolution(width: number, height: number): void {
    this.renderWidth = width;
    this.renderHeight = height;
  }

  // ========================================================================
  // Getters
  // ========================================================================

  getCamera(): THREE.PerspectiveCamera | null {
    return this.camera;
  }

  getControls(): OrbitControls | null {
    return this.controls;
  }

  getMode(): CameraMode {
    return this.mode;
  }

  getFrameId(): number {
    return this.frameId;
  }

  getRenderResolution(): { width: number; height: number } {
    return { width: this.renderWidth, height: this.renderHeight };
  }
}
