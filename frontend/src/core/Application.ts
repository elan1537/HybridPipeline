/**
 * Application - Main application class
 * CP4: Central orchestration of all systems with parallel initialization
 * This runs alongside the legacy code to verify identical behavior (Phase 1)
 */

import * as THREE from "three";
import { RenderingContext } from "./RenderingContext";
import { EventBus } from "./EventBus";
import { ApplicationState } from "./ApplicationState";
import { System, SystemContext, ApplicationConfig, RenderMode } from "../types";
import { WebSocketSystem } from "../systems/WebSocketSystem";
import { TextureManager } from "../systems/TextureManager";
import { CameraController } from "../systems/CameraController";
import { DebugSystem } from "../systems/DebugSystem";
import { RenderingSystem } from "../systems/RenderingSystem";
import { PhysicsSystem, CollisionResponseType } from "../systems/PhysicsSystem";

export class Application {
  // Core components
  private renderingContext: RenderingContext | null = null;
  private eventBus: EventBus;
  private state: ApplicationState;

  // Systems
  private systems: Map<string, System> = new Map();
  private websocketSystem: WebSocketSystem | null = null;
  private textureManager: TextureManager | null = null;
  private cameraController: CameraController | null = null;
  private debugSystem: DebugSystem | null = null;
  private renderingSystem: RenderingSystem | null = null;
  private physicsSystem: PhysicsSystem | null = null;

  // Render loop
  private clock: THREE.Clock = new THREE.Clock();
  private animationFrameId: number | null = null;
  private isRunning: boolean = false;

  // Time control for dynamic gaussians
  public timeController: TimeController;

  // Configuration
  private config: ApplicationConfig;

  constructor(config: ApplicationConfig) {
    this.config = config;

    // Create core components
    this.eventBus = new EventBus();
    this.state = new ApplicationState();

    // Initialize time controller for dynamic gaussians
    this.timeController = new TimeController();

    // Initialize state from config
    this.state.set("rendering:mode", config.renderMode);
    this.state.set("rendering:width", config.width);
    this.state.set("rendering:height", config.height);
    this.state.set("debug:consoleLogging", config.debugMode);

    // Store camera config if provided
    if (config.cameraConfig) {
      this.state.set("camera:config", config.cameraConfig);
      console.log("[Application] Camera config:", config.cameraConfig);
    }

    console.log("[Application] Created with config:", config);
  }

  /**
   * Phase 1: Wrap existing THREE.js objects
   * This allows the Application to work alongside legacy code
   */
  async initializeWithExistingObjects(
    scene: THREE.Scene,
    camera: THREE.PerspectiveCamera,
    renderer: THREE.WebGLRenderer,
    controls?: any,
    existingWorker?: Worker
  ): Promise<void> {
    console.log("[Application] Initializing with existing objects (Phase 1)...");

    // Wrap existing objects
    this.renderingContext = new RenderingContext(scene, camera, renderer, controls);

    // Create systems
    this.websocketSystem = new WebSocketSystem();
    this.textureManager = new TextureManager();
    this.cameraController = new CameraController();
    this.debugSystem = new DebugSystem();
    this.renderingSystem = new RenderingSystem();
    this.physicsSystem = new PhysicsSystem();

    // Register systems
    this.systems.set("websocket", this.websocketSystem);
    this.systems.set("texture", this.textureManager);
    this.systems.set("camera", this.cameraController);
    this.systems.set("debug", this.debugSystem);
    this.systems.set("rendering", this.renderingSystem);
    this.systems.set("physics", this.physicsSystem);

    // Set existing worker before initialization
    if (existingWorker && this.websocketSystem) {
      this.websocketSystem.setWorker(existingWorker);
      console.log("[Application] Using existing worker for WebSocketSystem");
    }

    // Create system context (after systems are registered)
    const context: SystemContext = {
      renderingContext: this.renderingContext,
      eventBus: this.eventBus,
      state: this.state,
      systems: this.systems,
    };

    // Initialize all systems in parallel
    console.log("[Application] Initializing systems in parallel...");
    await Promise.all(
      Array.from(this.systems.values()).map((system) => system.initialize(context))
    );

    console.log("[Application] All systems initialized");

    // Setup event listeners
    this.setupEventListeners();
  }

  /**
   * Phase 2: Create all objects internally (Future)
   * This will be used when we fully migrate away from global variables
   */
  async initialize(): Promise<void> {
    console.log("[Application] Initializing (Phase 2)...");

    // Create rendering context
    this.renderingContext = RenderingContext.create(
      {
        width: this.config.width,
        height: this.config.height,
        antialias: true,
      },
      this.config.canvas.parentElement || document.body
    );

    // Create system context
    const context: SystemContext = {
      renderingContext: this.renderingContext,
      eventBus: this.eventBus,
      state: this.state,
    };

    // Create and initialize systems
    this.websocketSystem = new WebSocketSystem();
    this.textureManager = new TextureManager();
    this.cameraController = new CameraController();
    this.debugSystem = new DebugSystem();

    this.systems.set("websocket", this.websocketSystem);
    this.systems.set("texture", this.textureManager);
    this.systems.set("camera", this.cameraController);
    this.systems.set("debug", this.debugSystem);

    await Promise.all(
      Array.from(this.systems.values()).map((system) => system.initialize(context))
    );

    // Setup event listeners
    this.setupEventListeners();

    console.log("[Application] Initialization complete");
  }

  /**
   * Setup event listeners for system communication
   */
  private setupEventListeners(): void {
    // Listen for video frames and update textures
    this.eventBus.on("frame:received", ({ data }) => {
      console.log('[Application] frame:received event fired, frameId:', data.frameId);
      if (this.textureManager) {
        console.log('[Application] Calling textureManager.updateFromVideoFrame()');
        this.textureManager.updateFromVideoFrame(data);
      } else {
        console.error('[Application] TextureManager not available!');
      }
    });

    // Listen for connection state changes
    this.eventBus.on("connection:open", () => {
      console.log("[Application] WebSocket connected");
    });

    this.eventBus.on("connection:close", () => {
      console.log("[Application] WebSocket disconnected");
    });

    // Listen for errors
    this.eventBus.on("system:error", ({ name, error }) => {
      console.error(`[Application] System error in ${name}:`, error);
    });
  }

  // ========================================================================
  // Render Loop
  // ========================================================================

  /**
   * Start the render loop
   */
  start(): void {
    if (this.isRunning) {
      console.warn("[Application] Already running");
      return;
    }

    this.isRunning = true;
    this.clock.start();
    this.animate();

    console.log("[Application] Started");
  }

  /**
   * Stop the render loop
   */
  stop(): void {
    if (!this.isRunning) {
      return;
    }

    this.isRunning = false;

    if (this.animationFrameId !== null) {
      cancelAnimationFrame(this.animationFrameId);
      this.animationFrameId = null;
    }

    console.log("[Application] Stopped");
  }

  /**
   * Main animation loop
   */
  private animate = (): void => {
    if (!this.isRunning) {
      return;
    }

    this.animationFrameId = requestAnimationFrame(this.animate);

    const deltaTime = this.clock.getDelta();
    this.update(deltaTime);
    this.render();
  };

  /**
   * Update all systems
   */
  private update(deltaTime: number): void {
    // Update all systems
    this.systems.forEach((system) => {
      system.update(deltaTime);
    });

    // Update frame counter
    const isPlaying = this.state.get("animation:playing");
    if (isPlaying) {
      const frameCounter = this.state.getOrDefault("frame:counter", 0);
      const newCounter = (frameCounter + 1) % 300;
      this.state.set("frame:counter", newCounter);
      this.state.set("frame:timeIndex", newCounter / 299.0);
    }
  }

  /**
   * Render the scene
   */
  private render(): void {
    // Delegate to RenderingSystem if configured
    if (this.renderingSystem) {
      this.renderingSystem.render();
    }
  }

  // ========================================================================
  // Public API
  // ========================================================================

  /**
   * Connect to WebSocket server
   */
  connectWebSocket(url: string): void {
    if (this.websocketSystem) {
      const width = this.state.getOrDefault("rendering:width", 1280);
      const height = this.state.getOrDefault("rendering:height", 720);
      this.websocketSystem.connect(url, width, height);
    }
  }

  /**
   * Disconnect from WebSocket server
   */
  disconnectWebSocket(): void {
    if (this.websocketSystem) {
      this.websocketSystem.disconnect();
    }
  }

  /**
   * Send camera frame to server
   * Uses TimeController to get current time index for dynamic gaussians
   */
  sendCameraFrame(): void {
    console.log('[Application] sendCameraFrame() called');

    if (!this.cameraController) {
      console.warn('[Application] sendCameraFrame: cameraController not available');
      return;
    }

    console.log('[Application] Getting camera frame...');

    // Get time index from TimeController
    // Note: TimeController.update() is called externally in the render loop
    const timeIndex = this.timeController.getCurrentTime();

    const frame = this.cameraController.getCameraFrame(timeIndex);
    console.log('[Application] Got camera frame:', frame.frameId, 'timeIndex:', timeIndex.toFixed(4));

    console.log('[Application] websocketSystem exists:', !!this.websocketSystem);

    if (this.websocketSystem) {
      console.log('[Application] Calling websocketSystem.sendCameraFrame()');
      try {
        this.websocketSystem.sendCameraFrame(frame);
        console.log('[Application] websocketSystem.sendCameraFrame() returned');
      } catch (error) {
        console.error('[Application] Error calling sendCameraFrame:', error);
      }
    } else {
      console.warn('[Application] sendCameraFrame: websocketSystem not available');
    }
  }

  /**
   * Set render mode
   */
  setRenderMode(mode: RenderMode): void {
    this.state.set("rendering:mode", mode);
    this.eventBus.emit("ui:mode-changed", { mode });
  }

  /**
   * Resize application
   */
  resize(width: number, height: number): void {
    this.state.set("rendering:width", width);
    this.state.set("rendering:height", height);

    if (this.renderingContext) {
      this.renderingContext.resize(width, height);
    }
  }

  // ========================================================================
  // System Getters
  // ========================================================================

  getSystem<T extends System>(name: string): T | undefined {
    return this.systems.get(name) as T | undefined;
  }

  getWebSocketSystem(): WebSocketSystem | null {
    return this.websocketSystem;
  }

  getTextureManager(): TextureManager | null {
    return this.textureManager;
  }

  getCameraController(): CameraController | null {
    return this.cameraController;
  }

  /**
   * Update camera configuration
   * This will update near/far/fov and notify all dependent systems
   */
  updateCameraConfig(params: Partial<import("../types").CameraConfig>): void {
    if (!this.cameraController) {
      console.warn("[Application] CameraController not available");
      return;
    }

    this.cameraController.updateCameraParams(params);

    // PhysicsSystem will automatically pick up the new values on next update
    if (this.physicsSystem) {
      const camera = this.cameraController.getCamera();
      if (camera) {
        console.log("[Application] Updating PhysicsSystem with new camera params");
        // PhysicsSystem needs to update its collision detector
        const collisionDetector = (this.physicsSystem as any).collisionDetector;
        if (collisionDetector) {
          collisionDetector.updateClippingPlanes(camera.near, camera.far);
        }
      }
    }
  }

  getDebugSystem(): DebugSystem | null {
    return this.debugSystem;
  }

  getRenderingSystem(): RenderingSystem | null {
    return this.renderingSystem;
  }

  getPhysicsSystem(): PhysicsSystem | null {
    return this.physicsSystem;
  }

  getRenderingContext(): RenderingContext | null {
    return this.renderingContext;
  }

  getEventBus(): EventBus {
    return this.eventBus;
  }

  getState(): ApplicationState {
    return this.state;
  }

  // ========================================================================
  // Physics API
  // ========================================================================

  /**
   * Add a mesh to physics simulation
   *
   * @param mesh - Three.js mesh to add
   * @param options - Physics properties (velocity, acceleration, mass, etc.)
   * @returns PhysicsMesh object for direct manipulation
   */
  addPhysicsMesh(mesh: THREE.Mesh, options?: {
    velocity?: THREE.Vector3;
    acceleration?: THREE.Vector3;
    mass?: number;
    restitution?: number;
    friction?: number;
  }) {
    if (!this.physicsSystem) {
      console.warn("[Application] PhysicsSystem not available");
      return null;
    }

    return this.physicsSystem.addMesh(mesh, options);
  }

  /**
   * Remove a mesh from physics simulation
   */
  removePhysicsMesh(mesh: THREE.Mesh): boolean {
    if (!this.physicsSystem) {
      console.warn("[Application] PhysicsSystem not available");
      return false;
    }

    return this.physicsSystem.removeMesh(mesh);
  }

  /**
   * Set collision response type for all physics objects
   *
   * @param type - "stop" | "bounce" | "slide"
   */
  setPhysicsResponseType(type: "stop" | "bounce" | "slide"): void {
    if (!this.physicsSystem) {
      console.warn("[Application] PhysicsSystem not available");
      return;
    }

    switch (type) {
      case "stop":
        this.physicsSystem.setResponseType(CollisionResponseType.Stop);
        break;
      case "bounce":
        this.physicsSystem.setResponseType(CollisionResponseType.Bounce);
        break;
      case "slide":
        this.physicsSystem.setResponseType(CollisionResponseType.Slide);
        break;
    }
  }

  /**
   * Set global gravity
   */
  setGravity(gravity: THREE.Vector3): void {
    if (!this.physicsSystem) {
      console.warn("[Application] PhysicsSystem not available");
      return;
    }

    this.physicsSystem.setGravity(gravity);
  }

  /**
   * Set collision detection threshold (in meters)
   */
  setCollisionEpsilon(epsilon: number): void {
    if (!this.physicsSystem) {
      console.warn("[Application] PhysicsSystem not available");
      return;
    }

    this.physicsSystem.setCollisionEpsilon(epsilon);
  }

  // ========================================================================
  // Cleanup
  // ========================================================================

  /**
   * Dispose of all resources
   */
  dispose(): void {
    console.log("[Application] Disposing...");

    // Stop render loop
    this.stop();

    // Dispose all systems
    this.systems.forEach((system) => {
      system.dispose();
    });
    this.systems.clear();

    // Clear event listeners and state
    this.eventBus.clear();
    this.state.clear();

    // Dispose rendering context
    if (this.renderingContext) {
      this.renderingContext.dispose();
      this.renderingContext = null;
    }

    console.log("[Application] Disposed");
  }
}

/**
 * TimeController - Controls time index for dynamic gaussian rendering
 *
 * Supports three modes:
 * 1. Automatic playback: time advances based on deltaTime and playback speed
 * 2. Manual scrubbing: time is set explicitly via seek()
 * 3. Hybrid: automatic playback with manual override
 *
 * Time is normalized to [0.0, 1.0] range and loops automatically
 */
export class TimeController {
  private isPlaying: boolean = true;  // Auto-play by default for dynamic gaussians
  private playbackSpeed: number = 1.0;
  private currentTime: number = 0.0;
  private manualOverride: boolean = false;

  /**
   * Update time based on deltaTime
   * Called every frame in the render loop
   *
   * @param deltaTime Time since last frame in seconds
   * @returns Current time index [0.0, 1.0]
   */
  update(deltaTime: number): number {
    if (this.isPlaying && !this.manualOverride) {
      // Advance time based on playback speed
      this.currentTime += deltaTime * this.playbackSpeed;

      // Loop time in [0.0, 1.0] range
      this.currentTime = this.currentTime % 1.0;

      // Handle negative modulo (when playbackSpeed < 0)
      if (this.currentTime < 0) {
        this.currentTime += 1.0;
      }
    }

    return this.currentTime;
  }

  /**
   * Start automatic playback
   */
  play(): void {
    this.isPlaying = true;
    this.manualOverride = false;
  }

  /**
   * Pause automatic playback
   */
  pause(): void {
    this.isPlaying = false;
  }

  /**
   * Toggle play/pause
   */
  togglePlay(): void {
    this.isPlaying = !this.isPlaying;
    if (this.isPlaying) {
      this.manualOverride = false;
    }
  }

  /**
   * Set playback speed
   * @param speed Playback speed multiplier (1.0 = real-time, 2.0 = 2x speed, etc.)
   */
  setSpeed(speed: number): void {
    this.playbackSpeed = speed;
  }

  /**
   * Seek to specific time (manual override)
   * @param time Time index [0.0, 1.0]
   */
  seek(time: number): void {
    this.currentTime = Math.max(0.0, Math.min(1.0, time));
    this.manualOverride = true;
  }

  /**
   * Resume automatic playback after manual override
   */
  resumeAuto(): void {
    this.manualOverride = false;
  }

  /**
   * Get current time index
   */
  getCurrentTime(): number {
    return this.currentTime;
  }

  /**
   * Get playback state
   */
  isCurrentlyPlaying(): boolean {
    return this.isPlaying;
  }

  /**
   * Get playback speed
   */
  getSpeed(): number {
    return this.playbackSpeed;
  }

  /**
   * Check if in manual override mode
   */
  isManual(): boolean {
    return this.manualOverride;
  }

  /**
   * Reset to initial state
   */
  reset(): void {
    this.currentTime = 0.0;
    this.isPlaying = false;
    this.playbackSpeed = 1.0;
    this.manualOverride = false;
  }
}
