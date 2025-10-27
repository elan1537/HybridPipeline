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

  // Render loop
  private clock: THREE.Clock = new THREE.Clock();
  private animationFrameId: number | null = null;
  private isRunning: boolean = false;

  // Configuration
  private config: ApplicationConfig;

  constructor(config: ApplicationConfig) {
    this.config = config;

    // Create core components
    this.eventBus = new EventBus();
    this.state = new ApplicationState();

    // Initialize state from config
    this.state.set("rendering:mode", config.renderMode);
    this.state.set("rendering:width", config.width);
    this.state.set("rendering:height", config.height);
    this.state.set("debug:consoleLogging", config.debugMode);

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

    // Create system context
    const context: SystemContext = {
      renderingContext: this.renderingContext,
      eventBus: this.eventBus,
      state: this.state,
    };

    // Create systems
    this.websocketSystem = new WebSocketSystem();
    this.textureManager = new TextureManager();
    this.cameraController = new CameraController();
    this.debugSystem = new DebugSystem();

    // Register systems
    this.systems.set("websocket", this.websocketSystem);
    this.systems.set("texture", this.textureManager);
    this.systems.set("camera", this.cameraController);
    this.systems.set("debug", this.debugSystem);

    // Set existing worker before initialization
    if (existingWorker && this.websocketSystem) {
      this.websocketSystem.setWorker(existingWorker);
      console.log("[Application] Using existing worker for WebSocketSystem");
    }

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
   * Note: In Phase 1, this is a no-op since legacy code handles rendering
   * In Phase 2, this will handle all rendering
   */
  private render(): void {
    // Phase 1: No-op, legacy code handles rendering
    // Phase 2: Implement full rendering pipeline here
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
   */
  sendCameraFrame(): void {
    if (!this.cameraController) {
      return;
    }

    const timeIndex = this.state.getOrDefault("frame:timeIndex", 0);
    const frame = this.cameraController.getCameraFrame(timeIndex);

    if (this.websocketSystem) {
      this.websocketSystem.sendCameraFrame(frame);
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

  getDebugSystem(): DebugSystem | null {
    return this.debugSystem;
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
