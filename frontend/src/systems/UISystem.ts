/**
 * UISystem - Central UI management system
 * Manages all UI panels and coordinates with other systems
 * Uses constructor injection for Application dependency
 */

import * as THREE from 'three';
import { OrbitControls } from 'three/addons';
import { System, SystemContext } from '../types';
import { Application } from '../core/Application';
import { WebSocketSystem } from './WebSocketSystem';
import { TextureManager } from './TextureManager';
import { LatencyTracker } from '../latency-tracker';
import { debug } from '../debug-logger';

// Import panels
import { StatsPanel } from '../ui/panels/StatsPanel';
import { ControlPanel, RenderMode } from '../ui/panels/ControlPanel';
import { DebugPanel } from '../ui/panels/DebugPanel';
import { RecordingPanel } from '../ui/panels/RecordingPanel';
import { FPSTestPanel } from '../ui/panels/FPSTestPanel';
import { TimeControlUI } from '../ui/TimeControlUI';
import { FrameScrubber } from '../ui/FrameScrubber';

// Import managers
import { CameraStateManager } from '../ui/managers/CameraStateManager';

export class UISystem implements System {
  readonly name = 'ui';

  // Panels (public for external access)
  readonly stats: StatsPanel;
  readonly control: ControlPanel;
  readonly debug: DebugPanel;
  readonly recording: RecordingPanel;
  readonly fpsTest: FPSTestPanel;
  readonly timeControl: TimeControlUI;
  readonly frameScrubber: FrameScrubber;

  // Application reference (injected via constructor)
  private app: Application;
  private latencyTracker: LatencyTracker;

  // Camera references (for debug panel)
  private camera: THREE.PerspectiveCamera | null = null;
  private controls: OrbitControls | null = null;

  constructor(app: Application, latencyTracker: LatencyTracker) {
    this.app = app;
    this.latencyTracker = latencyTracker;

    debug.logMain('[UISystem] Creating panels...');

    // Create panels with callbacks
    this.stats = new StatsPanel();

    this.control = new ControlPanel({
      onConnect: () => this.handleConnect(),
      onDisconnect: () => this.handleDisconnect(),
      onEncoderChange: (isJpeg) => this.handleEncoderChange(isJpeg),
      onDepthDebugToggle: (enabled) => this.handleDepthDebugToggle(enabled),
      onRenderModeChange: (mode) => this.handleRenderModeChange(mode),
    });

    this.debug = new DebugPanel({
      onConsoleDebugToggle: (enabled) => this.handleConsoleDebugToggle(enabled),
      onCameraDebugToggle: (enabled) => this.handleCameraDebugToggle(enabled),
      onSaveCamera: () => this.handleSaveCamera(),
      onLoadCamera: () => this.handleLoadCamera(),
    });

    this.recording = new RecordingPanel();

    this.fpsTest = new FPSTestPanel({
      onTestStart: () => this.handleFPSTestStart(),
      onTestStop: () => this.handleFPSTestStop(),
      onDownloadResults: () => this.handleFPSResultsDownload(),
    });

    // Create time control UI for dynamic gaussians
    this.timeControl = new TimeControlUI(app.timeController);
    this.frameScrubber = new FrameScrubber(app.timeController, 300); // 300 frames default

    debug.logMain('[UISystem] Panels created');
  }

  async initialize(_context: SystemContext): Promise<void> {
    // Get camera references from rendering context
    const renderingContext = this.app.getRenderingContext();
    if (renderingContext) {
      this.camera = renderingContext.camera as THREE.PerspectiveCamera;
      this.controls = renderingContext.controls as OrbitControls;

      // Set camera references in debug panel
      this.debug.setCameraReferences(this.camera, this.controls);

      // Auto-load saved camera position
      if (this.camera && this.controls) {
        CameraStateManager.load(this.camera, this.controls);
      }
    }

    // Get canvas for recording panel
    const canvas = renderingContext?.renderer?.domElement as HTMLCanvasElement;
    if (canvas) {
      this.recording.setCanvas(canvas);
    }

    // Add time control UI to DOM
    const timeControlContainer = document.getElementById('time-controls');
    if (timeControlContainer) {
      timeControlContainer.appendChild(this.timeControl.getElement());
      timeControlContainer.appendChild(this.frameScrubber.getElement());
      debug.logMain('[UISystem] Time controls added to DOM');
    } else {
      debug.logMain('[UISystem] Warning: #time-controls container not found, time controls not mounted');
    }

    debug.logMain('[UISystem] Initialized');
  }

  update(_deltaTime: number): void {
    // Update debug panel camera info if enabled
    if (this.debug.isCameraDebugEnabled()) {
      this.debug.updateCameraInfo();
    }

    // Update time control UI
    this.timeControl.update();
    this.frameScrubber.update();
  }

  dispose(): void {
    this.stats.cleanup();
    this.control.cleanup();
    this.debug.cleanup();
    this.recording.cleanup();
    this.fpsTest.cleanup();
    this.timeControl.dispose();
    this.frameScrubber.dispose();

    debug.logMain('[UISystem] Disposed');
  }

  // ========================================================================
  // Control Panel Handlers
  // ========================================================================

  private handleConnect(): void {
    const ws = this.app.getSystem<WebSocketSystem>('websocket');
    if (!ws) {
      debug.error('[UISystem] WebSocketSystem not found');
      return;
    }

    const isJpeg = this.control.isJpegMode();
    const wsURL = `wss://${location.host}/ws/${isJpeg ? 'jpeg' : 'h264'}`;

    debug.logMain(`[UISystem] Connecting to ${wsURL}`);

    // Get rendering dimensions
    const width = this.app.getState().getOrDefault('rendering:width', 1280);
    const height = this.app.getState().getOrDefault('rendering:height', 720);

    ws.reconnect(wsURL, width, height);
  }

  private handleDisconnect(): void {
    const ws = this.app.getSystem<WebSocketSystem>('websocket');
    if (!ws) {
      debug.error('[UISystem] WebSocketSystem not found');
      return;
    }

    debug.logMain('[UISystem] Disconnecting WebSocket');
    ws.disconnect();
  }

  private handleEncoderChange(isJpeg: boolean): void {
    const ws = this.app.getSystem<WebSocketSystem>('websocket');
    const tex = this.app.getSystem<TextureManager>('texture');

    if (!ws || !tex) {
      debug.error('[UISystem] Required systems not found');
      return;
    }

    debug.logMain(`[UISystem] Changing encoder to ${isJpeg ? 'JPEG' : 'H264'}`);

    // 1. Backend encoder change
    const encoderType = isJpeg ? 'jpeg' : 'h264';
    ws.changeEncoderType(encoderType);

    // 2. Frontend decoder update
    ws.toggleJPEGFallback(isJpeg);

    // 3. TextureManager update
    tex.setJpegMode(isJpeg);
  }

  private handleDepthDebugToggle(enabled: boolean): void {
    debug.logMain(`[UISystem] Depth debug ${enabled ? 'enabled' : 'disabled'}`);
    // TODO: Implement depth debug visualization
  }

  private handleRenderModeChange(mode: RenderMode): void {
    debug.logMain(`[UISystem] Render mode changed to ${mode}`);

    // Update recording panel
    this.recording.setRenderMode(mode);

    // Handle feed-forward mode special case
    if (mode === RenderMode.FEED_FORWARD) {
      const ws = this.app.getSystem<WebSocketSystem>('websocket');
      if (ws) {
        ws.disconnect();

        setTimeout(() => {
          const width = this.app.getState().getOrDefault('rendering:width', 1280);
          const height = this.app.getState().getOrDefault('rendering:height', 720);
          const wsURL = `wss://${location.host}/ws/feedforward`;

          ws.reconnect(wsURL, width, height);
          debug.logMain(`[UISystem] Reconnected for feed-forward mode`);
        }, 100);
      }
    }
  }

  // ========================================================================
  // Debug Panel Handlers
  // ========================================================================

  private handleConsoleDebugToggle(enabled: boolean): void {
    debug.setDebugEnabled(enabled);

    // Notify worker
    const ws = this.app.getSystem<WebSocketSystem>('websocket');
    if (ws) {
      // TODO: Add method to WebSocketSystem to toggle worker debug
      debug.logMain('[UISystem] Worker debug toggle - implementation needed');
    }
  }

  private handleCameraDebugToggle(enabled: boolean): void {
    debug.logMain(`[UISystem] Camera debug ${enabled ? 'enabled' : 'disabled'}`);
  }

  private handleSaveCamera(): void {
    debug.logMain('[UISystem] Camera saved via DebugPanel');
  }

  private handleLoadCamera(): void {
    debug.logMain('[UISystem] Camera loaded via DebugPanel');
  }

  // ========================================================================
  // FPS Test Panel Handlers
  // ========================================================================

  private handleFPSTestStart(): void {
    const ws = this.app.getSystem<WebSocketSystem>('websocket');
    if (!ws) {
      debug.error('[UISystem] WebSocketSystem not found');
      return;
    }

    // Send message to worker
    ws.startFPSMeasurement();

    // Start LatencyTracker measurement
    this.latencyTracker.startFPSMeasurement();

    // Update UI
    this.fpsTest.showTestStarted();

    debug.logFPS('[UISystem] FPS test started - Worker and LatencyTracker active');
  }

  private handleFPSTestStop(): void {
    const ws = this.app.getSystem<WebSocketSystem>('websocket');
    if (!ws) {
      debug.error('[UISystem] WebSocketSystem not found');
      return;
    }

    // Send message to worker
    ws.stopFPSMeasurement();

    // Get results from LatencyTracker
    const result = this.latencyTracker.stopFPSMeasurement();
    if (result) {
      this.fpsTest.displayResult(result);
      debug.logFPS('[UISystem] FPS measurement completed successfully', result);
    } else {
      debug.error('[UISystem] Failed to get FPS measurement result');
    }

    // Update UI
    this.fpsTest.showTestStopped();

    debug.logFPS('[UISystem] FPS test stopped');
  }

  private handleFPSResultsDownload(): void {
    debug.logFPS('[UISystem] FPS results download handled by FPSTestPanel');
    // Note: Download functionality is handled directly in FPSTestPanel via button click
  }

  // ========================================================================
  // Public API (called from main.ts)
  // ========================================================================

  /**
   * Update latency statistics display
   */
  updateLatencyStats(stats: any, clockOffset: number): void {
    this.stats.updateLatencyStats(stats, clockOffset);
  }

  /**
   * Update FPS display
   */
  updateFPS(decodeFps: number, renderFps?: number): void {
    this.stats.updateFPS(decodeFps, renderFps);
  }

  /**
   * Update decode FPS with additional info
   */
  updateDecodeFPS(decodeFps: number, avgTime: number): void {
    this.stats.updateDecodeFPS(decodeFps, avgTime);
  }

  /**
   * Update render FPS with additional info
   */
  updateRenderFPS(renderFps: number, avgTime: number): void {
    this.stats.updateRenderFPS(renderFps, avgTime);
  }

  /**
   * Set WebSocket connection state
   */
  setConnectionState(state: 'connected' | 'disconnected' | 'error' | 'closed'): void {
    this.control.setConnectionState(state);
  }

  /**
   * Get current render mode
   */
  getRenderMode(): RenderMode {
    return this.control.getRenderMode();
  }

  /**
   * Update FPS test progress UI (called from render loop)
   */
  updateFPSTestUI(): void {
    if (!this.latencyTracker.isFPSMeasurementActive()) return;

    const progress = this.latencyTracker.getFPSMeasurementProgress();
    const currentStats = this.latencyTracker.getCurrentFPSTestStats();

    if (progress) {
      this.fpsTest.updateProgress(progress);
    }

    if (currentStats) {
      this.fpsTest.updateCurrent(currentStats);
    }

    // Auto-completion when time runs out
    if (progress && progress.remainingMs <= 0) {
      debug.logFPS('[UISystem] Auto-completion triggered by progress timer');
      this.handleFPSTestStop();
    }
  }
}
