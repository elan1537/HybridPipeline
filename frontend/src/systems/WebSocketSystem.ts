/**
 * WebSocketSystem - Manages WebSocket communication via Worker
 * CP3: Wraps existing worker-based WebSocket communication
 */

import { System, SystemContext, CameraFrame, VideoFrame, ConnectionState, FormatType } from "../types";

export type VideoFrameHandler = (frame: VideoFrame) => void;
export type ConnectionStateHandler = (state: ConnectionState) => void;

export class WebSocketSystem implements System {
  readonly name = "websocket";

  private worker: Worker | null = null;
  private context: SystemContext | null = null;
  private videoFrameHandlers: Set<VideoFrameHandler> = new Set();
  private connectionStateHandlers: Set<ConnectionStateHandler> = new Set();
  private currentState: ConnectionState = ConnectionState.Disconnected;

  // Configuration
  private width: number = 1280;
  private height: number = 720;
  private wsURL: string = "";

  async initialize(context: SystemContext): Promise<void> {
    this.context = context;

    // Create worker
    this.worker = new Worker(new URL("../decode-worker.ts", import.meta.url), {
      type: "module",
    });

    // Set up message handler
    this.worker.onmessage = (e) => this.handleWorkerMessage(e);

    // Subscribe to state changes
    context.state.subscribe("connection:state", (state: ConnectionState) => {
      this.currentState = state;
      this.notifyConnectionStateHandlers(state);
    });

    console.log("[WebSocketSystem] Initialized");
  }

  update(deltaTime: number): void {
    // WebSocket system doesn't need frame updates
  }

  dispose(): void {
    if (this.worker) {
      this.worker.terminate();
      this.worker = null;
    }
    this.videoFrameHandlers.clear();
    this.connectionStateHandlers.clear();
  }

  // ========================================================================
  // Public API
  // ========================================================================

  /**
   * Connect to WebSocket server
   */
  connect(url: string, width: number, height: number): void {
    if (!this.worker) {
      console.error("[WebSocketSystem] Worker not initialized");
      return;
    }

    this.wsURL = url;
    this.width = width;
    this.height = height;

    this.worker.postMessage({
      type: "init",
      width,
      height,
      wsURL: url,
    });

    if (this.context) {
      this.context.state.set("connection:state", ConnectionState.Connecting);
      this.context.state.set("connection:url", url);
    }
  }

  /**
   * Disconnect from WebSocket server
   */
  disconnect(): void {
    if (!this.worker) {
      console.error("[WebSocketSystem] Worker not initialized");
      return;
    }

    this.worker.postMessage({ type: "ws-close" });

    if (this.context) {
      this.context.state.set("connection:state", ConnectionState.Disconnected);
    }
  }

  /**
   * Reconnect with new resolution
   */
  reconnect(url: string, width: number, height: number): void {
    this.disconnect();
    setTimeout(() => {
      this.connect(url, width, height);
    }, 100);
  }

  /**
   * Send camera frame to server
   */
  sendCameraFrame(frame: CameraFrame): void {
    if (!this.worker) {
      console.error("[WebSocketSystem] Worker not initialized");
      return;
    }

    if (this.currentState !== ConnectionState.Connected) {
      return;
    }

    this.worker.postMessage({
      type: "send-camera",
      eye: frame.eye,
      target: frame.target,
      intrinsics: frame.intrinsics,
      frameId: frame.frameId,
      timestamp: frame.timestamp,
      timeIndex: frame.timeIndex,
    });

    if (this.context) {
      this.context.eventBus.emit("frame:sent", { frameId: frame.frameId });
    }
  }

  /**
   * Register video frame handler
   */
  onVideoFrame(handler: VideoFrameHandler): () => void {
    this.videoFrameHandlers.add(handler);
    return () => {
      this.videoFrameHandlers.delete(handler);
    };
  }

  /**
   * Register connection state handler
   */
  onConnectionState(handler: ConnectionStateHandler): () => void {
    this.connectionStateHandlers.add(handler);
    // Immediately call with current state
    handler(this.currentState);
    return () => {
      this.connectionStateHandlers.delete(handler);
    };
  }

  /**
   * Start FPS measurement
   */
  startFPSMeasurement(): void {
    if (this.worker) {
      this.worker.postMessage({ type: "fps-measurement-start" });
    }
  }

  /**
   * Stop FPS measurement
   */
  stopFPSMeasurement(): void {
    if (this.worker) {
      this.worker.postMessage({ type: "fps-measurement-stop" });
    }
  }

  /**
   * Toggle JPEG fallback mode
   */
  toggleJPEGFallback(enabled: boolean): void {
    if (this.worker) {
      this.worker.postMessage({
        type: "toggle-jpeg-fallback",
        enabled,
      });
    }
  }

  // ========================================================================
  // Private Methods
  // ========================================================================

  private handleWorkerMessage(e: MessageEvent): void {
    const msg = e.data;

    switch (msg.type) {
      case "video-frame":
        this.handleVideoFrame(msg);
        break;

      case "ws-state":
        this.handleConnectionState(msg.state);
        break;

      case "decode-fps":
        if (this.context) {
          this.context.state.set("performance:decodeFps", msg.fps);
        }
        break;

      case "fps-measurement-progress":
        if (this.context) {
          this.context.eventBus.emit("system:initialized", {
            name: "fps-measurement-progress",
          });
        }
        break;

      case "fps-measurement-complete":
        if (this.context) {
          this.context.eventBus.emit("system:initialized", {
            name: "fps-measurement-complete",
          });
        }
        break;

      default:
        console.warn("[WebSocketSystem] Unknown message type:", msg.type);
    }
  }

  private handleVideoFrame(msg: any): void {
    const frame: VideoFrame = {
      frameId: msg.frameId,
      formatType: msg.formatType as FormatType,
      colorData: msg.colorData,
      depthData: msg.depthData,
      width: msg.width,
      height: msg.height,
      timestamps: {
        client: msg.clientTimestamp || msg.clientSendTime,
        server: msg.serverTimestamp || msg.serverReceiveTime,
        renderStart: msg.renderStartTimestamp || msg.serverProcessEndTime,
        encodeEnd: msg.encodeEndTimestamp || msg.serverSendTime,
      },
    };

    // Notify handlers
    this.videoFrameHandlers.forEach((handler) => {
      try {
        handler(frame);
      } catch (error) {
        console.error("[WebSocketSystem] Error in video frame handler:", error);
      }
    });

    // Emit event
    if (this.context) {
      this.context.eventBus.emit("frame:received", {
        frameId: frame.frameId,
        data: frame,
      });
    }
  }

  private handleConnectionState(state: string): void {
    let connectionState: ConnectionState;

    switch (state) {
      case "open":
        connectionState = ConnectionState.Connected;
        break;
      case "connecting":
        connectionState = ConnectionState.Connecting;
        break;
      case "closed":
        connectionState = ConnectionState.Disconnected;
        break;
      case "error":
        connectionState = ConnectionState.Error;
        break;
      default:
        connectionState = ConnectionState.Disconnected;
    }

    if (this.context) {
      this.context.state.set("connection:state", connectionState);
    }

    this.notifyConnectionStateHandlers(connectionState);
  }

  private notifyConnectionStateHandlers(state: ConnectionState): void {
    this.connectionStateHandlers.forEach((handler) => {
      try {
        handler(state);
      } catch (error) {
        console.error("[WebSocketSystem] Error in connection state handler:", error);
      }
    });
  }

  // ========================================================================
  // Getters
  // ========================================================================

  getConnectionState(): ConnectionState {
    return this.currentState;
  }

  isConnected(): boolean {
    return this.currentState === ConnectionState.Connected;
  }

  getURL(): string {
    return this.wsURL;
  }

  getResolution(): { width: number; height: number } {
    return { width: this.width, height: this.height };
  }
}
