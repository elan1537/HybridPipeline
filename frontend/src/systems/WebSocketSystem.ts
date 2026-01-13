/**
 * WebSocketSystem - Manages WebSocket communication via Worker
 * CP3: Wraps existing worker-based WebSocket communication
 */

import { System, SystemContext, CameraFrame, VideoFrame, ConnectionState, FormatType } from "../types";

export type VideoFrameHandler = (frame: VideoFrame) => void;
export type ConnectionStateHandler = (state: ConnectionState) => void;

/**
 * Callbacks for WebSocketSystem events
 * Used by main.ts to handle UI updates and latency tracking
 */
export interface WebSocketSystemCallbacks {
  onConnectionStateChange?: (state: 'connected' | 'error' | 'closed') => void;
  onDecodeStats?: (stats: {
    pureFPS: number;
    avgDecodeTime: number;
    minDecodeTime: number;
    maxDecodeTime: number;
    recentAvg: number;
    totalFrames: number;
    fpsMeasurementData?: { totalCount: number; avgTime: number };
  }) => void;
  onFrameReceive?: (frameId: number, serverTimestamps: any) => void;
  onFrameDecoded?: (frameId: number, decodeCompleteTime: number) => void;
  onClockSync?: (clientRequestTime: number, serverReceiveTime: number, serverSendTime: number) => void;
}

export class WebSocketSystem implements System {
  readonly name = "websocket";

  private worker: Worker | null = null;
  private context: SystemContext | null = null;
  private videoFrameHandlers: Set<VideoFrameHandler> = new Set();
  private connectionStateHandlers: Set<ConnectionStateHandler> = new Set();
  private currentState: ConnectionState = ConnectionState.Disconnected;
  private isUsingExistingWorker: boolean = false;

  // Configuration
  private width: number = 1280;
  private height: number = 720;
  private wsURL: string = "";

  // Callbacks for external event handling
  private callbacks: WebSocketSystemCallbacks = {};

  /**
   * Set callbacks for external event handling
   * Used by main.ts to connect UI updates and latency tracking
   */
  setCallbacks(callbacks: WebSocketSystemCallbacks): void {
    this.callbacks = callbacks;
    console.log("[WebSocketSystem] Callbacks configured");
  }

  /**
   * Set existing worker to reuse (Phase 1 compatibility)
   */
  setWorker(worker: Worker): void {
    this.worker = worker;
    this.isUsingExistingWorker = true;
    console.log("[WebSocketSystem] Using existing worker");
  }

  /**
   * Handle worker message (called from main.ts worker.onmessage)
   */
  handleMessage(data: any): void {
    this.handleWorkerMessage({ data } as MessageEvent);
  }

  async initialize(context: SystemContext): Promise<void> {
    this.context = context;

    // Only create worker if not already set (Phase 2)
    if (!this.worker) {
      this.worker = new Worker(new URL("../decode-worker.ts", import.meta.url), {
        type: "module",
      });
      console.log("[WebSocketSystem] Created new worker");
    }

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
    console.log(`[WebSocketSystem] sendCameraFrame() ENTRY - frameId=${frame.frameId}, worker=${!!this.worker}, state=${this.currentState}`);

    if (!this.worker) {
      console.error("[WebSocketSystem] Worker not initialized");
      return;
    }

    if (this.currentState !== ConnectionState.Connected) {
      console.warn(`[WebSocketSystem] Cannot send camera frame: not connected (state=${this.currentState})`);
      return;
    }

    // Log only first frame to verify sending is working
    console.log(`[WebSocketSystem] Sending camera frame ${frame.frameId}`);
    if (frame.frameId % 60 === 0) {
      console.log(`[WebSocketSystem] Frame ${frame.frameId} sent (logging every 60th frame)`);
    }

    this.worker.postMessage({
      type: "camera",  // Worker expects "camera" not "send-camera"
      frame: frame
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
    console.log(`[WebSocketSystem] toggleJPEGFallback(${enabled}) called`);
    if (this.worker) {
      console.log(`[WebSocketSystem] Sending toggle-jpeg-fallback message to worker`);
      this.worker.postMessage({
        type: "toggle-jpeg-fallback",
        enabled,
      });
    } else {
      console.error(`[WebSocketSystem] Worker not available!`);
    }
  }

  /**
   * Change encoder type on Backend (JPEG/H264)
   * Sends control message to Renderer via Transport
   */
  changeEncoderType(encoderType: 'jpeg' | 'h264'): void {
    if (!this.worker) {
      console.error("[WebSocketSystem] Worker not initialized");
      return;
    }

    // Map encoder type to format code (0=JPEG, 1=H264)
    const formatCode = encoderType === 'jpeg' ? 0 : 1;

    console.log(`[WebSocketSystem] Requesting encoder change to ${encoderType} (code=${formatCode})`);

    // Send control message via worker
    this.worker.postMessage({
      type: "change-encoder",
      encoderType: formatCode,
    });
  }

  // ========================================================================
  // Private Methods
  // ========================================================================

  private handleWorkerMessage(e: MessageEvent): void {
    const msg = e.data;

    // Only log non-frequent messages
    if (msg.type !== 'frame-receive' && msg.type !== 'pure-decode-stats' && msg.type !== 'pong-received') {
      console.log('[WebSocketSystem] Received worker message:', msg.type);
    }

    switch (msg.type) {
      case "frame": // Legacy decode-worker sends "frame"
      case "video-frame":
        console.log('[WebSocketSystem] Handling video frame:', msg.frameId, {
          hasImage: !!msg.image,
          hasDepth: !!msg.depth,
          depthType: msg.depth?.constructor?.name
        });
        this.handleVideoFrame(msg);
        break;

      case "ws-state":
        this.handleConnectionState(msg.state);
        break;

      case "frame-receive":
        // Frame receive notification (for latency tracking)
        if (this.callbacks.onFrameReceive) {
          this.callbacks.onFrameReceive(msg.frameId, msg.serverTimestamps);
        }
        break;

      case "pure-decode-stats":
        // Pure decode statistics (for performance monitoring)
        if (this.callbacks.onDecodeStats) {
          this.callbacks.onDecodeStats({
            pureFPS: msg.pureFPS,
            avgDecodeTime: msg.avgDecodeTime,
            minDecodeTime: msg.minDecodeTime,
            maxDecodeTime: msg.maxDecodeTime,
            recentAvg: msg.recentAvg,
            totalFrames: msg.totalFrames,
            fpsMeasurementData: msg.fpsMeasurementData,
          });
        }
        break;

      case "pong-received":
        // Clock sync pong (for latency tracking)
        if (this.callbacks.onClockSync) {
          this.callbacks.onClockSync(
            msg.clientRequestTime,
            msg.serverReceiveTime,
            msg.serverSendTime
          );
        }
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

      case "error":
        console.error("[WebSocketSystem] Worker error:", msg.error);
        break;

      case "ws-ready":
        console.log("[WebSocketSystem] Connection ready");
        this.handleConnectionState("open");
        if (this.callbacks.onConnectionStateChange) {
          this.callbacks.onConnectionStateChange('connected');
        }
        break;

      case "ws-error":
        console.log("[WebSocketSystem] Connection error");
        this.handleConnectionState("error");
        if (this.callbacks.onConnectionStateChange) {
          this.callbacks.onConnectionStateChange('error');
        }
        break;

      case "ws-close":
        console.log("[WebSocketSystem] Connection closed");
        this.handleConnectionState("closed");
        if (this.callbacks.onConnectionStateChange) {
          this.callbacks.onConnectionStateChange('closed');
        }
        break;

      default:
        console.warn("[WebSocketSystem] Unknown message type:", msg.type);
    }
  }

  private handleVideoFrame(msg: any): void {
    // Notify frame decoded callback (for latency tracking)
    if (msg.frameId && msg.decodeCompleteTime && this.callbacks.onFrameDecoded) {
      this.callbacks.onFrameDecoded(msg.frameId, msg.decodeCompleteTime);
    }

    console.log("handleVideoFrame")
    // Convert legacy decode-worker 'frame' message to VideoFrame format
    let colorData: Uint8Array | undefined;
    let depthData: Float32Array | undefined;
    let colorBitmap: ImageBitmap | undefined;
    let depthBitmap: ImageBitmap | undefined;
    let depthRaw: Uint16Array | undefined;
    let width: number;
    let height: number;

    if (msg.image) {
      // decode-worker sends ImageBitmap
      if (msg.image instanceof ImageBitmap) {
        // H264 or JPEG mode: ImageBitmap format
        colorBitmap = msg.image;
        width = msg.image.width;
        height = msg.image.height;

        // Handle depth
        if (msg.depth) {
          if (msg.depth instanceof ImageBitmap) {
            // H264 mode: depth as ImageBitmap
            depthBitmap = msg.depth;
          } else if (msg.depth instanceof Uint16Array) {
            // JPEG mode: Float16 raw bytes - pass as-is without conversion
            depthRaw = msg.depth;
          } else if (msg.depth instanceof Uint8Array) {
            // Fallback: Uint8Array grayscale
            depthData = new Float32Array(msg.depth.length);
            for (let i = 0; i < msg.depth.length; i++) {
              depthData[i] = msg.depth[i] / 255.0; // Normalize
            }
          }
        }
      } else if (msg.image instanceof ImageData) {
        // Legacy ImageData format (should not happen with current decode-worker)
        const imageData = msg.image as ImageData;
        width = imageData.width;
        height = imageData.height;
        colorData = new Uint8Array(imageData.data.buffer);
      } else {
        console.error('[WebSocketSystem] Unknown image type:', msg.image);
        return;
      }
    } else {
      // New format: already has colorData, depthData, width, height
      colorData = msg.colorData;
      depthData = msg.depthData;
      colorBitmap = msg.colorBitmap;
      depthBitmap = msg.depthBitmap;
      depthRaw = msg.depthRaw;
      width = msg.width;
      height = msg.height;
    }

    const frame: VideoFrame = {
      frameId: msg.frameId,
      formatType: (msg.formatType as FormatType) || FormatType.JPEG,
      colorData,
      depthData,
      colorBitmap,
      depthBitmap,
      depthRaw,
      width,
      height,
      timestamps: {
        client: msg.clientTimestamp || msg.clientSendTime || 0,
        server: msg.serverTimestamp || msg.serverReceiveTime || 0,
        renderStart: msg.renderStartTimestamp || msg.serverProcessEndTime || 0,
        encodeEnd: msg.encodeEndTimestamp || msg.serverSendTime || 0,
      },
    };

    console.log('[WebSocketSystem] Converted frame:', frame.frameId, frame.width, 'x', frame.height, {
      hasColorBitmap: !!frame.colorBitmap,
      hasDepthBitmap: !!frame.depthBitmap,
      hasDepthRaw: !!frame.depthRaw,
      hasColorData: !!frame.colorData,
      hasDepthData: !!frame.depthData
    });

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

    // Update internal state
    this.currentState = connectionState;
    console.log(`[WebSocketSystem] State updated: "${state}" -> ${connectionState} (currentState=${this.currentState})`);

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
