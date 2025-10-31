/**
 * Frontend Type Definitions
 * Part of the refactoring effort to modularize the codebase
 * CP1: Type definitions and interfaces for refactoring
 *
 * Note: Physics and collision types will be added in a future phase
 */

import * as THREE from "three";

// ============================================================================
// Core Data Types
// ============================================================================

/**
 * Camera frame data sent to the backend
 * Total size: 260 bytes when serialized
 *
 * Protocol v3: Matrix + position/target for flexible rendering
 * - view: Camera → World transform (matrixWorld)
 * - projection: Camera → Clip transform (projectionMatrix)
 * - intrinsics: Pixel-space camera intrinsics (from getCameraIntrinsics)
 * - position: Camera position in world space
 * - target: Camera target (lookAt point) in world space
 * - up: Camera up vector
 */
export interface CameraFrame {
  view: Float32Array;        // 16 floats - view matrix (4×4)
  projection: Float32Array;  // 16 floats - projection matrix (4×4)
  intrinsics: Float32Array;  // 9 floats - intrinsics matrix (3×3)
  position: Float32Array;    // 3 floats - camera position
  target: Float32Array;      // 3 floats - camera target (lookAt)
  up: Float32Array;          // 3 floats - camera up vector
  frameId: number;
  timestamp: number;
  timeIndex: number;
}

/**
 * Video frame received from the backend
 * Header: 44 bytes + variable data
 * Supports both raw data (Uint8Array) and decoded images (ImageBitmap)
 */
export interface VideoFrame {
  frameId: number;
  formatType: FormatType; // 0=JPEG, 1=H264, 2=Raw
  colorData?: Uint8Array;
  depthData?: Float32Array;
  colorBitmap?: ImageBitmap; // Decoded color image from worker
  depthBitmap?: ImageBitmap; // Decoded depth image from worker (H264)
  depthRaw?: Uint16Array; // Raw Float16 depth data (JPEG mode)
  width: number;
  height: number;
  timestamps: {
    client: number;
    server: number;
    renderStart: number;
    encodeEnd: number;
  };
}

// ============================================================================
// System Configuration Types
// ============================================================================

/**
 * Main application configuration
 */
export interface ApplicationConfig {
  canvas: HTMLCanvasElement;
  wsUrl: string;
  width: number;
  height: number;
  renderMode: RenderMode;
  debugMode: boolean;
  cameraConfig?: CameraConfig; // Optional camera configuration
}

/**
 * Rendering configuration
 */
export interface RenderConfig {
  width: number;
  height: number;
  antialias: boolean;
  pixelRatio?: number;
  preserveDrawingBuffer?: boolean;
}

/**
 * WebSocket configuration
 */
export interface WebSocketConfig {
  url: string;
  reconnectInterval?: number;
  maxReconnectAttempts?: number;
  binaryType?: BinaryType;
}

/**
 * Context passed to systems during initialization
 */
export interface SystemContext {
  renderingContext: any; // Will be RenderingContext type
  eventBus: any; // Will be EventBus type
  state: any; // Will be ApplicationState type
  systems?: Map<string, System>; // Access to other systems
}

// ============================================================================
// UI and State Types
// ============================================================================

/**
 * Performance statistics
 */
export interface PerformanceStats {
  fps: number;
  frameTime: number;
  renderTime: number;
  networkLatency: number;
  memoryUsage?: number;
}

/**
 * Debug options
 */
export interface DebugOptions {
  showStats: boolean;
  showDepthMap: boolean;
  showWireframe: boolean;
  logPerformance: boolean;
}

/**
 * Camera state for transitions
 */
export interface CameraState {
  position: THREE.Vector3;
  target: THREE.Vector3;
  fov: number;
  near: number;
  far: number;
}

/**
 * Camera configuration for initialization
 */
export interface CameraConfig {
  fov?: number;
  near?: number;
  far?: number;
  aspect?: number;
  position?: THREE.Vector3;
  target?: THREE.Vector3;
}

// ============================================================================
// Event Types
// ============================================================================

/**
 * Application events
 */
export interface AppEvents {
  // Connection events
  "connection:open": void;
  "connection:close": CloseEvent;
  "connection:error": Event;

  // Frame events
  "frame:sent": { frameId: number };
  "frame:received": { frameId: number; data: VideoFrame };

  // System events
  "system:initialized": { name: string };
  "system:error": { name: string; error: Error };

  // UI events
  "ui:mode-changed": { mode: RenderMode };
  "ui:debug-toggled": { option: keyof DebugOptions; value: boolean };
}

/**
 * Event handler type
 */
export type EventHandler<T = any> = (data: T) => void | Promise<void>;

/**
 * State listener type
 */
export type StateListener<T = any> = (value: T, prevValue?: T) => void;

// ============================================================================
// Enumerations
// ============================================================================

/**
 * Rendering modes
 * - Mesh: Three.js local rendering only
 * - Gaussian: Server-side Gaussian splatting image only
 * - Hybrid: Mesh + Gaussian depth combined rendering
 */
export enum RenderMode {
  Mesh = "mesh",
  Gaussian = "gaussian",
  Hybrid = "hybrid",
}

/**
 * Camera modes
 */
export enum CameraMode {
  Orbit = "orbit",
  FirstPerson = "firstperson",
  ThirdPerson = "thirdperson",
}

/**
 * Video format types (must match backend)
 */
export enum FormatType {
  JPEG = 0,
  H264 = 1,
  Raw = 2,
}

/**
 * Connection states
 */
export enum ConnectionState {
  Disconnected = "disconnected",
  Connecting = "connecting",
  Connected = "connected",
  Reconnecting = "reconnecting",
  Error = "error",
}

/**
 * Latency types for tracking
 */
export enum LatencyType {
  Network = "network",
  Render = "render",
  Encode = "encode",
  Decode = "decode",
  Total = "total",
}

// ============================================================================
// System Interface
// ============================================================================

/**
 * Base interface for all systems
 */
export interface System {
  readonly name: string;
  initialize(context: SystemContext): Promise<void>;
  update(deltaTime: number): void;
  dispose(): void;
}

// ============================================================================
// Buffer Types
// ============================================================================

/**
 * Frame buffer types
 */
export enum BufferType {
  FIFO = "fifo",
  Latest = "latest",
}

/**
 * Frame buffer interface
 */
export interface FrameBuffer<T> {
  push(item: T): void;
  pop(): T | undefined;
  peek(): T | undefined;
  clear(): void;
  size(): number;
  isEmpty(): boolean;
}

// ============================================================================
// Protocol Types
// ============================================================================

/**
 * Binary protocol header for camera data
 */
export interface CameraProtocolHeader {
  frameId: number;
  timestamp: number;
  timeIndex: number;
}

/**
 * Binary protocol header for video data
 */
export interface VideoProtocolHeader {
  frameId: number;
  formatType: number;
  colorLen: number;
  depthLen: number;
  width: number;
  height: number;
  clientTimestamp: number;
  serverTimestamp: number;
  renderStartTimestamp: number;
  encodeEndTimestamp: number;
}

// ============================================================================
// Callback Types
// ============================================================================

/**
 * Video frame callback
 */
export type VideoFrameCallback = (frame: VideoFrame) => void;

/**
 * Connection state callback
 */
export type ConnectionStateCallback = (state: ConnectionState) => void;

/**
 * Error callback
 */
export type ErrorCallback = (error: Error) => void;

// ============================================================================
// Utility Types
// ============================================================================

/**
 * Make all properties optional recursively
 */
export type DeepPartial<T> = {
  [P in keyof T]?: T[P] extends object ? DeepPartial<T[P]> : T[P];
};

/**
 * Extract promise type
 */
export type Awaited<T> = T extends Promise<infer U> ? U : T;

/**
 * Disposable resource
 */
export interface Disposable {
  dispose(): void;
}
