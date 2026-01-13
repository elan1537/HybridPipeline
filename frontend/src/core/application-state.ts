/**
 * ApplicationState - Centralized state management
 * CP3: Replace 122 global variables with structured state
 */

import { StateListener, ConnectionState, RenderMode } from "../types";

export class ApplicationState {
  private state: Map<string, any>;
  private listeners: Map<string, Set<StateListener>>;

  constructor() {
    this.state = new Map();
    this.listeners = new Map();

    // Initialize default state
    this.initializeDefaults();
  }

  private initializeDefaults(): void {
    // Connection state
    this.set("connection:state", ConnectionState.Disconnected);
    this.set("connection:url", "");

    // Rendering state
    this.set("rendering:mode", RenderMode.Hybrid);
    this.set("rendering:width", 1280);
    this.set("rendering:height", 720);
    this.set("rendering:rescaleFactor", 0.8);

    // Frame counters
    this.set("frame:counter", 0);
    this.set("frame:timeIndex", 0.0);
    this.set("frame:id", 0);

    // Performance
    this.set("performance:fps", 0);
    this.set("performance:renderTime", 0);
    this.set("performance:networkLatency", 0);

    // Debug flags
    this.set("debug:showStats", false);
    this.set("debug:showDepthMap", false);
    this.set("debug:showWireframe", false);
    this.set("debug:consoleLogging", false);
    this.set("debug:cameraLogging", false);

    // Animation
    this.set("animation:playing", true);

    // Recording
    this.set("recording:active", false);
    this.set("recording:startTime", 0);

    // Texture update flags
    this.set("texture:colorNeedsUpdate", false);
    this.set("texture:depthNeedsUpdate", false);
  }

  /**
   * Get a state value
   */
  get<T>(key: string): T | undefined {
    return this.state.get(key);
  }

  /**
   * Get a state value with a default fallback
   */
  getOrDefault<T>(key: string, defaultValue: T): T {
    return this.state.has(key) ? this.state.get(key) : defaultValue;
  }

  /**
   * Set a state value and notify listeners
   */
  set<T>(key: string, value: T): void {
    const prevValue = this.state.get(key);
    this.state.set(key, value);

    // Notify listeners only if value changed
    if (prevValue !== value) {
      this.notifyListeners(key, value, prevValue);
    }
  }

  /**
   * Update multiple state values at once
   */
  update(updates: Record<string, any>): void {
    Object.entries(updates).forEach(([key, value]) => {
      this.set(key, value);
    });
  }

  /**
   * Subscribe to state changes
   */
  subscribe<T>(key: string, listener: StateListener<T>): () => void {
    if (!this.listeners.has(key)) {
      this.listeners.set(key, new Set());
    }
    this.listeners.get(key)!.add(listener);

    // Return unsubscribe function
    return () => this.unsubscribe(key, listener);
  }

  /**
   * Unsubscribe from state changes
   */
  unsubscribe<T>(key: string, listener: StateListener<T>): void {
    const listeners = this.listeners.get(key);
    if (listeners) {
      listeners.delete(listener);
      if (listeners.size === 0) {
        this.listeners.delete(key);
      }
    }
  }

  /**
   * Notify all listeners for a key
   */
  private notifyListeners(key: string, value: any, prevValue: any): void {
    const listeners = this.listeners.get(key);
    if (listeners) {
      listeners.forEach((listener) => {
        try {
          listener(value, prevValue);
        } catch (error) {
          console.error(`Error in state listener for "${key}":`, error);
        }
      });
    }
  }

  /**
   * Check if a key exists
   */
  has(key: string): boolean {
    return this.state.has(key);
  }

  /**
   * Delete a key
   */
  delete(key: string): void {
    this.state.delete(key);
    this.listeners.delete(key);
  }

  /**
   * Clear all state
   */
  clear(): void {
    this.state.clear();
    this.listeners.clear();
    this.initializeDefaults();
  }

  /**
   * Get all keys
   */
  keys(): string[] {
    return Array.from(this.state.keys());
  }

  /**
   * Get all state as an object
   */
  toObject(): Record<string, any> {
    const obj: Record<string, any> = {};
    this.state.forEach((value, key) => {
      obj[key] = value;
    });
    return obj;
  }

  /**
   * Debug: Print current state
   */
  debug(): void {
    console.log("=== Application State ===");
    this.state.forEach((value, key) => {
      console.log(`  ${key}:`, value);
    });
    console.log("========================");
  }
}
