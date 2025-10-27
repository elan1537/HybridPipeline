/**
 * DebugSystem - Manages debug visualization and performance monitoring
 * CP3: Centralized debug utilities
 */

import { System, SystemContext, PerformanceStats, DebugOptions } from "../types";

export class DebugSystem implements System {
  readonly name = "debug";

  private context: SystemContext | null = null;
  private options: DebugOptions = {
    showStats: false,
    showDepthMap: false,
    showWireframe: false,
    logPerformance: false,
  };

  // Performance tracking
  private performanceHistory: PerformanceStats[] = [];
  private maxHistorySize: number = 100;

  async initialize(context: SystemContext): Promise<void> {
    this.context = context;

    // Subscribe to debug option changes from state
    context.state.subscribe("debug:showStats", (value: boolean) => {
      this.options.showStats = value;
    });

    context.state.subscribe("debug:showDepthMap", (value: boolean) => {
      this.options.showDepthMap = value;
    });

    context.state.subscribe("debug:showWireframe", (value: boolean) => {
      this.options.showWireframe = value;
    });

    context.state.subscribe("debug:logPerformance", (value: boolean) => {
      this.options.logPerformance = value;
    });

    console.log("[DebugSystem] Initialized");
  }

  update(deltaTime: number): void {
    if (this.options.logPerformance) {
      this.updatePerformanceStats(deltaTime);
    }
  }

  dispose(): void {
    this.performanceHistory = [];
  }

  // ========================================================================
  // Debug Options
  // ========================================================================

  setOption(option: keyof DebugOptions, value: boolean): void {
    this.options[option] = value;

    if (this.context) {
      this.context.state.set(`debug:${option}`, value);
      this.context.eventBus.emit("ui:debug-toggled", { option, value });
    }
  }

  getOption(option: keyof DebugOptions): boolean {
    return this.options[option];
  }

  getOptions(): DebugOptions {
    return { ...this.options };
  }

  // ========================================================================
  // Performance Monitoring
  // ========================================================================

  private updatePerformanceStats(deltaTime: number): void {
    if (!this.context) return;

    const stats: PerformanceStats = {
      fps: 1 / deltaTime,
      frameTime: deltaTime * 1000,
      renderTime: this.context.state.getOrDefault("performance:renderTime", 0),
      networkLatency: this.context.state.getOrDefault("performance:networkLatency", 0),
    };

    this.performanceHistory.push(stats);

    // Keep history size limited
    if (this.performanceHistory.length > this.maxHistorySize) {
      this.performanceHistory.shift();
    }

    // Update state with current stats
    this.context.state.set("performance:fps", stats.fps);
    this.context.state.set("performance:frameTime", stats.frameTime);
  }

  /**
   * Log performance metric
   */
  logPerformance(metric: string, value: number): void {
    if (this.options.logPerformance) {
      console.log(`[Performance] ${metric}: ${value.toFixed(2)}`);
    }
  }

  /**
   * Get recent performance stats
   */
  getPerformanceStats(count: number = 10): PerformanceStats[] {
    return this.performanceHistory.slice(-count);
  }

  /**
   * Get average performance over recent frames
   */
  getAveragePerformance(count: number = 10): PerformanceStats {
    const recent = this.getPerformanceStats(count);
    if (recent.length === 0) {
      return {
        fps: 0,
        frameTime: 0,
        renderTime: 0,
        networkLatency: 0,
      };
    }

    const sum = recent.reduce(
      (acc, stat) => ({
        fps: acc.fps + stat.fps,
        frameTime: acc.frameTime + stat.frameTime,
        renderTime: acc.renderTime + stat.renderTime,
        networkLatency: acc.networkLatency + stat.networkLatency,
      }),
      { fps: 0, frameTime: 0, renderTime: 0, networkLatency: 0 }
    );

    return {
      fps: sum.fps / recent.length,
      frameTime: sum.frameTime / recent.length,
      renderTime: sum.renderTime / recent.length,
      networkLatency: sum.networkLatency / recent.length,
    };
  }

  // ========================================================================
  // Debug Logging
  // ========================================================================

  /**
   * Log message (respects console logging setting)
   */
  log(message: string, ...args: any[]): void {
    const consoleLogging = this.context?.state.get("debug:consoleLogging");
    if (consoleLogging) {
      console.log(`[Debug] ${message}`, ...args);
    }
  }

  /**
   * Log warning
   */
  warn(message: string, ...args: any[]): void {
    console.warn(`[Debug] ${message}`, ...args);
  }

  /**
   * Log error
   */
  error(message: string, ...args: any[]): void {
    console.error(`[Debug] ${message}`, ...args);
  }

  // ========================================================================
  // Debug Information
  // ========================================================================

  /**
   * Get system information for debugging
   */
  getSystemInfo(): Record<string, any> {
    if (!this.context) {
      return {};
    }

    const state = this.context.state.toObject();

    return {
      // Connection
      connectionState: state["connection:state"],
      connectionUrl: state["connection:url"],

      // Rendering
      renderMode: state["rendering:mode"],
      resolution: `${state["rendering:width"]}x${state["rendering:height"]}`,

      // Performance
      fps: state["performance:fps"]?.toFixed(1),
      networkLatency: state["performance:networkLatency"]?.toFixed(1),

      // Frame info
      frameCounter: state["frame:counter"],
      frameId: state["frame:id"],

      // Debug options
      debugOptions: this.options,
    };
  }

  /**
   * Print system information to console
   */
  printSystemInfo(): void {
    const info = this.getSystemInfo();
    console.log("=== System Information ===");
    Object.entries(info).forEach(([key, value]) => {
      console.log(`  ${key}:`, value);
    });
    console.log("=========================");
  }

  /**
   * Clear performance history
   */
  clearPerformanceHistory(): void {
    this.performanceHistory = [];
  }
}
