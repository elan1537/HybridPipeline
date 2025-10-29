/**
 * StatsPanel - Displays FPS and latency statistics
 * Extracted from main.ts (lines 91-103, 1567-1615)
 */

import { debug } from '../../debug-logger';
import { LatencyStats } from '../../latency-tracker';

export class StatsPanel {
  // DOM elements
  private fpsDiv: HTMLDivElement | null = null;
  private renderFpsDiv: HTMLDivElement | null = null;
  private totalLatencyDiv: HTMLDivElement | null = null;
  private networkLatencyDiv: HTMLDivElement | null = null;
  private serverLatencyDiv: HTMLDivElement | null = null;
  private decodeLatencyDiv: HTMLDivElement | null = null;
  private clockOffsetDiv: HTMLDivElement | null = null;

  // Update throttling
  private lastStatsUpdate = 0;
  private readonly statsUpdateInterval = 500; // 0.5 seconds

  constructor() {
    this.initializeElements();
  }

  private initializeElements(): void {
    this.fpsDiv = document.getElementById('decode-fps') as HTMLDivElement;
    this.renderFpsDiv = document.getElementById('render-fps') as HTMLDivElement;
    this.totalLatencyDiv = document.getElementById('total-latency') as HTMLDivElement;
    this.networkLatencyDiv = document.getElementById('network-latency') as HTMLDivElement;
    this.serverLatencyDiv = document.getElementById('server-latency') as HTMLDivElement;
    this.decodeLatencyDiv = document.getElementById('decode-latency') as HTMLDivElement;
    this.clockOffsetDiv = document.getElementById('clock-offset') as HTMLDivElement;

    if (!this.fpsDiv || !this.renderFpsDiv) {
      debug.warn('[StatsPanel] Some FPS display elements not found');
    }
  }

  /**
   * Update FPS display
   */
  updateFPS(decodeFps: number, renderFps?: number): void {
    if (this.fpsDiv) {
      this.fpsDiv.textContent = `Decode FPS: ${decodeFps.toFixed(2)}`;
    }

    if (this.renderFpsDiv && renderFps !== undefined) {
      this.renderFpsDiv.textContent = `Render FPS: ${renderFps.toFixed(2)}`;
    }
  }

  /**
   * Update decode FPS with additional info
   */
  updateDecodeFPS(decodeFps: number, avgTime: number): void {
    if (this.fpsDiv) {
      this.fpsDiv.textContent = `Decode FPS: ${decodeFps.toFixed(2)} (Pure: ${avgTime.toFixed(1)}ms avg)`;
    }
  }

  /**
   * Update render FPS with additional info
   */
  updateRenderFPS(renderFps: number, avgTime: number): void {
    if (this.renderFpsDiv) {
      this.renderFpsDiv.textContent = `GPU Processing FPS: ${renderFps.toFixed(2)} (${avgTime.toFixed(1)}ms avg)`;
    }
  }

  /**
   * Update latency statistics (throttled)
   * Called from render loop - automatically throttles updates
   */
  updateLatencyStats(stats: LatencyStats, clockOffset: number): void {
    const now = performance.now();
    if (now - this.lastStatsUpdate < this.statsUpdateInterval) {
      return;
    }

    this.lastStatsUpdate = now;

    // Update total latency
    if (this.totalLatencyDiv && stats.avg.totalLatency !== undefined) {
      this.totalLatencyDiv.textContent =
        `Total: ${stats.avg.totalLatency.toFixed(1)}ms (${stats.p95.totalLatency?.toFixed(1)}ms p95)`;
    }

    // Update network latency
    if (this.networkLatencyDiv &&
        stats.avg.networkUploadTime !== undefined &&
        stats.avg.networkDownloadTime !== undefined) {
      const totalNetwork = stats.avg.networkUploadTime + stats.avg.networkDownloadTime;
      this.networkLatencyDiv.textContent = `Network: ${totalNetwork.toFixed(1)}ms`;
    }

    // Update server latency
    if (this.serverLatencyDiv && stats.avg.serverProcessingTime !== undefined) {
      this.serverLatencyDiv.textContent = `Server: ${stats.avg.serverProcessingTime.toFixed(1)}ms`;
    }

    // Update client decode latency
    if (this.decodeLatencyDiv && stats.avg.clientDecodeTime !== undefined) {
      const totalClient = stats.avg.clientDecodeTime + (stats.avg.clientRenderTime || 0);
      this.decodeLatencyDiv.textContent = `Client: ${totalClient.toFixed(1)}ms`;
    }

    // Update clock offset
    if (this.clockOffsetDiv) {
      this.clockOffsetDiv.textContent = `Clock offset: ${clockOffset.toFixed(1)}ms`;
    }

    // Log detailed stats (for development)
    if (stats.avg.totalLatency !== undefined) {
      debug.logLatency(
        `Stats - Total: ${stats.avg.totalLatency.toFixed(1)}ms, ` +
        `Network: ${((stats.avg.networkUploadTime || 0) + (stats.avg.networkDownloadTime || 0)).toFixed(1)}ms, ` +
        `Server: ${(stats.avg.serverProcessingTime || 0).toFixed(1)}ms, ` +
        `Decode: ${(stats.avg.clientDecodeTime || 0).toFixed(1)}ms, ` +
        `Render: ${(stats.avg.clientRenderTime || 0).toFixed(1)}ms, ` +
        `Offset: ${clockOffset.toFixed(1)}ms`
      );
    }
  }

  /**
   * Reset all displays
   */
  reset(): void {
    if (this.fpsDiv) this.fpsDiv.textContent = 'Decode FPS: --';
    if (this.renderFpsDiv) this.renderFpsDiv.textContent = 'Render FPS: --';
    if (this.totalLatencyDiv) this.totalLatencyDiv.textContent = 'Total: --';
    if (this.networkLatencyDiv) this.networkLatencyDiv.textContent = 'Network: --';
    if (this.serverLatencyDiv) this.serverLatencyDiv.textContent = 'Server: --';
    if (this.decodeLatencyDiv) this.decodeLatencyDiv.textContent = 'Client: --';
    if (this.clockOffsetDiv) this.clockOffsetDiv.textContent = 'Clock offset: --';
  }

  /**
   * Cleanup resources
   */
  cleanup(): void {
    // No resources to clean up (DOM elements are managed by HTML)
  }
}
