/**
 * StatsPanel - Displays FPS and latency statistics
 * Extracted from main.ts (lines 91-103, 1567-1615)
 */

import { debug } from '../../debug-logger';
import type { LatencyStats } from '../../latency-tracker';
import { BasePanel } from './BasePanel';

// Type for getRecentStats() return value
interface AggregatedLatencyStats {
  avg: Partial<LatencyStats>;
  min: Partial<LatencyStats>;
  max: Partial<LatencyStats>;
  p95: Partial<LatencyStats>;
  p99: Partial<LatencyStats>;
}

export class StatsPanel extends BasePanel {
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
    super();
    this.initializeElements();
  }

  private initializeElements(): void {
    this.fpsDiv = this.getElement('decode-fps');
    this.renderFpsDiv = this.getElement('render-fps');
    this.totalLatencyDiv = this.getElement('total-latency');
    this.networkLatencyDiv = this.getElement('network-latency');
    this.serverLatencyDiv = this.getElement('server-latency');
    this.decodeLatencyDiv = this.getElement('decode-latency');
    this.clockOffsetDiv = this.getElement('clock-offset');

    if (!this.fpsDiv || !this.renderFpsDiv) {
      debug.warn('[StatsPanel] Some FPS display elements not found');
    }
  }

  /**
   * Update FPS display
   */
  updateFPS(decodeFps: number, renderFps?: number): void {
    this.updateText(this.fpsDiv, `Decode FPS: ${decodeFps.toFixed(2)}`);

    if (renderFps !== undefined) {
      this.updateText(this.renderFpsDiv, `Render FPS: ${renderFps.toFixed(2)}`);
    }
  }

  /**
   * Update decode FPS with additional info
   */
  updateDecodeFPS(decodeFps: number, avgTime: number): void {
    this.updateText(this.fpsDiv, `Decode FPS: ${decodeFps.toFixed(2)} (Pure: ${avgTime.toFixed(1)}ms avg)`);
  }

  /**
   * Update render FPS with additional info
   */
  updateRenderFPS(renderFps: number, avgTime: number): void {
    this.updateText(this.renderFpsDiv, `GPU Processing FPS: ${renderFps.toFixed(2)} (${avgTime.toFixed(1)}ms avg)`);
  }

  /**
   * Update latency statistics (throttled)
   * Called from render loop - automatically throttles updates
   */
  updateLatencyStats(stats: AggregatedLatencyStats, clockOffset: number): void {
    const now = performance.now();
    if (now - this.lastStatsUpdate < this.statsUpdateInterval) {
      return;
    }

    this.lastStatsUpdate = now;

    // Update total latency
    if (stats.avg.totalLatency !== undefined) {
      this.updateText(
        this.totalLatencyDiv,
        `Total: ${stats.avg.totalLatency.toFixed(1)}ms (${stats.p95.totalLatency?.toFixed(1)}ms p95)`
      );
    }

    // Update network latency
    if (stats.avg.networkUploadTime !== undefined && stats.avg.networkDownloadTime !== undefined) {
      const totalNetwork = stats.avg.networkUploadTime + stats.avg.networkDownloadTime;
      this.updateText(this.networkLatencyDiv, `Network: ${totalNetwork.toFixed(1)}ms`);
    }

    // Update server latency
    if (stats.avg.serverProcessingTime !== undefined) {
      this.updateText(this.serverLatencyDiv, `Server: ${stats.avg.serverProcessingTime.toFixed(1)}ms`);
    }

    // Update client decode latency
    if (stats.avg.clientDecodeTime !== undefined) {
      const totalClient = stats.avg.clientDecodeTime + (stats.avg.clientRenderTime || 0);
      this.updateText(this.decodeLatencyDiv, `Client: ${totalClient.toFixed(1)}ms`);
    }

    // Update clock offset
    this.updateText(this.clockOffsetDiv, `Clock offset: ${clockOffset.toFixed(1)}ms`);

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
    this.updateText(this.fpsDiv, 'Decode FPS: --');
    this.updateText(this.renderFpsDiv, 'Render FPS: --');
    this.updateText(this.totalLatencyDiv, 'Total: --');
    this.updateText(this.networkLatencyDiv, 'Network: --');
    this.updateText(this.serverLatencyDiv, 'Server: --');
    this.updateText(this.decodeLatencyDiv, 'Client: --');
    this.updateText(this.clockOffsetDiv, 'Clock offset: --');
  }

  /**
   * Cleanup resources
   */
  cleanup(): void {
    // No resources to clean up (DOM elements are managed by HTML)
  }
}
