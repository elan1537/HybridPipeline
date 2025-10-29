/**
 * FPSTestPanel - FPS measurement tool
 * Fully migrated from main.ts (lines 639-810, 1289-1391)
 */

import { debug } from '../../debug-logger';
import { FPSMeasurementResult } from '../../latency-tracker';

export interface FPSTestPanelCallbacks {
  onTestStart?: () => void;
  onTestStop?: () => void;
}

export class FPSTestPanel {
  // DOM elements
  private fpsTestButton: HTMLInputElement | null = null;
  private fpsTestProgress: HTMLDivElement | null = null;
  private fpsTestCurrent: HTMLDivElement | null = null;
  private fpsTestResult: HTMLDivElement | null = null;
  private fpsResultDownload: HTMLInputElement | null = null;

  // State
  private lastFPSResult: FPSMeasurementResult | null = null;
  private callbacks: FPSTestPanelCallbacks = {};

  constructor(callbacks: FPSTestPanelCallbacks = {}) {
    this.callbacks = callbacks;
    this.initializeElements();
    this.setupListeners();
  }

  private initializeElements(): void {
    this.fpsTestButton = document.getElementById('fps-measurement-button') as HTMLInputElement;
    this.fpsTestProgress = document.getElementById('fps-measurement-progress') as HTMLDivElement;
    this.fpsTestCurrent = document.getElementById('fps-measurement-current') as HTMLDivElement;
    this.fpsTestResult = document.getElementById('fps-measurement-result') as HTMLDivElement;
    this.fpsResultDownload = document.getElementById('fps-result-download') as HTMLInputElement;
  }

  private setupListeners(): void {
    this.fpsTestButton?.addEventListener('click', () => this.handleTestToggle());
    this.fpsResultDownload?.addEventListener('click', () => this.downloadResults());
  }

  private handleTestToggle(): void {
    const isActive = this.fpsTestButton?.value.includes('Stop');

    if (isActive) {
      this.callbacks.onTestStop?.();
    } else {
      this.callbacks.onTestStart?.();
    }
  }

  // Public methods
  showTestStarted(): void {
    if (this.fpsTestButton) this.fpsTestButton.value = 'Stop FPS Test';
    if (this.fpsTestProgress) this.fpsTestProgress.style.display = 'block';
    if (this.fpsTestCurrent) this.fpsTestCurrent.style.display = 'block';
    if (this.fpsTestResult) this.fpsTestResult.style.display = 'none';
    if (this.fpsResultDownload) this.fpsResultDownload.style.display = 'none';

    debug.logFPS('[FPSTestPanel] Test started');
  }

  showTestStopped(): void {
    if (this.fpsTestButton) this.fpsTestButton.value = 'Start FPS Test (60s)';
    if (this.fpsTestProgress) this.fpsTestProgress.style.display = 'none';
    if (this.fpsTestCurrent) this.fpsTestCurrent.style.display = 'none';

    debug.logFPS('[FPSTestPanel] Test stopped');
  }

  updateProgress(progress: { progress: number; remainingMs: number }): void {
    if (!this.fpsTestProgress) return;

    const remainingSeconds = Math.ceil(progress.remainingMs / 1000);
    const progressPercent = (progress.progress * 100).toFixed(1);
    this.fpsTestProgress.textContent = `Progress: ${progressPercent}% (${remainingSeconds}s left)`;
  }

  updateCurrent(stats: { decodeFPS: number; renderFPS: number }): void {
    if (!this.fpsTestCurrent) return;

    this.fpsTestCurrent.textContent =
      `Current: Decode ${stats.decodeFPS.toFixed(1)}fps, Render ${stats.renderFPS.toFixed(1)}fps`;
  }

  displayResult(result: FPSMeasurementResult): void {
    if (!this.fpsTestResult) return;

    debug.logFPS('[FPSTestPanel] Displaying FPS result:', result);

    // Save result for download
    this.lastFPSResult = result;

    // 데이터 검증
    const hasValidData = result && result.measurementDurationMs > 0;
    const duration = hasValidData ? (result.measurementDurationMs / 1000).toFixed(1) : '0.0';

    // 각 메트릭의 유효성 검사
    const pureDecodeFPS = (result.pureDecodeFPS && isFinite(result.pureDecodeFPS)) ? result.pureDecodeFPS : 0;
    const frameProcessingFPS = (result.frameProcessingFPS && isFinite(result.frameProcessingFPS)) ? result.frameProcessingFPS : 0;
    const renderFPS = (result.renderFPS && isFinite(result.renderFPS)) ? result.renderFPS : 0;
    const legacyDecodeFPS = (result.decodeFPS && isFinite(result.decodeFPS)) ? result.decodeFPS : 0;

    const avgDecodeTime = (result.avgDecodeTime && isFinite(result.avgDecodeTime)) ? result.avgDecodeTime : 0;
    const avgProcessingTime = (result.avgProcessingTime && isFinite(result.avgProcessingTime)) ? result.avgProcessingTime : 0;
    const avgRenderTime = (result.avgRenderTime && isFinite(result.avgRenderTime)) ? result.avgRenderTime : 0;

    const averageLatency = (result.averageLatency && isFinite(result.averageLatency)) ? result.averageLatency : 0;
    const minLatency = (result.minLatency && isFinite(result.minLatency)) ? result.minLatency : 0;
    const maxLatency = (result.maxLatency && isFinite(result.maxLatency)) ? result.maxLatency : 0;
    const totalFrames = result.totalFrames || 0;

    // 경고 메시지 생성
    let warningHtml = '';
    const warnings: string[] = [];

    if (pureDecodeFPS === 0) warnings.push('No decode data collected');
    if (frameProcessingFPS === 0) warnings.push('No frame processing data collected');
    if (renderFPS === 0) warnings.push('No render data collected');
    if (totalFrames < 10) warnings.push(`Low frame count (${totalFrames})`);
    if (!hasValidData) warnings.push('Invalid measurement duration');

    if (warnings.length > 0) {
      warningHtml = `
        <div style="margin-bottom: 4px; padding: 4px; background: rgba(255,170,0,0.1); border-left: 2px solid #ffaa00; font-size: 11px;">
          <div style="font-weight: bold; color: #ffaa00; margin-bottom: 2px;">⚠️ Data Quality Warnings:</div>
          ${warnings.map(warning => `<div style="color: #cccccc; font-size: 10px;">• ${warning}</div>`).join('')}
          <div style="color: #aaaaaa; font-size: 10px; margin-top: 2px;">Results may be incomplete. Try reconnecting and retesting.</div>
        </div>
      `;
    }

    // 병목 구간 HTML 생성
    let bottleneckHtml = '';
    if (result.bottlenecks && result.bottlenecks.length > 0) {
      bottleneckHtml = '<div style="margin-top: 4px; padding-top: 4px; border-top: 1px solid #555;">';
      bottleneckHtml += '<div style="font-weight: bold; color: #ff6666; margin-bottom: 2px;">Performance Issues:</div>';

      result.bottlenecks.forEach(bottleneck => {
        const severityColor = bottleneck.severity === 'critical' ? '#ff4444' : '#ffaa00';
        bottleneckHtml += `
          <div style="margin-bottom: 3px; padding: 2px 4px; background: rgba(255,68,68,0.1); border-left: 2px solid ${severityColor}; font-size: 11px;">
            <div style="font-weight: bold; color: ${severityColor};">${bottleneck.stage} (${bottleneck.avgTime.toFixed(1)}ms)</div>
            <div style="color: #cccccc; font-size: 10px;">${bottleneck.suggestion}</div>
          </div>
        `;
      });

      bottleneckHtml += '</div>';
    }

    this.fpsTestResult.innerHTML = `
      <div style="margin-bottom: 2px; font-weight: bold;">Performance Test Complete (${duration}s):</div>

      ${warningHtml}

      <div style="margin-bottom: 4px; padding: 2px 0;">
        <div style="color: #00ff00; font-size: 12px; margin-bottom: 1px;">📊 FPS Metrics:</div>
        <div style="margin-left: 8px; font-size: 11px;">
          <div>Pure Decode: ${pureDecodeFPS.toFixed(2)} fps (${avgDecodeTime.toFixed(1)}ms avg)</div>
          <div>Frame Processing: ${frameProcessingFPS.toFixed(2)} fps (${avgProcessingTime.toFixed(1)}ms avg)</div>
          <div>Render: ${renderFPS.toFixed(2)} fps (${avgRenderTime.toFixed(1)}ms avg)</div>
          <div>Legacy Decode: ${legacyDecodeFPS.toFixed(2)} fps</div>
        </div>
      </div>

      <div style="margin-bottom: 4px; padding: 2px 0;">
        <div style="color: #00ff00; font-size: 12px; margin-bottom: 1px;">⚡ Latency:</div>
        <div style="margin-left: 8px; font-size: 11px;">
          <div>Average: ${averageLatency.toFixed(1)}ms</div>
          <div>Range: ${minLatency.toFixed(1)}ms - ${maxLatency.toFixed(1)}ms</div>
          <div>Total Frames: ${totalFrames}</div>
        </div>
      </div>

      ${bottleneckHtml}
    `;

    this.fpsTestResult.style.display = 'block';

    // 마지막 결과 저장 및 다운로드 버튼 표시 (유효한 데이터가 있을 때만)
    if (hasValidData && (pureDecodeFPS > 0 || frameProcessingFPS > 0 || renderFPS > 0)) {
      if (this.fpsResultDownload) {
        this.fpsResultDownload.style.display = 'block';
      }
      debug.logFPS('[FPSTestPanel] Result displayed successfully with download option');
    } else {
      debug.warn('[FPSTestPanel] Result displayed but insufficient data for download');
    }

    // 병목 구간이 있으면 별도 로그 출력
    if (result.bottlenecks && result.bottlenecks.length > 0) {
      debug.logFPS('[FPSTestPanel] 🚨 Performance Bottlenecks Detected:');
      result.bottlenecks.forEach(bottleneck => {
        debug.logFPS(`  ${bottleneck.severity.toUpperCase()}: ${bottleneck.stage} - ${bottleneck.avgTime.toFixed(1)}ms (${bottleneck.percentage.toFixed(1)}%)`);
        debug.logFPS(`  Suggestion: ${bottleneck.suggestion}`);
      });
    }
  }

  private downloadResults(): void {
    if (!this.lastFPSResult) {
      debug.warn('[FPSTestPanel] No FPS test results available for download');
      return;
    }

    const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
    const filename = `fps-test-results-${timestamp}.txt`;

    // 현재 설정 정보 수집
    const currentResolution = document.querySelector('#resolution-select') as HTMLSelectElement;
    const selectedResolution = currentResolution ? currentResolution.value : 'unknown';
    const jpegFallbackCheckbox = document.getElementById('jpeg-fallback-checkbox') as HTMLInputElement;
    const jpegMode = jpegFallbackCheckbox?.checked ?? false;

    // TODO: renderMode를 외부에서 주입받도록 개선 필요
    const renderMode = 'unknown';

    // txt 내용 생성
    const content = this.generateReportText(this.lastFPSResult, {
      resolution: selectedResolution,
      jpegMode,
      renderMode,
      timestamp: new Date().toISOString()
    });

    // 파일 다운로드
    const blob = new Blob([content], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);

    debug.logFPS(`[FPSTestPanel] Downloaded test results: ${filename}`);
  }

  private generateReportText(result: FPSMeasurementResult, config: {
    resolution: string;
    jpegMode: boolean;
    renderMode: string;
    timestamp: string;
  }): string {
    const duration = (result.measurementDurationMs / 1000).toFixed(1);

    // 병목 구간 텍스트 생성
    let bottleneckText = '';
    if (result.bottlenecks && result.bottlenecks.length > 0) {
      bottleneckText = '\nPerformance Bottlenecks Detected:\n';
      bottleneckText += '----------------------------------------\n';
      result.bottlenecks.forEach((bottleneck, index) => {
        bottleneckText += `${index + 1}. ${bottleneck.stage} [${bottleneck.severity.toUpperCase()}]\n`;
        bottleneckText += `   - Average Time: ${bottleneck.avgTime.toFixed(2)} ms\n`;
        bottleneckText += `   - Threshold: ${bottleneck.threshold.toFixed(2)} ms\n`;
        bottleneckText += `   - Percentage of Total: ${bottleneck.percentage.toFixed(1)}%\n`;
        bottleneckText += `   - Recommendation: ${bottleneck.suggestion}\n\n`;
      });
    }

    return `StreamSplat Enhanced Performance Test Results
=====================================================

Test Configuration:
- Timestamp: ${config.timestamp}
- Resolution: ${config.resolution}
- Encoding: ${config.jpegMode ? 'JPEG' : 'H.264'}
- Render Mode: ${config.renderMode}
- Test Duration: ${duration} seconds

Enhanced Performance Metrics:
- Pure Decode FPS: ${result.pureDecodeFPS.toFixed(2)} fps (${result.avgDecodeTime.toFixed(2)} ms avg per frame)
- Frame Processing FPS: ${result.frameProcessingFPS.toFixed(2)} fps (${result.avgProcessingTime.toFixed(2)} ms avg per frame)
- Render FPS: ${result.renderFPS.toFixed(2)} fps (${result.avgRenderTime.toFixed(2)} ms avg per frame)
- Legacy Decode FPS: ${result.decodeFPS.toFixed(2)} fps (main thread measurement)
- Total Frames Processed: ${result.totalFrames}

Performance Analysis:
- Most accurate FPS metric: Pure Decode FPS (${result.pureDecodeFPS.toFixed(2)} fps)
- Decode processing is ${result.pureDecodeFPS > result.decodeFPS ? 'faster' : 'slower'} than legacy measurement suggests
- Frame processing overhead: ${result.avgProcessingTime.toFixed(2)} ms per frame
- Rendering overhead: ${result.avgRenderTime.toFixed(2)} ms per frame

Latency Statistics:
- Average Total Latency: ${result.averageLatency.toFixed(2)} ms
- Minimum Latency: ${result.minLatency.toFixed(2)} ms
- Maximum Latency: ${result.maxLatency.toFixed(2)} ms
- Latency Range: ${(result.maxLatency - result.minLatency).toFixed(2)} ms
${bottleneckText}
System Information:
- User Agent: ${navigator.userAgent}
- Window Size: ${window.innerWidth}×${window.innerHeight}
- Pixel Ratio: ${window.devicePixelRatio}
- Hardware Decoding: ${config.jpegMode ? 'Not Used (JPEG Mode)' : 'Available (H.264 Mode)'}

Performance Insights:
- Your decode performance: ${result.pureDecodeFPS.toFixed(0)} fps is ${result.pureDecodeFPS >= 60 ? 'excellent' : result.pureDecodeFPS >= 30 ? 'good' : 'below optimal'} for real-time streaming
- Frame processing latency: ${result.avgProcessingTime.toFixed(1)} ms is ${result.avgProcessingTime <= 3 ? 'excellent' : result.avgProcessingTime <= 8 ? 'acceptable' : 'high'}
- Overall system performance: ${result.averageLatency <= 50 ? 'Excellent' : result.averageLatency <= 100 ? 'Good' : 'Needs optimization'} (${result.averageLatency.toFixed(0)} ms total latency)

=====================================================
Generated by StreamSplat Enhanced Performance Testing Tool
`;
  }

  cleanup(): void {
    // Event listeners are automatically removed when elements are removed from DOM
  }
}
