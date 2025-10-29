/**
 * RecordingPanel - Screen recording functionality
 * Fully migrated from main.ts (lines 1394-1584)
 */

import { debug } from '../../debug-logger';
import { RenderMode } from './ControlPanel';

export class RecordingPanel {
  // DOM elements
  private recordingButton: HTMLInputElement | null = null;
  private recordingStatus: HTMLDivElement | null = null;
  private recordingTime: HTMLDivElement | null = null;
  private recordingMode: HTMLDivElement | null = null;
  private recordingSize: HTMLDivElement | null = null;
  private recordingDownload: HTMLInputElement | null = null;
  private recordingCompatibility: HTMLDivElement | null = null;

  // State
  private mediaRecorder: MediaRecorder | null = null;
  private recordedChunks: Blob[] = [];
  private recordingStartTime = 0;
  private recordingTimer: number | null = null;
  private recordingStream: MediaStream | null = null;
  private isRecordingSupported = false;
  private recordingBlob: Blob | null = null;
  private canvas: HTMLCanvasElement | null = null;
  private currentRenderMode: RenderMode = RenderMode.FUSION;

  constructor() {
    this.initializeElements();
    this.setupListeners();
    this.checkSupport();
  }

  private initializeElements(): void {
    this.recordingButton = document.getElementById('recording-button') as HTMLInputElement;
    this.recordingStatus = document.getElementById('recording-status') as HTMLDivElement;
    this.recordingTime = document.getElementById('recording-time') as HTMLDivElement;
    this.recordingMode = document.getElementById('recording-mode') as HTMLDivElement;
    this.recordingSize = document.getElementById('recording-size') as HTMLDivElement;
    this.recordingDownload = document.getElementById('recording-download') as HTMLInputElement;
    this.recordingCompatibility = document.getElementById('recording-compatibility') as HTMLDivElement;
  }

  private setupListeners(): void {
    this.recordingButton?.addEventListener('click', () => this.handleRecordingToggle());
    this.recordingDownload?.addEventListener('click', () => this.downloadRecording());
  }

  private checkSupport(): void {
    this.isRecordingSupported = !!(window.MediaRecorder && HTMLCanvasElement.prototype.captureStream);

    if (!this.isRecordingSupported) {
      if (this.recordingCompatibility) {
        this.recordingCompatibility.style.display = 'block';
      }
      if (this.recordingButton) {
        this.recordingButton.disabled = true;
      }
      debug.warn('[RecordingPanel] Screen recording not supported in this browser');
    } else {
      debug.logMain('[RecordingPanel] Screen recording is supported');
    }
  }

  private handleRecordingToggle(): void {
    if (!this.isRecordingSupported) {
      if (this.recordingCompatibility) {
        this.recordingCompatibility.style.display = 'block';
        this.recordingCompatibility.textContent = 'Screen recording not supported in this browser';
      }
      return;
    }

    if (this.mediaRecorder?.state === 'recording') {
      this.stopRecording();
    } else {
      this.startRecording();
    }
  }

  private setupMediaRecorder(): boolean {
    try {
      if (!this.canvas) {
        debug.error('[RecordingPanel] Canvas not available');
        return false;
      }

      // Canvas에서 MediaStream 생성 (60fps)
      this.recordingStream = this.canvas.captureStream(60);
      if (!this.recordingStream) {
        debug.error('[RecordingPanel] Failed to capture stream from canvas');
        return false;
      }

      debug.logMain('[RecordingPanel] Canvas stream created successfully');

      // MediaRecorder 생성
      const options: MediaRecorderOptions = {
        mimeType: 'video/webm;codecs=vp9',
        videoBitsPerSecond: 8000000, // 8 Mbps
      };

      // VP9가 지원되지 않으면 VP8 시도
      if (!MediaRecorder.isTypeSupported(options.mimeType!)) {
        options.mimeType = 'video/webm;codecs=vp8';
        debug.logMain('[RecordingPanel] VP9 not supported, using VP8');
      }

      // VP8도 지원되지 않으면 기본 webm
      if (!MediaRecorder.isTypeSupported(options.mimeType!)) {
        options.mimeType = 'video/webm';
        debug.logMain('[RecordingPanel] VP8 not supported, using default webm');
      }

      this.mediaRecorder = new MediaRecorder(this.recordingStream, options);

      // 녹화 이벤트 핸들러
      this.mediaRecorder.ondataavailable = (event) => {
        if (event.data && event.data.size > 0) {
          this.recordedChunks.push(event.data);
          debug.logMain(`[RecordingPanel] Data chunk received: ${event.data.size} bytes`);
        }
      };

      this.mediaRecorder.onstop = () => {
        debug.logMain('[RecordingPanel] Recording stopped');
        this.recordingBlob = new Blob(this.recordedChunks, { type: 'video/webm' });
        debug.logMain(`[RecordingPanel] Final video blob size: ${this.recordingBlob.size} bytes`);

        // 다운로드 버튼 활성화
        if (this.recordingDownload) {
          this.recordingDownload.style.display = 'block';
          this.recordingDownload.value = `Download Recording (${(this.recordingBlob.size / 1024 / 1024).toFixed(1)}MB)`;
        }
      };

      this.mediaRecorder.onerror = (event) => {
        debug.error('[RecordingPanel] MediaRecorder error:', event);
      };

      return true;

    } catch (error) {
      debug.error('[RecordingPanel] Setup failed:', error);
      return false;
    }
  }

  private startRecording(): void {
    try {
      // MediaRecorder 설정
      if (!this.setupMediaRecorder()) {
        debug.error('[RecordingPanel] Failed to setup recording');
        return;
      }

      // 녹화 데이터 초기화
      this.recordedChunks = [];
      this.recordingBlob = null;
      this.recordingStartTime = performance.now();

      // MediaRecorder 시작
      this.mediaRecorder!.start(1000); // 1초마다 데이터 청크 생성

      // UI 업데이트
      if (this.recordingButton) this.recordingButton.value = 'Stop Recording';
      if (this.recordingStatus) {
        this.recordingStatus.style.display = 'block';
        this.recordingStatus.textContent = 'Status: Recording...';
      }
      if (this.recordingTime) this.recordingTime.style.display = 'block';
      if (this.recordingMode) {
        this.recordingMode.style.display = 'block';
        this.recordingMode.textContent = `Mode: ${this.currentRenderMode}`;
      }
      if (this.recordingSize) this.recordingSize.style.display = 'block';
      if (this.recordingDownload) this.recordingDownload.style.display = 'none';

      // 타이머 시작
      this.recordingTimer = window.setInterval(() => this.updateRecordingUI(), 100);

      debug.logMain(`[RecordingPanel] Started recording in ${this.currentRenderMode} mode`);

    } catch (error) {
      debug.error('[RecordingPanel] Failed to start recording:', error);
      if (this.recordingStatus) {
        this.recordingStatus.textContent = 'Status: Failed to start recording';
      }
    }
  }

  private stopRecording(): void {
    try {
      if (this.mediaRecorder && this.mediaRecorder.state === 'recording') {
        this.mediaRecorder.stop();
      }

      // 타이머 정리
      if (this.recordingTimer) {
        clearInterval(this.recordingTimer);
        this.recordingTimer = null;
      }

      // 스트림 정리
      if (this.recordingStream) {
        this.recordingStream.getTracks().forEach(track => track.stop());
        this.recordingStream = null;
      }

      // UI 업데이트
      if (this.recordingButton) this.recordingButton.value = 'Start Recording';
      if (this.recordingStatus) this.recordingStatus.textContent = 'Status: Stopped';

      debug.logMain('[RecordingPanel] Recording stopped successfully');

    } catch (error) {
      debug.error('[RecordingPanel] Error stopping recording:', error);
    }
  }

  private updateRecordingUI(): void {
    if (this.recordingStartTime > 0) {
      const elapsedMs = performance.now() - this.recordingStartTime;
      const elapsedSeconds = Math.floor(elapsedMs / 1000);
      const minutes = Math.floor(elapsedSeconds / 60);
      const seconds = elapsedSeconds % 60;

      if (this.recordingTime) {
        this.recordingTime.textContent = `Duration: ${minutes}:${seconds.toString().padStart(2, '0')}`;
      }

      // 현재까지 녹화된 데이터 크기 표시
      if (this.recordedChunks.length > 0 && this.recordingSize) {
        const totalSize = this.recordedChunks.reduce((total, chunk) => total + chunk.size, 0);
        this.recordingSize.textContent = `Size: ${(totalSize / 1024 / 1024).toFixed(1)}MB`;
      }
    }
  }

  private downloadRecording(): void {
    if (!this.recordingBlob) {
      debug.warn('[RecordingPanel] No recording available for download');
      return;
    }

    const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
    const filename = `recording-${this.currentRenderMode}-${timestamp}.webm`;

    const url = URL.createObjectURL(this.recordingBlob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);

    debug.logMain(`[RecordingPanel] Downloaded: ${filename}`);
  }

  // Public methods
  setCanvas(canvas: HTMLCanvasElement): void {
    this.canvas = canvas;
  }

  setRenderMode(mode: RenderMode): void {
    this.currentRenderMode = mode;
  }

  cleanup(): void {
    if (this.recordingTimer) {
      clearInterval(this.recordingTimer);
      this.recordingTimer = null;
    }
    if (this.recordingStream) {
      this.recordingStream.getTracks().forEach(track => track.stop());
      this.recordingStream = null;
    }
    if (this.mediaRecorder && this.mediaRecorder.state === 'recording') {
      this.mediaRecorder.stop();
    }
  }
}
