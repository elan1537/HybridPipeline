import { debug } from './debug-logger';

export interface TimingData {
    frameId: number;
    clientSendTime: number;          // 클라이언트가 카메라 데이터를 보낸 시간
    serverReceiveTime?: number;      // 서버가 받은 시간  
    serverProcessStartTime?: number; // 서버 처리 시작 시간
    serverProcessEndTime?: number;   // 서버 처리 완료 시간
    serverSendTime?: number;         // 서버가 응답을 보낸 시간
    clientReceiveTime?: number;      // 클라이언트가 받은 시간
    clientDecodeTime?: number;       // 클라이언트 디코딩 완료 시간
    clientRenderTime?: number;       // 클라이언트 렌더링 완료 시간
}

export interface LatencyStats {
    roundTripTime: number;           // 전체 왕복 시간
    networkUploadTime: number;       // 업로드 네트워크 시간
    serverProcessingTime: number;    // 서버 처리 시간
    networkDownloadTime: number;     // 다운로드 네트워크 시간
    clientDecodeTime: number;        // 클라이언트 디코딩 시간
    clientRenderTime: number;        // 클라이언트 렌더링 시간
    totalLatency: number;           // 총 레이턴시 (send -> render)
}

export interface ClockSyncData {
    clientRequestTime: number;
    serverReceiveTime: number;
    serverSendTime: number;
    clientResponseTime: number;
    roundTripTime: number;
    estimatedOffset: number;
}

export interface PerformanceBottleneck {
    stage: string;                  // 병목 단계 이름
    severity: 'warning' | 'critical'; // 심각도
    avgTime: number;                // 평균 처리 시간
    threshold: number;              // 임계값
    percentage: number;             // 전체 대비 비율
    suggestion: string;             // 개선 제안
}

export interface FPSMeasurementResult {
    decodeFPS: number;              // 기존 decode FPS (main thread 기준)
    pureDecodeFPS: number;          // 순수 decode FPS (worker 내부 측정)
    frameProcessingFPS: number;     // 프레임 처리 FPS (texture update 등)
    renderFPS: number;              // 실제 렌더링 FPS (requestAnimationFrame 기준)
    totalFrames: number;
    measurementDurationMs: number;
    averageLatency: number;
    minLatency: number;
    maxLatency: number;
    // 추가 통계
    avgDecodeTime: number;          // 평균 디코딩 시간
    avgProcessingTime: number;      // 평균 프레임 처리 시간
    avgRenderTime: number;          // 평균 렌더링 시간
    // 성능 병목 분석
    bottlenecks: PerformanceBottleneck[]; // 감지된 병목 구간들
}

export class LatencyTracker {
    private timingData: Map<number, TimingData> = new Map();
    private latencyHistory: LatencyStats[] = [];
    private clockOffset: number = 0;
    private clockSyncHistory: ClockSyncData[] = [];
    private frameIdCounter: number = 0;
    private maxHistorySize: number = 1000;
    private pingInterval: number = 5000; // 5초마다 클럭 동기화
    private lastPingTime: number = 0;
    
    // FPS 측정 관련 필드
    private fpsTestActive: boolean = false;
    private fpsTestStartTime: number = 0;
    private fpsTestDecodeCount: number = 0;
    private fpsTestRenderCount: number = 0;
    private fpsTestLatencySum: number = 0;
    private fpsTestMinLatency: number = Infinity;
    private fpsTestMaxLatency: number = 0;
    private fpsTestTimeout: number | null = null;
    
    // 샘플링 기반 성능 측정 필드들
    private fpsTestPureDecodeSamples: number[] = [];
    private fpsTestFrameProcessingSamples: number[] = [];
    private fpsTestDecodeTimeSamples: number[] = [];
    private fpsTestProcessingTimeSamples: number[] = [];
    private fpsTestRenderTimeSamples: number[] = [];

    constructor() {
        this.startClockSync();
    }

    // 새 프레임 ID 생성
    generateFrameId(): number {
        return ++this.frameIdCounter;
    }

    // 카메라 데이터 전송 시작 시점 기록
    recordCameraSend(frameId: number): void {
        const timing: TimingData = {
            frameId,
            clientSendTime: performance.now()
        };
        this.timingData.set(frameId, timing);
        
        // 오래된 데이터 정리
        if (this.timingData.size > this.maxHistorySize) {
            const oldestFrameId = Math.min(...this.timingData.keys());
            this.timingData.delete(oldestFrameId);
        }
    }

    // 서버로부터 응답 수신 시점 기록
    recordFrameReceive(frameId: number, serverTimestamps?: {
        serverReceiveTime: number;
        serverProcessStartTime: number;
        serverProcessEndTime: number;
        serverSendTime: number;
    }): void {
        const timing = this.timingData.get(frameId);
        if (!timing) return;

        timing.clientReceiveTime = performance.now();
        
        if (serverTimestamps) {
            // 서버 타임스탬프를 원본 그대로 저장 (계산시에 오프셋 적용)
            timing.serverReceiveTime = serverTimestamps.serverReceiveTime;
            timing.serverProcessStartTime = serverTimestamps.serverProcessStartTime;
            timing.serverProcessEndTime = serverTimestamps.serverProcessEndTime;
            timing.serverSendTime = serverTimestamps.serverSendTime;
        }
    }

    // 디코딩 완료 시점 기록
    recordDecodeComplete(frameId: number): void {
        const timing = this.timingData.get(frameId);
        if (!timing) return;

        timing.clientDecodeTime = performance.now();
        
        // FPS 측정 중이면 카운트
        if (this.fpsTestActive) {
            this.fpsTestDecodeCount++;
        }
    }

    // 렌더링 완료 시점 기록 및 레이턴시 계산
    recordRenderComplete(frameId: number): LatencyStats | null {
        const timing = this.timingData.get(frameId);
        if (!timing) return null;

        timing.clientRenderTime = performance.now();

        // 레이턴시 통계 계산
        const stats = this.calculateLatencyStats(timing);
        if (stats) {
            this.latencyHistory.push(stats);
            
            // 히스토리 크기 제한
            if (this.latencyHistory.length > this.maxHistorySize) {
                this.latencyHistory.shift();
            }
            
            // FPS 측정 중이면 카운트 및 통계 업데이트
            if (this.fpsTestActive) {
                this.fpsTestRenderCount++;
                this.fpsTestLatencySum += stats.totalLatency;
                this.fpsTestMinLatency = Math.min(this.fpsTestMinLatency, stats.totalLatency);
                this.fpsTestMaxLatency = Math.max(this.fpsTestMaxLatency, stats.totalLatency);
                
                // 렌더링 시간 샘플 수집
                const renderTime = stats.clientRenderTime;
                if (renderTime > 0 && renderTime <= 100) { // 100ms 이하만 유효한 것으로 간주
                    this.fpsTestRenderTimeSamples.push(renderTime);
                }
            }
        }

        return stats;
    }

    private calculateLatencyStats(timing: TimingData): LatencyStats | null {
        const { 
            clientSendTime, 
            serverReceiveTime, 
            serverProcessStartTime,
            serverProcessEndTime,
            serverSendTime,
            clientReceiveTime, 
            clientDecodeTime, 
            clientRenderTime 
        } = timing;

        if (!clientReceiveTime || !clientDecodeTime || !clientRenderTime) {
            return null;
        }

        const roundTripTime = clientReceiveTime - clientSendTime;
        const totalLatency = clientRenderTime - clientSendTime;

        let networkUploadTime = 0;
        let serverProcessingTime = 0;
        let networkDownloadTime = 0;

        // 서버 타임스탬프가 있는 경우 더 정확한 계산 (클럭 오프셋 적용)
        if (serverReceiveTime && serverProcessStartTime && serverProcessEndTime && serverSendTime) {
            // 서버 시간을 클라이언트 시간으로 변환 (클럭 오프셋 차감)
            const adjustedServerReceiveTime = serverReceiveTime - this.clockOffset;
            const adjustedServerProcessStartTime = serverProcessStartTime - this.clockOffset;
            const adjustedServerProcessEndTime = serverProcessEndTime - this.clockOffset;
            const adjustedServerSendTime = serverSendTime - this.clockOffset;
            
            networkUploadTime = adjustedServerReceiveTime - clientSendTime;
            serverProcessingTime = adjustedServerProcessEndTime - adjustedServerProcessStartTime;
            networkDownloadTime = clientReceiveTime - adjustedServerSendTime;
            
            // 네트워크 지연시간이 음수가 되는 경우 클럭 동기화 오차로 간주하여 보정
            if (networkUploadTime < 0) {
                debug.warn(`Negative upload time detected: ${networkUploadTime.toFixed(1)}ms, clock sync may need adjustment`);
                networkUploadTime = Math.max(0, networkUploadTime);
            }
            if (networkDownloadTime < 0) {
                debug.warn(`Negative download time detected: ${networkDownloadTime.toFixed(1)}ms, clock sync may need adjustment`);
                networkDownloadTime = Math.max(0, networkDownloadTime);
            }
        } else {
            // 서버 타임스탬프가 없는 경우 추정
            networkUploadTime = roundTripTime * 0.3; // 추정
            serverProcessingTime = roundTripTime * 0.4; // 추정  
            networkDownloadTime = roundTripTime * 0.3; // 추정
        }

        return {
            roundTripTime,
            networkUploadTime: Math.max(0, networkUploadTime),
            serverProcessingTime: Math.max(0, serverProcessingTime),
            networkDownloadTime: Math.max(0, networkDownloadTime),
            clientDecodeTime: clientDecodeTime - clientReceiveTime,
            clientRenderTime: clientRenderTime - clientDecodeTime,
            totalLatency
        };
    }

    // 클럭 동기화 (개선된 알고리즘)
    recordClockSync(clientRequestTime: number, serverReceiveTime: number, serverSendTime: number): void {
        const clientResponseTime = performance.now();
        const roundTripTime = clientResponseTime - clientRequestTime;
        
        // 네트워크 지연시간의 절반을 가정하여 오프셋 계산
        const networkDelay = roundTripTime / 2;
        const serverMidTime = (serverReceiveTime + serverSendTime) / 2;
        const clientMidTime = clientRequestTime + networkDelay;
        const estimatedOffset = serverMidTime - clientMidTime;

        const syncData: ClockSyncData = {
            clientRequestTime,
            serverReceiveTime,
            serverSendTime,
            clientResponseTime,
            roundTripTime,
            estimatedOffset
        };

        this.clockSyncHistory.push(syncData);
        
        // 이상치 제거 및 이동평균을 사용한 오프셋 계산
        const recentSyncs = this.clockSyncHistory.slice(-20);
        if (recentSyncs.length >= 3) {
            // RTT가 너무 큰 측정값 제거 (이상치)
            const rttValues = recentSyncs.map(sync => sync.roundTripTime);
            const avgRtt = rttValues.reduce((a, b) => a + b, 0) / rttValues.length;
            const rttStdDev = Math.sqrt(rttValues.reduce((sq, val) => sq + Math.pow(val - avgRtt, 2), 0) / rttValues.length);
            
            const filteredSyncs = recentSyncs.filter(sync => 
                Math.abs(sync.roundTripTime - avgRtt) <= 2 * rttStdDev
            );
            
            if (filteredSyncs.length >= 3) {
                // 지수 이동평균으로 오프셋 업데이트
                const newOffset = filteredSyncs.reduce((sum, sync) => sum + sync.estimatedOffset, 0) / filteredSyncs.length;
                const alpha = 0.3; // EMA 가중치
                this.clockOffset = this.clockOffset * (1 - alpha) + newOffset * alpha;
            }
        } else {
            // 초기 몇 개 측정값은 단순 평균 사용
            const avgOffset = recentSyncs.reduce((sum, sync) => sum + sync.estimatedOffset, 0) / recentSyncs.length;
            this.clockOffset = avgOffset;
        }

        // 히스토리 크기 제한
        if (this.clockSyncHistory.length > 100) {
            this.clockSyncHistory.shift();
        }
    }

    // 주기적으로 클럭 동기화 필요한지 확인
    shouldSendPing(): boolean {
        const now = performance.now();
        if (now - this.lastPingTime > this.pingInterval) {
            this.lastPingTime = now;
            return true;
        }
        return false;
    }

    private startClockSync(): void {
        // 초기 클럭 동기화는 외부에서 트리거
    }

    // 통계 조회 (이상치 제거 적용)
    getRecentStats(windowSize: number = 100): {
        avg: Partial<LatencyStats>;
        min: Partial<LatencyStats>;
        max: Partial<LatencyStats>;
        p95: Partial<LatencyStats>;
        p99: Partial<LatencyStats>;
    } {
        const recentStats = this.latencyHistory.slice(-windowSize);
        if (recentStats.length === 0) {
            return { avg: {}, min: {}, max: {}, p95: {}, p99: {} };
        }

        const keys: (keyof LatencyStats)[] = [
            'roundTripTime', 'networkUploadTime', 'serverProcessingTime', 
            'networkDownloadTime', 'clientDecodeTime', 'clientRenderTime', 'totalLatency'
        ];

        const result: any = {
            avg: {},
            min: {},
            max: {},
            p95: {},
            p99: {}
        };

        keys.forEach(key => {
            let values = recentStats.map(stat => stat[key]).filter(val => val !== undefined && !isNaN(val));
            
            if (values.length === 0) return;
            
            // 이상치 제거 (IQR 방법 사용)
            if (values.length >= 10) {
                values.sort((a, b) => a - b);
                const q1 = values[Math.floor(values.length * 0.25)];
                const q3 = values[Math.floor(values.length * 0.75)];
                const iqr = q3 - q1;
                const lowerBound = q1 - 1.5 * iqr;
                const upperBound = q3 + 1.5 * iqr;
                
                values = values.filter(val => val >= lowerBound && val <= upperBound);
            } else {
                values.sort((a, b) => a - b);
            }
            
            if (values.length === 0) return;
            
            const sum = values.reduce((a, b) => a + b, 0);
            
            result.avg[key] = sum / values.length;
            result.min[key] = values[0];
            result.max[key] = values[values.length - 1];
            result.p95[key] = values[Math.floor(values.length * 0.95)] || values[values.length - 1];
            result.p99[key] = values[Math.floor(values.length * 0.99)] || values[values.length - 1];
        });

        return result;
    }

    // 현재 클럭 오프셋 조회
    getClockOffset(): number {
        return this.clockOffset;
    }

    // 모든 히스토리 데이터 조회 (디버깅용)
    getLatencyHistory(): LatencyStats[] {
        return [...this.latencyHistory];
    }

    // 클럭 동기화 히스토리 조회 (디버깅용)
    getClockSyncHistory(): ClockSyncData[] {
        return [...this.clockSyncHistory];
    }
    
    // 성능 병목 구간 분석
    private analyzeBottlenecks(avgDecodeTime: number, avgProcessingTime: number, avgRenderTime: number, stats: any): PerformanceBottleneck[] {
        const bottlenecks: PerformanceBottleneck[] = [];
        
        // 각 단계별 임계값 정의 (밀리초)
        const thresholds = {
            networkUpload: 5.0,     // 5ms 이상
            serverProcessing: 10.0,  // 10ms 이상  
            networkDownload: 5.0,    // 5ms 이상
            decoding: 8.0,          // 8ms 이상 (60fps = 16.67ms 프레임 기준)
            processing: 3.0,        // 3ms 이상
            rendering: 2.0          // 2ms 이상
        };
        
        const totalLatency = stats.avg.totalLatency || 0;
        
        // Network Upload 병목 체크
        const networkUploadTime = stats.avg.networkUploadTime || 0;
        if (networkUploadTime > thresholds.networkUpload) {
            bottlenecks.push({
                stage: 'Network Upload',
                severity: networkUploadTime > thresholds.networkUpload * 2 ? 'critical' : 'warning',
                avgTime: networkUploadTime,
                threshold: thresholds.networkUpload,
                percentage: (networkUploadTime / totalLatency) * 100,
                suggestion: 'Check network connectivity or server location. Consider using a closer server or optimizing camera data compression.'
            });
        }
        
        // Server Processing 병목 체크
        const serverProcessingTime = stats.avg.serverProcessingTime || 0;
        if (serverProcessingTime > thresholds.serverProcessing) {
            bottlenecks.push({
                stage: 'Server Processing',
                severity: serverProcessingTime > thresholds.serverProcessing * 2 ? 'critical' : 'warning',
                avgTime: serverProcessingTime,
                threshold: thresholds.serverProcessing,
                percentage: (serverProcessingTime / totalLatency) * 100,
                suggestion: 'Server GPU may be overloaded. Try reducing resolution or using JPEG fallback mode.'
            });
        }
        
        // Network Download 병목 체크
        const networkDownloadTime = stats.avg.networkDownloadTime || 0;
        if (networkDownloadTime > thresholds.networkDownload) {
            bottlenecks.push({
                stage: 'Network Download',
                severity: networkDownloadTime > thresholds.networkDownload * 2 ? 'critical' : 'warning',
                avgTime: networkDownloadTime,
                threshold: thresholds.networkDownload,
                percentage: (networkDownloadTime / totalLatency) * 100,
                suggestion: 'Download bandwidth may be limited. Check network speed or reduce stream quality.'
            });
        }
        
        // Decoding 병목 체크
        if (avgDecodeTime > thresholds.decoding) {
            bottlenecks.push({
                stage: 'Decoding',
                severity: avgDecodeTime > thresholds.decoding * 2 ? 'critical' : 'warning',
                avgTime: avgDecodeTime,
                threshold: thresholds.decoding,
                percentage: (avgDecodeTime / totalLatency) * 100,
                suggestion: 'Hardware decoding may not be available or CPU is overloaded. Try JPEG fallback mode or close other applications.'
            });
        }
        
        // Frame Processing 병목 체크
        if (avgProcessingTime > thresholds.processing) {
            bottlenecks.push({
                stage: 'Frame Processing',
                severity: avgProcessingTime > thresholds.processing * 2 ? 'critical' : 'warning',
                avgTime: avgProcessingTime,
                threshold: thresholds.processing,
                percentage: (avgProcessingTime / totalLatency) * 100,
                suggestion: 'Main thread is busy with texture updates. This is typically not a major bottleneck.'
            });
        }
        
        // Rendering 병목 체크  
        if (avgRenderTime > thresholds.rendering) {
            bottlenecks.push({
                stage: 'Rendering',
                severity: avgRenderTime > thresholds.rendering * 2 ? 'critical' : 'warning',
                avgTime: avgRenderTime,
                threshold: thresholds.rendering,
                percentage: (avgRenderTime / totalLatency) * 100,
                suggestion: 'GPU rendering is slow. Try reducing window size or switching to Local Only mode.'
            });
        }
        
        // 병목 구간을 심각도와 시간순으로 정렬
        return bottlenecks.sort((a, b) => {
            if (a.severity !== b.severity) {
                return a.severity === 'critical' ? -1 : 1;
            }
            return b.avgTime - a.avgTime;
        });
    }
    
    // 순수 디코딩 FPS 기록 (기존 방식 - 평상시 사용)
    recordPureDecodeFPS(count: number, avgTime: number): void {
        // 평상시에만 사용하는 기존 방식 (FPS 측정 중이 아닐 때)
        // FPS 측정 중이면 이 메소드는 호출되지 않음
    }
    
    // 순수 디코딩 FPS 샘플 기록 (FPS 측정 시 사용)
    recordPureDecodeFPSSample(count: number, avgTime: number): void {
        if (this.fpsTestActive && avgTime > 0) {
            const fps = count > 0 ? 1000 / avgTime : 0;
            if (fps > 0 && fps <= 240 && isFinite(fps)) {
                this.fpsTestPureDecodeSamples.push(fps);
                this.fpsTestDecodeTimeSamples.push(avgTime);
                debug.logLatency(`Pure decode FPS sample: ${fps.toFixed(2)} fps (${avgTime.toFixed(2)}ms)`);
            }
        }
    }
    
    // 프레임 처리 FPS 기록 (기존 방식 - 평상시 사용)
    recordFrameProcessingFPS(count: number, avgTime: number): void {
        // 평상시에만 사용하는 기존 방식 (FPS 측정 중이 아닐 때)
        // FPS 측정 중이면 이 메소드는 호출되지 않음
    }
    
    // 프레임 처리 FPS 샘플 기록 (FPS 측정 시 사용)
    recordFrameProcessingFPSSample(sampleCount: number, avgTime: number): void {
        if (this.fpsTestActive && sampleCount > 0 && avgTime > 0) {
            const fps = 1000 / avgTime;
            if (fps > 0 && fps <= 240 && isFinite(fps)) {
                this.fpsTestFrameProcessingSamples.push(fps);
                this.fpsTestProcessingTimeSamples.push(avgTime);
                debug.logLatency(`Frame processing FPS sample: ${fps.toFixed(2)} fps (${avgTime.toFixed(2)}ms)`);
            }
        }
    }

    // FPS 측정 시작
    startFPSMeasurement(): void {
        if (this.fpsTestActive) return;
        
        this.fpsTestActive = true;
        this.fpsTestStartTime = performance.now();
        this.fpsTestDecodeCount = 0;
        this.fpsTestRenderCount = 0;
        this.fpsTestLatencySum = 0;
        this.fpsTestMinLatency = Infinity;
        this.fpsTestMaxLatency = 0;
        
        // 샘플 배열들 초기화
        this.fpsTestPureDecodeSamples = [];
        this.fpsTestFrameProcessingSamples = [];
        this.fpsTestDecodeTimeSamples = [];
        this.fpsTestProcessingTimeSamples = [];
        this.fpsTestRenderTimeSamples = [];
        
        debug.logLatency('FPS measurement started - all counters reset');
        
        // 60초 후 자동 중지 (보조 타이머)
        this.fpsTestTimeout = window.setTimeout(() => {
            debug.logLatency('Auto-completion triggered by backup timer');
            // UI에서 처리하도록 이벤트만 발생
        }, 60000);
    }
    
    // FPS 측정 중지 및 결과 반환
    stopFPSMeasurement(): FPSMeasurementResult | null {
        if (!this.fpsTestActive) {
            debug.warn('stopFPSMeasurement called but measurement is not active');
            return null;
        }
        
        this.fpsTestActive = false;
        
        if (this.fpsTestTimeout) {
            clearTimeout(this.fpsTestTimeout);
            this.fpsTestTimeout = null;
        }
        
        const measurementDurationMs = performance.now() - this.fpsTestStartTime;
        const measurementDurationSeconds = measurementDurationMs / 1000;
        
        debug.logLatency(`FPS measurement stopped after ${measurementDurationSeconds.toFixed(1)}s`);
        debug.logLatency(`Data collected - Pure decode samples: ${this.fpsTestPureDecodeSamples.length}, Processing samples: ${this.fpsTestFrameProcessingSamples.length}, Render: ${this.fpsTestRenderCount}`);
        
        // 샘플 기반 평균 계산
        const avgDecodeTime = this.fpsTestDecodeTimeSamples.length > 0 
            ? this.fpsTestDecodeTimeSamples.reduce((a, b) => a + b, 0) / this.fpsTestDecodeTimeSamples.length 
            : 0;
        const avgProcessingTime = this.fpsTestProcessingTimeSamples.length > 0 
            ? this.fpsTestProcessingTimeSamples.reduce((a, b) => a + b, 0) / this.fpsTestProcessingTimeSamples.length 
            : 0;
        const avgRenderTime = this.fpsTestRenderTimeSamples.length > 0 
            ? this.fpsTestRenderTimeSamples.reduce((a, b) => a + b, 0) / this.fpsTestRenderTimeSamples.length 
            : 0;
        
        // 데이터 부족 경고
        if (this.fpsTestPureDecodeSamples.length === 0) {
            debug.warn('No pure decode samples collected!');
        }
        if (this.fpsTestFrameProcessingSamples.length === 0) {
            debug.warn('No frame processing samples collected!');
        }
        
        // 최근 통계 가져와서 병목 분석
        const recentStats = this.getRecentStats(100);
        const bottlenecks = this.analyzeBottlenecks(avgDecodeTime, avgProcessingTime, avgRenderTime, recentStats);
        
        // 샘플 기반 FPS 계산 (측정 시간과 무관하게 안정적)
        const calculatedPureDecodeFPS = this.fpsTestPureDecodeSamples.length > 0 
            ? this.fpsTestPureDecodeSamples.reduce((a, b) => a + b, 0) / this.fpsTestPureDecodeSamples.length 
            : 0;
        const calculatedFrameProcessingFPS = this.fpsTestFrameProcessingSamples.length > 0 
            ? this.fpsTestFrameProcessingSamples.reduce((a, b) => a + b, 0) / this.fpsTestFrameProcessingSamples.length 
            : 0;
        const calculatedRenderFPS = this.fpsTestRenderCount / measurementDurationSeconds;
        
        // 검증: FPS와 평균 시간이 일치하는지 확인
        this.validateFPSCalculations(calculatedPureDecodeFPS, avgDecodeTime, 'Pure Decode');
        this.validateFPSCalculations(calculatedFrameProcessingFPS, avgProcessingTime, 'Frame Processing');
        this.validateFPSCalculations(calculatedRenderFPS, avgRenderTime, 'Render');

        const result: FPSMeasurementResult = {
            // 기존 FPS 측정값들
            decodeFPS: this.fpsTestDecodeCount / measurementDurationSeconds,
            renderFPS: calculatedRenderFPS,
            totalFrames: this.fpsTestRenderCount,
            measurementDurationMs,
            averageLatency: this.fpsTestRenderCount > 0 ? this.fpsTestLatencySum / this.fpsTestRenderCount : 0,
            minLatency: this.fpsTestMinLatency === Infinity ? 0 : this.fpsTestMinLatency,
            maxLatency: this.fpsTestMaxLatency,
            
            // 새로운 성능 지표들 (검증된 값들)
            pureDecodeFPS: calculatedPureDecodeFPS,
            frameProcessingFPS: calculatedFrameProcessingFPS,
            avgDecodeTime,
            avgProcessingTime,
            avgRenderTime,
            
            // 성능 병목 분석 결과
            bottlenecks
        };
        
        debug.logLatency('Final result:', {
            pureDecodeFPS: result.pureDecodeFPS.toFixed(2),
            frameProcessingFPS: result.frameProcessingFPS.toFixed(2),
            renderFPS: result.renderFPS.toFixed(2),
            bottlenecks: result.bottlenecks.length
        });
        
        return result;
    }
    
    // FPS 측정 활성 상태 확인
    isFPSMeasurementActive(): boolean {
        return this.fpsTestActive;
    }
    
    // FPS 측정 진행률 반환
    getFPSMeasurementProgress(): { elapsedMs: number; remainingMs: number; progress: number } | null {
        if (!this.fpsTestActive) return null;
        
        const elapsedMs = performance.now() - this.fpsTestStartTime;
        const remainingMs = Math.max(0, 60000 - elapsedMs);
        const progress = Math.min(1, elapsedMs / 60000);
        
        return { elapsedMs, remainingMs, progress };
    }
    
    // FPS 측정 중간 결과 (실시간 표시용)
    getCurrentFPSTestStats(): { decodeFPS: number; renderFPS: number; currentLatency: number } | null {
        if (!this.fpsTestActive) return null;
        
        const elapsedSeconds = (performance.now() - this.fpsTestStartTime) / 1000;
        if (elapsedSeconds <= 0) return { decodeFPS: 0, renderFPS: 0, currentLatency: 0 };
        
        return {
            decodeFPS: this.fpsTestDecodeCount / elapsedSeconds,
            renderFPS: this.fpsTestRenderCount / elapsedSeconds,
            currentLatency: this.fpsTestRenderCount > 0 ? this.fpsTestLatencySum / this.fpsTestRenderCount : 0
        };
    }
    
    // FPS와 평균 시간 계산 검증 (개선된 버전)
    private validateFPSCalculations(fps: number, avgTime: number, stageName: string): void {
        // 기본 데이터 유효성 검사
        if (fps <= 0 && avgTime <= 0) {
            debug.warn(`${stageName}: No valid data to validate`);
            return;
        }
        
        // 극단값 검사
        const minRealisticFPS = 5; // 5fps 미만은 비현실적
        const maxRealisticFPS = 240; // 240fps 초과는 비현실적
        const minRealisticTime = 0.1; // 0.1ms 미만은 비현실적
        const maxRealisticTime = 200; // 200ms 초과는 너무 느림
        
        let isValid = true;
        const warnings: string[] = [];
        
        if (fps > 0) {
            if (fps < minRealisticFPS) {
                warnings.push(`Very low FPS: ${fps.toFixed(2)} fps (< ${minRealisticFPS} fps)`);
                isValid = false;
            }
            if (fps > maxRealisticFPS) {
                warnings.push(`Unrealistically high FPS: ${fps.toFixed(2)} fps (> ${maxRealisticFPS} fps)`);
                isValid = false;
            }
            if (!isFinite(fps)) {
                warnings.push(`Invalid FPS value: ${fps}`);
                isValid = false;
            }
        }
        
        if (avgTime > 0) {
            if (avgTime < minRealisticTime) {
                warnings.push(`Very low processing time: ${avgTime.toFixed(2)} ms (< ${minRealisticTime} ms)`);
                isValid = false;
            }
            if (avgTime > maxRealisticTime) {
                warnings.push(`Very high processing time: ${avgTime.toFixed(2)} ms (> ${maxRealisticTime} ms)`);
                isValid = false;
            }
            if (!isFinite(avgTime)) {
                warnings.push(`Invalid time value: ${avgTime}`);
                isValid = false;
            }
        }
        
        // FPS와 평균 시간의 일관성 검사 (둘 다 유효할 때만)
        if (fps > 0 && avgTime > 0) {
            const expectedFPS = 1000 / avgTime;
            const fpsDifference = Math.abs(fps - expectedFPS);
            const tolerancePercent = 10; // 10% 허용 오차로 완화
            const tolerance = Math.max(expectedFPS * (tolerancePercent / 100), 1); // 최소 1fps 오차 허용
            
            if (fpsDifference > tolerance) {
                warnings.push(`FPS/Time inconsistency: ${fps.toFixed(2)} fps vs expected ${expectedFPS.toFixed(2)} fps (${fpsDifference.toFixed(2)} difference)`);
                isValid = false;
            }
        }
        
        // 결과 출력
        if (!isValid && warnings.length > 0) {
            debug.error(`${stageName} validation failed:`);
            warnings.forEach(warning => debug.error(`  ❌ ${warning}`));
            debug.error(`  📊 Data: FPS=${fps.toFixed(2)}, AvgTime=${avgTime.toFixed(2)}ms`);
        } else {
            debug.logLatency(`${stageName} validated: ${fps.toFixed(2)} fps ↔ ${avgTime.toFixed(2)}ms ✅`);
        }
    }
}