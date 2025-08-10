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

export class LatencyTracker {
    private timingData: Map<number, TimingData> = new Map();
    private latencyHistory: LatencyStats[] = [];
    private clockOffset: number = 0;
    private clockSyncHistory: ClockSyncData[] = [];
    private frameIdCounter: number = 0;
    private maxHistorySize: number = 1000;
    private pingInterval: number = 5000; // 5초마다 클럭 동기화
    private lastPingTime: number = 0;

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
                console.warn(`Negative upload time detected: ${networkUploadTime.toFixed(1)}ms, clock sync may need adjustment`);
                networkUploadTime = Math.max(0, networkUploadTime);
            }
            if (networkDownloadTime < 0) {
                console.warn(`Negative download time detected: ${networkDownloadTime.toFixed(1)}ms, clock sync may need adjustment`);
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
}