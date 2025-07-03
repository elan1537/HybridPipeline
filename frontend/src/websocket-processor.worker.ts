/**
 * WebSocket 메시지 처리를 위한 Worker
 * 메인 스레드의 블로킹을 방지하기 위해 메시지 파싱과 디코딩을 담당합니다.
 * 분리된 Color/Depth 파이프라인을 지원합니다.
 */

// Worker 내부에서 사용할 타입 정의
interface ParsedData {
    colorData: ArrayBuffer;
    depthData: ArrayBuffer;
    frameCount: number;
}

interface ProcessedFrame {
    colorImageBitmap?: ImageBitmap;
    colorRawRGB?: ArrayBuffer;
    depthData: ArrayBuffer;          // 원본 비트스트림
    depthNDC?: ArrayBuffer;          // 디코딩된 깊이 (float32, -1~1)
    frameTimings: { [k: string]: number };
}

// 분리된 메시지 타입 정의
interface ColorFrameMessage {
    type: 'color_frame';
    data: ArrayBuffer;
    frameId: number;
    timestamp: number;
}

interface DepthFrameMessage {
    type: 'depth_frame';
    data: ArrayBuffer;
    frameId: number;
    timestamp: number;
}

// 설정 플래그들 (메인 스레드에서 전달받음)
let USE_H264_DECODER = false;
let USE_RAW_RGB = false;
let CLIENT_ASSUMED_SCALE_Y = 1.0;
let rtWidth = 1920;
let rtHeight = 1080;

// 분리된 파이프라인을 위한 변수
let colorWorker: Worker | null = null;
let depthWorker: Worker | null = null;
let pendingColorFrames = new Map<number, ColorFrameMessage>();
let pendingDepthFrames = new Map<number, DepthFrameMessage>();
let processedFrames = new Map<number, ProcessedFrame>();
let frameIdCounter = 0;

// 성능 모니터링을 위한 유틸리티
const perfMonitor = {
    timings: {
        total_onmessage: 0.0,
        header_parse: 0.0,
        color_processing: 0.0,
        h264_decode_call: 0.0,
        raw_rgb_processing: 0.0,
        depth_processing: 0.0,
    },
    frameCount: 0,
    lastPrintTime: 0,

    init() {
        this.lastPrintTime = performance.now();
    },

    reset() {
        this.frameCount = 0;
        for (const key in this.timings) {
            this.timings[key] = 0.0;
        }
        this.lastPrintTime = performance.now();
    },

    record(stats: { [key: string]: number }) {
        for (const key in stats) {
            if (key in this.timings) {
                this.timings[key] += stats[key];
            }
        }
        this.frameCount++;
    },

    logStatsIfReady(periodInMs = 60000) {
        const now = performance.now();
        if (now - this.lastPrintTime > periodInMs) {
            if (this.frameCount > 0) {
                console.log(`--- Worker Performance Stats (avg over ${this.frameCount} frames) ---`);
                for (const key in this.timings) {
                    const avgTime = this.timings[key] / this.frameCount;
                    console.log(`  ${key}: ${avgTime.toFixed(3)} ms`);
                }
                console.log('--------------------------------------------------');

                // 메인 스레드로 성능 통계 전송
                (self as any).postMessage({
                    type: 'performance_stats',
                    stats: {
                        frameCount: this.frameCount,
                        timings: { ...this.timings }
                    }
                });
            }
            this.reset();
        }
    }
};

perfMonitor.init();

// 분리된 Worker 초기화
function initializeSeparatedWorkers() {
    if (!colorWorker) {
        colorWorker = new Worker(new URL('./color-worker.ts', import.meta.url), { type: 'module' });
        colorWorker.onmessage = handleColorWorkerMessage;
    }

    if (!depthWorker) {
        depthWorker = new Worker(new URL('./depth-worker.ts', import.meta.url), { type: 'module' });
        depthWorker.onmessage = handleDepthWorkerMessage;
    }

    // 설정 전달
    const config = {
        USE_H264_DECODER,
        USE_RAW_RGB,
        rtWidth,
        rtHeight
    };

    colorWorker.postMessage({ type: 'config', config });
    depthWorker.postMessage({ type: 'config', config });
}

// Color Worker 메시지 처리
function handleColorWorkerMessage(event: MessageEvent) {
    const { type, frameId, timestamp, result, processingTime } = event.data;

    if (type === 'color_result') {
        const processedFrame = processedFrames.get(frameId) || {
            depthData: new ArrayBuffer(0),
            frameTimings: {}
        };

        if (result.type === 'image_bitmap') {
            processedFrame.colorImageBitmap = result.data;
        } else if (result.type === 'raw_rgb') {
            processedFrame.colorRawRGB = result.data;
        }

        processedFrame.frameTimings.color_processing = processingTime;
        processedFrames.set(frameId, processedFrame);

        // Color와 Depth가 모두 처리되었는지 확인
        checkFrameCompletion(frameId);
    }
}

// Depth Worker 메시지 처리
function handleDepthWorkerMessage(event: MessageEvent) {
    const { type, frameId, timestamp, result, processingTime } = event.data;

    if (type === 'depth_result') {
        const processedFrame = processedFrames.get(frameId) || {
            depthData: new ArrayBuffer(0),
            frameTimings: {}
        };

        if (result.type === 'depth_ndc') {
            processedFrame.depthNDC = result.data;
        } else if (result.type === 'depth_raw') {
            processedFrame.depthData = result.data;
        }

        processedFrame.frameTimings.depth_processing = processingTime;
        processedFrames.set(frameId, processedFrame);

        // Color와 Depth가 모두 처리되었는지 확인
        checkFrameCompletion(frameId);
    }
}

// 프레임 완성 확인 및 메인 스레드로 전송
function checkFrameCompletion(frameId: number) {
    const processedFrame = processedFrames.get(frameId);
    if (!processedFrame) return;

    // Color와 Depth가 모두 처리되었는지 확인 (간단한 체크)
    const hasColor = processedFrame.colorImageBitmap || processedFrame.colorRawRGB;
    const hasDepth = processedFrame.depthNDC || processedFrame.depthData.byteLength > 0;

    if (hasColor && hasDepth) {
        // 메인 스레드로 전송
        const transferList: Transferable[] = [];
        if (processedFrame.depthNDC) {
            transferList.push(processedFrame.depthNDC);
        }
        if (processedFrame.colorImageBitmap) {
            transferList.push(processedFrame.colorImageBitmap);
        }
        if (processedFrame.colorRawRGB) {
            transferList.push(processedFrame.colorRawRGB);
        }

        (self as any).postMessage({
            type: 'processed_frame',
            processedFrame
        }, transferList);

        // 처리된 프레임 제거
        processedFrames.delete(frameId);
    }
}

// 메인 메시지 처리 핸들러
(self as any).onmessage = async (event: MessageEvent) => {
    const { type, data, config } = event.data;

    // 설정 업데이트
    if (type === 'config') {
        USE_H264_DECODER = config.USE_H264_DECODER;
        USE_RAW_RGB = config.USE_RAW_RGB;
        CLIENT_ASSUMED_SCALE_Y = config.CLIENT_ASSUMED_SCALE_Y;
        rtWidth = config.rtWidth;
        rtHeight = config.rtHeight;
        return;
    }

    if (type === 'ws_open') {
        console.log(`[DEBUG] WebSocket opened, initializing separated workers...`);

        // 분리된 Worker 초기화
        initializeSeparatedWorkers();

        return;
    }

    // 분리된 프레임 처리
    if (type === 'color_frame') {
        const colorMessage: ColorFrameMessage = event.data;

        if (colorWorker) {
            colorWorker.postMessage(colorMessage, [colorMessage.data]);
        }
        return;
    }

    if (type === 'depth_frame') {
        const depthMessage: DepthFrameMessage = event.data;

        if (depthWorker) {
            depthWorker.postMessage(depthMessage, [depthMessage.data]);
        }
        return;
    }
};

console.log('WebSocket Processor Worker initialized with separated pipeline support.');