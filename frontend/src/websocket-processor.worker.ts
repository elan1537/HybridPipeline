interface ParsedData {
    videoData: ArrayBuffer;
}

interface ProcessedFrame {
    colorImageBitmap?: ImageBitmap;
    depthNDC?: ArrayBuffer; // Float32Array 버퍼
    frameTimings: { [k: string]: number };
}

const perfMonitor = {
    timings: {
        total_onmessage: 0.0,
        header_parse: 0.0,
        h264_decode_call: 0.0,
        frame_split_and_process: 0.0,
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

    logStatsIfReady(periodInMs = 10000) {
        const now = performance.now();
        if (now - this.lastPrintTime > periodInMs) {
            if (this.frameCount > 0) {
                console.log(`--- Worker Performance Stats (avg over ${this.frameCount} frames) ---`);
                for (const key in this.timings) {
                    const avgTime = this.timings[key] / this.frameCount;
                    console.log(`  ${key}: ${avgTime.toFixed(3)} ms`);
                }
                console.log('--------------------------------------------------');
            }
            this.reset();
        }
    }
};

perfMonitor.init();

let rtWidth = 1920;
let rtHeight = 1080;
let near = 0.1;
let far = 30.0;

let videoDecoder: VideoDecoder | null = null;
let nextCombinedFrame: VideoFrame | null = null;
let isDecoderConfigured = false;


function parseMessage(data: ArrayBuffer): ParsedData | null {
    const dv = new DataView(data);
    const HEADER_SIZE = 4 + 8 + 8 + 8;
    if (data.byteLength < HEADER_SIZE) return null;

    const videoLen = dv.getUint32(0, true);
    const videoStart = HEADER_SIZE;
    if (data.byteLength < videoStart + videoLen) return null;

    return {
        videoData: data.slice(videoStart, videoStart + videoLen),
    };
}

async function timeExecution(fn: () => Promise<any>, timings: { [key: string]: number }, key: string) {
    const start = performance.now();
    const result = await fn();
    timings[key] = performance.now() - start;
    return result;
}

let depthCanvas: OffscreenCanvas | null = null;
let depthCtx: OffscreenCanvasRenderingContext2D | null = null;

function convertDepthBitmapToFloat(depthBitmap: ImageBitmap): ArrayBuffer {
    if (!depthCanvas || depthCanvas.width !== depthBitmap.width || depthCanvas.height !== depthBitmap.height) {
        depthCanvas = new OffscreenCanvas(depthBitmap.width, depthBitmap.height);
        depthCtx = depthCanvas.getContext('2d', { willReadFrequently: true }) as OffscreenCanvasRenderingContext2D;
    }
    if (!depthCtx) throw new Error("Could not create OffscreenCanvas context");

    depthCtx.drawImage(depthBitmap, 0, 0);
    const imageData = depthCtx.getImageData(0, 0, depthBitmap.width, depthBitmap.height);
    const pixels = imageData.data;
    const depthValues = new Uint8Array(depthBitmap.width * depthBitmap.height);

    for (let i = 0; i < depthValues.length; i++) {
        depthValues[i] = pixels[i * 4] % 256;
    }
    return depthValues.buffer;
}

async function decodeCombinedFrame(videoData: ArrayBuffer): Promise<{ color: ImageBitmap, depth: ArrayBuffer } | null> {
    if (!isDecoderConfigured) {
        try {
            const config: VideoDecoderConfig = {
                codec: 'avc1.42001E',
                // codec: 'hvc1.01.1.L93.B0', // HEVC 코덱 예시
                description: videoData,
            };
            if (!(await VideoDecoder.isConfigSupported(config)).supported) {
                console.error('H.264 decoder config not supported');
                return null;
            }
            videoDecoder = new VideoDecoder({
                output: (frame: VideoFrame) => {
                    if (nextCombinedFrame) nextCombinedFrame.close();
                    nextCombinedFrame = frame;
                },
                error: (e) => {
                    console.error('VideoDecoder error:', e);
                    isDecoderConfigured = false;
                    if (videoDecoder && videoDecoder.state !== 'closed') videoDecoder.close();
                }
            });
            await videoDecoder.configure(config);
            console.log('✅ Combined H.264 VideoDecoder configured.');
            isDecoderConfigured = true;
            // ✨ 첫 프레임은 설정에만 사용하고, 디코딩 결과는 버립니다.
            return null;
        } catch (e) {
            console.error('Failed to configure decoder:', e);
            isDecoderConfigured = false;
            return null;
        }
    }

    // 2. 두 번째 프레임부터 디코딩
    if (!videoDecoder) return null;

    try {
        videoDecoder.decode(new EncodedVideoChunk({
            type: 'key', timestamp: performance.now(), data: videoData,
        }));

        return new Promise((resolve) => {
            let attempts = 0;
            const waitFordecodedFrame = async () => {
                if (nextCombinedFrame) {
                    const frame = nextCombinedFrame;
                    nextCombinedFrame = null;

                    const w = frame.codedWidth;
                    const h = frame.codedHeight / 2;

                    const colorPromise = createImageBitmap(frame, 0, 0, w, h);
                    const depthBitmapPromise = createImageBitmap(frame, 0, h, w, h);

                    const [color, depthBitmap] = await Promise.all([colorPromise, depthBitmapPromise]);
                    frame.close();

                    const depth = convertDepthBitmapToFloat(depthBitmap);
                    depthBitmap.close();

                    resolve({ color, depth });
                }
            };
            waitFordecodedFrame();
        });
    } catch (e) {
        console.error('Combined H.264 decoding error:', e);
        isDecoderConfigured = false;
        if (videoDecoder && videoDecoder.state !== 'closed') videoDecoder.close();
        return null;
    }
}

// 메인 메시지 핸들러
(self as any).onmessage = async (event: MessageEvent) => {
    const { type, data, config } = event.data;
    if (type === 'config') {
        rtWidth = config.rtWidth;
        rtHeight = config.rtHeight;
        near = config.near; // ✨ near, far 값 수신
        far = config.far;
        return;
    }
    if (type === 'ws_open') { return; }

    if (type === 'frame') {
        const t_onmessage_start = performance.now();
        const frameTimings: { [key: string]: number } = {};

        const parsedData = await timeExecution(() => Promise.resolve(parseMessage(data)), frameTimings, 'header_parse');
        if (!parsedData) return;

        const processedFrame: ProcessedFrame = { frameTimings };

        const decodedData = await timeExecution(
            () => decodeCombinedFrame(parsedData.videoData),
            frameTimings, 'h264_decode_call'
        );

        if (decodedData) {
            processedFrame.colorImageBitmap = decodedData.color;
            processedFrame.depthNDC = decodedData.depth;
        }

        frameTimings.total_onmessage = performance.now() - t_onmessage_start;
        perfMonitor.record(frameTimings);

        if (processedFrame.colorImageBitmap && processedFrame.depthNDC) {
            const transferList: Transferable[] = [processedFrame.colorImageBitmap, processedFrame.depthNDC];
            (self as any).postMessage({ type: 'processed_frame', processedFrame }, transferList);
        }
    }
};

console.log('WebSocket Processor Worker initialized.');
console.log('wsDepthTexture size:', rtWidth, rtHeight, 'depthValues.length:', rtWidth * rtHeight);