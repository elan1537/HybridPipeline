// /**
//  * WebSocket 메시지 처리를 위한 Worker
//  * 메인 스레드의 블로킹을 방지하기 위해 메시지 파싱과 디코딩을 담당합니다.
//  */

// // Worker 내부에서 사용할 타입 정의
// interface ParsedData {
//     colorData: ArrayBuffer;
//     depthData: ArrayBuffer;
// }

// interface ProcessedFrame {
//     colorImageBitmap?: ImageBitmap;
//     depthNDC?: ArrayBuffer;          // 디코딩 및 변환된 최종 깊이 (Float32Array)
//     frameTimings: { [k: string]: number };
// }

// // 성능 모니터링을 위한 유틸리티
// const perfMonitor = {
//     timings: {
//         total_onmessage: 0.0,
//         header_parse: 0.0,
//         h264_decode_call: 0.0,
//         depth_processing: 0.0,
//     },
//     frameCount: 0,
//     lastPrintTime: 0,

//     init() {
//         this.lastPrintTime = performance.now();
//     },

//     reset() {
//         this.frameCount = 0;
//         for (const key in this.timings) {
//             this.timings[key] = 0.0;
//         }
//         this.lastPrintTime = performance.now();
//     },

//     record(stats: { [key: string]: number }) {
//         for (const key in stats) {
//             if (key in this.timings) {
//                 this.timings[key] += stats[key];
//             }
//         }
//         this.frameCount++;
//     },

//     logStatsIfReady(periodInMs = 10000) {
//         const now = performance.now();
//         if (now - this.lastPrintTime > periodInMs) {
//             if (this.frameCount > 0) {
//                 console.log(`--- Worker Performance Stats (avg over ${this.frameCount} frames) ---`);
//                 for (const key in this.timings) {
//                     const avgTime = this.timings[key] / this.frameCount;
//                     console.log(`  ${key}: ${avgTime.toFixed(3)} ms`);
//                 }
//                 console.log('--------------------------------------------------');
//             }
//             this.reset();
//         }
//     }
// };

// perfMonitor.init();


// // 설정 플래그들
// let USE_H264_DECODER = true;
// let CLIENT_ASSUMED_SCALE_Y = 1.0;
// let rtWidth = 1920;
// let rtHeight = 1080;

// // H.264 디코더 인스턴스 및 상태
// let videoDecoder: VideoDecoder | null = null;
// let depthDecoder: VideoDecoder | null = null;
// let nextImageBitmap: ImageBitmap | null = null;
// let nextDepthFrame: VideoFrame | null = null;
// let isColorDecoderConfigured = false;
// let isDepthDecoderConfigured = false;


// // 데이터 파싱 함수
// function parseMessage(data: ArrayBuffer): ParsedData | null {
//     if (!(data instanceof ArrayBuffer)) {
//         console.error("Invalid data type received.");
//         return null;
//     }

//     const dv = new DataView(data);
//     const HEADER_SIZE = 4 + 4 + 8 + 8 + 8;

//     if (data.byteLength < HEADER_SIZE) {
//         console.error("Received data is too short for headers.");
//         return null;
//     }

//     const rgbLen = dv.getUint32(0, true);
//     const depthLen = dv.getUint32(4, true);

//     const rgbStart = HEADER_SIZE;
//     const depthStart = HEADER_SIZE + rgbLen;
//     const totalExpectedSize = depthStart + depthLen;

//     if (data.byteLength < totalExpectedSize) {
//         console.error(`Received data is shorter (${data.byteLength}) than specified by headers (${totalExpectedSize}).`);
//         return null;
//     }

//     return {
//         colorData: data.slice(rgbStart, depthStart),
//         depthData: data.slice(depthStart, totalExpectedSize),
//     };
// }

// // 함수 실행 시간 측정 헬퍼
// async function timeExecution(
//     fn: () => Promise<any>,
//     timings: { [key: string]: number },
//     key: string
// ) {
//     const start = performance.now();
//     const result = await fn();
//     timings[key] = performance.now() - start;
//     return result;
// }

// // 컬러 디코딩 함수
// async function decodeH264(videoData: ArrayBuffer): Promise<ImageBitmap | null> {
//     // 1. 첫 프레임인 경우, 디코더를 설정하고 프레임은 버림
//     if (!isColorDecoderConfigured) {
//         try {
//             const config: VideoDecoderConfig = {
//                 codec: 'avc1.42001E',
//                 description: videoData, // 첫 프레임 전체를 description으로 사용
//             };
//             if (!(await VideoDecoder.isConfigSupported(config)).supported) {
//                 console.error('H.264 color decoder config not supported');
//                 return null;
//             }
//             videoDecoder = new VideoDecoder({
//                 output: (frame: VideoFrame) => {
//                     createImageBitmap(frame).then(bmp => {
//                         frame.close();
//                         if (nextImageBitmap) nextImageBitmap.close();
//                         nextImageBitmap = bmp;
//                     });
//                 },
//                 error: (e) => {
//                     console.error('Color VideoDecoder error:', e);
//                     isColorDecoderConfigured = false; // 에러 시 재설정
//                     if (videoDecoder && videoDecoder.state !== 'closed') videoDecoder.close();
//                 }
//             });
//             await videoDecoder.configure(config);
//             await videoDecoder.flush(); // ✨ 상태 전환을 위해 flush() 호출
//             console.log('✅ Color H.264 VideoDecoder configured and flushed.');
//             isColorDecoderConfigured = true;
//             // 첫 프레임은 설정에만 사용하고 디코딩 결과는 버림
//             return null;
//         } catch (e) {
//             console.error('Failed to configure color decoder with first frame:', e);
//             isColorDecoderConfigured = false;
//             return null;
//         }
//     }

//     // 2. 두 번째 프레임부터 디코딩
//     if (!videoDecoder) return null;

//     try {
//         const chunk = new EncodedVideoChunk({
//             type: 'key',
//             timestamp: performance.now(),
//             data: videoData,
//         });
//         videoDecoder.decode(chunk);

//         return new Promise((resolve) => {
//             let attempts = 0;
//             const maxAttempts = 100;
//             const checkForFrame = () => {
//                 if (nextImageBitmap) {
//                     const bmp = nextImageBitmap;
//                     nextImageBitmap = null;
//                     resolve(bmp);
//                 } else if (++attempts < maxAttempts) {
//                     setTimeout(checkForFrame, 1);
//                 } else {
//                     console.warn('H.264 color frame timeout');
//                     resolve(null);
//                 }
//             };
//             checkForFrame();
//         });
//     } catch (e) {
//         console.error('H.264 color decoding error:', e);
//         isColorDecoderConfigured = false;
//         if (videoDecoder && videoDecoder.state !== 'closed') videoDecoder.close();
//         return null;
//     }
// }

// // 뎁스 디코딩 함수
// async function decodeDepthH264(depthData: ArrayBuffer): Promise<ArrayBuffer | null> {
//     // 1. 첫 프레임인 경우, 디코더를 설정하고 프레임은 버림
//     if (!isDepthDecoderConfigured) {
//         try {
//             const config: VideoDecoderConfig = {
//                 codec: 'avc1.42001E',
//                 description: depthData, // 첫 프레임 전체를 description으로 사용
//             };
//             if (!(await VideoDecoder.isConfigSupported(config)).supported) {
//                 console.error('H.264 depth decoder config not supported');
//                 return null;
//             }
//             depthDecoder = new VideoDecoder({
//                 output: (frame: VideoFrame) => {
//                     if (nextDepthFrame) nextDepthFrame.close();
//                     nextDepthFrame = frame;
//                 },
//                 error: (e) => {
//                     console.error('Depth VideoDecoder error:', e);
//                     isDepthDecoderConfigured = false; // 에러 시 재설정
//                     if (depthDecoder && depthDecoder.state !== 'closed') depthDecoder.close();
//                 },
//             });
//             await depthDecoder.configure(config);
//             await depthDecoder.flush(); // ✨ 상태 전환을 위해 flush() 호출
//             console.log('✅ Depth H.264 VideoDecoder configured and flushed.');
//             isDepthDecoderConfigured = true;
//             return null;
//         } catch (e) {
//             console.error('Failed to configure depth decoder with first frame:', e);
//             isDepthDecoderConfigured = false;
//             return null;
//         }
//     }

//     // 2. 두 번째 프레임부터 디코딩
//     if (!depthDecoder) return null;

//     try {
//         const chunk = new EncodedVideoChunk({
//             type: 'key',
//             timestamp: performance.now(),
//             data: depthData,
//         });
//         depthDecoder.decode(chunk);

//         return new Promise((resolve) => {
//             let attempts = 0;
//             const maxAttempts = 100;
//             const waitFordecodedFrame = async () => {
//                 if (nextDepthFrame) {
//                     const frame = nextDepthFrame;
//                     nextDepthFrame = null;
//                     const w = frame.codedWidth, h = frame.codedHeight;
//                     const yPlane = new Uint8Array(w * h);
//                     await frame.copyTo(yPlane, {
//                         rect: { x: 0, y: 0, width: w, height: h },
//                         planeIndex: 0
//                     });
//                     frame.close();
//                     const depthValues = new Float32Array(w * h);
//                     for (let i = 0; i < yPlane.length; i++) {
//                         depthValues[i] = yPlane[i] / 255.0;
//                     }
//                     resolve(depthValues.buffer);
//                 } else if (++attempts < maxAttempts) {
//                     setTimeout(waitFordecodedFrame, 1);
//                 } else {
//                     console.warn('Depth H.264 frame timeout');
//                     resolve(null);
//                 }
//             };
//             waitFordecodedFrame();
//         });
//     } catch (e) {
//         console.error('Depth H.264 decoding error:', e);
//         isDepthDecoderConfigured = false;
//         if (depthDecoder && depthDecoder.state !== 'closed') depthDecoder.close();
//         return null;
//     }
// }

// // 메인 메시지 핸들러
// (self as any).onmessage = async (event: MessageEvent) => {
//     const { type, data, config } = event.data;

//     if (type === 'config') {
//         USE_H264_DECODER = config.USE_H264_DECODER;
//         CLIENT_ASSUMED_SCALE_Y = config.CLIENT_ASSUMED_SCALE_Y;
//         rtWidth = config.rtWidth;
//         rtHeight = config.rtHeight;
//         return;
//     }

//     if (type === 'ws_open') {
//         // 첫 프레임에서 디코더를 생성하므로 여기서 할 일 없음
//         return;
//     }

//     if (type === 'frame') {
//         const t_onmessage_start = performance.now();
//         const frameTimings: { [key: string]: number } = {};

//         const t_parse_start = performance.now();
//         const parsedData = parseMessage(data);
//         frameTimings.header_parse = performance.now() - t_parse_start;

//         if (!parsedData) {
//             self.postMessage({ type: 'error', error: 'Failed to parse message' });
//             return;
//         }

//         const processedFrame: ProcessedFrame = { frameTimings };

//         // H.264 경로
//         processedFrame.colorImageBitmap = await timeExecution(
//             () => decodeH264(parsedData.colorData),
//             frameTimings,
//             'h264_decode_call'
//         );

//         processedFrame.depthNDC = await timeExecution(
//             () => decodeDepthH264(parsedData.depthData),
//             frameTimings,
//             'depth_processing'
//         );

//         frameTimings.total_onmessage = performance.now() - t_onmessage_start;
//         perfMonitor.record(frameTimings);

//         // 첫 프레임은 디코딩 결과가 null일 수 있으므로, 결과가 있을 때만 전송
//         if (processedFrame.colorImageBitmap || processedFrame.depthNDC) {
//             const transferList: Transferable[] = [];
//             if (processedFrame.depthNDC) {
//                 transferList.push(processedFrame.depthNDC);
//             }
//             if (processedFrame.colorImageBitmap) {
//                 transferList.push(processedFrame.colorImageBitmap);
//             }

//             (self as any).postMessage({
//                 type: 'processed_frame',
//                 processedFrame
//             }, transferList);
//         }

//         perfMonitor.logStatsIfReady();
//     }
// };

// console.log('WebSocket Processor Worker initialized.');

/**
 * WebSocket 메시지 처리를 위한 Worker
 * 하나의 합쳐진 비디오 스트림을 받아 컬러와 뎁스로 분리합니다.
 */

// Worker 내부에서 사용할 타입 정의
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

// 설정 플래그들
let rtWidth = 1920;
let rtHeight = 1080;

// ✨ 디코더는 하나만 필요
let videoDecoder: VideoDecoder | null = null;
let nextCombinedFrame: VideoFrame | null = null;
let isDecoderConfigured = false;

// ✨ 파싱 함수 단순화
function parseMessage(data: ArrayBuffer): ParsedData | null {
    const dv = new DataView(data);
    // 서버 struct.pack("<Iddd",...) -> 길이(I, 4), 타임스탬프(d, 8)*3
    const HEADER_SIZE = 4 + 8 + 8 + 8;
    if (data.byteLength < HEADER_SIZE) return null;

    const videoLen = dv.getUint32(0, true);
    const videoStart = HEADER_SIZE;
    if (data.byteLength < videoStart + videoLen) return null;

    return {
        videoData: data.slice(videoStart, videoStart + videoLen),
    };
}

// 함수 실행 시간 측정 헬퍼
async function timeExecution(fn: () => Promise<any>, timings: { [key: string]: number }, key: string) {
    const start = performance.now();
    const result = await fn();
    timings[key] = performance.now() - start;
    return result;
}

// ✨ OffscreenCanvas를 이용한 뎁스 변환
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
    const depthValues = new Float32Array(depthBitmap.width * depthBitmap.height);

    for (let i = 0; i < depthValues.length; i++) {
        // 서버에서 grayscale로 보냈으므로 R 채널 값만 사용
        depthValues[i] = pixels[i * 4] / 255.0;
    }
    return depthValues.buffer;
}

// ✨ 디코딩 함수 하나로 통합
async function decodeCombinedFrame(videoData: ArrayBuffer): Promise<{ color: ImageBitmap, depth: ArrayBuffer } | null> {
    // 1. 첫 프레임으로 디코더 설정
    if (!isDecoderConfigured) {
        try {
            const config: VideoDecoderConfig = {
                codec: 'avc1.42001E',
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
            await videoDecoder.flush();
            console.log('✅ Combined H.264 VideoDecoder configured and flushed.');
            isDecoderConfigured = true;
            return null; // 첫 프레임은 설정에만 사용
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

                    // 프레임을 컬러와 뎁스 비트맵으로 분리
                    const colorPromise = createImageBitmap(frame, 0, 0, w, h);
                    const depthBitmapPromise = createImageBitmap(frame, 0, h, w, h);

                    const [color, depthBitmap] = await Promise.all([colorPromise, depthBitmapPromise]);
                    frame.close();

                    // 뎁스 비트맵을 Float32Array로 변환
                    const depth = convertDepthBitmapToFloat(depthBitmap);
                    depthBitmap.close();

                    resolve({ color, depth });
                } else if (++attempts < 100) {
                    setTimeout(waitFordecodedFrame, 1);
                } else {
                    console.warn('Combined H.264 frame timeout');
                    resolve(null);
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

        // 첫 프레임은 결과가 null이므로, 결과가 있을 때만 전송
        if (processedFrame.colorImageBitmap && processedFrame.depthNDC) {
            const transferList: Transferable[] = [processedFrame.colorImageBitmap, processedFrame.depthNDC];
            (self as any).postMessage({ type: 'processed_frame', processedFrame }, transferList);
        }
    }
};

console.log('WebSocket Processor Worker initialized.');
