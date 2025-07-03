/**
 * WebSocket 메시지 처리를 위한 Worker
 * 메인 스레드의 블로킹을 방지하기 위해 메시지 파싱과 디코딩을 담당합니다.
 */

// Worker 내부에서 사용할 타입 정의
interface ParsedData {
    colorData: ArrayBuffer;
    depthData: ArrayBuffer;
}

interface ProcessedFrame {
    colorImageBitmap?: ImageBitmap;
    colorRawRGB?: ArrayBuffer;
    depthData: ArrayBuffer;          // 원본 비트스트림
    depthNDC?: ArrayBuffer;          // 디코딩된 깊이 (float32, -1~1)
    frameTimings: { [k: string]: number };
}

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

// 설정 플래그들 (메인 스레드에서 전달받음)
let USE_H264_DECODER = false;
let USE_RAW_RGB = false;
let CLIENT_ASSUMED_SCALE_Y = 1.0;
let rtWidth = 1920;
let rtHeight = 1080;

// H.264 디코더 관련 변수
let videoDecoder: VideoDecoder | null = null;
let isFirstFrame = true;
let h264ErrorCount = 0;
const MAX_H264_ERRORS = 5;
let nextImageBitmap: ImageBitmap | null = null;

// 깊이 HEVC(P010) 디코더
let depthDecoder: VideoDecoder | null = null;
let depthErrorCount = 0;
const MAX_DEPTH_ERRORS = 5;
let nextDepthFrame: VideoFrame | null = null;
let depthDecoderInitialized = false; // 디코더 초기화 상태 추적

// 데이터 파싱 함수
function parseMessage(data: ArrayBuffer): ParsedData | null {
    if (!(data instanceof ArrayBuffer)) {
        console.error("Invalid data type received.");
        return null;
    }

    const dv = new DataView(data);
    const HEADER_SIZE = 4 + 4 + 8 + 8 + 8;

    if (data.byteLength < HEADER_SIZE) {
        console.error("Received data is too short for headers.");
        return null;
    }

    const rgbLen = dv.getUint32(0, true);
    const depthLen = dv.getUint32(4, true);

    const client_send_timestamp_ms = dv.getFloat64(8, true);
    const server_recv_timestamp_ms = dv.getFloat64(16, true);
    const server_process_end_timestamp_ms = dv.getFloat64(24, true);

    const rgbStart = HEADER_SIZE;
    const depthStart = HEADER_SIZE + rgbLen;
    const totalExpectedSize = depthStart + depthLen;

    if (data.byteLength < totalExpectedSize) {
        console.error(`Received data is shorter (${data.byteLength}) than specified by headers (${totalExpectedSize}).`);
        return null;
    }

    return {
        colorData: data.slice(rgbStart, depthStart),
        depthData: data.slice(depthStart, totalExpectedSize),
    };
}

// 함수 실행 시간 측정 헬퍼
async function timeExecution(
    fn: () => Promise<void> | void,
    timings: { [key: string]: number },
    key: string
) {
    const start = performance.now();
    await fn();
    timings[key] = performance.now() - start;
}

// H.264 디코더 초기화
function initializeVideoDecoder() {
    if (!('VideoDecoder' in self)) {
        console.error('WebCodecs VideoDecoder is not supported.');
        return;
    }

    if (videoDecoder && videoDecoder.state === 'configured') {
        return;
    }

    if (videoDecoder && videoDecoder.state !== 'closed') {
        videoDecoder.close();
    }

    videoDecoder = new VideoDecoder({
        output: async (frame: VideoFrame) => {
            const bmp = await createImageBitmap(frame);
            frame.close();

            if (nextImageBitmap) {
                nextImageBitmap.close();
            }
            nextImageBitmap = bmp;
        },
        error: (e) => {
            console.error('VideoDecoder error:', e);
            h264ErrorCount++;
            if (h264ErrorCount >= 3) {
                (self as any).USE_H264_FALLBACK_DISABLED = true;
            }
        },
    });

    videoDecoder.configure({
        codec: 'avc1.42E01E',
        hardwareAcceleration: 'prefer-hardware',
        optimizeForLatency: true,
        colorSpace: {
            primaries: 'bt709',
            transfer: 'bt709',
            matrix: 'bt709',
            fullRange: true
        }
    });

    console.log('VideoDecoder ready for H.264 streaming.');
}

// ──────────────────────────────────────────────────────────────
// HEVC Main 10 (P010) Depth Decoder 초기화
async function initializeDepthDecoder(hevcData: ArrayBuffer) {
    console.log('[DEBUG] Initializing depth decoder...');

    // 이미 초기화되었거나 비활성화된 경우 건너뛰기
    if (depthDecoderInitialized) {
        console.log('[DEBUG] Decoder already initialized, skipping');
        return;
    }

    if ((self as any).USE_DEPTH_DECODER_DISABLED) {
        console.log('[DEBUG] Decoder is disabled, skipping');
        return;
    }

    // 2. 서버에서 받은 필수 정보가 있는지 확인
    if (rtWidth === 0 || rtHeight === 0) {
        console.error('Decoder cannot be initialized: dimensions are missing.');
        return;
    }

    try {
        // HEVC 데이터에서 VPS, SPS, PPS 추출
        const hevcDescription = extractHEVCHeaders(hevcData);
        if (!hevcDescription) {
            console.error('Failed to extract HEVC headers from data');
            return;
        }

        console.log(`[DEBUG] Extracted HEVC headers: ${hevcDescription.byteLength} bytes`);

        // 4. HEVC Main 10 프로파일에 맞는 설정으로 시도
        const config: VideoDecoderConfig = {
            codec: 'hvc1.1.6.L120.00',  // HEVC Main 10 프로파일
            description: hevcDescription,
            codedWidth: rtWidth,
            codedHeight: rtHeight,
        };


        const { supported } = await VideoDecoder.isConfigSupported(config);
        console.log(`[DEBUG] HEVC Main 10 decoder support check result: ${supported}`);

        if (!supported) {
            console.error('HEVC Main 10 decoder is not supported, trying alternative configurations...');

            console.error('No HEVC decoder configuration is supported, disabling depth decoder');
            (self as any).USE_DEPTH_DECODER_DISABLED = true;
            depthDecoderInitialized = false; // 명시적으로 false로 설정
            return;

        }
        console.log('Depth decoder configuration is supported.');

        // 5. 디코더 인스턴스 생성 및 설정
        if (depthDecoder && depthDecoder.state !== 'closed') {
            depthDecoder.close();
        }

        depthDecoder = new VideoDecoder({
            output: (frame) => {
                if (nextDepthFrame) {
                    nextDepthFrame.close();
                }
                nextDepthFrame = frame;
            },
            error: (e) => {
                console.error('Depth VideoDecoder runtime error:', e);
                depthErrorCount++;
                if (depthErrorCount >= MAX_DEPTH_ERRORS) {
                    (self as any).USE_DEPTH_DECODER_DISABLED = true;
                }
            },
        });

        console.log('[DEBUG] Configuring decoder with:', config);
        depthDecoder.configure(config);
        console.log('✅ Depth decoder successfully configured and ready.');
        (self as any).USE_DEPTH_DECODER_DISABLED = false;
        depthDecoderInitialized = true;
        console.log('[DEBUG] depthDecoderInitialized set to true');

    } catch (e) {
        console.error('A critical error occurred during depth decoder initialization:', e);
        (self as any).USE_DEPTH_DECODER_DISABLED = true;
        depthDecoderInitialized = false; // 명시적으로 false로 설정
    }
}

// HEVC 헤더 추출 함수
function extractHEVCHeaders(hevcData: ArrayBuffer): ArrayBuffer | null {
    const data = new Uint8Array(hevcData);
    const startCode = [0x00, 0x00, 0x00, 0x01];

    console.log(`[DEBUG] Analyzing HEVC data: ${data.length} bytes`);
    console.log(`[DEBUG] First 16 bytes:`, Array.from(data.slice(0, 16)).map(b => b.toString(16).padStart(2, '0')).join(' '));

    let vpsData: Uint8Array | null = null;
    let spsData: Uint8Array | null = null;
    let ppsData: Uint8Array | null = null;
    let nalCount = 0;

    let i = 0;
    while (i < data.length - 4) {
        // 시작 코드 찾기
        if (data[i] === startCode[0] &&
            data[i + 1] === startCode[1] &&
            data[i + 2] === startCode[2] &&
            data[i + 3] === startCode[3]) {

            if (i + 4 >= data.length) break;

            // NAL 유닛 타입 확인 (첫 바이트의 상위 1비트 제거 후 하위 6비트)
            const nalType = (data[i + 4] >> 1) & 0x3F;
            nalCount++;

            console.log(`[DEBUG] Found NAL unit ${nalCount}: type=${nalType} at position ${i}`);

            // 다음 시작 코드 찾기
            let nextStart = i + 4;
            while (nextStart < data.length - 4) {
                if (data[nextStart] === startCode[0] &&
                    data[nextStart + 1] === startCode[1] &&
                    data[nextStart + 2] === startCode[2] &&
                    data[nextStart + 3] === startCode[3]) {
                    break;
                }
                nextStart++;
            }

            // NAL 유닛 데이터 추출
            const nalData = data.slice(i, nextStart);

            // NAL 타입에 따라 분류
            switch (nalType) {
                case 32: // VPS
                    vpsData = nalData;
                    console.log('[DEBUG] Found VPS:', nalData.length, 'bytes');
                    break;
                case 33: // SPS
                    spsData = nalData;
                    console.log('[DEBUG] Found SPS:', nalData.length, 'bytes');
                    break;
                case 34: // PPS
                    ppsData = nalData;
                    console.log('[DEBUG] Found PPS:', nalData.length, 'bytes');
                    break;
                default:
                    console.log(`[DEBUG] Found other NAL type ${nalType}:`, nalData.length, 'bytes');
                    break;
            }

            i = nextStart;
        } else {
            i++;
        }
    }

    console.log(`[DEBUG] Total NAL units found: ${nalCount}`);
    console.log(`[DEBUG] VPS: ${vpsData ? 'found' : 'not found'}, SPS: ${spsData ? 'found' : 'not found'}, PPS: ${ppsData ? 'found' : 'not found'}`);

    // VPS, SPS, PPS가 모두 있으면 조합
    if (vpsData && spsData && ppsData) {
        const totalLength = vpsData.length + spsData.length + ppsData.length;
        const combined = new Uint8Array(totalLength);
        let offset = 0;

        combined.set(vpsData, offset);
        offset += vpsData.length;
        combined.set(spsData, offset);
        offset += spsData.length;
        combined.set(ppsData, offset);

        console.log('[DEBUG] Combined HEVC headers:', totalLength, 'bytes');
        return combined.buffer;
    }

    // VPS가 없으면 SPS + PPS만 사용
    if (spsData && ppsData) {
        const totalLength = spsData.length + ppsData.length;
        const combined = new Uint8Array(totalLength);
        combined.set(spsData, 0);
        combined.set(ppsData, spsData.length);

        console.log('[DEBUG] Combined SPS+PPS headers:', totalLength, 'bytes');
        return combined.buffer;
    }

    // SPS만 있어도 시도
    if (spsData) {
        console.log('[DEBUG] Using SPS only:', spsData.length, 'bytes');
        return spsData.buffer;
    }

    console.error('[DEBUG] No valid HEVC headers found');
    return null;
}

async function decodeH264(videoData: ArrayBuffer): Promise<ImageBitmap | null> {
    if (!videoData || videoData.byteLength === 0) {
        return null;
    }

    try {
        if (!videoDecoder || videoDecoder.state === 'closed') {
            initializeVideoDecoder();
        }

        const chunk = new EncodedVideoChunk({
            type: 'key',
            timestamp: performance.now(),
            data: videoData,
            duration: 0,
        });
        videoDecoder!.decode(chunk);

        h264ErrorCount = 0;

        // 다음 프레임이 준비될 때까지 대기 (최대 100ms)
        return new Promise((resolve) => {
            let attempts = 0;
            const maxAttempts = 100; // 100ms 타임아웃

            const checkForFrame = () => {
                attempts++;
                if (nextImageBitmap) {
                    const bmp = nextImageBitmap;
                    nextImageBitmap = null;
                    resolve(bmp);
                } else if (attempts < maxAttempts) {
                    setTimeout(checkForFrame, 1);
                } else {
                    console.warn('H.264 frame timeout');
                    resolve(null);
                }
            };
            checkForFrame();
        });

    } catch (e) {
        console.error('H.264 decoding error:', e);
        h264ErrorCount++;
        if (h264ErrorCount >= 3) {
            (self as any).USE_H264_FALLBACK_DISABLED = true;
        }
        return null;
    }
}

async function decodeDepthHEVC(depthData: ArrayBuffer): Promise<ArrayBuffer | null> {
    try {
        // 디코더 상태 확인
        if (!depthDecoder || depthDecoder.state !== 'configured') {
            console.warn('Depth decoder not ready, skipping decode');
            return null;
        }

        const chunk = new EncodedVideoChunk({
            type: 'key',
            timestamp: performance.now(),
            data: depthData,
            duration: 0,
        });
        depthDecoder.decode(chunk);

        return new Promise((res) => {
            let tries = 0;
            (async function wait() {
                if (nextDepthFrame) {
                    const f = nextDepthFrame; nextDepthFrame = null;
                    const w = f.displayWidth, h = f.displayHeight;

                    // 1) Copy P010 Y‑plane (16‑bit words, top‑10 bits significant)
                    const yPlane16 = new Uint16Array(w * h);
                    await f.copyTo(yPlane16);
                    f.close();

                    // 2) Convert to float32 in WebGL NDC: 0‑1023 → 0‑1 → -1‑1
                    const depthNDC = new Float32Array(w * h);
                    for (let i = 0; i < yPlane16.length; i++) {
                        const v10 = yPlane16[i] >> 6;
                        depthNDC[i] = (v10 / 1023.0) * 2.0 - 1.0; // -1 ~ 1
                    }

                    res(depthNDC.buffer);
                } else if (++tries < 100) {
                    setTimeout(wait, 1);
                } else {
                    console.warn('Depth frame timeout');
                    res(null);
                }
            })();
        });

    } catch (e) {
        console.error('Depth HEVC decoding error:', e);
        depthErrorCount++;
        if (depthErrorCount >= MAX_DEPTH_ERRORS) {
            (self as any).USE_DEPTH_DECODER_DISABLED = true;
        }
        return null;
    }
}

// JPEG 디코딩
async function decodeJPEG(jpegBlob: Blob): Promise<ImageBitmap> {
    return await createImageBitmap(jpegBlob);
}

// 메인 메시지 처리 핸들러
// Main message handler (async!)
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
        console.log(`[DEBUG] WebSocket opened, initializing decoders...`);
        console.log(`[DEBUG] VideoDecoder support:`, 'VideoDecoder' in self);
        console.log(`[DEBUG] USE_H264_DECODER:`, USE_H264_DECODER);
        console.log(`[DEBUG] USE_DEPTH_DECODER_DISABLED:`, (self as any).USE_DEPTH_DECODER_DISABLED);



        return;
    }

    // 프레임 데이터 처리
    if (type === 'frame') {
        // console.log(`[DEBUG] Worker received frame data, size: ${data.byteLength} bytes`);
        const t_onmessage_start = performance.now();
        const frameTimings: { [key: string]: number } = {};

        // 1. 메시지 파싱
        const t_parse_start = performance.now();
        const parsedData = parseMessage(data);
        frameTimings.header_parse = performance.now() - t_parse_start;

        if (!parsedData) {
            self.postMessage({ type: 'error', error: 'Failed to parse message' });
            return;
        }

        const processedFrame: ProcessedFrame = {
            depthData: parsedData.depthData,
            frameTimings
        };

        // 2. 디코딩 경로 분기
        if (USE_RAW_RGB) {
            // RAW RGB 경로
            await timeExecution(() => {
                processedFrame.colorRawRGB = parsedData.colorData;
            }, frameTimings, 'raw_rgb_processing');
        } else if (USE_H264_DECODER && !(self as any).USE_H264_FALLBACK_DISABLED) {
            // H.264 경로
            await timeExecution(async () => {
                const imageBitmap = await decodeH264(parsedData.colorData);
                if (imageBitmap) {
                    processedFrame.colorImageBitmap = imageBitmap;
                } else {
                    // H.264 디코딩 실패 시 JPEG로 폴백
                    try {
                        const jpegBlob = new Blob([parsedData.colorData], { type: 'image/jpeg' });
                        const fallbackImageBitmap = await decodeJPEG(jpegBlob);
                        processedFrame.colorImageBitmap = fallbackImageBitmap;
                    } catch (e) {
                        console.error('H.264 fallback to JPEG failed:', e);
                    }
                }
            }, frameTimings, 'h264_decode_call');
            // 3. Depth data decoding
            await timeExecution(async () => {
                console.log(`[DEBUG] Starting depth processing, decoder initialized: ${depthDecoderInitialized}, disabled: ${(self as any).USE_DEPTH_DECODER_DISABLED}`);

                if (depthDecoderInitialized && !(self as any).USE_DEPTH_DECODER_DISABLED) {
                    console.log(`[DEBUG] Attempting HEVC depth decode, data size: ${parsedData.depthData.byteLength} bytes`);
                    const decoded = await decodeDepthHEVC(parsedData.depthData);
                    if (decoded) {
                        console.log(`[DEBUG] HEVC depth decode successful, decoded size: ${decoded.byteLength} bytes`);
                        processedFrame.depthNDC = decoded;   // 성공 시 디코딩된 버퍼
                        return;
                    } else {
                        console.log(`[DEBUG] HEVC depth decode failed, returning null`);
                    }
                } else {
                    console.log(`[DEBUG] Depth decoder not ready, skipping HEVC decode`);
                }
                // 실패하거나 비활성이면 원본 유지
                console.log(`[DEBUG] Keeping original depthData, size: ${parsedData.depthData.byteLength} bytes`);
            }, frameTimings, 'depth_processing');
        } else {
            // JPEG 경로
            await timeExecution(async () => {
                try {
                    const jpegBlob = new Blob([parsedData.colorData], { type: 'image/jpeg' });
                    const imageBitmap = await decodeJPEG(jpegBlob);
                    processedFrame.colorImageBitmap = imageBitmap;
                } catch (e) {
                    console.error('Color processing error:', e);
                }
            }, frameTimings, 'color_processing');
        }

        // 3. 뎁스 데이터 처리
        await timeExecution(() => {
            // 뎁스 데이터는 그대로 전달 (메인 스레드에서 처리)
        }, frameTimings, 'depth_processing');

        // 4. 최종 통계 기록
        frameTimings.total_onmessage = performance.now() - t_onmessage_start;
        perfMonitor.record(frameTimings);

        // 5. 처리된 프레임을 메인 스레드로 전송
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

        perfMonitor.logStatsIfReady(10000); // 10초마다 통계 출력
    }
};

console.log('WebSocket Processor Worker initialized.');