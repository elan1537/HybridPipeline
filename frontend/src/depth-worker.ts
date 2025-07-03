/**
 * Depth 데이터 처리를 전담하는 Worker
 * HEVC 디코딩을 담당합니다.
 */

// Depth Worker 내부 변수
let depthDecoder: VideoDecoder | null = null;
let depthErrorCount = 0;
let nextDepthFrame: VideoFrame | null = null;
let depthDecoderInitialized = false;
let rtWidth = 1920;
let rtHeight = 1080;

// HEVC 헤더 추출 함수
function extractHEVCHeaders(hevcData: ArrayBuffer): ArrayBuffer | null {
    const data = new Uint8Array(hevcData);
    const startCode = [0x00, 0x00, 0x00, 0x01];

    let vpsData: Uint8Array | null = null;
    let spsData: Uint8Array | null = null;
    let ppsData: Uint8Array | null = null;

    console.log(`[DEPTH WORKER] Analyzing HEVC data: ${data.length} bytes`);

    let i = 0;
    while (i < data.length - 4) {
        if (data[i] === startCode[0] &&
            data[i + 1] === startCode[1] &&
            data[i + 2] === startCode[2] &&
            data[i + 3] === startCode[3]) {

            if (i + 4 >= data.length) break;

            const nalType = (data[i + 4] >> 1) & 0x3F;

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

            const nalData = data.slice(i, nextStart);
            console.log(`[DEPTH WORKER] Found NAL unit type ${nalType}, length: ${nalData.length} bytes`);

            switch (nalType) {
                case 32: // VPS
                    vpsData = nalData;
                    console.log(`[DEPTH WORKER] VPS found: ${nalData.length} bytes`);
                    break;
                case 33: // SPS
                    spsData = nalData;
                    console.log(`[DEPTH WORKER] SPS found: ${nalData.length} bytes`);
                    break;
                case 34: // PPS
                    ppsData = nalData;
                    console.log(`[DEPTH WORKER] PPS found: ${nalData.length} bytes`);
                    break;
            }

            i = nextStart;
        } else {
            i++;
        }
    }

    if (vpsData && spsData && ppsData) {
        const totalLength = vpsData.length + spsData.length + ppsData.length;
        const combined = new Uint8Array(totalLength);
        let offset = 0;
        combined.set(vpsData, offset);
        offset += vpsData.length;
        combined.set(spsData, offset);
        offset += spsData.length;
        combined.set(ppsData, offset);
        console.log(`[DEPTH WORKER] Combined VPS+SPS+PPS: ${totalLength} bytes`);
        return combined.buffer;
    }

    if (spsData && ppsData) {
        const totalLength = spsData.length + ppsData.length;
        const combined = new Uint8Array(totalLength);
        combined.set(spsData, 0);
        combined.set(ppsData, spsData.length);
        console.log(`[DEPTH WORKER] Combined SPS+PPS: ${totalLength} bytes`);
        return combined.buffer;
    }

    if (spsData) {
        console.log(`[DEPTH WORKER] Using SPS only: ${spsData.length} bytes`);
        return spsData.buffer;
    }

    console.error('[DEPTH WORKER] No valid HEVC headers found');
    return null;
}

// HEVC 디코더 초기화
async function initializeDepthDecoder(hevcData: ArrayBuffer): Promise<boolean> {
    if (depthDecoderInitialized) {
        return true;
    }

    try {
        console.log(`[DEPTH WORKER] Starting HEVC decoder initialization with ${hevcData.byteLength} bytes`);

        const hevcDescription = extractHEVCHeaders(hevcData);
        if (!hevcDescription) {
            console.error('[DEPTH WORKER] Failed to extract HEVC headers');
            return false;
        }

        console.log(`[DEPTH WORKER] HEVC headers extracted: ${hevcDescription.byteLength} bytes`);

        const config: VideoDecoderConfig = {
            codec: 'hvc1.2.4.L120.B0',
            codedWidth: rtWidth,
            codedHeight: rtHeight,
            optimizeForLatency: true,
        };

        console.log(`[DEPTH WORKER] Checking codec support for: ${config.codec}, size: ${config.codedWidth}x${config.codedHeight}`);

        const { supported } = await VideoDecoder.isConfigSupported(config);
        if (!supported) {
            console.error('[DEPTH WORKER] HEVC decoder is not supported');
            return false;
        }

        console.log('[DEPTH WORKER] Codec is supported, creating VideoDecoder');

        if (depthDecoder && depthDecoder.state !== 'closed') {
            depthDecoder.close();
        }

        depthDecoder = new VideoDecoder({
            output: (frame) => {
                console.log(`[DEPTH WORKER] VideoDecoder output: format=${frame.format}, size=${frame.displayWidth}x${frame.displayHeight}`);
                if (nextDepthFrame) {
                    nextDepthFrame.close();
                }
                nextDepthFrame = frame;
            },
            error: (e) => {
                console.error('[DEPTH WORKER] Depth VideoDecoder runtime error:', e);
                depthErrorCount++;
            },
        });

        depthDecoder.configure(config);
        depthDecoderInitialized = true;
        console.log('[DEPTH WORKER] Depth decoder initialized successfully');

        return true;

    } catch (e) {
        console.error('[DEPTH WORKER] Depth decoder initialization error:', e);
        return false;
    }
}

// Depth HEVC 디코딩
async function decodeDepthHEVC(depthData: ArrayBuffer): Promise<ArrayBuffer | null> {
    try {
        if (!depthDecoder || depthDecoder.state !== 'configured') {
            console.warn('[DEPTH WORKER] Depth decoder not ready');
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
                    const f = nextDepthFrame;
                    nextDepthFrame = null;
                    const w = f.displayWidth, h = f.displayHeight;

                    try {
                        // VideoFrame의 format 체크
                        if (!f.format) {
                            console.warn('[DEPTH WORKER] VideoFrame format is null, cannot copy data');
                            f.close();
                            res(null);
                            return;
                        }

                        console.log(`[DEPTH WORKER] VideoFrame format: ${f.format}, size: ${w}x${h}`);

                        // P010 포맷에서 Y 평면만 추출 (16비트)
                        const yPlane16 = new Uint16Array(w * h);
                        await f.copyTo(yPlane16);
                        f.close();

                        const depthNDC = new Float32Array(w * h);
                        for (let i = 0; i < yPlane16.length; i++) {
                            const v10 = yPlane16[i] >> 6;
                            depthNDC[i] = (v10 / 1023.0) * 2.0 - 1.0;
                        }

                        res(depthNDC.buffer);
                    } catch (copyError) {
                        console.error('[DEPTH WORKER] Error copying VideoFrame data:', copyError);
                        f.close();
                        res(null);
                    }
                } else if (++tries < 100) {
                    setTimeout(wait, 1);
                } else {
                    console.warn('[DEPTH WORKER] Depth frame timeout');
                    res(null);
                }
            })();
        });

    } catch (e) {
        console.error('[DEPTH WORKER] Depth HEVC decoding error:', e);
        return null;
    }
}

// 메시지 처리
self.onmessage = async function (event: MessageEvent) {
    const { type, data, frameId, timestamp, config } = event.data;

    if (type === 'config') {
        rtWidth = config.rtWidth;
        rtHeight = config.rtHeight;
        console.log(`[DEPTH WORKER] Config updated: size=${rtWidth}x${rtHeight}`);
        return;
    }

    if (type === 'depth_frame') {
        const startTime = performance.now();
        let result = null;

        try {
            // Depth 디코더가 초기화되지 않은 경우 초기화
            if (!depthDecoderInitialized) {
                console.log(`[DEPTH WORKER] Initializing depth decoder with data: ${data.byteLength} bytes`);
                const initSuccess = await initializeDepthDecoder(data);
                if (!initSuccess) {
                    console.warn('[DEPTH WORKER] Failed to initialize depth decoder, using raw data');
                    result = { type: 'depth_raw', data: data };
                }
            }

            if (depthDecoderInitialized) {
                console.log(`[DEPTH WORKER] Attempting HEVC decode: ${data.byteLength} bytes`);
                const decoded = await decodeDepthHEVC(data);
                if (decoded) {
                    result = { type: 'depth_ndc', data: decoded };
                    console.log(`[DEPTH WORKER] HEVC decode successful: ${decoded.byteLength} bytes`);
                } else {
                    console.warn('[DEPTH WORKER] HEVC decode failed, using raw data');
                    result = { type: 'depth_raw', data: data };
                }
            }

            // 실패하거나 비활성이면 원본 유지
            if (!result) {
                console.log(`[DEPTH WORKER] Using raw depth data: ${data.byteLength} bytes`);
                result = { type: 'depth_raw', data: data };
            }
        } catch (error) {
            console.error('[DEPTH WORKER] Unexpected error during depth processing:', error);
            result = { type: 'depth_raw', data: data };
        }

        const processingTime = performance.now() - startTime;

        // 결과 전송
        self.postMessage({
            type: 'depth_result',
            frameId,
            timestamp,
            result,
            processingTime
        }, result && result.data ? [result.data] : []);
    }
};

console.log('[DEPTH WORKER] Depth Worker initialized'); 