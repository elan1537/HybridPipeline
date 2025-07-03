/**
 * Color 데이터 처리를 전담하는 Worker
 * H.264, JPEG, RAW RGB 디코딩을 담당합니다.
 */

// Color Worker 내부 변수
let videoDecoder: VideoDecoder | null = null;
let h264ErrorCount = 0;
let nextImageBitmap: ImageBitmap | null = null;
let USE_H264_DECODER = false;
let USE_RAW_RGB = false;
let rtWidth = 1920;
let rtHeight = 1080;

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

    console.log('[COLOR WORKER] VideoDecoder initialized');
}

// H.264 디코딩
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
        videoDecoder.decode(chunk);

        return new Promise((resolve) => {
            let attempts = 0;
            const maxAttempts = 100;

            const checkForFrame = () => {
                attempts++;
                if (nextImageBitmap) {
                    const bmp = nextImageBitmap;
                    nextImageBitmap = null;
                    resolve(bmp);
                } else if (attempts < maxAttempts) {
                    setTimeout(checkForFrame, 1);
                } else {
                    console.warn('[COLOR WORKER] H.264 frame timeout');
                    resolve(null);
                }
            };
            checkForFrame();
        });

    } catch (e) {
        console.error('[COLOR WORKER] H.264 decoding error:', e);
        return null;
    }
}

// JPEG 디코딩
async function decodeJPEG(jpegBlob: Blob): Promise<ImageBitmap> {
    return await createImageBitmap(jpegBlob);
}

// 메시지 처리
self.onmessage = async function (event: MessageEvent) {
    const { type, data, frameId, timestamp, config } = event.data;

    if (type === 'config') {
        USE_H264_DECODER = config.USE_H264_DECODER;
        USE_RAW_RGB = config.USE_RAW_RGB;
        rtWidth = config.rtWidth;
        rtHeight = config.rtHeight;
        console.log(`[COLOR WORKER] Config updated: H.264=${USE_H264_DECODER}, RAW_RGB=${USE_RAW_RGB}, size=${rtWidth}x${rtHeight}`);
        return;
    }

    if (type === 'color_frame') {
        const startTime = performance.now();
        let result = null;

        try {
            if (USE_RAW_RGB) {
                // RAW RGB 데이터는 그대로 전달
                result = { type: 'raw_rgb', data: data };
                console.log(`[COLOR WORKER] Processing RAW RGB data: ${data.byteLength} bytes`);
            } else if (USE_H264_DECODER) {
                // H.264 디코딩 시도
                const imageBitmap = await decodeH264(data);
                if (imageBitmap) {
                    result = { type: 'image_bitmap', data: imageBitmap };
                    console.log(`[COLOR WORKER] H.264 decode successful: ${imageBitmap.width}x${imageBitmap.height}`);
                } else {
                    // H.264 디코딩 실패 시 JPEG로 폴백
                    console.log(`[COLOR WORKER] H.264 decode failed, falling back to JPEG`);
                    try {
                        const jpegBlob = new Blob([data], { type: 'image/jpeg' });
                        const fallbackImageBitmap = await decodeJPEG(jpegBlob);
                        result = { type: 'image_bitmap', data: fallbackImageBitmap };
                        console.log(`[COLOR WORKER] JPEG fallback successful: ${fallbackImageBitmap.width}x${fallbackImageBitmap.height}`);
                    } catch (e) {
                        console.error('[COLOR WORKER] H.264 fallback to JPEG failed:', e);
                    }
                }
            } else {
                // JPEG 경로
                console.log(`[COLOR WORKER] Processing JPEG data: ${data.byteLength} bytes`);
                try {
                    const jpegBlob = new Blob([data], { type: 'image/jpeg' });
                    const imageBitmap = await decodeJPEG(jpegBlob);
                    result = { type: 'image_bitmap', data: imageBitmap };
                    console.log(`[COLOR WORKER] JPEG decode successful: ${imageBitmap.width}x${imageBitmap.height}`);
                } catch (e) {
                    console.error('[COLOR WORKER] Color processing error:', e);
                }
            }
        } catch (error) {
            console.error('[COLOR WORKER] Unexpected error during color processing:', error);
        }

        const processingTime = performance.now() - startTime;

        // 결과 전송
        self.postMessage({
            type: 'color_result',
            frameId,
            timestamp,
            result,
            processingTime
        }, result && result.data ? [result.data] : []);
    }
};

console.log('[COLOR WORKER] Color Worker initialized'); 