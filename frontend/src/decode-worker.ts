let videoDecoder: VideoDecoder
let ws: WebSocket
let decoderInitialized = false;

let rtWidth = 1920
let rtHeight = 1080

let rgbaTemp: Uint8Array

let decodeCnt = 0
let renderCnt = 0
let decodeStart = performance.now()
let renderStart = performance.now()

interface InitMessage {
    type: 'init'
    canvas: OffscreenCanvas
    width: number
    height: number
    wsURL: string
}

interface ChunkMessage {
    type: 'chunk'
    data: ArrayBuffer
}

interface CameraMessage {
    type: 'camera'
    position: Float32Array
    target: Float32Array
    intrinsics: Float32Array
    projection: Float32Array
}

self.onmessage = async (evt: MessageEvent<InitMessage | ChunkMessage | CameraMessage>) => {
    const data = evt.data

    switch (data.type) {
        case 'init':
            console.log(data)
            await init(data)
            break
        case 'chunk':
            decodeChunk(data.data)
            break
        case 'camera':
            updateCamera(data)
            break
    }
}

function parseMessage(data: ArrayBuffer) {
    const dv = new DataView(data)
    const HEADER_SIZE = 4 + 8 + 8 + 8;
    if (data.byteLength < HEADER_SIZE) return null;

    const videoLen = dv.getUint32(0, true);
    const videoStart = HEADER_SIZE;

    if (data.byteLength < videoStart + videoLen) return null;

    return {
        videoData: data.slice(videoStart, videoStart + videoLen)
    }
}

function updateCamera(data: CameraMessage) {
    const position = new Float32Array(data.position);
    const target = new Float32Array(data.target);
    const intrinsics = new Float32Array(data.intrinsics);
    const projection = new Float32Array(data.projection);

    const dv = new Float32Array(32); // 32 * 4 = 128 bytes

    // position (3 floats)
    dv.set(position, 0);

    // target (3 floats) 
    dv.set(target, 3);

    // intrinsics (9 floats)
    dv.set(intrinsics, 6);

    // padding (1 float) - 서버에서 기대하는 0.0 값
    dv[15] = 0.0;

    // projection (16 floats)
    dv.set(projection, 16);

    // 타임스탬프 추가 (8 bytes)
    const timestamp = performance.timeOrigin + performance.now();
    const timestampBuffer = new Float64Array([timestamp]);

    // 최종 버퍼 생성 (128 + 8 = 136 bytes)
    const finalBuffer = new ArrayBuffer(136);
    const floatView = new Float32Array(finalBuffer, 0, 32);
    const timestampView = new Float64Array(finalBuffer, 128);

    floatView.set(dv);
    timestampView.set(timestampBuffer);

    ws.send(finalBuffer);
}

async function initDecoder() {
    if (decoderInitialized) return;

    console.log("initDecoder")

    // Safari 호환성 체크
    if (typeof VideoDecoder === 'undefined') {
        console.error('VideoDecoder API not supported in this browser');
        return;
    }

    try {
        videoDecoder = new VideoDecoder({
            output: handleFrame,
            error: e => {
                console.error('Decoder error', e)
                console.error('Decoder state:', videoDecoder?.state)
                console.error('Decoder config:', {
                    codec: 'avc1.42E01E',
                    codedWidth: rtWidth,
                    codedHeight: rtHeight * 2
                })
                decoderInitialized = false;
                // 에러 발생 시 디코더를 닫고 재시도 준비
                if (videoDecoder && videoDecoder.state !== 'closed') {
                    videoDecoder.close();
                }
            }
        });

        const config: VideoDecoderConfig = {
            codec: 'avc1.42E01E',
            codedWidth: rtWidth,
            codedHeight: rtHeight * 2,
        };

        const { supported } = await VideoDecoder.isConfigSupported(config);
        if (!supported) {
            console.error('VideoDecoder config unsupported');
            return;
        }

        videoDecoder.configure(config);
        decoderInitialized = true;
        console.log("Decoder initialized successfully");

    } catch (error) {
        console.error('Failed to initialize decoder:', error);
        decoderInitialized = false;
    }
}

async function initWebSocket(wsURL: string) {
    console.log("initWebSocket")
    console.log("URL: ", wsURL)
    ws = new WebSocket(wsURL);
    console.log("WS: ", ws)
    ws.binaryType = 'arraybuffer';
    ws.onopen = () => {
        console.log('[MediaWorker] WS opened')
        self.postMessage({ type: 'ws-ready' })
        const buf = new ArrayBuffer(4);
        new DataView(buf).setUint16(0, rtWidth, true);
        new DataView(buf).setUint16(2, rtHeight, true);
        ws.send(buf);
    }
    ws.onmessage = async e => {
        const arrBuf = e.data as ArrayBuffer;

        if (!decoderInitialized) {
            await initDecoder();
        }

        const chunk = new EncodedVideoChunk({
            type: 'key',
            timestamp: performance.now(),
            data: arrBuf
        });
        videoDecoder.decode(chunk);
    };
    ws.onerror = e => console.error('WS error', e);
    ws.onclose = e => console.log('WS closed', e.code);
}

async function init({ width, height, wsURL }: InitMessage) {
    rtWidth = width
    rtHeight = height

    rgbaTemp = new Uint8Array(rtWidth * rtHeight * 4)

    await initWebSocket(wsURL)

}

async function splitFrameCanvas(frame: VideoFrame, w: number, h: number) {
    const full = await createImageBitmap(frame);   // 1280×1440
    const [cCan, dCan] = [new OffscreenCanvas(w, h), new OffscreenCanvas(w, h)];

    cCan.getContext('2d')!.drawImage(full, 0, 0, w, h, 0, 0, w, h);
    dCan.getContext('2d')!.drawImage(full, 0, h, w, h, 0, 0, w, h);

    return Promise.all([createImageBitmap(cCan), createImageBitmap(dCan)]);
}

async function handleFrame(frame: VideoFrame) {
    decodeCnt++;

    const w = frame.codedWidth
    const h = frame.codedHeight / 2;

    const [colorBitmap, depthBitmap] = await splitFrameCanvas(frame, w, h)

    frame.close()

    self.postMessage({
        type: 'frame',
        image: colorBitmap,
        depth: depthBitmap,
    })

    const now = performance.now()

    if (now - decodeStart > 1000) {
        const fps = decodeCnt / ((now - decodeStart) / 1000)
        self.postMessage({ type: 'fps', decode: fps })
        decodeCnt = 0
        decodeStart = now
    }
}

function decodeChunk(arrayBuf: ArrayBuffer) {
    if (!decoderInitialized) {
        console.warn('Decoder not initialized, skipping chunk');
        return;
    }

    const chunk = new EncodedVideoChunk({
        type: 'key',
        timestamp: performance.now(),
        data: arrayBuf,
    })

    videoDecoder.decode(chunk)
}
