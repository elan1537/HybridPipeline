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
let jpegFallback = false

// H264 디코딩용 pending frames 추적
const pendingFrames = new Map<number, {
    frameId: number,
    serverTimestamps: {
        serverReceiveTime: number,
        serverProcessStartTime: number,
        serverProcessEndTime: number,
        serverSendTime: number
    }
}>()


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
    frameId?: number
}

interface ConnectionMessage {
    type: 'change'
    wsURL: string
}

interface CloseMessage {
    type: 'ws-close'
}

interface PingMessage {
    type: 'ping'
    clientTime: number
}

interface LatencyMessage {
    type: 'latency-update'
    frameId: number
    serverTimestamps?: {
        serverReceiveTime: number
        serverProcessStartTime: number
        serverProcessEndTime: number
        serverSendTime: number
    }
}

self.onmessage = async (evt: MessageEvent<InitMessage | ChunkMessage | CameraMessage | ConnectionMessage | CloseMessage | PingMessage | LatencyMessage>) => {
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
        case 'change':
            ws.close(1000, "Connection changed")
            jpegFallback = !jpegFallback
            await initWebSocket(data.wsURL)
            break
        case 'ws-close':
            ws.close(1000, "Connection closed")
            self.postMessage({ type: 'ws-close' })
            break
        case 'ping':
            sendPing(data.clientTime)
            break
        case 'latency-update':
            // Main thread에서 latency tracker로 전달
            self.postMessage({ 
                type: 'latency-update', 
                frameId: data.frameId,
                serverTimestamps: data.serverTimestamps
            })
            break
    }
}

function parseH264Message(data: ArrayBuffer) {
    const dv = new DataView(data)
    const HEADER_SIZE = 4 + 4 + 8 + 8 + 8 + 8; // videoLen + frameId + 4 timestamps
    if (data.byteLength < HEADER_SIZE) return null;

    const videoLen = dv.getUint32(0, true);
    const frameId = dv.getUint32(4, true);
    const clientSendTime = dv.getFloat64(8, true);
    const serverReceiveTime = dv.getFloat64(16, true);
    const serverProcessEndTime = dv.getFloat64(24, true);
    const serverSendTime = dv.getFloat64(32, true);
    const videoStart = HEADER_SIZE;

    if (data.byteLength < videoStart + videoLen) return null;

    return {
        frameId,
        videoData: data.slice(videoStart, videoStart + videoLen),
        serverTimestamps: {
            serverReceiveTime,
            serverProcessStartTime: serverReceiveTime, // 처리 시작은 수신과 같다고 가정
            serverProcessEndTime,
            serverSendTime
        }
    }
}

function parseJPEGMessage(data: ArrayBuffer): { 
    frameId: number,
    jpegData: ArrayBuffer, 
    depthData: ArrayBuffer,
    serverTimestamps: {
        serverReceiveTime: number,
        serverProcessStartTime: number,
        serverProcessEndTime: number,
        serverSendTime: number
    }
} | null {
    const dv = new DataView(data)
    const HEADER_SIZE = 4 + 4 + 4 + 8 + 8 + 8 + 8; // jpeg_len + depth_len + frameId + 4 timestamps

    if (data.byteLength < HEADER_SIZE) {
        console.error(`JPEG message too short: ${data.byteLength} < ${HEADER_SIZE}`)
        return null;
    }

    const jpegLen = dv.getUint32(0, true)
    const depthLen = dv.getUint32(4, true)
    const frameId = dv.getUint32(8, true)
    const clientSendTime = dv.getFloat64(12, true)
    const serverReceiveTime = dv.getFloat64(20, true)
    const serverProcessEndTime = dv.getFloat64(28, true)
    const serverSendTime = dv.getFloat64(36, true)
    const jpegStart = HEADER_SIZE

    if (data.byteLength < jpegStart + jpegLen + depthLen) {
        console.error(`JPEG message incomplete: expected ${jpegStart + jpegLen + depthLen}, got ${data.byteLength}`)
        return null;
    }

    return {
        frameId,
        jpegData: data.slice(jpegStart, jpegStart + jpegLen),
        depthData: data.slice(jpegStart + jpegLen, jpegStart + jpegLen + depthLen),
        serverTimestamps: {
            serverReceiveTime,
            serverProcessStartTime: serverReceiveTime,
            serverProcessEndTime,
            serverSendTime
        }
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

    // frameId와 타임스탬프 추가 (4 + 4(패딩) + 8 = 16 bytes)
    const timestamp = performance.timeOrigin + performance.now();
    const frameId = data.frameId || 0;

    // 최종 버퍼 생성 (128 + 16 = 144 bytes, 8바이트 정렬을 위해 패딩 추가)
    const finalBuffer = new ArrayBuffer(144);
    const floatView = new Float32Array(finalBuffer, 0, 32);
    const frameIdView = new Uint32Array(finalBuffer, 128, 1);
    // 132는 8의 배수가 아니므로 136으로 조정 (8바이트 정렬)
    const timestampView = new Float64Array(finalBuffer, 136, 1);

    floatView.set(dv);
    frameIdView[0] = frameId;
    timestampView[0] = timestamp;

    ws.send(finalBuffer);
}

function sendPing(clientTime: number) {
    if (!ws || ws.readyState !== WebSocket.OPEN) return;

    // Ping 메시지: type(1 byte) + padding(7 bytes) + clientTime(8 bytes) = 16 bytes (8바이트 정렬)
    const buffer = new ArrayBuffer(16);
    const typeView = new Uint8Array(buffer, 0, 1);
    const timeView = new Float64Array(buffer, 8, 1);  // 8바이트 정렬된 위치
    
    typeView[0] = 255; // ping message type
    timeView[0] = clientTime;
    
    ws.send(buffer);
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
        const clientReceiveTime = performance.now();

        // Ping response 처리 (type(1) + padding(7) + clientTime(8) + serverTime(8) = 24 bytes)
        if (arrBuf.byteLength === 24) { 
            const dv = new DataView(arrBuf);
            const type = dv.getUint8(0);
            if (type === 254) { // pong message type
                const clientTime = dv.getFloat64(8, true);   // 8바이트 정렬된 위치
                const serverTime = dv.getFloat64(16, true);  // 16바이트 위치
                
                self.postMessage({
                    type: 'pong-received',
                    clientRequestTime: clientTime,
                    serverReceiveTime: serverTime,
                    serverSendTime: serverTime, // 간단히 동일하다고 가정
                    clientResponseTime: clientReceiveTime
                });
                return;
            }
        }

        if (jpegFallback) {
            const parseResult = parseJPEGMessage(arrBuf);
            if (!parseResult) {
                console.error("Failed to parse JPEG message");
                return;
            }

            const { frameId, jpegData, depthData, serverTimestamps } = parseResult;
            const colorBitmap = await createImageBitmap(new Blob([jpegData], { type: 'image/jpeg' }));
            const depthFloat16 = new Uint16Array(depthData);

            // 레이턴시 추적 정보 전달
            self.postMessage({
                type: 'frame-receive',
                frameId,
                serverTimestamps,
                clientReceiveTime
            });

            decodeCnt++;
            const decodeCompleteTime = performance.now();

            self.postMessage({
                type: 'frame',
                frameId,
                image: colorBitmap,
                depth: depthFloat16,
                decodeCompleteTime
            });

            const now = performance.now();
            if (now - decodeStart > 1000) {
                const fps = decodeCnt / ((now - decodeStart) / 1000);
                self.postMessage({ type: 'fps', decode: fps });
                decodeCnt = 0;
                decodeStart = now;
            }
        } else {
            const parseResult = parseH264Message(arrBuf);
            if (!parseResult) {
                console.error("Failed to parse H264 message");
                return;
            }

            const { frameId, videoData, serverTimestamps } = parseResult;

            // 레이턴시 추적 정보 전달
            self.postMessage({
                type: 'frame-receive',
                frameId,
                serverTimestamps,
                clientReceiveTime
            });

            if (!decoderInitialized) {
                await initDecoder();
            }
            
            const chunk = new EncodedVideoChunk({
                type: 'key',
                timestamp: performance.now(),
                data: videoData
            });
            
            // frameId를 VideoFrame에 연결하기 위해 저장
            pendingFrames.set(chunk.timestamp, { frameId, serverTimestamps });
            
            videoDecoder.decode(chunk);
        }
    };
    ws.onerror = e => {
        console.error('WS error', e)
        self.postMessage({ type: 'ws-error' })
    }
    ws.onclose = e => {
        console.log('WS closed', e.code)
        self.postMessage({ type: 'ws-close' })
    }
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
    const decodeCompleteTime = performance.now();

    const w = frame.codedWidth
    const h = frame.codedHeight / 2;

    const [colorBitmap, depthBitmap] = await splitFrameCanvas(frame, w, h);

    // pending frame 정보 찾기
    const frameInfo = pendingFrames.get(frame.timestamp);
    let frameId = 0;
    
    if (frameInfo) {
        frameId = frameInfo.frameId;
        pendingFrames.delete(frame.timestamp);
    }

    frame.close();

    self.postMessage({
        type: 'frame',
        frameId,
        image: colorBitmap,
        depth: depthBitmap,
        decodeCompleteTime
    });

    const now = performance.now();

    if (now - decodeStart > 1000) {
        const fps = decodeCnt / ((now - decodeStart) / 1000);
        self.postMessage({ type: 'fps', decode: fps });
        decodeCnt = 0;
        decodeStart = now;
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
