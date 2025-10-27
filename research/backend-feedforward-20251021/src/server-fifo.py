# server-fifo.py
# Transport Service: WebSocket <-> Unix Domain Socket <-> Renderer
import asyncio
import struct
import time
import os
import websockets

from session import UserSession

# 설정
WEBSOCKET_PORT = 8765
CAMERA_SOCKET = "/run/ipc/camera.sock"
VIDEO_SOCKET = "/run/ipc/video.sock"

# 소켓 디렉토리 생성
SOCKET_DIR = "/run/ipc"
if not os.path.exists(SOCKET_DIR):
    os.makedirs(SOCKET_DIR, exist_ok=True)
    print(f"Created socket directory: {SOCKET_DIR}")

# 기존 소켓 파일 제거
for sock_path in [CAMERA_SOCKET, VIDEO_SOCKET]:
    if os.path.exists(sock_path):
        os.remove(sock_path)
        print(f"Removed existing socket: {sock_path}")

# 전역 session 관리 (단일 클라이언트 가정)
current_session: UserSession = None


async def recv_loop(session: UserSession):
    """Frontend로부터 카메라 데이터 및 ping/pong 메시지 수신"""
    ws = session.ws
    q = session.q
    width = session.width
    height = session.height
    camera_frame_count = 0  # 카메라 프레임 카운터

    print(f"Receive loop started for {ws.remote_address}")
    try:
        while True:
            raw_with_ts = await ws.recv()

            # Ping 메시지 처리
            if len(raw_with_ts) == 16:
                message_type = struct.unpack_from("<B", raw_with_ts, 0)[0]
                if message_type == 255:  # ping message
                    client_time = struct.unpack_from("<d", raw_with_ts, 8)[0]
                    server_time = time.time_ns() / 1_000_000
                    pong_response = struct.pack("<B7xdd", 254, client_time, server_time)
                    await ws.send(pong_response)
                    continue

            # 핸드셰이크 처리 (해상도 변경)
            if len(raw_with_ts) == 4:
                W, H = struct.unpack("<HH", raw_with_ts)
                if width != W or height != H:
                    print(f"[+] Peer {ws.remote_address} resized to {W}x{H}")
                    width, height = W, H
                    session.width = width
                    session.height = height
                    # 큐 초기화
                    while not q.empty():
                        try:
                            q.get_nowait()
                            q.task_done()
                        except asyncio.QueueEmpty:
                            break

            # 카메라 데이터 (160 bytes - 확장 프로토콜)
            elif len(raw_with_ts) == 160:
                server_recv_timestamp_ms = time.time_ns() / 1_000_000

                # frameId 추출 (128번째 바이트부터 4바이트)
                frame_id = struct.unpack_from("<I", raw_with_ts, 128)[0]
                # client timestamp 추출
                client_send_timestamp_ms = struct.unpack_from("<d", raw_with_ts, 136)[0]
                # time_index 추출
                time_index = struct.unpack_from("<f", raw_with_ts, 144)[0]
                # 실제 카메라 데이터 (처음 128바이트)
                actual_payload = raw_with_ts[:128]

                if q.full():
                    try:
                        q.get_nowait()
                        q.task_done()
                    except asyncio.QueueEmpty:
                        pass

                await q.put((actual_payload, client_send_timestamp_ms, server_recv_timestamp_ms, time_index, frame_id))

                # 카메라 데이터 수신 로그 (60프레임마다)
                if camera_frame_count % 60 == 0:
                    # view_matrix와 intrinsics 파싱 (간단히)
                    view_mat_sample = struct.unpack_from("<4f", actual_payload, 0)  # 처음 4개 float
                    intrinsics_sample = struct.unpack_from("<4f", actual_payload, 64)  # 중간 4개 float
                    print(f"📷 [Camera Recv] Frame {frame_id}: time_index={time_index:.3f}, "
                          f"view_mat[0:4]=[{view_mat_sample[0]:.2f}, {view_mat_sample[1]:.2f}, {view_mat_sample[2]:.2f}, {view_mat_sample[3]:.2f}], "
                          f"intrinsics[0:4]=[{intrinsics_sample[0]:.1f}, {intrinsics_sample[1]:.1f}, {intrinsics_sample[2]:.1f}, {intrinsics_sample[3]:.1f}]")

                camera_frame_count += 1

            # 카메라 데이터 (148 bytes - test_feedforward.py 호환)
            elif len(raw_with_ts) == 148:
                server_recv_timestamp_ms = time.time_ns() / 1_000_000

                # 패킷 구조: payload(128) + frameId(4) + padding(4) + clientTime(8) + time_index(4)
                frame_id = struct.unpack_from("<I", raw_with_ts, 128)[0]  # offset 128
                # padding 4 bytes는 건너뜀
                client_send_timestamp_ms = struct.unpack_from("<d", raw_with_ts, 136)[0]  # offset 136
                time_index = struct.unpack_from("<f", raw_with_ts, 144)[0]  # offset 144
                actual_payload = raw_with_ts[:128]

                if q.full():
                    try:
                        q.get_nowait()
                        q.task_done()
                    except asyncio.QueueEmpty:
                        pass

                await q.put((actual_payload, client_send_timestamp_ms, server_recv_timestamp_ms, time_index, frame_id))

                # 카메라 데이터 수신 로그 (60프레임마다)
                if camera_frame_count % 60 == 0:
                    view_mat_sample = struct.unpack_from("<4f", actual_payload, 0)
                    intrinsics_sample = struct.unpack_from("<4f", actual_payload, 64)
                    print(f"📷 [Camera Recv] Frame {frame_id}: time_index={time_index:.3f}, "
                          f"view_mat[0:4]=[{view_mat_sample[0]:.2f}, {view_mat_sample[1]:.2f}, {view_mat_sample[2]:.2f}, {view_mat_sample[3]:.2f}], "
                          f"intrinsics[0:4]=[{intrinsics_sample[0]:.1f}, {intrinsics_sample[1]:.1f}, {intrinsics_sample[2]:.1f}, {intrinsics_sample[3]:.1f}]")

                camera_frame_count += 1

            # 기존 프로토콜 호환성 (136 bytes)
            elif len(raw_with_ts) == 136:
                server_recv_timestamp_ms = time.time_ns() / 1_000_000
                client_send_timestamp_ms = struct.unpack_from("<d", raw_with_ts, len(raw_with_ts) - 8)[0]
                actual_payload = raw_with_ts[:-8]
                if q.full():
                    try:
                        q.get_nowait()
                        q.task_done()
                    except asyncio.QueueEmpty:
                        pass
                await q.put((actual_payload, client_send_timestamp_ms, server_recv_timestamp_ms, 0, 0))

    except websockets.exceptions.ConnectionClosed:
        print(f"Connection closed for {ws.remote_address}")
    finally:
        print(f"Receive loop finished for {ws.remote_address}")


async def send_loop(session: UserSession):
    """Frontend로 비디오 데이터 전송"""
    target_fps = 60
    frame_interval = 1 / target_fps
    last_send = time.perf_counter()

    ws = session.ws
    send_q = session.send_q

    try:
        while True:
            queue_item = await send_q.get()

            # 확장된 큐 아이템 형식 지원
            if len(queue_item) >= 4:
                header, video_bitstream, frame_count, server_process_end_ms = queue_item[:4]

                # 실제 전송 시점 측정
                server_send_timestamp_ms = time.time_ns() / 1_000_000

                # 헤더의 마지막 8바이트(serverSendTime)를 실제 전송 시점으로 업데이트
                header_bytes = bytearray(header)
                send_time_offset = len(header_bytes) - 8
                struct.pack_into("<d", header_bytes, send_time_offset, server_send_timestamp_ms)
                header = bytes(header_bytes)

            else:
                # 기존 형식 호환성
                header, video_bitstream, frame_count = queue_item

            now = time.perf_counter()
            elapsed = now - last_send

            if elapsed < frame_interval:
                await asyncio.sleep(frame_interval - elapsed)

            await ws.send(header + video_bitstream)
            send_q.task_done()
            last_send = time.perf_counter()

    except asyncio.CancelledError:
        print(f"Send loop cancelled.")
    except Exception as e:
        print(f"Error in send loop: {e}")


async def camera_server_handler(reader, writer):
    """Renderer로 카메라 데이터 전송 (Unix Socket Server)"""
    global current_session
    addr = writer.get_extra_info('peername')
    print(f"Camera client connected: {addr}")

    forward_count = 0  # 전달 프레임 카운터

    try:
        while True:
            # WebSocket 세션이 준비될 때까지 대기
            while current_session is None:
                await asyncio.sleep(0.1)

            q = current_session.q

            # recv_loop에서 받은 카메라 데이터 가져오기
            queue_data = await q.get()

            if len(queue_data) == 5:
                raw_payload, client_ts, server_ts, time_index, frame_id = queue_data
            else:
                raw_payload, client_ts, server_ts = queue_data
                time_index = 0.0
                frame_id = 0

            # 카메라 데이터 패킷 구성
            # payload(128) + client_ts(8) + server_ts(8) + time_index(4) + frame_id(4) = 152 bytes
            packet = raw_payload + struct.pack("<ddfi", client_ts, server_ts, time_index, frame_id)

            writer.write(packet)
            await writer.drain()
            q.task_done()

            # Renderer로 전송 로그 (60프레임마다)
            if forward_count % 60 == 0:
                print(f"➡️  [Camera Forward] Frame {frame_id} → Renderer: time_index={time_index:.3f}, packet_size={len(packet)} bytes")

            forward_count += 1

    except asyncio.CancelledError:
        print(f"Camera server handler cancelled for {addr}")
    except Exception as e:
        print(f"Error in camera server handler: {e}")
    finally:
        writer.close()
        await writer.wait_closed()
        print(f"Camera client disconnected: {addr}")


async def video_server_handler(reader, writer):
    global current_session
    addr = writer.get_extra_info('peername')
    print(f"Video client connected: {addr}")

    while current_session is None:
        await asyncio.sleep(0.1)

    send_q = current_session.send_q
    frame_count = 0

    os.makedirs("transport_output", exist_ok=True)

    try:
        while True:
            header_bytes = await reader.read(16)

            if len(header_bytes) < 16:
                print("Video connection closed or incomplete header")
                break

            frame_id, color_size, depth_size = struct.unpack("<QII", header_bytes)

            if color_size <= 0 or depth_size <= 0 or color_size > 10 * 1024 * 1024 or depth_size > 10 * 1024 * 1024:
                print(f"Invalid sizes: color={color_size}, depth={depth_size}")
                continue

            color_jpeg = await reader.read(color_size)
            depth_jpeg = await reader.read(depth_size)

            if len(color_jpeg) < color_size or len(depth_jpeg) < depth_size:
                print(f"Incomplete frame")
                break

            with open(f"transport_output/frame_{frame_id:06d}_color.jpg", 'wb') as f:
                f.write(color_jpeg)
            with open(f"transport_output/frame_{frame_id:06d}_depth.jpg", 'wb') as f:
                f.write(depth_jpeg)

            transport_recv_time = time.time_ns() / 1_000_000

            combined = color_jpeg + depth_jpeg
            header = struct.pack("<IIdddd",
                len(combined), int(frame_id), 0.0, 0.0, transport_recv_time, 0.0)

            await send_q.put((header, combined, frame_count, transport_recv_time))

            if frame_count % 60 == 0:
                print(f"[Video] Frame {frame_id}: color={color_size} depth={depth_size} bytes")

            frame_count += 1

    except asyncio.CancelledError:
        print(f"Video server handler cancelled for {addr}")
    except Exception as e:
        print(f"Error in video server handler: {e}")
        import traceback
        traceback.print_exc()
    finally:
        writer.close()
        await writer.wait_closed()
        print(f"Video client disconnected: {addr}")


async def start_camera_server():
    """카메라 데이터 전송 서버 시작 (전역 session 사용)"""
    server = await asyncio.start_unix_server(
        camera_server_handler,
        CAMERA_SOCKET
    )
    os.chmod(CAMERA_SOCKET, 0o666)
    print(f"✅ Camera server listening on {CAMERA_SOCKET}")

    async with server:
        await server.serve_forever()


async def start_video_server():
    """비디오 데이터 수신 서버 시작 (전역 session 사용)"""
    server = await asyncio.start_unix_server(
        video_server_handler,
        VIDEO_SOCKET
    )
    os.chmod(VIDEO_SOCKET, 0o666)
    print(f"✅ Video server listening on {VIDEO_SOCKET}")

    async with server:
        await server.serve_forever()


async def handler(ws: websockets.WebSocketServerProtocol):
    """WebSocket 연결 핸들러"""
    global current_session
    remote_addr = ws.remote_address

    print(f"Connection opened from {remote_addr}")
    session = None

    try:
        handshake = await ws.recv()
        if isinstance(handshake, bytes) and len(handshake) == 4:
            width, height = struct.unpack("<HH", handshake)
            # feedforward 모드는 encoder 불필요 (Renderer에서 인코딩)
            session = UserSession(ws, width, height, use_encoder=False)
            print(f"✅ Session created without encoder for {remote_addr} ({width}x{height})")
            print(f"[+] Session created for {remote_addr} => {width}x{height}")
        else:
            await ws.close()
            return

        print(ws.request.path)

        if ws.request.path == "/ws/feedforward":
            # 전역 session 업데이트
            current_session = session

            # 2개 태스크만 실행 (Unix Socket 서버는 이미 실행 중)
            recv_task = asyncio.create_task(recv_loop(session))
            send_task = asyncio.create_task(send_loop(session))

            print(f"[+] Feedforward mode started for {remote_addr}")

            done, pending = await asyncio.wait(
                [recv_task, send_task],
                return_when=asyncio.FIRST_COMPLETED
            )
            for task in pending:
                task.cancel()
        else:
            print(f"[!] Unsupported path: {ws.request.path}")
            await ws.close()
            return

    except Exception as e:
        print(f"Handler error for {remote_addr}: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 세션 종료 시 전역 session 초기화
        if current_session == session:
            current_session = None
        print(f"Connection handler finished for {remote_addr}")


async def main():
    # Unix Socket 서버 먼저 시작 (백그라운드 태스크)
    camera_task = asyncio.create_task(start_camera_server())
    video_task = asyncio.create_task(start_video_server())

    # 잠시 대기 (소켓 파일 생성 대기)
    await asyncio.sleep(0.5)

    # WebSocket 서버 시작
    async with websockets.serve(handler, "0.0.0.0", WEBSOCKET_PORT,
                                max_size=None, ping_interval=None, ping_timeout=None):
        print(f"🚀 Transport WebSocket server listening on ws://0.0.0.0:{WEBSOCKET_PORT}")
        print(f"✅ Renderer can now connect to Unix sockets")
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())