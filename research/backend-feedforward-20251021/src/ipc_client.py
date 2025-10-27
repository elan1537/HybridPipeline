"""
IPC Client for Transport Service
Connects to 4DGS Feed-Forward Renderer via Unix Domain Socket + Shared Memory
"""
import socket
import struct
import os
from multiprocessing import shared_memory
from multiprocessing import resource_tracker
import threading
from queue import Queue
import time

SOCK_HDR_FMT = "<BQI"
SOCK_HDR_SIZE = struct.calcsize(SOCK_HDR_FMT)
SHM_HDR_FMT  = "<QfIdb"
SHM_HDR_SIZE = struct.calcsize(SHM_HDR_FMT)
READY_OFFSET = struct.calcsize("<QfId")
DATA_OFFSET  = 64


class IPCClient:
    def __init__(self, socket_path: str, shm_name: str):
        self.socket_path = socket_path
        self.shm_name = shm_name
        self.shm_size = 20 * 1024 * 1024
        self.num_slots = 4
        self.slot_size = self.shm_size // self.num_slots

        self.shm = None
        self.socket = None
        self.connected = False

        self.frame_queue = Queue(maxsize=10)
        self.running = True

        self.recv_thread = None

    def connect(self, timeout=10):
        start_time = time.time()

        print(f"Waiting for renderer socket: {self.socket_path}")
        while not os.path.exists(self.socket_path):
            if time.time() - start_time > timeout:
                print(f"Timeout: Socket not found after {timeout}s")
                return False
            time.sleep(0.1)

        try:
            self.socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            self.socket.setblocking(True)
            # Optional: enlarge recv buffer to reduce syscalls when frames are frequent
            try:
                self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1 << 20)
            except Exception:
                pass
            self.socket.connect(self.socket_path)
            print(f"Connected to renderer socket: {self.socket_path}")
        except Exception as e:
            print(f"Failed to connect to socket: {e}")
            return False

        try:
            self.shm = shared_memory.SharedMemory(name=self.shm_name)
            resource_tracker.unregister(self.shm._name, "shared_memory")
            print(f"Attached to shared memory: {self.shm_name} ({self.shm.size // 1024 // 1024}MB)")
        except FileNotFoundError:
            print(f"Shared memory not found: {self.shm_name}")
            self.socket.close()
            return False

        self.connected = True

        self.recv_thread = threading.Thread(target=self._receive_loop, daemon=True)
        self.recv_thread.start()
        print("IPC Client ready, receiving frames...")

        return True

    def _recv_exact(self, n: int):
        """Receive exactly n bytes from a stream socket or return None on EOF/timeout."""
        buf = bytearray()
        while len(buf) < n and self.running and self.connected:
            try:
                chunk = self.socket.recv(n - len(buf))
                if not chunk:
                    # EOF or peer closed
                    return None
                buf += chunk
            except InterruptedError:
                continue
            except BlockingIOError:
                # Non-blocking transient; small sleep avoids busy loop
                time.sleep(0.001)
                continue
            except OSError as e:
                print(f"Recv error: {e}")
                return None
        return bytes(buf) if len(buf) == n else None

    def _receive_loop(self):
        while self.running and self.connected:
            try:
                header = self._recv_exact(SOCK_HDR_SIZE)
                if header is None:
                    print("Renderer disconnected or failed to read header")
                    self.connected = False
                    break
                cmd, frame_id, slot = struct.unpack(SOCK_HDR_FMT, header)

                if cmd == 0x01:
                    frame_data = self._read_frame_from_shm(slot)
                    if frame_data is None:
                        # brief retry to bridge tiny races on first cycle / heavy load
                        for _ in range(5):
                            time.sleep(0.0005)
                            frame_data = self._read_frame_from_shm(slot)
                            if frame_data:
                                break
                    if frame_data is not None:
                        sz = len(frame_data['video_bitstream']) if 'video_bitstream' in frame_data else -1
                        # Filter warmup frames (typically 1 byte) - real H.264 frames are 100+ bytes
                        if sz < 100:
                            print(f"Skipped warmup frame {frame_id} from slot {slot} ({sz} bytes)")
                        else:
                            self.frame_queue.put(frame_data)
                            print(f"Received frame {frame_id} from slot {slot} ({sz} bytes)")
                    else:
                        print(f"Dropped notification for slot {slot}: not ready after retry")

            except Exception as e:
                print(f"Receive error: {e}")
                self.connected = False
                break

    def _read_frame_from_shm(self, slot):
        try:
            slot_offset = slot * self.slot_size

            # Wait for ready=1 with a short bounded spin (avoids racing payload write)
            t0 = time.time()
            ready = 0
            for _ in range(1000):  # ~1ms at 1us sleep, adjust as needed
                ready = int(self.shm.buf[slot_offset + READY_OFFSET])
                if ready == 1:
                    break
                # tiny sleep to avoid busy spinning; shared memory has no blocking prims
                time.sleep(0.0005)
            if ready != 1:
                # Not ready; skip
                return None

            # Re-read full header now that ready==1 to get a consistent size
            header_bytes = bytes(self.shm.buf[slot_offset:slot_offset + SHM_HDR_SIZE])
            frame_id, time_index, frame_size, timestamp, ready_flag = struct.unpack(SHM_HDR_FMT, header_bytes)
            if ready_flag != 1:
                return None
            if frame_size <= 0:
                # consume and clear, but return an explicit empty frame to avoid misleading 'not ready'
                try:
                    self.shm.buf[slot_offset + READY_OFFSET:slot_offset + READY_OFFSET + 1] = b"\x00"
                except Exception:
                    pass
                return {
                    'frame_id': frame_id,
                    'time_index': time_index,
                    'video_bitstream': b"",
                    'timestamp': timestamp,
                    'slot': slot
                }

            data_offset = slot_offset + DATA_OFFSET
            video_bitstream = bytes(self.shm.buf[data_offset:data_offset + frame_size])

            # Mark slot as consumed (ready=0) to avoid stale/first-cycle effects
            try:
                self.shm.buf[slot_offset + READY_OFFSET:slot_offset + READY_OFFSET + 1] = b"\x00"
            except Exception:
                pass

            return {
                'frame_id': frame_id,
                'time_index': time_index,
                'video_bitstream': video_bitstream,
                'timestamp': timestamp,
                'slot': slot
            }
        except Exception as e:
            print(f"Error reading from shared memory: {e}")
            return None

    def get_frame(self, timeout=None):
        try:
            return self.frame_queue.get(timeout=timeout)
        except:
            return None

    def get_frame_nowait(self):
        try:
            return self.frame_queue.get_nowait()
        except:
            return None

    def is_connected(self):
        return self.connected

    def close(self):
        self.running = False
        self.connected = False

        if self.recv_thread and self.recv_thread.is_alive():
            self.recv_thread.join(timeout=1.0)

        if self.socket:
            try:
                self.socket.close()
            except:
                pass

        if self.shm:
            try:
                self.shm.close()
            except:
                pass

        print("IPC Client closed")


if __name__ == "__main__":
    client = IPCClient(
        socket_path="/run/ipc/renderer.sock",
        shm_name="3dgstream_frames"
    )

    if not client.connect(timeout=10):
        print("Failed to connect to renderer")
        exit(1)

    print("Receiving frames...")
    frame_count = 0

    try:
        while True:
            frame = client.get_frame()

            if frame is None:
                print("Timeout waiting for frame")
                break

            print(f"Frame {frame['frame_id']}: "
                  f"{len(frame['video_bitstream'])} bytes, "
                  f"time_index={frame['time_index']:.4f}, "
                  f"slot={frame['slot']}")

            frame_count += 1

            if frame_count >= 10:
                print("Received 10 frames, stopping test")
                break

    except KeyboardInterrupt:
        print("Interrupted by user")

    finally:
        client.close()
        print(f"Test completed: {frame_count} frames received")
