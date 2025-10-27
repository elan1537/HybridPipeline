# Transport Service - Usage Guide

## 🎯 Overview

Transport Service mediates between Frontend (WebSocket) and Renderer (Unix Socket).

**Architecture:**
```
Frontend (Browser) ↔ WebSocket ↔ Transport Service ↔ Unix Socket ↔ Renderer Service
```

---

## 🚀 Quick Start

### Terminal 1: Start Renderer Service

```bash
cd /workspace/backend

python -m renderer.main \
  --scene-type static \
  --model-path livinglab-no-ur5.ply \
  --encoder-type jpeg
```

**Expected output:**
```
[INIT] Initializing Renderer Service...
[INIT] Loading Gaussian scene: 780,112 Gaussians
[INIT] Encoder initialized: JPEG (OpenCV, quality=90)
[INIT] Waiting for Transport Service...
```

---

### Terminal 2: Start Transport Service

```bash
cd /workspace/backend

python -m transport.main \
  --host 0.0.0.0 \
  --port 8765 \
  --camera-sock /run/ipc/camera.sock \
  --video-sock /run/ipc/video.sock
```

**Expected output:**
```
============================================================
Transport Service - HybridPipeline
============================================================

Configuration:
  WebSocket:     ws://0.0.0.0:8765
  Camera Socket: /run/ipc/camera.sock
  Video Socket:  /run/ipc/video.sock
  Viewport:      640x480

============================================================

[Transport] Starting Transport Service...
[UnixSocket] Created socket directory: /run/ipc
[UnixSocket] Camera socket listening: /run/ipc/camera.sock
[UnixSocket] Video socket listening: /run/ipc/video.sock
[UnixSocket] Waiting for Renderer to connect...
[UnixSocket] Renderer connected successfully
[Transport] Connected to Renderer successfully
[WebSocket] Starting server on 0.0.0.0:8765...
[WebSocket] Server started, waiting for connections...
[Transport] Camera forward loop started
[Transport] Video broadcast loop started
```

---

### Terminal 3: Run Mock Client

```bash
cd /workspace/backend

python test_client.py \
  --host localhost \
  --port 8765 \
  --frames 10 \
  --output-dir output
```

**Expected output:**
```
============================================================
Mock WebSocket Client - HybridPipeline
============================================================
  Server:   ws://localhost:8765
  Frames:   10
  Output:   output
  Viewport: 640x480
============================================================

[Client] Connecting to ws://localhost:8765...
[Client] Connected successfully
[Client] Sent handshake: 640x480

[Client] Frame 0:
  📤 Sent camera data (160 bytes)
  ✅ Frame 0: color=650234 bytes, depth=461824 bytes

[Client] Frame 1:
  📤 Sent camera data (160 bytes)
  ✅ Frame 1: color=648901 bytes, depth=461824 bytes

...

============================================================
[Client] Test completed!
  Sent:     10 frames
  Received: 10 frames
  Success:  100.0%
  Output:   output
============================================================
```

---

### Verify Results

```bash
ls output/

# Output:
# color_000000.jpg  depth_000000.bin
# color_000001.jpg  depth_000001.bin
# ...

# View images
eog output/color_000000.jpg
```

---

## 📋 Command Reference

### Transport Service

```bash
python -m transport.main [OPTIONS]
```

**Options:**
- `--host HOST`: WebSocket server host (default: 0.0.0.0)
- `--port PORT`: WebSocket server port (default: 8765)
- `--camera-sock PATH`: Unix socket for camera (default: /run/ipc/camera.sock)
- `--video-sock PATH`: Unix socket for video (default: /run/ipc/video.sock)
- `--width WIDTH`: Default viewport width (default: 640)
- `--height HEIGHT`: Default viewport height (default: 480)

**Examples:**
```bash
# Custom port
python -m transport.main --port 9000

# Custom socket paths
python -m transport.main \
  --camera-sock /tmp/camera.sock \
  --video-sock /tmp/video.sock

# Custom resolution
python -m transport.main --width 1280 --height 720
```

---

### Mock Client

```bash
python backend/test_client.py [OPTIONS]
```

**Options:**
- `--host HOST`: WebSocket server host (default: localhost)
- `--port PORT`: WebSocket server port (default: 8765)
- `--frames N`: Number of frames to send (default: 10)
- `--output-dir DIR`: Output directory (default: output)
- `--width WIDTH`: Viewport width (default: 640)
- `--height HEIGHT`: Viewport height (default: 480)

**Examples:**
```bash
# Send 100 frames
python backend/test_client.py --frames 100

# Custom server
python backend/test_client.py --host 192.168.1.100 --port 9000

# Custom output directory
python backend/test_client.py --output-dir results/test1/

# HD resolution
python backend/test_client.py --width 1280 --height 720
```

---

## 🔧 Troubleshooting

### Error: "Failed to connect to Renderer"

**Cause:** Renderer Service not running

**Solution:**
1. Start Renderer Service first (Terminal 1)
2. Wait for "Waiting for Transport Service..." message
3. Then start Transport Service (Terminal 2)

---

### Error: "WebSocket connection refused"

**Cause:** Transport Service not running or port in use

**Solution:**
```bash
# Check if port is in use
lsof -i :8765

# Kill process if needed
kill -9 <PID>

# Use different port
python -m transport.main --port 9000
```

---

### Error: "Permission denied: /run/ipc/camera.sock"

**Cause:** No permission to create socket in /run/ipc

**Solution:**
```bash
# Use alternative directory
python -m transport.main \
  --camera-sock /tmp/camera.sock \
  --video-sock /tmp/video.sock

# Or create directory with proper permissions
sudo mkdir -p /run/ipc
sudo chown $USER:$USER /run/ipc
```

---

### No frames received

**Cause:** Renderer Service initialization failed or not rendering

**Solution:**
1. Check Renderer Service logs for errors
2. Verify PLY file path is correct
3. Check GPU availability: `nvidia-smi`

---

## 📊 Protocol Details

### Camera Data (Frontend → Transport → Renderer)

**Frontend Protocol (160 bytes):**
- [0:128] - Camera data (32 floats)
- [128:132] - frame_id (uint32)
- [132:136] - padding
- [136:144] - client_timestamp (float64)
- [144:148] - time_index (float32)
- [148:160] - padding

**Renderer Protocol (168 bytes):**
- [0:64] - view_matrix (4×4 float32)
- [64:100] - intrinsics (3×3 float32)
- [100:168] - metadata (width, height, near, far, timestamps, etc.)

---

### Video Data (Renderer → Transport → Frontend)

**Renderer Protocol (56 bytes header + data):**
- [0:4] - frame_id (uint32)
- [4:5] - format_type (uint8: 0=JPEG, 1=H264, 2=Raw)
- [5:8] - padding
- [8:12] - color_len (uint32)
- [12:16] - depth_len (uint32)
- [16:20] - width (uint32)
- [20:24] - height (uint32)
- [24:56] - timestamps (4 × float64)

**Frontend Protocol - JPEG (48 bytes header + data):**
- [0:4] - jpegLen (uint32)
- [4:8] - depthLen (uint32)
- [8:12] - frameId (uint32)
- [12:48] - timestamps (4 × float64)

**Frontend Protocol - H.264 (44 bytes header + data):**
- [0:4] - videoLen (uint32)
- [4:8] - frameId (uint32)
- [8:44] - timestamps (4 × float64)

---

## 🎨 Architecture

### Protocol Adapter Pattern

```
TransportService
├── Frontend Adapter (WebSocket)
│   ├── recv_camera(): 160 bytes → CameraFrame
│   └── send_video(): RenderPayload → 44/48 bytes + data
└── Backend Adapter (Unix Socket)
    ├── send_camera(): CameraFrame → 168 bytes
    └── recv_video(): 56 bytes + data → RenderPayload
```

### Data Flow

```
Mock Client
    ↓ WebSocket (160 bytes camera)
WebSocketAdapter
    ↓ parse_frontend_camera()
TransportService
    ↓ camera_forward_loop()
UnixSocketAdapter
    ↓ Unix Socket (168 bytes camera)
Renderer Service
    ↓ render() + encode()
    ↓ Unix Socket (56 bytes header + data)
UnixSocketAdapter
    ↓ parse_render_payload()
TransportService
    ↓ video_broadcast_loop()
WebSocketAdapter
    ↓ create_frontend_video_header()
    ↓ WebSocket (44/48 bytes header + data)
Mock Client
    → Save to output/
```

---

## ✅ Success Checklist

- [ ] Renderer Service running
- [ ] Transport Service connected to Renderer
- [ ] WebSocket server started
- [ ] Mock Client connected
- [ ] Frames sent and received
- [ ] Results saved in `output/` directory
- [ ] Images viewable

---

## 📚 Next Steps

1. **Test with real Frontend** (browser-based client)
2. **Add WebRTC adapter** for P2P communication
3. **Add UDP adapter** for low-latency streaming
4. **Implement multi-client support**
5. **Add error recovery and reconnection logic**

---

## 🐛 Debug Tips

### Enable verbose logging

```python
# Add to service.py
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Monitor socket connections

```bash
# Watch socket files
watch -n 1 'ls -lh /run/ipc/'

# Monitor network connections
netstat -an | grep 8765
```

### Check frame processing

```bash
# Monitor output directory
watch -n 1 'ls -lh output/ | tail -10'

# Count frames
ls output/*.jpg | wc -l
```

---

For more information, see:
- `architecture.md` - Overall system architecture
- `CLAUDE.md` - Project memory and progress
- `backend/renderer/README.md` - Renderer Service documentation
