# JPEG Image Transfer Test

## Test Steps

Terminal 1 - Transport:
```bash
cd backend/src
python server-fifo.py
```

Terminal 2 - Renderer:
```bash
cd research/3DGStream
python feed-forward-renderer-socket.py
```

Terminal 3 - Camera Sender:
```bash
cd backend
python test_camera_sender.py
```

## Check Saved Images

Renderer output:
```bash
ls -lh research/3DGStream/renderer_output/
```

Transport output:
```bash
ls -lh backend/transport_output/
```

View images:
```bash
eog research/3DGStream/renderer_output/frame_000001_color.jpg
eog backend/transport_output/frame_000001_color.jpg
```

## Verify Transfer

```bash
cd backend
python verify_transfer.py
```

Expected output:
```
Renderer: 2 files
Transport: 2 files
✅ All files match! Transfer successful.
```

## Protocol

Header: frame_id(8) + color_size(4) + depth_size(4) = 16 bytes
Body: color_jpeg + depth_jpeg
