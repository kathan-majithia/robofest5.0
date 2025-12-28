import cv2
import time
import json
import numpy as np
import threading
import signal
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn

# ---------------- CONFIG ----------------
MODEL_PATH = "gesture_nchw.onnx"
CLASS_INDEX_PATH = "class_indices.json"
IMG_SIZE = 224
CONF_THRESHOLD = 0.6
STREAM_PORT = 8080

# Performance Tuning
STREAM_WIDTH = 640  # Resize output to this width (Lower = Faster)
JPEG_QUALITY = 60   # 0 to 100 (Lower = Faster/Less Lag)
# ---------------------------------------

# ---------- Load ONNX model ----------
try:
    net = cv2.dnn.readNetFromONNX(MODEL_PATH)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    print("ONNX model loaded")
except Exception as e:
    print(f"Error loading model: {e}")
    sys.exit(1)

# ---------- Load class labels ----------
try:
    with open(CLASS_INDEX_PATH) as f:
        class_indices = json.load(f)
    idx_to_class = {v: k for k, v in class_indices.items()}
except Exception as e:
    idx_to_class = {}

# ---------- Camera ----------
cap = cv2.VideoCapture(0)
# Optional: Force camera to a lower resolution to save CPU from the start
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("ERROR: Camera not accessible")
    sys.exit(1)

# ---------- Globals for streaming ----------
output_frame = None
lock = threading.Lock()

# ---------- Graceful exit ----------
def handle_exit(sig, frame):
    print("\nStopping...")
    cap.release()
    sys.exit(0)

signal.signal(signal.SIGINT, handle_exit)

# ---------- HTML Template (Visuals Only - No CPU Cost) ----------
PAGE_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Team Arcnet - Command Center</title>
    <style>
        :root {
            --primary: #00e5ff;
            --secondary: #ff3d00;
            --bg: #0b0c10;
            --panel: #1f2833;
            --text: #c5c6c7;
        }
        body {
            background-color: var(--bg);
            color: var(--text);
            font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
            margin: 0;
            padding: 0;
            display: flex;
            flex-direction: column;
            align-items: center;
            min-height: 100vh;
        }
        header {
            width: 100%;
            background: linear-gradient(90deg, #1f2833 0%, #000 100%);
            padding: 15px 0;
            text-align: center;
            border-bottom: 2px solid var(--primary);
        }
        h1 { margin: 0; font-size: 2rem; color: #fff; text-transform: uppercase; letter-spacing: 2px; }
        .sub-header { color: var(--primary); font-size: 1.1rem; font-weight: 600; text-transform: uppercase; }
        .challenge-name { color: var(--secondary); font-style: italic; font-size: 0.85rem; margin-top: 5px; }
        .main-container {
            margin-top: 30px;
            padding: 15px;
            background-color: var(--panel);
            border-radius: 8px;
            border: 1px solid #333;
            text-align: center;
        }
        .video-wrapper {
            border: 2px solid #45a29e;
            border-radius: 4px;
            overflow: hidden;
            background: #000;
        }
        img { display: block; max-width: 100%; height: auto; }
        .status-bar {
            margin-top: 10px;
            display: flex;
            justify-content: space-between;
            font-family: 'Courier New', monospace;
            font-size: 0.8rem;
            color: #66fcf1;
        }
    </style>
</head>
<body>
    <header>
        <h1>Robofest 5.0</h1>
        <div class="sub-header">Team Arcnet</div>
        <div class="challenge-name">Aerial Robotics: Minefield Navigation Challenge</div>
    </header>

    <div class="main-container">
        <div class="video-wrapper">
            <img src="/stream" alt="Live Feed" width="640">
        </div>
        <div class="status-bar">
            <span>SYSTEM: ONLINE</span>
            <span>LATENCY: OPTIMIZED</span>
        </div>
    </div>
</body>
</html>
"""

class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    """Handle requests in a separate thread."""

class StreamHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(PAGE_HTML.encode('utf-8'))

        elif self.path == '/stream':
            self.send_response(200)
            self.send_header('Content-type', 'multipart/x-mixed-replace; boundary=frame')
            self.end_headers()
            try:
                while True:
                    with lock:
                        if output_frame is None:
                            continue

                        # --- OPTIMIZATION: Resize & Compress ---
                        # 1. Resize for transmission (does not affect AI logic)
                        h, w = output_frame.shape[:2]
                        if w > STREAM_WIDTH:
                            scale = STREAM_WIDTH / w
                            dim = (STREAM_WIDTH, int(h * scale))
                            # INTER_NEAREST is fastest, though slightly jagged
                            stream_img = cv2.resize(output_frame, dim, interpolation=cv2.INTER_NEAREST)
                        else:
                            stream_img = output_frame

                        # 2. Compress JPEG (Quality vs Size trade-off)
                        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
                        ret, jpg = cv2.imencode('.jpg', stream_img, encode_param)

                    if not ret:
                        continue

                    frame_bytes = jpg.tobytes()
                    self.wfile.write(b"--frame\r\n")
                    self.send_header("Content-Type", "image/jpeg")
                    self.send_header("Content-Length", len(frame_bytes))
                    self.end_headers()
                    self.wfile.write(frame_bytes)

                    # Prevent zero-delay loop
                    time.sleep(0.015)
            except Exception:
                pass
        else:
            self.send_error(404)

StreamServer = ThreadedHTTPServer(("0.0.0.0", STREAM_PORT), StreamHandler)
threading.Thread(target=StreamServer.serve_forever, daemon=True).start()

print(f"DASHBOARD: http://localhost:{STREAM_PORT}")

# ---------- Main loop ----------
prev_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    # --- Preprocess (Standard) ---
    blob = cv2.dnn.blobFromImage(
        frame, scalefactor=1.0 / 255.0, size=(IMG_SIZE, IMG_SIZE),
        mean=(0, 0, 0), swapRB=True, crop=False
    )

    # --- Inference ---
    net.setInput(blob)
    preds = net.forward()[0]
    cid = int(np.argmax(preds))
    conf = float(preds[cid])
    gesture = idx_to_class.get(cid, "Unknown") if conf > CONF_THRESHOLD else "..."

    # --- FPS ---
    curr_time = time.time()
    fps = 1.0 / (curr_time - prev_time)
    prev_time = curr_time

    # --- Draw Minimal Overlay (Fast) ---
    # We draw directly on the frame without fancy transparency math
    h, w = frame.shape[:2]

    # Simple solid black box at bottom for text readability
    cv2.rectangle(frame, (0, h-40), (w, h), (0, 0, 0), -1)

    # Text info
    info_text = f"CMD: {gesture.upper()} | CONF: {conf:.2f} | FPS: {int(fps)}"
    cv2.putText(frame, info_text, (10, h-12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

    # --- Share frame with streamer ---
    with lock:
        output_frame = frame.copy()

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()