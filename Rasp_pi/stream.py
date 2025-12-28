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
# ---------------------------------------

# ---------- Load ONNX model ----------
# (Ensure your model and json files are in the same directory)
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
    print("Classes:", idx_to_class)
except Exception as e:
    print(f"Error loading class indices: {e}")
    # Fallback if file missing
    idx_to_class = {}

# ---------- Camera ----------
cap = cv2.VideoCapture(0)
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

# ---------- HTML Template ----------
# This HTML includes CSS for the Robofest/Arcnet theme
PAGE_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Team Arcnet - Command Center</title>
    <style>
        :root {
            --primary: #00e5ff; /* Cyan Arcnet color */
            --secondary: #ff3d00; /* Minefield Orange */
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
            padding: 20px 0;
            text-align: center;
            border-bottom: 2px solid var(--primary);
            box-shadow: 0 0 15px rgba(0, 229, 255, 0.2);
        }
        h1 {
            margin: 0;
            font-size: 2.5rem;
            color: #fff;
            text-transform: uppercase;
            letter-spacing: 3px;
        }
        .sub-header {
            color: var(--primary);
            font-size: 1.2rem;
            font-weight: 600;
            margin-top: 5px;
            text-transform: uppercase;
            letter-spacing: 1.5px;
        }
        .challenge-name {
            color: var(--secondary);
            font-style: italic;
            font-size: 0.9rem;
            margin-top: 5px;
        }
        .main-container {
            margin-top: 40px;
            padding: 20px;
            background-color: var(--panel);
            border-radius: 10px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.5);
            border: 1px solid #333;
            text-align: center;
        }
        .video-wrapper {
            position: relative;
            border: 2px solid #45a29e;
            display: inline-block;
            border-radius: 4px;
            overflow: hidden;
            box-shadow: 0 0 20px rgba(0, 229, 255, 0.1);
        }
        img {
            display: block;
            max-width: 100%;
            height: auto;
        }
        .status-bar {
            margin-top: 15px;
            display: flex;
            justify-content: space-between;
            font-family: 'Courier New', monospace;
            font-size: 0.9rem;
            color: #66fcf1;
        }
        .badge {
            background: #333;
            padding: 5px 10px;
            border-radius: 3px;
        }
        footer {
            margin-top: auto;
            padding: 20px;
            font-size: 0.8rem;
            color: #666;
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
            <img src="/stream" alt="Live Feed" width="800">
        </div>
        <div class="status-bar">
            <span class="badge">SYSTEM: ONLINE</span>
            <span class="badge">MODE: GESTURE CONTROL</span>
            <span class="badge">LATENCY: LOW</span>
        </div>
    </div>

    <footer>
        &copy; 2024 Team Arcnet. All Systems Nominal.
    </footer>
</body>
</html>
"""

# ---------- MJPEG Server with Threading ----------
# We need ThreadingMixIn so the server can handle the HTML request
# AND the video stream request simultaneously.
class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    """Handle requests in a separate thread."""

class StreamHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        # Route 1: The Website (Root URL)
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(PAGE_HTML.encode('utf-8'))

        # Route 2: The Video Stream
        elif self.path == '/stream':
            self.send_response(200)
            self.send_header('Content-type', 'multipart/x-mixed-replace; boundary=frame')
            self.end_headers()
            try:
                while True:
                    with lock:
                        if output_frame is None:
                            continue
                        # Encode the frame in JPEG format
                        ret, jpg = cv2.imencode('.jpg', output_frame)

                    if not ret:
                        continue

                    frame_bytes = jpg.tobytes()
                    self.wfile.write(b"--frame\r\n")
                    self.send_header("Content-Type", "image/jpeg")
                    self.send_header("Content-Length", len(frame_bytes))
                    self.end_headers()
                    self.wfile.write(frame_bytes)

                    # Control stream framerate slightly to prevent browser lag
                    time.sleep(0.01)
            except Exception as e:
                # Browser closed connection
                pass
        else:
            self.send_error(404)

class StreamServer(threading.Thread):
    def run(self):
        # Listen on all interfaces
        server = ThreadedHTTPServer(("0.0.0.0", STREAM_PORT), StreamHandler)
        print(f"--------------------------------------------------")
        print(f" DASHBOARD ONLINE: http://localhost:{STREAM_PORT}")
        print(f"--------------------------------------------------")
        server.serve_forever()

# ---------- Start server ----------
StreamServer(daemon=True).start()

# ---------- Main loop ----------
prev_time = time.time()

# Colors for overlay (BGR)
COLOR_TEXT = (255, 255, 255)
COLOR_BOX = (50, 50, 50)
COLOR_ACCENT = (0, 255, 255) # Yellow/Cyan

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    # --- Preprocess ---
    blob = cv2.dnn.blobFromImage(
        frame,
        scalefactor=1.0 / 255.0,
        size=(IMG_SIZE, IMG_SIZE),
        mean=(0, 0, 0),
        swapRB=True,
        crop=False
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

    # --- Draw Modern Overlay ---
    h, w, _ = frame.shape

    # 1. Create a translucent footer bar on the video feed
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h-60), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    # 2. Draw Text (Gesture, Confidence, FPS)
    # Gesture (Left)
    cv2.putText(frame, f"CMD: {gesture.upper()}", (20, h-25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Confidence (Center)
    cv2.putText(frame, f"CONF: {conf:.0%}", (w//2 - 50, h-25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    # FPS (Right)
    cv2.putText(frame, f"FPS: {int(fps)}", (w - 120, h-25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)

    # 3. Target Reticle (Optional - adds 'Drone' feel)
    cx, cy = w // 2, h // 2
    cv2.line(frame, (cx-20, cy), (cx+20, cy), (255, 255, 255), 1)
    cv2.line(frame, (cx, cy-20), (cx, cy+20), (255, 255, 255), 1)
    cv2.circle(frame, (cx, cy), 150, (255, 255, 255), 1)

    # --- Share frame with streamer ---
    with lock:
        output_frame = frame.copy()

    # Optional: Display local window for debugging
    # cv2.imshow("Local Feed", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()