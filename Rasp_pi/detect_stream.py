import cv2
import time
import json
import numpy as np
import threading
import signal
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer

# ---------------- CONFIG ----------------
MODEL_PATH = "gesture_nchw.onnx"
CLASS_INDEX_PATH = "class_indices.json"
IMG_SIZE = 224
CONF_THRESHOLD = 0.6
STREAM_PORT = 8080
# ---------------------------------------

# ---------- Load ONNX model ----------
net = cv2.dnn.readNetFromONNX(MODEL_PATH)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
print("ONNX model loaded")

# ---------- Load class labels ----------
with open(CLASS_INDEX_PATH) as f:
    class_indices = json.load(f)
idx_to_class = {v: k for k, v in class_indices.items()}
print("Classes:", idx_to_class)

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

# ---------- MJPEG Server ----------
class StreamHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header(
            'Content-type',
            'multipart/x-mixed-replace; boundary=frame'
        )
        self.end_headers()

        while True:
            with lock:
                if output_frame is None:
                    continue
                ret, jpg = cv2.imencode('.jpg', output_frame)
                if not ret:
                    continue
                frame_bytes = jpg.tobytes()

            self.wfile.write(b"--frame\r\n")
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", len(frame_bytes))
            self.end_headers()
            self.wfile.write(frame_bytes)

class StreamServer(threading.Thread):
    def run(self):
        server = HTTPServer(("0.0.0.0", STREAM_PORT), StreamHandler)
        print(f"MJPEG stream running at http://0.0.0.0:{STREAM_PORT}")
        server.serve_forever()

# ---------- Start server ----------
StreamServer(daemon=True).start()

# ---------- Main loop ----------
prev_time = time.time()

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
    gesture = idx_to_class[cid] if conf > CONF_THRESHOLD else "Unknown"

    # --- FPS ---
    curr_time = time.time()
    fps = 1.0 / (curr_time - prev_time)
    prev_time = curr_time

    # --- Draw overlay ---
    cv2.putText(frame, f"Gesture: {gesture}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.putText(frame, f"Conf: {conf:.2f}", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    cv2.putText(frame, f"FPS: {fps:.1f}", (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

    # --- Share frame with streamer ---
    with lock:
        output_frame = frame.copy()