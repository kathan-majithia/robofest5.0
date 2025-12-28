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

# Camera & Stream Tuning (Crucial for Pi Stability)
CAM_WIDTH = 640
CAM_HEIGHT = 480
CAM_FPS = 30
JPEG_QUALITY = 60       # Balance between speed and quality
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
except Exception:
    idx_to_class = {}

# ---------- Camera Setup (Stable V4L2) ----------
print("Opening camera with V4L2 backend...")
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)

# Force MJPG to prevent USB bandwidth freeze
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
cap.set(cv2.CAP_PROP_FPS, CAM_FPS)

if not cap.isOpened():
    print("ERROR: Camera not accessible")
    sys.exit(1)

# ---------- Globals ----------
output_frame = None
lock = threading.Lock()

# ---------- Graceful Exit ----------
def handle_exit(sig, frame):
    print("\nStopping...")
    cap.release()
    sys.exit(0)
signal.signal(signal.SIGINT, handle_exit)

# ---------- MJPEG Server ----------
class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    """Handle requests in a separate thread."""

class StreamHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        # Serve the raw video stream at root '/'
        if self.path == '/' or self.path == '/stream':
            self.send_response(200)
            self.send_header('Content-type', 'multipart/x-mixed-replace; boundary=frame')
            self.end_headers()
            try:
                while True:
                    with lock:
                        if output_frame is None:
                            continue

                        # Encode to JPEG
                        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
                        ret, jpg = cv2.imencode('.jpg', output_frame, encode_param)

                    if not ret:
                        continue

                    frame_bytes = jpg.tobytes()
                    self.wfile.write(b"--frame\r\n")
                    self.send_header("Content-Type", "image/jpeg")
                    self.send_header("Content-Length", len(frame_bytes))
                    self.end_headers()
                    self.wfile.write(frame_bytes)

                    time.sleep(0.015) # Small delay to prevent browser buffer overflow
            except Exception:
                pass
        else:
            self.send_error(404)

# Start Server
StreamServer = ThreadedHTTPServer(("0.0.0.0", STREAM_PORT), StreamHandler)
threading.Thread(target=StreamServer.serve_forever, daemon=True).start()
print(f"Stream running at: http://0.0.0.0:{STREAM_PORT}")

# ---------- Main Loop ----------
prev_time = time.time()

while True:
    ret, frame = cap.read()

    # Auto-reconnect if camera freezes
    if not ret:
        print("Camera freeze detected. Reconnecting...")
        cap.release()
        time.sleep(1)
        cap.open(0, cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
        continue

    # --- 1. Preprocess ---
    blob = cv2.dnn.blobFromImage(
        frame, scalefactor=1.0 / 255.0, size=(IMG_SIZE, IMG_SIZE),
        mean=(0, 0, 0), swapRB=True, crop=False
    )

    # --- 2. Inference ---
    net.setInput(blob)
    preds = net.forward()[0]

    cid = int(np.argmax(preds))
    conf = float(preds[cid])
    gesture = idx_to_class.get(cid, "Unknown") if conf > CONF_THRESHOLD else "..."

    # --- 3. FPS Calculation ---
    curr_time = time.time()
    fps = 1.0 / (curr_time - prev_time)
    prev_time = curr_time

    # --- 4. Draw Overlays (The data you wanted to keep) ---
    # Draw Gesture
    cv2.putText(frame, f"Gesture: {gesture}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Draw Confidence
    cv2.putText(frame, f"Conf: {conf:.2f}", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    # Draw FPS
    cv2.putText(frame, f"FPS: {fps:.1f}", (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

    # --- 5. Update Stream ---
    with lock:
        output_frame = frame.copy()

    # Optional local display
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()