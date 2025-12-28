import cv2
import time
import json
import numpy as np
import threading
import signal
import sys
import os
import select # Needed for keyboard input on Linux
from http.server import BaseHTTPRequestHandler, HTTPServer

# Import Dronekit
try:
    from dronekit import connect, VehicleMode
except ImportError:
    print("Error: dronekit not installed. Please install: pip3 install dronekit")
    sys.exit(1)

# ====== CONFIGURATION ======
# --- Vision Config ---
MODEL_PATH = "gesture_nchw.onnx"
# Ensure this file contains: {"fist": 0, "idle": 1, "open_palm": 2, "thumb_up": 3, "two_fingers": 4}
CLASS_INDEX_PATH = "class_indices.json"
IMG_SIZE = 224
CONF_THRESHOLD = 0.6
STREAM_PORT = 8080

# --- Drone Config ---
DRONE_PORT = "/dev/ttyACM0"
BAUD_RATE = 115200
THROTTLE_CHANNEL = '3'

# Throttle PWM Mapping based on gestures
PWM_PALM = 1150       # For "open_palm"
PWM_FIST = 1125       # For "fist"
PWM_THUMBS_UP = 1100  # For "thumb_up"
PWM_STOP = 1000       # For "two_fingers" or 'q'

# ====== SHARED GLOBAL STATE ======
# For MJPEG streaming
output_frame = None
frame_lock = threading.Lock()

# For Gesture-to-Drone communication
shared_gesture = "Unknown"
gesture_lock = threading.Lock()

# Global flag to cleanly stop all threads
stop_event = threading.Event()

# ====== DRONE HELPER FUNCTIONS ======
def force_arm(vehicle):
    print("\n=== FORCE ARMING ===")
    # Switch to Stabilize mode first to allow arming without GPS lock if needed
    vehicle.mode = VehicleMode("STABILIZE")
    time.sleep(1)

    print("Sending Magic Arm command...")
    # MAV_CMD_COMPONENT_ARM_DISARM = 400, Arm = 1, Magic Code = 21196
    vehicle._master.mav.command_long_send(
        vehicle._master.target_system,
        vehicle._master.target_component,
        400, 0, 1, 21196, 0, 0, 0, 0, 0
    )

    # Wait for arming
    start_time = time.time()
    while not vehicle.armed:
        if time.time() - start_time > 10:
            print("  Arming timed out!")
            return False
        print("  Waiting for forced arm...")
        time.sleep(1)

    print("Vehicle ARMED.\n")
    return True

def force_disarm(vehicle):
    if vehicle and vehicle.armed:
        print("\n=== FORCE DISARMING ===")
        # MAV_CMD_COMPONENT_ARM_DISARM = 400, Disarm = 0, Magic Code = 21196
        vehicle._master.mav.command_long_send(
            vehicle._master.target_system,
            vehicle._master.target_component,
            400, 0, 0, 21196, 0, 0, 0, 0, 0
        )
        time.sleep(2)
        print("Vehicle Disarmed command sent.")

def set_throttle(vehicle, pwm):
    # Only send command if vehicle is connected and armed (for safety)
    if vehicle and vehicle.armed:
        vehicle.channels.overrides[THROTTLE_CHANNEL] = pwm
        # Using carriage return '\r' to update line instead of spamming console
        sys.stdout.write(f"\r[DRONE] Throttle set to: {pwm}   ")
        sys.stdout.flush()

# ====== DRONE CONTROL THREAD ======
def drone_control_loop():
    vehicle = None
    try:
        print(f"[DRONE] Connecting to Pixhawk on {DRONE_PORT}...")
        # wait_ready=False for faster connection, we force arm anyway
        vehicle = connect(DRONE_PORT, baud=BAUD_RATE, wait_ready=False)
        print("[DRONE] Connected.")

        if not force_arm(vehicle):
            print("[DRONE] Failed to arm. Exiting drone thread.")
            stop_event.set()
            return

        # Initial throttle safety
        set_throttle(vehicle, PWM_STOP)
        print("\n[DRONE] Ready for gesture control.")

        # Initialize current state to STOP
        currentState = PWM_STOP

        while not stop_event.is_set():
            # 1. Get the latest gesture safely
            current_gesture_cmd = None
            with gesture_lock:
                current_gesture_cmd = shared_gesture

            # 2. Determine target PWM based on gesture.
            # Start assuming we hold the current state (for "idle" or "Unknown")
            target_pwm = currentState

            # --- UPDATED MAPPING BASED ON NEW JSON ---
            if current_gesture_cmd == "open_palm":
                target_pwm = PWM_PALM
            elif current_gesture_cmd == "fist":
                target_pwm = PWM_FIST
            elif current_gesture_cmd == "thumb_up":
                target_pwm = PWM_THUMBS_UP
            elif current_gesture_cmd == "two_fingers":
                print("\n[DRONE] 'two_fingers' detected. Initiating stop.")
                target_pwm = PWM_STOP
                stop_event.set() # Signal main loop to quit

            # 3. Apply throttle only if the target is different from current state
            if target_pwm != currentState:
                set_throttle(vehicle, target_pwm)
                currentState = target_pwm

            # Control loop rate (e.g., 10Hz)
            time.sleep(0.1)

    except Exception as e:
        print(f"\n[DRONE ERROR] {e}")
    finally:
        # Cleanup routine
        print("\n[DRONE] Cleaning up...")
        if vehicle:
            # Reset channel overrides immediately
            vehicle.channels.overrides = {}
            vehicle.flush()
            time.sleep(0.5)
            force_disarm(vehicle)
            vehicle.close()
        print("[DRONE] Disconnected.")

# ====== KEYBOARD INPUT THREAD (for 'q' quit) ======
# Needed because standard input blocking interferes with other threads on Linux
def keyboard_watcher_thread():
    # Set stdin to non-blocking mode to read 'q' without halting everything
    if os.name == 'posix':
        import termios, tty
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setcbreak(fd)
            while not stop_event.is_set():
                # Check if data is available on stdin
                if select.select([sys.stdin], [], [], 0)[0]:
                    key = sys.stdin.read(1)
                    if key.lower() == 'q':
                        print("\n[KEYBOARD] 'q' pressed. Stopping...")
                        stop_event.set()
                        break
                time.sleep(0.1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    else:
         # Fallback for non-posix systems if needed (unlikely for Pi)
         print("Keyboard watcher not fully supported on this OS.")
         while not stop_event.is_set(): time.sleep(1)

# ====== MJPEG SERVER ======
class StreamHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        # Check if the server should be stopping before processing request
        if stop_event.is_set():
             self.send_error(503, "Server Shutting Down")
             return

        if self.path == '/stream':
            self.send_response(200)
            self.send_header('Content-type', 'multipart/x-mixed-replace; boundary=frame')
            self.end_headers()
            try:
                while not stop_event.is_set():
                    with frame_lock:
                        if output_frame is None:
                            time.sleep(0.01)
                            continue
                        # Compress frame to JPEG
                        ret, jpg = cv2.imencode('.jpg', output_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
                        if not ret: continue
                        frame_bytes = jpg.tobytes()

                    self.wfile.write(b"--frame\r\n")
                    self.send_header("Content-Type", "image/jpeg")
                    self.send_header("Content-Length", len(frame_bytes))
                    self.end_headers()
                    self.wfile.write(frame_bytes)
                    time.sleep(0.04) # Limit stream FPS slightly
            except Exception:
                 pass # Handle client disconnect quietly
        else:
            self.send_error(404)

class StreamServerThread(threading.Thread):
    def run(self):
        httpd = None
        try:
            httpd = HTTPServer(("0.0.0.0", STREAM_PORT), StreamHandler)
            httpd.timeout = 1 # Set socket timeout
            print(f"[STREAM] MJPEG stream running at http://0.0.0.0:{STREAM_PORT}/stream")
            # Serve until stop_event is set.
            while not stop_event.is_set():
                 httpd.handle_request()
        except Exception as e:
             print(f"[STREAM ERROR] {e}")
        finally:
             if httpd: httpd.server_close()
             print("[STREAM] Server stopped.")

# ====== MAIN VISION LOOP ======
def main():
    # Handle Ctrl+C
    def signal_handler(sig, frame):
        print("\n[MAIN] Caught Ctrl+C. Stopping...")
        stop_event.set()
    signal.signal(signal.SIGINT, signal_handler)

    # 1. Load Models
    try:
        net = cv2.dnn.readNetFromONNX(MODEL_PATH)
        # Use OpenCV CPU backend for Pi compatibility
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        with open(CLASS_INDEX_PATH) as f:
            class_indices = json.load(f)
        idx_to_class = {v: k for k, v in class_indices.items()}
        print(f"[VISION] Model loaded. Classes: {list(idx_to_class.values())}")
    except Exception as e:
        print(f"Error loading models: {e}")
        sys.exit(1)

    # 2. Start Camera
    cap = cv2.VideoCapture(0)
    # Lower resolution for better performance on Pi
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not cap.isOpened():
        print("ERROR: Camera not accessible")
        sys.exit(1)

    # 3. Start Background Threads
    # Start video streaming thread
    stream_thread = StreamServerThread(daemon=True)
    stream_thread.start()

    # Start drone control thread
    drone_thread = threading.Thread(target=drone_control_loop, daemon=True)
    drone_thread.start()

    # Start keyboard watcher
    keyboard_thread = threading.Thread(target=keyboard_watcher_thread, daemon=True)
    keyboard_thread.start()

    print("\n=== SYSTEM READY ===")
    print(f"open_palm: {PWM_PALM}, fist: {PWM_FIST}, thumb_up: {PWM_THUMBS_UP}")
    print(f"two_fingers OR 'q': Stop & Quit ({PWM_STOP})")
    print(f"idle/Unknown: Hold last state")
    print("====================\n")

    prev_time = time.time()

    # 4. Main Vision processing loop
    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.01)
            continue

        # --- Preprocess & Inference ---
        blob = cv2.dnn.blobFromImage(frame, 1.0 / 255.0, (IMG_SIZE, IMG_SIZE), (0, 0, 0), swapRB=True, crop=False)
        net.setInput(blob)
        preds = net.forward()[0]

        # --- Post-process ---
        cid = int(np.argmax(preds))
        conf = float(preds[cid])
        detected_gesture = idx_to_class.get(cid, "Unknown")

        final_gesture_label = "Unknown"
        if conf > CONF_THRESHOLD:
            final_gesture_label = detected_gesture

        # --- UPDATE SHARED STATE FOR DRONE THREAD ---
        with gesture_lock:
            shared_gesture = final_gesture_label

        # --- Visualization ---
        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
        prev_time = curr_time

        # Draw overlays
        cv2.rectangle(frame, (0,0), (250, 140), (0,0,0), -1) # Background box for text
        cv2.putText(frame, f"Gesture: {final_gesture_label}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f"Conf: {conf:.2f}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)

        # --- Update Streaming Buffer ---
        with frame_lock:
            output_frame = frame.copy()

        # Optional: display locally if a monitor is connected
        # cv2.imshow("Vision Feed", frame)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #      stop_event.set()

    # Cleanup
    print("\n[MAIN] Closing camera.")
    cap.release()
    cv2.destroyAllWindows()
    # Wait a moment for threads to finish cleaning up
    print("[MAIN] Waiting for threads to exit...")
    time.sleep(3)
    print("[MAIN] System shutdown complete.")

if __name__ == '__main__':
    main()