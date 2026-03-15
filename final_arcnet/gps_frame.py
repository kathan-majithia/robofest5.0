"""
ARCNET â€” Manual Capture Script (Press ENTER to capture)
========================================================
Press ENTER to capture a frame with GPS coordinates stamped.
Press Ctrl+C to stop.
"""

import cv2
import os
import time
import json
import threading
import numpy as np
from picamera2 import Picamera2
from pymavlink import mavutil

# â”€â”€ CONFIG â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
SERIAL_PORT   = '/dev/ttyAMA0'  # Pixhawk TELEM2 â†’ RPi UART
BAUD_RATE     = 57600           # confirmed working baud
OUTPUT_FOLDER = "frames"        # folder to save frames

# â”€â”€ GLOBAL GPS STATE â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
gps_data = {
    "lat":  None,
    "lon":  None,
    "alt":  None,
    "fix":  0,
    "sats": 0
}

def gps_reader():
    """Background thread â€” reads GPS from Pixhawk via MAVLink."""
    global gps_data
    try:
        master = mavutil.mavlink_connection(SERIAL_PORT, baud=BAUD_RATE)
        print("[GPS] Waiting for Pixhawk heartbeat...")
        master.wait_heartbeat()
        print("[GPS] Pixhawk connected! Reading GPS...")

        while True:
            msg = master.recv_match(type='GPS_RAW_INT', blocking=True, timeout=2)
            if msg:
                gps_data["lat"]  = round(msg.lat  / 1e7, 7)
                gps_data["lon"]  = round(msg.lon  / 1e7, 7)
                gps_data["alt"]  = round(msg.alt  / 1000, 2)
                gps_data["fix"]  = msg.fix_type
                gps_data["sats"] = msg.satellites_visible

    except Exception as e:
        print(f"[GPS] ERROR: {e}")

def overlay_gps(frame, frame_id):
    """Draw GPS coordinates on the frame."""
    h, w = frame.shape[:2]

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - 90), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    fix_color  = (0, 255, 80) if gps_data["fix"] >= 3 else (0, 80, 255)
    fix_text   = "3D FIX" if gps_data["fix"] >= 3 else f"NO FIX (type={gps_data['fix']})"

    if gps_data["lat"] and gps_data["lon"]:
        coord_text = f"LAT: {gps_data['lat']:.7f}   LON: {gps_data['lon']:.7f}"
    else:
        coord_text = "LAT: ---.-------   LON: ---.-------"

    alt_text   = f"ALT: {gps_data['alt']}m" if gps_data["alt"] else "ALT: ---"
    sats_text  = f"SATS: {gps_data['sats']}"
    frame_text = f"FRAME: {frame_id:04d}"

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, coord_text, (10, h-62), font, 0.58, (255,255,255), 1)
    cv2.putText(frame, f"{alt_text}   {sats_text}   {frame_text}",
                (10, h-36), font, 0.52, (200,200,200), 1)
    cv2.putText(frame, fix_text, (10, h-10), font, 0.52, fix_color, 1)

    cx, cy = w - 30, 30
    cv2.line(frame, (cx-12, cy), (cx+12, cy), (0, 255, 80), 1)
    cv2.line(frame, (cx, cy-12), (cx, cy+12), (0, 255, 80), 1)
    cv2.circle(frame, (cx, cy), 8, (0, 255, 80), 1)

    return frame

def main():
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    gps_thread = threading.Thread(target=gps_reader, daemon=True)
    gps_thread.start()

    print("[SYS] Waiting for Pixhawk GPS (5s)...")
    time.sleep(5)

    # â”€â”€ PICAMERA2 SETUP â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    cam = Picamera2()
    cam.configure(cam.create_still_configuration(main={"size": (1280, 720)}))
    cam.start()
    time.sleep(2)  # warm up camera
    print("[CAM] PiCamera2 ready")

    print("=" * 50)
    print("  Press ENTER to capture a frame")
    print("  Press Ctrl+C to stop")
    print("=" * 50 + "\n")

    frame_id = 0
    gps_log  = []

    try:
        while True:
            input(f"[READY] Press ENTER to capture frame {frame_id:04d}...")

            # Capture frame as numpy array (RGB) and convert to BGR for OpenCV
            frame = cam.capture_array()
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            frame = overlay_gps(frame, frame_id)

            fname = f"{OUTPUT_FOLDER}/frame_{frame_id:04d}.jpg"
            cv2.imwrite(fname, frame, [cv2.IMWRITE_JPEG_QUALITY, 92])

            gps_log.append({
                "frame":     fname,
                "frame_id":  frame_id,
                "lat":       gps_data["lat"],
                "lon":       gps_data["lon"],
                "alt":       gps_data["alt"],
                "fix":       gps_data["fix"],
                "sats":      gps_data["sats"],
                "timestamp": round(time.time(), 3)
            })

            with open(f"{OUTPUT_FOLDER}/gps_log.json", "w") as f:
                json.dump(gps_log, f, indent=2)

            fix_str = "3D FIX âœ“" if gps_data["fix"] >= 3 else "NO FIX âœ—"
            print(f"  âœ“ Saved {fname}")
            print(f"    LAT={gps_data['lat']}  LON={gps_data['lon']}  ALT={gps_data['alt']}m  {fix_str}\n")

            frame_id += 1

    except KeyboardInterrupt:
        print(f"\n[SYS] Capture stopped. {frame_id} frames saved.")

    finally:
        cam.stop()
        print(f"[SYS] GPS log saved â†’ {OUTPUT_FOLDER}/gps_log.json")
        print(f"[SYS] Total frames captured: {frame_id}")

if __name__ == "__main__":
    main()