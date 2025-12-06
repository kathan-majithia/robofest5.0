import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import json
import cv2
import numpy as np
import tensorflow as tf
import time

# ========= CONFIG =========
MODEL_PATH = "best_model.h5"
CLASS_INDICES_PATH = "class_indices.json"
IMG_SIZE = 224
CAMERA_INDEX = 0
# ==========================

# Load class index mapping
with open(CLASS_INDICES_PATH, "r") as f:
    class_indices = json.load(f)

id_to_label = {v: k for k, v in class_indices.items()}
print("Loaded class map:", id_to_label)

# Load model
print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded!")

# Open camera
cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    print("ERROR: Cannot open camera. Try CAMERA_INDEX = 1 or 2.")
    exit()

print("Press 'q' to quit.")

# FPS measurement
prev_time = time.time()
fps = 0.0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Frame not captured!")
        break

    frame = cv2.flip(frame, 1)

    # Preprocess for model
    img = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)

    # Predict
    preds = model.predict(img, verbose=0)[0]
    class_id = int(np.argmax(preds))
    conf = float(np.max(preds))
    label = id_to_label[class_id]

    # FPS calculation
    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    prev_time = current_time

    # Draw prediction
    text = f"{label} ({conf*100:.1f}%)"
    fps_text = f"FPS: {fps:.1f}"

    cv2.putText(frame, text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(frame, fps_text, (10, 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Show frame
    cv2.imshow("Gesture Test (Keras Model)", frame)

    # Quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()