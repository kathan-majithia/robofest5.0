import json
import cv2
import numpy as np
import tensorflow as tf

# ========= CONFIG =========
MODEL_PATH = "best_model.h5"
CLASS_INDICES_PATH = "class_indices.json"
IMG_SIZE = 224
CAMERA_INDEX = 0
# ==========================

# Load class index mapping
with open(CLASS_INDICES_PATH, "r") as f:
    class_indices = json.load(f)

# Invert mapping {0:"fist", 1:"idle", ...}
id_to_label = {v: k for k, v in class_indices.items()}

print("Class mapping:", id_to_label)

# Load the trained Keras model
print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded.")

# Open camera
cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    print("Error: Cannot open camera")
    exit()

print("Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    frame = cv2.flip(frame, 1)  # mirror for natural feel

    # Preprocess frame
    img = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)  # shape: (1, 224, 224, 3)

    # Predict
    preds = model.predict(img, verbose=0)[0]
    class_id = int(np.argmax(preds))
    confidence = float(np.max(preds))

    # Decode class
    label = id_to_label[class_id]

    # Draw prediction on screen
    text = f"{label} ({confidence*100:.1f}%)"
    cv2.putText(frame, text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.imshow("Gesture Test (Keras Model)", frame)

    # Quit on Q
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()