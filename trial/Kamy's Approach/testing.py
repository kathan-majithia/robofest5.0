import cv2
import numpy as np
import tensorflow as tf
import time

class GestureDetector:
    def __init__(self, model_path, class_names):
        # Load the trained model
        self.model = tf.keras.models.load_model(model_path)
        self.class_names = class_names
        
        # Gesture mapping to drone commands
        self.gesture_commands = {
            'index_up': 'FORWARD',
            'three_fingers': 'TURN_LEFT', 
            'all_fingers': 'LAND',
            'thumb_index': 'TURN_RIGHT',
            'thumb_index_little': 'SWARM',
            'index_little': 'STOP'
        }
        
        self.current_gesture = "No Gesture"
        self.prev_time = 0
        self.fps = 0
        
    def calculate_fps(self):
        current_time = time.time()
        self.fps = 1 / (current_time - self.prev_time) if self.prev_time != 0 else 0
        self.prev_time = current_time
        return self.fps

    def preprocess_frame(self, frame):
        """Preprocess frame for model prediction"""
        # Crop to square (center)
        height, width = frame.shape[:2]
        size = min(height, width)
        start_x = (width - size) // 2
        start_y = (height - size) // 2
        cropped = frame[start_y:start_y+size, start_x:start_x+size]
        
        # Resize to model input size
        resized = cv2.resize(cropped, (128, 128))
        
        # Normalize
        normalized = resized.astype('float32') / 255.0
        
        # Add batch dimension
        batched = np.expand_dims(normalized, axis=0)
        
        return batched, cropped

    def detect_gesture(self, frame):
        """Detect gesture in frame"""
        processed_frame, cropped = self.preprocess_frame(frame)
        
        # Predict
        predictions = self.model.predict(processed_frame, verbose=0)
        predicted_class = np.argmax(predictions[0])
        confidence = predictions[0][predicted_class]
        
        gesture_name = self.class_names[predicted_class]
        drone_command = self.gesture_commands.get(gesture_name, 'UNKNOWN')
        
        return drone_command, confidence, cropped

    def process_frame(self, frame):
        fps = self.calculate_fps()
        frame = cv2.flip(frame, 1)
        
        gesture, confidence, cropped_roi = self.detect_gesture(frame)
        
        if confidence > 0.7:  # Confidence threshold
            self.current_gesture = gesture
        
        # Display results
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Gesture: {self.current_gesture}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.putText(frame, f"Confidence: {confidence:.2f}", (10, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Show detection ROI
        roi_resized = cv2.resize(cropped_roi, (200, 200))
        frame[10:210, frame.shape[1]-210:frame.shape[1]-10] = roi_resized
        
        # Draw gesture guide
        guide_text = "1:Forward 2:Left 3:Land 4:Right 5:Swarm 6:Stop"
        cv2.putText(frame, guide_text, (10, frame.shape[0]-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame, self.current_gesture

def main():
    # Gesture classes (must match training)
    class_names = ['index_up', 'three_fingers', 'all_fingers', 
                   'thumb_index', 'thumb_index_little', 'index_little']
    
    # Initialize detector
    detector = GestureDetector('gesture_model.h5', class_names)
    
    # Camera setup
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
    cap.set(cv2.CAP_PROP_FPS, 20)
    
    print("Custom Gesture Detector Started!")
    print("Expected FPS: 15-25")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        processed_frame, gesture = detector.process_frame(frame)
        cv2.imshow('Gesture Control', processed_frame)
        
        # Execute drone command when gesture is detected
        if gesture != "No Gesture":
            print(f"DRONE COMMAND: {gesture}")
            # Add your drone control code here
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()