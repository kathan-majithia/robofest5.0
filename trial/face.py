import cv2
import time

class FaceGestureController:
    def __init__(self):
        # Load ultra-light face detector
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.current_gesture = "No Gesture"
        self.prev_time = 0
        self.fps = 0
        
    def calculate_fps(self):
        current_time = time.time()
        self.fps = 1 / (current_time - self.prev_time) if self.prev_time != 0 else 0
        self.prev_time = current_time
        return self.fps

    def detect_face_gesture(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
        
        if len(faces) == 0:
            return "No Gesture"
        
        # Get the largest face
        x, y, w, h = faces[0]
        center_x = x + w // 2
        center_y = y + h // 2
        
        # Draw face bounding box
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.circle(frame, (center_x, center_y), 5, (0, 255, 0), -1)
        
        # Screen regions
        height, width = frame.shape[:2]
        left_region = width // 3
        right_region = 2 * width // 3
        
        # Head position based gestures
        if center_x < left_region:
            return "TURN_LEFT"
        elif center_x > right_region:
            return "TURN_RIGHT"
        elif w > width // 3:  # Face close to camera
            return "FORWARD"
        elif w < width // 6:  # Face far from camera
            return "LAND"
        elif len(faces) >= 2:  # Multiple faces detected
            return "SWARM"
        
        return "STOP"  # Default when face is centered

    def process_frame(self, frame):
        fps = self.calculate_fps()
        frame = cv2.flip(frame, 1)
        
        gesture = self.detect_face_gesture(frame)
        if gesture != "No Gesture":
            self.current_gesture = gesture
        
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Gesture: {self.current_gesture}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.putText(frame, "Move your head position for gestures", 
                   (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame, self.current_gesture

def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
    
    controller = FaceGestureController()
    
    print("Face-Based Gesture Controller Started!")
    print("Move your head to different screen positions")
    print("Expected FPS: 30+")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        processed_frame, gesture = controller.process_frame(frame)
        cv2.imshow('Face Gesture Control', processed_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()