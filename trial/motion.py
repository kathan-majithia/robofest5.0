import cv2
import numpy as np
import time

class MotionGestureController:
    def __init__(self):
        self.prev_frame = None
        self.current_gesture = "No Gesture"
        self.gesture_start_time = 0
        self.prev_time = 0
        self.fps = 0
        
    def calculate_fps(self):
        current_time = time.time()
        self.fps = 1 / (current_time - self.prev_time) if self.prev_time != 0 else 0
        self.prev_time = current_time
        return self.fps

    def detect_motion_gesture(self, current_frame):
        if self.prev_frame is None:
            self.prev_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
            return "No Gesture"
        
        # Convert to grayscale
        gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate frame difference
        frame_diff = cv2.absdiff(self.prev_frame, gray)
        _, thresh = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)
        
        # Update previous frame
        self.prev_frame = gray
        
        # Find contours in the thresholded image
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return "No Gesture"
        
        # Get the largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        
        if cv2.contourArea(largest_contour) < 1000:
            return "No Gesture"
        
        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(largest_contour)
        center_x = x + w // 2
        center_y = y + h // 2
        
        # Draw motion area
        cv2.rectangle(current_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.circle(current_frame, (center_x, center_y), 5, (0, 0, 255), -1)
        
        # Define screen regions
        height, width = current_frame.shape[:2]
        left_region = width // 3
        right_region = 2 * width // 3
        top_region = height // 3
        bottom_region = 2 * height // 3
        
        # Detect gesture based on motion position
        if center_x < left_region and center_y < top_region:
            return "SWARM"
        elif center_x > right_region and center_y < top_region:
            return "FORWARD"
        elif center_x < left_region and center_y > bottom_region:
            return "TURN_LEFT"
        elif center_x > right_region and center_y > bottom_region:
            return "TURN_RIGHT"
        elif w > h and w > width // 2:  # Wide horizontal motion
            return "STOP"
        elif h > w and h > height // 2:  # Tall vertical motion
            return "LAND"
        
        return "No Gesture"

    def process_frame(self, frame):
        fps = self.calculate_fps()
        
        frame = cv2.flip(frame, 1)
        
        gesture = self.detect_motion_gesture(frame)
        
        if gesture != "No Gesture":
            self.current_gesture = gesture
        
        # Display info
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Gesture: {self.current_gesture}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.putText(frame, "Move in screen regions: Top-Left=Swarm, Top-Right=Forward", 
                   (10, frame.shape[0] - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(frame, "Bottom-Left=Turn Left, Bottom-Right=Turn Right", 
                   (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        return frame, self.current_gesture

def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
    
    controller = MotionGestureController()
    
    print("Motion-Based Gesture Controller Started!")
    print("Move in different screen regions to trigger gestures")
    print("Expected FPS: 30+")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        processed_frame, gesture = controller.process_frame(frame)
        cv2.imshow('Motion Gesture Control', processed_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()