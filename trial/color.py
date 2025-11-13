"""
Concept: Wear colored gloves (Red on right hand, Blue on left hand) and the system tracks their positions.

Gesture Mappings:
FORWARD (Move drone forward)

Action: Both arms extended straight out horizontally (T-pose)

Detection: Both hands at shoulder height level

Visual: Red and blue circles at mid-screen height, far apart

LAND (Land the drone)

Action: Both arms straight down at your sides

Detection: Both hands at waist/lower screen level

Visual: Red and blue circles at bottom of screen

TURN LEFT (Rotate drone counter-clockwise)

Action: Right arm raised up, left arm down

Detection: Right hand (red) at top of screen, left hand (blue) at bottom

Visual: Red circle top, blue circle bottom

TURN RIGHT (Rotate drone clockwise)

Action: Left arm raised up, right arm down

Detection: Left hand (blue) at top of screen, right hand (red) at bottom

Visual: Blue circle top, red circle bottom

SWARM FORMATION (Activate swarm mode)

Action: Both hands on your head

Detection: Both hands at top of screen, close together

Visual: Red and blue circles both at top, close to each other

STOP (Emergency stop/hover)

Action: Cross your arms in front of you (X shape)

Detection: Hands close together horizontally

Visual: Red and blue circles very close to each other
"""


import cv2
import numpy as np
import time

class ColorGestureController:
    def __init__(self):
        # Define color ranges for gloves (HSV format)
        self.colors = {
            'right': {'lower': np.array([0, 120, 70]), 'upper': np.array([10, 255, 255])},  # Red
            'left': {'lower': np.array([110, 120, 70]), 'upper': np.array([130, 255, 255])}   # Blue
        }
        
        self.gestures = {
            'FORWARD': 'both hands horizontal',
            'LAND': 'both hands down', 
            'TURN_LEFT': 'right hand up',
            'TURN_RIGHT': 'left hand up',
            'SWARM': 'both hands head',
            'STOP': 'hands crossed'
        }
        
        self.current_gesture = "No Gesture"
        self.prev_time = 0
        self.fps = 0

    def calculate_fps(self):
        current_time = time.time()
        self.fps = 1 / (current_time - self.prev_time) if self.prev_time != 0 else 0
        self.prev_time = current_time
        return self.fps

    def detect_hands(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        hand_positions = {}
        
        for hand_name, color_range in self.colors.items():
            # Create mask for color
            mask = cv2.inRange(hsv, color_range['lower'], color_range['upper'])
            mask = cv2.erode(mask, None, iterations=2)
            mask = cv2.dilate(mask, None, iterations=2)
            
            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                if cv2.contourArea(largest_contour) > 1000:  # Filter small detections
                    moments = cv2.moments(largest_contour)
                    if moments["m00"] != 0:
                        cx = int(moments["m10"] / moments["m00"])
                        cy = int(moments["m01"] / moments["m00"])
                        hand_positions[hand_name] = (cx, cy)
                        
                        # Draw center and contour
                        cv2.circle(frame, (cx, cy), 10, (0, 255, 0), -1)
                        cv2.drawContours(frame, [largest_contour], -1, (255, 0, 0), 2)
        
        return hand_positions

    def detect_gesture(self, hand_positions, frame_height):
        if 'right' not in hand_positions or 'left' not in hand_positions:
            return "No Gesture"
        
        right_x, right_y = hand_positions['right']
        left_x, left_y = hand_positions['left']
        
        # Divide screen into regions
        head_region = frame_height * 0.3
        shoulder_region = frame_height * 0.5
        waist_region = frame_height * 0.7
        
        # Gesture detection
        right_high = right_y < head_region
        right_mid = head_region <= right_y < shoulder_region
        right_low = right_y >= waist_region
        
        left_high = left_y < head_region
        left_mid = head_region <= left_y < shoulder_region
        left_low = left_y >= waist_region
        
        # Both hands at head level
        if right_high and left_high:
            return "SWARM"
        
        # Both hands horizontal (mid level)
        elif right_mid and left_mid:
            return "FORWARD"
        
        # Both hands low
        elif right_low and left_low:
            return "LAND"
        
        # Right hand high, left hand low = TURN LEFT
        elif right_high and left_low:
            return "TURN_LEFT"
        
        # Left hand high, right hand low = TURN RIGHT
        elif left_high and right_low:
            return "TURN_RIGHT"
        
        # Hands crossed (close x positions)
        elif abs(right_x - left_x) < 50:
            return "STOP"
        
        return "No Gesture"

    def process_frame(self, frame):
        fps = self.calculate_fps()
        
        # Flip for mirror effect
        frame = cv2.flip(frame, 1)
        
        # Detect hands
        hand_positions = self.detect_hands(frame)
        
        # Detect gesture
        gesture = self.detect_gesture(hand_positions, frame.shape[0])
        
        if gesture != "No Gesture":
            self.current_gesture = gesture
        
        # Display info
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Gesture: {self.current_gesture}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.putText(frame, "Wear colored gloves (Red=Right, Blue=Left)", 
                   (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame, self.current_gesture

def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Lower resolution for speed
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    controller = ColorGestureController()
    
    print("Color-Based Gesture Controller Started!")
    print("Wear colored gloves: Red on right hand, Blue on left hand")
    print("Expected FPS: 25-30+")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        processed_frame, gesture = controller.process_frame(frame)
        cv2.imshow('Color Gesture Control', processed_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()