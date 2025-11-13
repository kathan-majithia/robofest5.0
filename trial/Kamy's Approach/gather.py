import cv2
import os
import time

def collect_gesture_data():
    gestures = {
        '1': 'index_up',
        '2': 'three_fingers', 
        '3': 'all_fingers',
        '4': 'thumb_index',
        '5': 'thumb_index_little',
        '6': 'index_little'
    }
    
    cap = cv2.VideoCapture(0)
    
    print("Gesture Data Collection")
    print("Press 1-6 to capture for each gesture, 'q' to quit")
    
    for gesture_num, gesture_name in gestures.items():
        count = 0
        target_count = 70  # 70 images per gesture
        
        # Create directory if not exists
        os.makedirs(f'dataset/{gesture_name}', exist_ok=True)
        
        print(f"\nCollecting {gesture_name}... Press spacebar to capture")
        
        while count < target_count:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame = cv2.flip(frame, 1)
            cv2.putText(frame, f"Gesture: {gesture_name}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Count: {count}/{target_count}", (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, "Press SPACE to capture", (10, 110), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            cv2.imshow('Data Collection', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord(' '):
                # Save image
                filename = f'dataset/{gesture_name}/img_{count:04d}.jpg'
                cv2.imwrite(filename, frame)
                count += 1
                print(f"Captured {filename}")
            elif key == ord('q'):
                break
                
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    collect_gesture_data()