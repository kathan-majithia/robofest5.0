import numpy as np
import cv2
import time

cap = cv2.VideoCapture(0)

fps = 0.0
alpha = 0.9
prev_time = time.time()

lower_blue = np.array([90, 50, 50])
upper_blue = np.array([130, 255, 255])

# Middle-to-dark green
lower_dark_green = np.array([35, 100, 100])
upper_dark_green = np.array([85, 255, 220])

lower_red1 = np.array([0, 120, 50])
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([170, 120, 50])
upper_red2 = np.array([180, 255, 255])

# --- New Threshold for "Bigness" ---
# <-- This is the important new variable.
# <-- Increase this value to detect only *larger* objects.
# <-- Decrease it to detect *smaller* objects.
MIN_AREA = 5000 

while True:
    ret, frame = cap.read()
    if not ret: 
        break

    cur_time = time.time()
    ins = 1.0 / (cur_time - prev_time) if (cur_time - prev_time) > 0 else 0.0
    prev_time = cur_time
    fps = (alpha * fps) + ((1.0 - alpha) * ins)

    width = int(cap.get(3))
    height = int(cap.get(4))

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)


    mask_b = cv2.inRange(hsv, lower_blue, upper_blue)
    mask_g = cv2.inRange(hsv,lower_dark_green,upper_dark_green)
    mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask_red = cv2.bitwise_or(mask_red1, mask_red2)

    com_mask = cv2.bitwise_or(mask_b,mask_g)
    com_mask = cv2.bitwise_or(com_mask,mask_red)


    result = cv2.bitwise_and(frame, frame, mask=com_mask)

    ftext = f"FPS : {fps:.2f}"

    cv2.putText(result,ftext,(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.9,(0,255,0),2,cv2.LINE_AA)

    # --- New Code for Color Detection (with Area Check) ---
    
    # 1. Check which colors are detected
    colors_str = ""
    
    # --- Check for Blue ---
    # Find all the blue "blobs"
    contours, _ = cv2.findContours(mask_b, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        # Check if the area of a blob is bigger than our minimum
        if cv2.contourArea(cnt) > MIN_AREA:
            colors_str += "B"
            break # Found a big one, stop checking for blue

    # --- Check for Green ---
    contours, _ = cv2.findContours(mask_g, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        if cv2.contourArea(cnt) > MIN_AREA:
            colors_str += "G"
            break # Found a big one

    # --- Check for Red ---
    contours, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        if cv2.contourArea(cnt) > MIN_AREA:
            colors_str += "R"
            break # Found a big one
        
    # 2. Create the final display string
    if not colors_str: # If the string is still empty, no colors were found
        display_text = "Colors: None"
    else:
        display_text = "Colors: " + colors_str

    # 3. Get text size to position it in the right corner
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.9
    thickness = 2
    (text_width, text_height), _ = cv2.getTextSize(display_text, font, font_scale, thickness)

    # 4. Calculate position (top-right corner)
    text_x = width - text_width - 10
    text_y = 30 

    # 5. Draw the text on the result frame
    cv2.putText(result, display_text, (text_x, text_y), font, font_scale, (0, 255, 0), thickness, cv2.LINE_AA)

    # --- End of New Code ---

    cv2.imshow('frame', result)
    cv2.imshow('mask', com_mask)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()