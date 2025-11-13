import numpy as np
import cv2
import time

cap = cv2.VideoCapture(0)

fps = 0.0
alpha = 0.9
prev_time = time.time()

lower_blue = np.array([90, 50, 50])
upper_blue = np.array([130, 255, 255])
lower_dark_green = np.array([35, 50, 30])
upper_dark_green = np.array([85, 255, 120])
lower_red1 = np.array([0, 120, 50])
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([170, 120, 50])
upper_red2 = np.array([180, 255, 255])
while True:
    ret, frame = cap.read()

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

    cv2.imshow('frame', result)
    cv2.imshow('mask', com_mask)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()