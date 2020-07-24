import cv2
import numpy as np


cap = cv2.VideoCapture(0)

while True:

    ret, frame  =  cap.read()
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    #Blue Filter
    low = np.array([94, 80 ,2])
    high = np.array([126, 255, 255])
    mask = cv2.inRange(hsv_frame, low, high)

    blue_frame = cv2.bitwise_and(frame, frame, mask=mask)    
    
    
    cv2.imshow("Frame", frame)
    cv2.imshow('Blue_HSV', blue_frame)

    if cv2.waitKey(1) == 27:
       break

cv2.destroyAllWindows()
cap.release()