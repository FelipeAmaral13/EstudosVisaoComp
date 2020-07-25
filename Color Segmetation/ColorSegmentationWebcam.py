import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Verm
    red_low = np.array([160, 155, 85])
    red_high = np.array([180, 255, 255])
    red_mask = cv2.inRange(frame, red_low, red_high)
    red = cv2.bitwise_and(frame, frame, mask=red_mask)

    #Blue Filter
    low = np.array([94, 80 ,2])
    high = np.array([126, 255, 255])
    mask = cv2.inRange(hsv_frame, low, high)
    blue_frame = cv2.bitwise_and(frame, frame, mask=mask)    


    # Todas as cores exceto Branco
    #low = np.array([0, 40, 0])
    #high = np.array([180, 255, 255])
    #mask = cv2.inRange(frame, low, high)
    #result = cv2.bitwise_and(frame, frame, mask=mask)


    #cv2.imshow("Frame", frame)
    #cv2.imshow("Frame", result)
    #cv2.imshow("RED", red)
    cv2.imshow('Blue_HSV', blue_frame)



    key = cv2.waitKey(1)

    if key == 27:
        break


cv2.destroyAllWindows()
cap.release()