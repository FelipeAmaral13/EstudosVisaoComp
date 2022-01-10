import cv2
import numpy as np

cap = cv2.VideoCapture(0)


def range_HSV(H: int, S: int, V: int):

    red_low = np.array([H, S, V])
    red_high = np.array([255, 255, 255])
    red_mask = cv2.inRange(frame, red_low, red_high)
    red = cv2.bitwise_and(frame, frame, mask=red_mask)

    return red


while True:

    ret, frame = cap.read()
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    color_hsv = range_HSV(160, 115, 85)

    # cv2.imshow("Frame", frame)
    # cv2.imshow("Frame", result)
    cv2.imshow("RED", color_hsv)
    # cv2.imshow('Blue_HSV', blue_frame)

    key = cv2.waitKey(1)

    if key == 27:
        break


cv2.destroyAllWindows()
cap.release()
