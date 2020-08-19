import numpy as np
import cv2

cap = cv2.VideoCapture(0)

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
fgbg_knn = cv2.createBackgroundSubtractorKNN()
#fgbg_mog = cv2.createBackgroundSubtractorMOG2()


while True:
    ret, frame = cap.read()

    fgmask = fgbg_knn.apply(frame)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)

    cv2.imshow('frame', fgmask)

    if cv2.waitKey(1) == 27:
        break

cv2.destroyAllWindows()
cap.release()