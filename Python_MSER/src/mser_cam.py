import cv2
import numpy as np

mser = cv2.MSER_create()

cap = cv2.VideoCapture(0)


while True:

    ret, frame = cap.read()

    #Convert to gray scale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #detect regions in gray scale image
    regions, _ = mser.detectRegions(gray)

    hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]

    mask = np.zeros((frame.shape[0], frame.shape[1], 1), dtype=np.uint8)


    for contour in hulls:

        cv2.drawContours(mask, [contour], -1, (255, 255, 255), -1)

    text_only = cv2.bitwise_and(frame, frame, mask=mask)

    cv2.imshow("RED", text_only)
    key = cv2.waitKey(1)

    if key == 27:
        break


cv2.destroyAllWindows()
cap.release()
