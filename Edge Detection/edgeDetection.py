import cv2

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Bilateral
    edges_bil = cv2.bilateralFilter(gray, 7, 50, 50)

    # Canny
    edges_Canny = cv2.Canny(gray, 60, 120)

    cv2.imshow('Frame_Canny', edges_Canny)

    if cv2.waitKey(1) == 27:
        break

cv2.destroyAllWindows()
cap.release()
