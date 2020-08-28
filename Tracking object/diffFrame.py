import cv2

cap = cv2.VideoCapture(0)
ret, frame = cap.read()
prox_frame = frame

while True:
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    prox_frame_gray = cv2.cvtColor(prox_frame, cv2.COLOR_BGR2GRAY)    

    frame_diff = cv2.absdiff(frame_gray,prox_frame_gray)

    cv2.imshow('frame diff ',frame_diff)      
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    prox_frame = frame.copy()
    ret, frame = cap.read()

cap.release()
cv2.destroyAllWindows()