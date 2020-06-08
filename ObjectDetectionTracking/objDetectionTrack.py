import numpy as np
import cv2

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    #Limites superior e inferior
    lim_inf = np.array([100,150,0])
    lim_sup = np.array([140,255,255])

    #Aplicacao dos lim
    color_mask = cv2.inRange(hsv, lim_inf, lim_sup) #Verifica se os elementos da matriz estão entre os elementos de duas outras matrizes

    (couts,hir) = cv2.findContours(color_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


    for cout in couts:
        area = cv2.contourArea(cout)

        if (area > 800):
            x,y,w,h = cv2.boundingRect(cout)
            frame = cv2.rectangle(frame, (x,y),(x+w,y+h),(0,0,255),2)

    cv2.imshow('ObjectDetctionTrack', frame)

    if cv2.waitKey(1) == 27:
        break

cv2.destroyAllWindows()
cap.release()