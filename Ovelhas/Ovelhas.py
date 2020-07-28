import cv2
import matplotlib.pyplot as plt
import os
import numpy as np

cap = cv2.VideoCapture(os.path.join(os.getcwd(),'Video.mp4'))

width = 640
height = 480

cap.set(3, width)
cap.set(4, height)

while True:
    ret, frame = cap.read()

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    #Limites superior e inferior
    lim_inf = np.array([0,0,100])
    lim_sup = np.array([180,120,255])

    #Aplicacao dos lim
    color_mask = cv2.inRange(hsv, lim_inf, lim_sup) #Verifica se os elementos da matriz estão entre os elementos de duas outras matrizes

    #Morfologia
    kernel_erode = np.ones((4,4), np.uint8)
    eroded_mask = cv2.erode(color_mask, kernel_erode, iterations=1)
    kernel_dilate = np.ones((6,6),np.uint8)
    dilated_mask = cv2.dilate(eroded_mask, kernel_dilate, iterations=1)
    
    
    (couts,hir) = cv2.findContours(dilated_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for cout in couts:
        area = cv2.contourArea(cout)

        if (area > 100  and len(couts) > 0):
            x,y,w,h = cv2.boundingRect(cout)
            frame = cv2.rectangle(frame, (x,y),(x+w,y+h),(0,0,255),2)
            cv2.putText(frame, "Ovelha", (x+5, y+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255),2)
            
            M = cv2.moments(couts[0])
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            print("Centroid da maior area: ({}, {})".format(cx, cy))
    
    cv2.imshow('Window', frame)

    if cv2.waitKey(30) & 0xFF == ord("q"):
        break

cap.realese()
cv2.destroyAllWindows()

