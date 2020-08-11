import cv2 
import numpy as np
import os


face_cascade = cv2.CascadeClassifier(os.path.join(os.getcwd(), "cascades\\haarcascade_frontalface_default.xml"))

cap = cv2.VideoCapture(0)

#Callback Retangulo
def draw_rect(event,x,y,flags,params):
    global pt1,pt2,topLeftClicked,bottomRightClicked
    if event == cv2.EVENT_LBUTTONDOWN:
        if topLeftClicked == True and bottomRightClicked == True:
            pt1 = (0,0)
            pt2 = (0,0)
            topLeftClicked = False
            bottomRightClicked = False
            
        if topLeftClicked == False:
            pt1 = (x,y)
            topLeftClicked = True
        elif bottomRightClicked == False:
            pt2 = (x,y)
            bottomRightClicked = True

## Variaveis Globais
pt1 = (0,0)
pt2 = (0,0)
topLeftClicked = False
bottomRightClicked = False

cv2.namedWindow('Teste')
cv2.setMouseCallback('Teste', draw_rect)

while True:
    
    ret, frame = cap.read()
    
    ## Desenhar o retangulo de acordo com as VG
    if topLeftClicked:
        cv2.circle(frame, center=pt1, radius=5, color=(0,0,255),thickness=-1)
    if topLeftClicked and bottomRightClicked:
        cv2.rectangle(frame, pt1,pt2,color = (0,0,255), thickness = 1)
#         cv2.imshow('Crop', frame[pt1,pt2])
    
    cv2.imshow('Teste', frame)
    
    faces = face_cascade.detectMultiScale(frame, 1.15,5)
    
    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (255,0,0),2)
        roi_gray = frame[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

    cv2.imshow('Face detectada', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
cap.release()
cv2.destroyAllWindows()