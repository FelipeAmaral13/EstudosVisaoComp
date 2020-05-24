import cv2
import numpy as np


face_classifier = cv2.CascadeClassifier('Haarcascades/haarcascade_frontalface_default.xml')

def detectorFace(img, size=0.5):
    
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    if faces is ():
        return img, []
    
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,255),2)
        roi = img[y:y+h, x:x+w]
        roi = cv2.resize(roi, (300, 300))
    return img, roi


# Capturar imagens
cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()
    
    image, face = detectorFace(frame)
    
    try:
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

       
        results = model.predict(face)
        print(results[1])
        
        if results[1] < 5000:
            confidence = int( 100 * (1 - (results[1])/400) )
            print(confidence)
            display_string = str(confidence) + '% Match usuario'
            
        cv2.putText(image, display_string, (100, 120), cv2.FONT_HERSHEY_COMPLEX, 1, (255,120,150), 2)
        
        if confidence < 600:
            cv2.putText(image, "Negado", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
            cv2.imshow('Reconhecimento face', image )
        else:
            cv2.putText(image, "Permitido", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)
            cv2.imshow('reconhecimento Face', image )

    except:
        cv2.putText(image, "Face não encontrada", (220, 120) , cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)
        cv2.putText(image, "Negado", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)
        cv2.imshow('Reconhecimento face', image )
        pass
        
    if cv2.waitKey(1) == 13: #13 = Enter 
        break
        
cap.release()
cv2.destroyAllWindows()     