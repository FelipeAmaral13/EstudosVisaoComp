
import cv2
import os

cam = cv2.VideoCapture(0)
cam.set(3, 640) # setar video width
cam.set(4, 480) # setar video height

# HaarCasacde Detect
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Para cada pessoa, insira um ID para identificacao
face_id = input('Pessoa ID')

count = 0

while(True):

    ret, img = cam.read()
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:

        cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)     
        count += 1

        # Salvar face  na pasta dataset
        cv2.imwrite("dataset/User." + str(face_id) + '.' + str(count) + ".jpg", gray[y:y+h,x:x+w])

        cv2.imshow('image', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
         break
    elif count >= 30: 
         break

cam.release()
cv2.destroyAllWindows()


