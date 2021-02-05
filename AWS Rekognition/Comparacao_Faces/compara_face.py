# Bibliotecas
import boto3
import csv
import numpy as np
import cv2
import os

path = os.getcwd()

# Credenciais
with open(path + r'\AWS Rekognition\Comparacao_Faces\new_user_credentials.csv', 'r') as input:
    next(input)
    reader = csv.reader(input)

    # Ler o ID e o Access
    for line in reader:
        access_key_id = line[2]
        secret_access_key = line[3]

#############################################################
#                                                           #
#            Captacao da Face pela webcam                   #
#                                                           #
#############################################################

# Haarcascade
faceCascade = cv2.CascadeClassifier(path + r'\AWS Rekognition\Comparacao_Faces\Haarcascades\haarcascade_frontalface_default.xml')

# WebCama
cap = cv2.VideoCapture(0)

cap.set(3,640) #  Width
cap.set(4,480) #  Heigh

while True:
    ret, img = cap.read()
    
    # Deteccao de faces
    faces = faceCascade.detectMultiScale(
        img,     
        scaleFactor=1.2,
        minNeighbors=5,     
        minSize=(20, 20)
    )

    # Localizar a face e mostrar por um retangulo
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_color = img[y:y+h, x:x+w] 

    cv2.putText(img,"Aperte ESC para salvar e sair",(10,30),cv2.FONT_HERSHEY_COMPLEX,1,255)
    cv2.imshow('video',img)

    k = cv2.waitKey(30) & 0xff
    if k == 27: # press 'ESC' para quitar e salvar a imagem
        cv2.imwrite(path + r'\\AWS Rekognition\Comparacao_Faces\face_detect.png', roi_color)
        break
    
cap.release()
cv2.destroyAllWindows()


#############################################################
#                                                           #
#                    AWS - Rekognition                      #
#                                                           #
#############################################################

# Client
client = boto3.client('rekognition', 
                        region_name='us-east-1',
                        aws_access_key_id=access_key_id,
                        aws_secret_access_key=secret_access_key)

#s3.Object('teste-rekognition-meganha', secret_access_key).delete()                        

# Salvar a imagem com o nome face_detect
photo = path + r'\AWS Rekognition\Comparacao_Faces\face_detect.png'

# Subir imagem para o Bucket S3 - AWS
s3 = boto3.resource('s3', 
                    aws_access_key_id=access_key_id,
                    aws_secret_access_key=secret_access_key)


s3.Bucket('teste-rekognition-meganha').upload_file(photo, "face_detect.png")

# Aplicar o reconhecimento de faces
response = client.compare_faces(
    SourceImage={'S3Object': {
        'Bucket': 'teste-rekognition-meganha',
        'Name':'Ronaldo_Uninove.jpeg',
        }
    },

    TargetImage={'S3Object': {
        'Bucket': 'teste-rekognition-meganha',
        'Name': 'face_detect.png',
        }
    }
)

#print(response)

for key, value in response.items():
    if key in ('FaceMatches', 'UnmatchedFaces'):
        print(key)
        for att in value:
            print(att)

