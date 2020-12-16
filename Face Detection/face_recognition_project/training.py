
import cv2
import numpy as np
from PIL import Image
import os

# Path do dataset
path = 'dataset'

recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

def getImagesAndLabels(path):
    ''' Função para pegar as imagens e seus respectivos labels
        Entrada: path do dataset.
        Saida: Faces detectadas e seus respectivos IDs
    '''

    imagePaths = [os.path.join(path,f) for f in os.listdir(path)]     
    faceSamples=[]
    ids = []

    for imagePath in imagePaths:

        PIL_img = Image.open(imagePath).convert('L') # converter em tons de cinza
        img_numpy = np.array(PIL_img,'uint8')

        id = int(os.path.split(imagePath)[-1].split(".")[1])
        faces = detector.detectMultiScale(img_numpy)

        for (x,y,w,h) in faces:
            faceSamples.append(img_numpy[y:y+h,x:x+w])
            ids.append(id)

    return faceSamples,ids

faces,ids = getImagesAndLabels(path)
recognizer.train(faces, np.array(ids))

# Salvar o modelo treinado trainer/trainer.yml
recognizer.write('trainer/trainer.yml') 

# Print o numeor de faces trinadas
print("\n [INFO] {0} faces treinadas.".format(len(np.unique(ids))))
