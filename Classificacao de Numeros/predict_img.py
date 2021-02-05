#! python3

# Bilioteca
import cv2
import os
import numpy as np
from keras.models import load_model

# Caminho
path = os.getcwd()

# Ler modelo treinado
model = load_model("test_model.h5")


# Imagens Salvas processadas
nomes  = [ f for f in os.listdir(path + r"\Repositorio") if f.endswith(".png") ]



leitura = []

for i in range(len(nomes)):
    im = cv2.imread(path + r"\Repositorio\\" + nomes[i])
    print(f'Lendo imagem {nomes[i]}')
    gray = np.dot(im[...,:3], [0.299, 0.587, 0.114])
    gray = gray.reshape(1, 28, 28, 1)
    gray /= 255
    # predict image-digito
    prediction = model.predict(gray)
    leitura.append(prediction.argmax())
print(leitura)