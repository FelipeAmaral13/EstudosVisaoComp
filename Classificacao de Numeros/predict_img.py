import cv2
import os
import numpy as np
from keras.models import load_model


class LeitorDeDigitos:
    def __init__(self, modelo):
        self.modelo = load_model(modelo)
        self.path = os.getcwd()

    def ler_imagens(self):
        nomes = [f for f in os.listdir(self.path + r"\Repositorio") if f.endswith(".png")]
        leitura = []

        for nome in nomes:
            im = cv2.imread(self.path + r"\Repositorio\\" + nome)
            print(f'Lendo imagem {nome}')
            gray = np.dot(im[..., :3], [0.299, 0.587, 0.114])
            gray = gray.reshape(1, 28, 28, 1)
            gray /= 255
            # predict image-digito
            prediction = self.modelo.predict(gray)
            leitura.append(prediction.argmax())

        return leitura


if __name__ == "__main__":
    ld = LeitorDeDigitos('test_model.h5')
    leitura = ld.ler_imagens()
    print(leitura)