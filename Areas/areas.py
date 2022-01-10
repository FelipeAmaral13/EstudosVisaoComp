# Bibliotecas
import cv2
import numpy as np
import os
import random
from shapely.geometry import Polygon

# Caminho
path = os.getcwd()

# Criacao do fundo
img = np.zeros((800, 800, 3), dtype='uint8')

# Dicionario dos objetos
geo_dict = {'retangulo1': cv2.rectangle(img, (50, 50), ((img.shape[1]//2)+50, (img.shape[0]//2)+50), (0, 255, 0), thickness=-1),
            'retangulo2': cv2.rectangle(img, (350, 500), ((img.shape[1]//4)+350,(img.shape[0]//4)+350), (255, 0, 0), thickness=-1),
            'elipse': cv2.ellipse(img, (600, 200), (100, 50), 0, 0, 180, 255, -1)}

# Escolha randomica dos objetos do dicionario
random.choice(list(geo_dict.values()))

# Mouse
pointsList = []


def mousePoints(event: int, x: int, y: int, flags, params):
    '''
    Função para captação dos clicks do mouse.
    '''
    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(img, (x, y), 5, (0, 0, 255), cv2.FILLED)
        pointsList.append([x, y])
        print(pointsList)


while True:

    cv2.imshow('Image', img)
    cv2.setMouseCallback('Image', mousePoints)

    #  Reset da imagem. Apagar todos os textos inseridos na imagem
    if cv2.waitKey(1) == ord('n'):
        img = cv2.imread(path)
        pointsList = []
        cv2.imshow('Image', img)

    # Aperte M para calculo da area
    if cv2.waitKey(1) == ord('m'):
        area1 = np.array(pointsList)
        cv2.fillPoly(img, [area1], (255, 255, 255))

        polygon = Polygon(pointsList)
        polygon.area
        polygon.length
        len(pointsList)

        cv2.putText(img, f"Area: {polygon.area}, Perimetro: {polygon.length}", (50, 700), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    #  Apertar 'ESC' para sair
    if cv2.waitKey(1) == 27:
        break

cv2.destroyAllWindows()
