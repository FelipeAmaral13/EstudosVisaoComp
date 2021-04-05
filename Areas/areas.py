import cv2
import math
import numpy as np
import copy
import os

import matplotlib.pyplot as plt
from shapely.geometry import Polygon

path = os.getcwd()

img = np.zeros((500,500,3), dtype='uint8')


geo_dict = {'retangulo1': cv2.rectangle(img, (50,50), ((img.shape[1]//2)+50, (img.shape[0]//2)+50), (0,255,0), thickness=-1), 
            'retangulo2': cv2.rectangle(img, (350,350), ((img.shape[1]//4)+350, (img.shape[0]//4)+350), (255,0,0), thickness=-1),
            'elipse':cv2.ellipse(img,(150,400),(100,50),0,0,180,255,-1)}

import random
random.choice(list(geo_dict.values()))


pointsList = []


def mousePoints(event,x,y,flags,params):
    '''
    Função para captação dos clicks do mouse.
    '''
    if event == cv2.EVENT_LBUTTONDOWN:
        size = len(pointsList)
        cv2.circle(img,(x,y),5,(0,0,255),cv2.FILLED)
        pointsList.append([x,y])
        print(pointsList)


while True:   
    

    cv2.imshow('Image',img)
    cv2.setMouseCallback('Image',mousePoints)


    # Reset da imagem. Apagar todos os textos inseridos na imagem
    if cv2.waitKey(1) == ord('n'):
        img = cv2.imread(path)
        pointsList = []      
        cv2.imshow('Image', img)

    if cv2.waitKey(1) == ord('m'):
        area1 = np.array(pointsList)
        cv2.fillPoly(img, [area1], (255, 255, 255))

        polygon = Polygon(pointsList)

        polygon.area
        polygon.length

        len(pointsList)


    # Apertar 'ESC' para sair
    if cv2.waitKey(1) == 27:
        break

cv2.destroyAllWindows()




