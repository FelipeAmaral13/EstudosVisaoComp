import cv2
import math
import numpy as np
import copy

 
path = '1.png'
img = cv2.imread(path)
img_copy = copy.copy(img)
pointsList = []
 
def mousePoints(event,x,y,flags,params):
    if event == cv2.EVENT_LBUTTONDOWN:
        size = len(pointsList)
        print(size)
        if size != 0 and  size % 3 != 0 :
            cv2.line(img, tuple(pointsList[round((size-1)/3)*3]), (x, y), (0,255,0), 2)
        cv2.circle(img,(x,y),5,(0,0,255),cv2.FILLED)
        pointsList.append([x,y])


def gradient(pt1, pt2): # Slope das retas
    '''
    Função para pegar o slope, ou também conhecido como gradiente, das retas
    '''
    return (pt2[1] - pt1[1])/(pt2[0] - pt1[0])


def getAngle(pointsList):
    '''
    Função para pegar o angulo a partir de três pontos
    '''

    pt1, pt2, pt3 = pointsList[-3:]
    m1 = gradient(pt1, pt2)
    m2 = gradient(pt1, pt3)
    angR = math.atan((m2-m1)/(1 + (m2*m1))) # Angulos em Radianos
    angD = round(math.degrees(angR)) # Angulos em Graus
    #print(angD)
    cv2.putText(img,str(angD),(pt1[0],pt1[1]),cv2.FONT_HERSHEY_COMPLEX,
                1.5,(0,0,255),2)

while True: 
    
    if len(pointsList) % 3 == 0 and len(pointsList) != 0:
        getAngle(pointsList)


    cv2.imshow('Image',img)
    cv2.setMouseCallback('Image',mousePoints)

    if cv2.waitKey(1) == ord('n'):
        img = cv2.imread(path)       
        cv2.imshow('Image', img)



    if cv2.waitKey(1) == 27:
        pointsList = []
        img = cv2.imread(path)
        break

cv2.destroyAllWindows()