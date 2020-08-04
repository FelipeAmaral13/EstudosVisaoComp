#Bibliotecas
import numpy as np
import cv2
from time import sleep
import os

#Variaveis
veiculos = 0
detects = []
total = 0

#Linhas Vertical
posL = 580 #posicao da linha da Vert
offset = 40 #pixel pra cima de pra baixo pra contagem

#Posicao da linha da faixa 
xy1 = (80, posL)
xy2 = (1090, posL)

#Funcao calcular o centro
def centro(x,y,w,h):
    x1 = int(w/2)
    y1 = int(h/2)
    cx = x + x1
    cy = y + y1
    return cx, cy


#Funcao mostrar contador
def show_veiculo(frame, closing):
    text = f'Veiculos: {veiculos}'
    cv2.putText(frame, text + str(total), (50, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.imshow("frame", frame)
    #cv2.imshow('opening', opening)
    #cv2.imshow('closing', closing)
    #cv2.imshow('fgmask', fgmask)
    #cv2.imshow('closing', closing)



#cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture(os.path.join(os.getcwd(),'video.mp4'))

fgbg = cv2.createBackgroundSubtractorMOG2(varThreshold=200 ,detectShadows=True)


while True:
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    fgmask = fgbg.apply(gray) #mascara

    #remocao das sombras na mascara
    retval, th = cv2.threshold(fgmask, 200, 255, cv2.THRESH_BINARY) #Retorno do frame e o TH

    #Morfologia
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    opening = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, iterations=3)
    dilation = cv2.dilate(opening, kernel, iterations=3)
    closing = cv2.morphologyEx(dilation, cv2.MORPH_CLOSE, kernel, iterations=4)
    
    

    #Extracao do contorno
    contours, hierarcky = cv2.findContours(closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    cv2.line(frame, (25, 600), (1200, 600), (0, 0, 255), 3)

    cv2.line(frame, (xy1[0], posL-offset), (xy2[0], posL-offset), (255,255,0), 2)

    id = 0
    for cnt in contours:
        (x,y,w,h) = cv2.boundingRect(cnt) #Quatro variavies para fazer o rect
        area = cv2.contourArea(cnt)

        if (w >= 50) and (h >= 50) :
            center = centro(x,y,w,h)
            cv2.circle(frame, center, 4, (0, 0, 255), -1)
            cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
            cv2.putText(frame, "veiculo: " + str(id), (x+5, y+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255),2)

            if len(detects) <= id:
                detects.append([])
            if center[1] > posL-offset and center[1] < posL+offset:
                detects[id].append(center)
            else:
                detects[id].clear()

            id += 1


    if id == 0:
        detects.clear()

    id = 0 


    if len(contours) == 0:
        detects.clear()

    else: 
        for detect in detects:
            for (c, l) in enumerate(detect):
                if detect[c-1][1] < posL and l[1] > posL:
                    detect.clear()
                    total+=1
                    cv2.line(frame, xy1, xy2, (0,255,0), 5)
                    continue
                
                

    print(detects)


    show_veiculo(frame, closing)
  

    if cv2.waitKey(30) & 0xFF == ord("q"):
        break

cap.realese()
cv2.destroyAllWindows()

