#-----------------------------------------#
#                                         #
#             Bibliotecas                 #
#                                         #
#-----------------------------------------#
import numpy as np
import cv2
from time import sleep


#-----------------------------------------#
#                                         #
#               VARIAVEIS                 #
#                                         #
#-----------------------------------------#

veiculos = 0
detects = []

#Linhas Vertical
posL = 530 #posicao da linha da Vert
offset = 30 #pixel pra cima de pra baixo pra contagem

#Posicao da linha
xy1 = (80, posL)
xy2 = (550, posL)

xy3 = (650, posL)
xy4 = (1090, posL)

#-----------------------------------------#
#                                         #
#                FUNCÕES                  #
#                                         #
#-----------------------------------------#
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
    cv2.putText(frame, text, (50, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.imshow("frame", frame)
    #cv2.imshow('Frame', frame)
    #cv2.imshow('Gray', gray)
    #cv2.imshow('Mask', fgmask)



#cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture('video.mp4')

fgbg = cv2.createBackgroundSubtractorMOG2()


while True:
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    fgmask = fgbg.apply(gray) #mascara

    #remocao das sombras na mascara
    retval, th = cv2.threshold(fgmask, 200, 255, cv2.THRESH_BINARY) #Retorno do frame e o TH

    #Morfologia
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    opening = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, iterations=2)
    dilation = cv2.dilate(opening, kernel, iterations=8)
    closing = cv2.morphologyEx(dilation, cv2.MORPH_CLOSE, kernel, iterations=8)    

    #Extracao do contorno
    contours, hierarcky = cv2.findContours(closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    cv2.line(frame, (25, 550), (1200, 550), (0, 0, 255), 3)

    cv2.line(frame, (xy1[0], posL-offset), (xy2[0], posL-offset), (255,255,0), 2)
    cv2.line(frame, (xy3[0], 500), (xy4[0], 500), (255,255,0), 2)
    


    id = 0
    for cnt in contours:
        (x,y,w,h) = cv2.boundingRect(cnt) #Quatro variavies para fazer o rect
        area = cv2.contourArea(cnt)

        if (w >= 80) and (h >= 80) :
            center = centro(x,y,w,h)
            cv2.circle(frame, center, 4, (0, 0, 255), -1)
            cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
            cv2.putText(frame, "veiculo: " + str(id), (x+5, y+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255),2)

            if len(detects) <= id:
                detects.append([])
            if center[1]> posL-offset and center[1] < posL+offset:
                detects[id].append(center)
            else:
                detects[id].clear()


        else:
            continue

        id +=1 


    if len(contours) == 0:
        detects.clear()

    if len(detects) > 0:
        for detect in detects:
            for (c, l) in enumerate(detect):
                print(id)

    print(detects)




    show_veiculo(frame, closing)
  

    if cv2.waitKey(30) & 0xFF == ord("q"):
        break

cap.realese()
cv2.destroyAllWindows()

