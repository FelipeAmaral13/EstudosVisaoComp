import numpy as np
import cv2


#Funcao calcular o centro
def centro(x,y,w,h):
    x1 = int(w/2)
    y1 = int(h/2)
    cx = x + x1
    cy = y + y1
    return cx, cy

#Linhas Vertical
posL = 150 #posicao da linha da Vert
offset = 30 #pixel pra cima de pra baixo pra contagem

#Posicao da linha
xy1 = (20, posL)
xy2 = (300, posL)

#Detectar as pessoas
detects = []

cap = cv2.VideoCapture(0)

#Subtracao do background
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

    #Linha
    cv2.line(frame, xy1, xy2, (255,0,0), 3)

    #Linhas do offset
    cv2.line(frame, (xy1[0], posL-offset), (xy2[0], posL-offset), (255,255,0), 2)
    cv2.line(frame, (xy1[0], posL+offset), (xy2[0], posL+offset), (255,255,0), 2)

    i=0 #id das pessoas
    for cnt in contours:
        (x,y,w,h) = cv2.boundingRect(cnt) #Quatro variavies para fazer o rect
        area = cv2.contourArea(cnt)

        #Ignorar pontos pequenos (ruidos)
        if int(area) > 3000:
            center = centro(x,y,w,h)
            #print(center)

            cv2.putText(frame, str(i), (x+5, y+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255),2)
            cv2.circle(frame, center, 4, (0, 0, 255), -1)
            cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)

            #Incrementar um lista de arrays
            if len(detects) <= i:
                detects.append([])
            if center[1]> posL-offset and center[1] < posL+offset:
                detects[i].append(center)
            else:
                detects[i].clear()

            i = i+1

    if len(contours) == 0:
        detects.clear()

    if len(detects) > 0:
        for detect in detects:
            for (c, l) in enumerate(detect):
                print(i)

    print(detects)

    cv2.imshow("Frame", frame)
    #cv2.imshow("fgmask", fgmask)
    #cv2.imshow("th", th)
    cv2.imshow("closing", closing)



    if cv2.waitKey(30) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()