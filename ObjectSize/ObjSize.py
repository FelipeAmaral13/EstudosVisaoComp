import cv2
import numpy as np
import imutils


cap = cv2.VideoCapture(0)


while True:

    ret,frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #Filtro Gaussiano
    blurimg = cv2.GaussianBlur(gray, (5,5), 0)

    #Deteccao de bordas por Canny
    edges = cv2.Canny(blurimg, 100, 255)

    #Opercao Morfologica Fechamento
    img_dilate = cv2.dilate(edges, None, iterations=1)
    img_erode = cv2.erode(img_dilate, None, iterations=1)

    #kernel = np.ones((5,5),np.uint8) 
    #img_close = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel) 

    #Contornos
    cnts = cv2.findContours(img_erode, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    (cnts, hir) = contours.sort_contours(cnts)

    #Remover contornos nos quais nao sao suficientemente grandes
    cnts = [x for in cnts if cv2.contourArea(x) > 100]

    cv2.drawContours(frame, cnts, -1, (0,255,0), 3)
    print(len(cnts))



    cv2.imshow("Frame", frame)
    cv2.imshow("Edges", edges)
    cv2.imshow("Close", img_erode)

    if cv2.waitKey(1) == 27:
        break

cv2.destroyAllWindows()
cap.release()