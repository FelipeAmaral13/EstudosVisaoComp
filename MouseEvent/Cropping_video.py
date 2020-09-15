import cv2
import numpy as np


#Funcao para desnehar o retangulo no video
def draw_rectangle(event, x, y, flags, params):
    global  x_init, y_init, mouse_pressed, top_left_pt, bottom_right_pt

    if event == cv2.EVENT_LBUTTONDOWN:
        mouse_pressed = True
        x_init, y_init = x, y

    elif event == cv2.EVENT_MOUSEMOVE:
        if mouse_pressed:
            top_left_pt = (min(x_init, x), min(y_init, y))
            bottom_right_pt = (max(x_init, x), max(y_init, y))
            frame[y_init:y, x_init:x] = 255 - frame[y_init:y, x_init:x]

    elif event == cv2.EVENT_LBUTTONUP:
        mouse_pressed = False
        top_left_pt = (min(x_init, x), min(y_init, y))
        bottom_right_pt = (max(x_init, x), max(y_init, y))
        frame[y_init:y, x_init:x] = 255 - frame[y_init:y, x_init:x]



#Variaveis para analise do estado do mouse
mouse_pressed = False
top_left_pt, bottom_right_pt = (0,0), (0,0)

#Captacao do video
cap = cv2.VideoCapture('1.mp4')

#Nome da janela do retangulo
cv2.namedWindow('Imagem')
cv2.setMouseCallback('Imagem', draw_rectangle)

while True:
    ret, frame = cap.read()

    #frame  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #frame = cv2.Canny(frame, 100, 180)

    (x0,y0), (x1,y1) = top_left_pt, bottom_right_pt


    # Tranformando a imagem em negativa
    frame[y0:y1, x0:x1] = 255 - frame[y0:y1, x0:x1]
    

    # Aplicando na imagem filtro mediano
    frame[y0:y1, x0:x1] = cv2.medianBlur(frame[y0:y1, x0:x1], 17)

    # Aplicando na imagem filtro bilateral
    #frame[y0:y1, x0:x1] = cv2.bilateralFilter(frame[y0:y1, x0:x1], 9, 200, 200)

    cv2.imshow('Imagem', frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()