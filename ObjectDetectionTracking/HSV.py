import cv2
import numpy as np

cap = cv2.VideoCapture('2.mp4')


def nothing(x):
    pass


cv2.namedWindow('Resultado')

# Começando com 100 para prevenir error da mask
h, s, v = 100, 100, 100

# Creating track bar
cv2.createTrackbar('h', 'Resultado', 0, 179, nothing)
cv2.createTrackbar('s', 'Resultado', 0, 255, nothing)
cv2.createTrackbar('v', 'Resultado', 0, 255, nothing)

while True:

    ret, frame = cap.read()

    # Convertando para HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Pegando as informacoes
    h = cv2.getTrackbarPos('h', 'Resultado')
    s = cv2.getTrackbarPos('s', 'Resultado')
    v = cv2.getTrackbarPos('v', 'Resultado')

    # Mascara
    lower_blue = np.array([h, s, v])
    upper_blue = np.array([180, 255, 255])

    # Aplicando a mascara
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    result = cv2.bitwise_and(frame, frame, mask=mask)

    cv2.imshow('Resultado', result)

    k = cv2.waitKey(30) & 0xFF
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
