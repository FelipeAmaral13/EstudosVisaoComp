# Bibliotecas
import numpy as np
import cv2

# Escala de tamanho
ds_factor = 0.6

# Valores de Inicio do espaco de cor HSV
h, s, v = 100, 100, 100


class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)

    def __del__(self):
        self.video.release()

    def get_frame(self):

        ret, frame = self.video.read()
        frame = cv2.resize(frame, None, fx=ds_factor,
                           fy=ds_factor, interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Mascara
        lower_blue = np.array([h, s, v])
        upper_blue = np.array([140, 255, 255])

        # Aplicando a mascara
        mask = cv2.inRange(hsv, lower_blue, upper_blue)

        result = cv2.bitwise_and(frame, frame, mask=mask)

        ret, jpeg = cv2.imencode('.jpg', result)
        return jpeg.tobytes()
