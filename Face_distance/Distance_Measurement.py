import cv2
import numpy as np
from utils import Dist_Focal, Dista_Medida

DIST_CAM = 45 # distancia conhecida da cam ao rosto
FACE_TAM = 17 # tamanho da face

face_detector = cv2.CascadeClassifier(r'Haarcascades/haarcascade_frontalface_default.xml')
FONT = cv2.FONT_HERSHEY_SIMPLEX 
reference_image =cv2.imread(r"face_cap\frame_cap.png")


def Face_Detection(image: np.ndarray):
    """
    Funcao para detectar a face com Haarcascade
        image: imagem de referenia 

        saida:
            f_width: Tamanho da caixa delimitadora calculada quando detectado o rosto com Haarcascade
            image: iamgem de referencia
    """

    face_w = 0
    Gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = face_detector.detectMultiScale(Gray_image, 1.3, 5)

    for (x, y, h, w) in faces:
        cv2.rectangle(image, (x,y), (x+w, y+h), (255,255,255), 1)
        face_w = w

    print(face_w)
    
    return face_w, image


face_w, image_read = Face_Detection(reference_image)

dist_focal_calc = Dist_Focal(DIST_CAM, FACE_TAM, face_w)

cam = cv2.VideoCapture(0)

while True:

    ret, frame = cam.read()
    img_text = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
    img_text = cv2.cvtColor(img_text, cv2.COLOR_BGR2RGB)

    Gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(Gray_image, 1.3, 5)

    for (x, y, h, w) in faces:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (255, 255, 255), 1)
        distance =Dista_Medida(FACE_TAM, dist_focal_calc, w)
        print(distance)

        cv2.putText(img_text, f"Distancia = {round(distance, 2)} cm", (50, 50), FONT, 1, (0, 0, 255), 3)

    img_concate = cv2.hconcat(
        [frame,  img_text])

    cv2.imshow('Distancia prevista:', img_concate)

    if cv2.waitKey(1) == ord('q'):
        break


cam.release()
cv2.destroyAllWindows()