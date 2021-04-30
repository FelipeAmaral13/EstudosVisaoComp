from imutils import face_utils
import dlib
import cv2
import imutils
import numpy as np


 
#Repositorio das faces
p = "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(p)

cap = cv2.VideoCapture(0)
 
while True:
    # Capturar os frames
    ret, image = cap.read()

    image = imutils.resize(image, width=500)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detectar a face em tons de cinza
    rects = detector(gray, 0)

    # Loop para detectcao de faces
    for (i, rect) in enumerate(rects):
        # Determina a facial landmarks na regiao da face
        #entao converte a para as coordenandas (x,y) as landmarks para NumpyArray
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        # Converter retangulo dlib's para Opencv-bounding box
        # [i.e., (x, y, w, h)], Desenha a face
        (x, y, w, h) = face_utils.rect_to_bb(rect)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Face Number
        cv2.putText(image, "Face #{}".format(i + 1), (x - 10, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Mostrar as landmarks
        for (x, y) in shape:
            cv2.circle(image, (x, y), 1, (0, 0, 255), -1)
    
    cv2.imshow("Frame", image)
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
cap.release()
