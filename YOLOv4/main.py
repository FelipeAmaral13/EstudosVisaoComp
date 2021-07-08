#! C:\Python\Python38\python.exe

# Bibliotecas
import cv2
import numpy as np
from random import randint


# Cores para as classes encontradas. 4 Classes
COLORS = [(randint(0,255), randint(0,255), randint(0,255)), 
          (randint(0,255), randint(0,255), randint(0,255)), 
          (randint(0,255), randint(0,255), randint(0,255)), 
          (randint(0,255), randint(0,255), randint(0,255))]


# Nomes das classes
class_names = []
with open("coco.names", "r") as f:
    class_names = [cname.strip() for cname in f.readlines()]

cap = cv2.VideoCapture(0)

# Carregando os pesos e as configuracoes da RNA treinada
net = cv2.dnn.readNet("yolov4-tiny.weights", 'yolov4-tiny.cfg')

# Criando o modelo de detecção
model = cv2.dnn_DetectionModel(net)

# Parametros baseados na configuracao da rede
model.setInputParams(size=(416,416), scale=1/255)

while True:
    ret, frame = cap.read()

    # Resultado das classes. Valores de th = [0.1, 0.2] 
    classes, scores, boxes = model.detect(frame, 0.01, 0.2)

    # 
    for (classid, score, box) in zip(classes, scores, boxes):
        # Analise para scores maor que X
        if score > 0.4 :
            for i in range(len(classid)):
                print(f"Classe: {classid[i]}")
            # Cores por classe. Mesma classe terá mesma cor
            x,y,w,h = box
            print(f"X:{x}, Y:{y}, W:{w}, H:{h}")
            color = COLORS[int(classid) % len(COLORS)]
            label = f"{class_names[classid[0]]} : {np.round(score, 2)}" # Label com a maior score e a sua respectiva classificação

            # Retangulo-BOX 
            cv2.rectangle(frame, box, color, 2)
            cv2.putText(frame, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    cv2.imshow("Captura", frame)
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()


