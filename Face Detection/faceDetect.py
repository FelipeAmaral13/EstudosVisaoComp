import cv2
import numpy as np

# Load classificador de face HAAR
face_classifier = cv2.CascadeClassifier('Haarcascades/haarcascade_frontalface_default.xml')

def faceDetect(img):
    #Função para detectar face e cropar.
    #Se face não detetctada, retornar a imagem de entrada
    
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    
    if faces is ():
        return None
    
    # Cropar as imagens
    for (x,y,w,h) in faces:
        cropped_face = img[y:y+h, x:x+w]

    return cropped_face

# Inicializar a captação de imagens
cap = cv2.VideoCapture(0)
count = 0

# Coletar 100 amostras
while True:

    ret, frame = cap.read()
    if faceDetect(frame) is not None:
        count += 1
        face = cv2.resize(faceDetect(frame), (200, 200))
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

        # Salvar as amostras
        face_path = './faces/user/' + str(count) + '.jpg'
        cv2.imwrite(face_path, face)

        # Mostrar o contador de amostras na imagem captada
        cv2.putText(face, str(count), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
        cv2.imshow('Face Cropper', face)
        
    else:
        print("Face Não Encontrada")
        pass

    if cv2.waitKey(1) == 13 or count == 100: #13 = Enter
        break
        
cap.release()
cv2.destroyAllWindows()      
print("Amostras coletadas com sucesso")