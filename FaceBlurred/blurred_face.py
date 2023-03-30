import cv2
import os


class FaceBlur:

    def __init__(self):
        # Verifica se a pasta com Haarcascades existe
        if os.path.isdir('Haarcascades') is False:
            print('A pasta Haarcascades não se encontra.')
        else:
            print('A pasta Haarcascades existe.')

        # Lê o xml
        self.cascade = cv2.CascadeClassifier(os.path.join(
            os.getcwd(), 'Haarcascades', 'haarcascade_frontalface_default.xml'))

    def find_and_blur(self, gray, frame):
        '''
        Função para detectar rosto e borrar:

        Entrada: Imagem em tons de cinza;
                 frame capturado

        Saída:
                Imagem com rosto borrado
        '''
        faces = self.cascade.detectMultiScale(gray, 1.1, 4)

        for (x, y, w, h) in faces:
            roi_frame = frame[y:y+h, x:x+w]
            blur = cv2.GaussianBlur(roi_frame, (101, 101), 0)
            frame[y:y+h, x:x+w] = blur

        return frame

    def run(self):
        # Captura os frames
        cap = cv2.VideoCapture(0)

        while True:
            ret, frame = cap.read()

            # Pre-processamento
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blur = self.find_and_blur(gray, frame)

            cv2.imshow('Video', blur)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


face_blur = FaceBlur()
face_blur.run()