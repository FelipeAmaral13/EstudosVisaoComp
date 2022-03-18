import cv2 
import numpy as np


num_face_cap = 5
capture_image = False
font = cv2.FONT_HERSHEY_SIMPLEX

cam = cv2.VideoCapture(0)

while True:
    
    ret, frame = cam.read()

    img_text = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)

    img_text = cv2.cvtColor(img_text, cv2.COLOR_BGR2RGB)
    cv2.putText(img_text, 'Instrucoes:', (25, 50), font, 1, (0,255, 255),2)
    cv2.putText(img_text, 'Aperte "s" para salvar a imagem', (50, 80), font, 1, (0,255, 255),2)
    cv2.putText(img_text, 'Aperte "q" para sair', (50, 110), font, 1, (0,255, 255),2)

    if capture_image == True:
        cv2.putText(frame, 'Salvo', (50, 70), font, 1, (0,255, 255),2)
        cv2.imwrite("face_cap/frame_cap.png", frame)

    else:
        cv2.putText(frame, 'Nao Capturada', (50, 70), font, 1, (255,0, 255),2)
        Cap_frame = 0
        capture_image = False

    img_concate = cv2.hconcat(
        [frame,  img_text])

    cv2.imshow("Image", img_concate)

    # Apertar 'q' para sair
    if cv2.waitKey(1) == ord('q'):
        break
    # Se apertado 'c' começa a captura das imagens. 
    if cv2.waitKey(1)==ord('c'):
        capture_image= True

   
cam.release()
cv2.destroyAllWindows()