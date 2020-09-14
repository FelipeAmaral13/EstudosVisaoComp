import cv2
import numpy as np

# Criar uma imagem teste (preta)
image = np.zeros((400,400,3), np.uint8)

# Variaveis globais
cropping = False 
x_start, y_start, x_end, y_end = 0, 0, 0, 0
 
oriImage = image.copy()
 
 
def mouse_crop(event, x, y, flags, param):
    
    # Pega a referencias das variaveis globais declaradas
    global x_start, y_start, x_end, y_end, cropping
 
    # Se o bt esquerdo for apertado, começar o crop
    # (x, y) coord, inicio do crop
    if event == cv2.EVENT_LBUTTONDOWN:
        x_start, y_start, x_end, y_end = x, y, x, y
        cropping = True
 
    # Mov Mouse
    elif event == cv2.EVENT_MOUSEMOVE:
        if cropping == True:
            x_end, y_end = x, y
 
    # Bt esquerdo solto
    elif event == cv2.EVENT_LBUTTONUP:
        # armazenar o final da  (x, y) coord
        x_end, y_end = x, y
        cropping = False 
 
        refPoint = [(x_start, y_start), (x_end, y_end)] # vetor de referencia
 
        if len(refPoint) == 2: # 2 pts encontrados
            roi = oriImage[refPoint[0][1]:refPoint[1][1], refPoint[0][0]:refPoint[1][0]]
            cv2.imshow("Cropped", roi)
            cv2.imwrite('C:\\Users\\megan\\Desktop\\GithubMeganha\\ProjetosVisaoComp\\MouseEvent\\CropingImage.py\\Cropped.jpg', roi) 
 
cv2.namedWindow("image")
cv2.setMouseCallback("image", mouse_crop)
 
while True:
 
    img = image.copy()
 
    if not cropping:
        cv2.imshow("image", image)
 
    elif cropping:
        cv2.rectangle(img, (x_start, y_start), (x_end, y_end), (255, 0, 0), 2)
        cv2.imshow("image", img)



 
    if cv2.waitKey(1) == ord('q'):
        break
 
cv2.destroyAllWindows()