# Bibliotecas
import os
import cv2
import matplotlib.pyplot as plt
import pytesseract
from pytesseract import Output
import numpy as np
import random

# Caminho atual
path = os.getcwd()

# Caminho do tessseract no Windows
pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

# Ler os Captchas de amostras
file_png = [f for f in os.listdir(path + '\\Captchas_Repo') if f.endswith(".png") ]


for i in range(0,5):
    # Escolher aleatoriamente um captcha
    imagem = random.choice(file_png)
    img = cv2.imread(path + '\\Captchas_Repo\\' + imagem)
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # Segmetacao
    th = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 17, 2)

    #Plot
    titles3 = ['Original', 'Adaptive']
    images3 = [img, th] 

    for i in range(2):
        plt.subplot(1, 2, i + 1), plt.imshow(images3[i], 'gray')
        plt.title(titles3[i])
        plt.xticks([]), plt.yticks([])

    plt.title('Letras')
    plt.show()
    
    # Contornos
    contours, hierarchy = cv2.findContours(th, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    img_th = cv2.cvtColor(th, cv2.COLOR_GRAY2RGB)

    image = cv2.drawContours(img_th, contours, -1, (0, 255, 0), 1)
    plt.imshow(image)
    plt.axis('off')
    plt.show()

    hImg, wImg,_ = image.shape

    boxes = pytesseract.image_to_boxes(image) # Letra detectada

    letras  = []
    for b in boxes.splitlines():
        b = b.split(' ')
        letras.append(b[0]) # Vetor com as letras detectadas
        x, y, w, h = int(b[1]), int(b[2]), int(b[3]), int(b[4])
        cv2.rectangle(image, (x,hImg- y), (w,hImg- h), (50, 50, 255), 1)
        print(letras)
        plt.imshow(image)
        plt.show()

    print(f"O arquivo orginal: {imagem[:4]}. Os caracteres reconhecidos foram: {''.join(letras)}")