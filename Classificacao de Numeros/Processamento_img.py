#! python3

# Bibliotecas
import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
import imutils

# Caminho
path = os.getcwd()


def verificar_pasta(caminho):
    # Verificar se pasta Repositorio existe
    if os.path.isdir(caminho) is False:
        print(f'A pasta {caminho} não existe. Criando diretório.')
        os.mkdir(caminho)
    else:
        print(f'A pasta {caminho} existe.')


verificar_pasta(path + r"\Repositorio")


def deletar_txt(caminho):
    '''Funcao para remover os arquivos txt existentes.
        entrada: caminho dos arquivos que serao removidos
    '''

    filelist = [f for f in os.listdir(caminho) if f.endswith(".png")]
    for f in filelist:
        os.remove(os.path.join(caminho, f))


deletar_txt(path + r"\Repositorio")

# Imagem com os dados
img = cv2.imread(path + r'\Teste.png')

# Pre-Processamento
# Imagem em tons de cinza
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Filtro de suavizacao
img_blur = cv2.medianBlur(gray, 3)

# Detectcao de bordas
edges = cv2.Canny(img_blur, 50, 255)

# Morfologia matematica
kernel = np.ones((9, 9), np.uint8)
img_dilation = cv2.dilate(edges, kernel, iterations=1)

# Segementacao
ret, thresh1 = cv2.threshold(img_dilation, 127, 255, cv2.THRESH_BINARY)


titles = ['Imagem Original', 'Tons de Cinza', 'Filtro Blur',
          'Deteccao de Bordas', 'Dilatacao', 'Segmentacao']
images = [img, gray, img_blur, edges, img_dilation, thresh1]


for i in range(6):
    plt.subplot(2, 3, i+1), plt.imshow(images[i], 'gray', vmin=0, vmax=255)
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])
plt.show()


# Localizar contornos
# cv2.RETR_TREE -> recupera todos os contornos e reconstrói uma hierarquia
# completa de contornos aninhados.
# cv2.CHAIN_APPROX_SIMPLE -> Armazena apenas os pontos iniciais e finais dos
# contornos detectados.
cnts = cv2.findContours(img_dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]

# Ordenacao dos contornos encontrados
(cnts, _) = imutils.contours.sort_contours(cnts, method='left-to-right')

ROI_number = 0

for c in cnts:
    # Calcular a area encontrada nos contornos.
    # Se area maior que 1100 provavelmente e um numero
    area = cv2.contourArea(c)
    if area > 1100:
        print(area)
        x, y, w, h = cv2.boundingRect(c)  # Coordenadas dos contornos
        # Selecionar um retangulo
        cv2.rectangle(
            img, (x - 1, y - 1), (x + 1 + w, y + 1 + h), (0, 0, 255), 1)

        ROI = thresh1[y:y+h, x:x+w]
        # Salvar a imagem 28x28 (Modelo treinado MNIST)
        ROI = cv2.resize(ROI, (28, 28), interpolation=cv2.INTER_AREA)
        #  Desenhar o contorno localizado
        cv2.drawContours(img, [c], -1, (0, 255, 0), -1)

        cv2.imwrite(path + f'\\Repositorio\\img_{ROI_number}.png', ROI)
        ROI_number += 1

        cv2.imshow('Digitos encontrados!', img)
        key = cv2.waitKey(0)
        if key == 27:  # (escape to quit)
            cv2.destroyAllWindows()
