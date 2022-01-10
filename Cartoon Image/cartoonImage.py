import cv2
import numpy as np
import os

img_rgb = cv2.imread(os.path.join(os.getcwd(), 'por_sol.png'))
print(img_rgb.shape)

image = cv2.resize(img_rgb, (800, 800))

# Imagem em escala de cinza
grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Filtro gaussianBlur
grayImage = cv2.GaussianBlur(grayImage, (3, 3), 0)

# Deteccao de bordas
edgeImage = cv2.Laplacian(grayImage, -1, ksize=5)
edgeImage = 255 - edgeImage

# threshold
ret, edgeImage = cv2.threshold(edgeImage, 150, 255, cv2.THRESH_BINARY)

# Desforque da imagem usando a funcao edgePreservingFilter
edgePreservingImage = cv2.edgePreservingFilter(
    image, flags=2, sigma_s=50, sigma_r=0.4
    )

# Criando uma matriz de saida
output = np.zeros(grayImage.shape)

# Combinando a imagem cartoon e a imagem com bordas detectadas
output = cv2.bitwise_and(
    edgePreservingImage, edgePreservingImage, mask=edgeImage)

cartoon_image = cv2.stylization(output, sigma_s=150, sigma_r=0.25)

cv2.imshow('cartoon', cartoon_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
