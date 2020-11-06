import cv2
import numpy as np  

num_down = 2
num_bilateral = 7

img_rgb = cv2.imread('me.jpg')
print(img_rgb.shape)

img_rgb = cv2.resize(img_rgb, (800, 800))

img_color = img_rgb

# Downsampling da imagem usando Gaussian Pyramid
for _ in range(num_down):
    img_color = cv2.pyrDown(img_color)

# Aplicacao de um filtro bilateral. O filtro bilateral irá diminuir o pallete das cores, necessario para o efeito de cartoon
for _ in range(num_bilateral):
    img_color = cv2.bilateralFilter(img_color, d=9, sigmaColor=9, sigmaSpace=7)

# Upsampling da imagem usando Gaussian Pyramid
for _ in range(num_down):
    img_color = cv2.pyrUp(img_color)

# Conversao da imagem em tons de cinza
img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

# Filtro mediano para realce
img_blur = cv2.medianBlur(img_gray, 7)

# Deteccao de bordas
img_edge = cv2.adaptiveThreshold(img_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, blockSize=9, C=2)

# Conversao da imagem para RGB
img_edge = cv2.cvtColor(img_edge, cv2.COLOR_GRAY2RGB)

# Combinando a imagem colorida com a imagem com bordas destacadas
img_cartoon = cv2.bitwise_and(img_color, img_edge)

stack = np.hstack([img_rgb, img_cartoon])
cv2.imshow("Cartoon", stack)
cv2.waitKey(0)