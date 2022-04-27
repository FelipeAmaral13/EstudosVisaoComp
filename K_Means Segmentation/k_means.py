import cv2
import numpy as np
import matplotlib.pyplot as plt


def mostrar(imagem):
  fig = plt.gcf()
  fig.set_size_inches(18,6)
  plt.imshow(cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB), cmap='gray')
  plt.axis('off')
  plt.show()

img = cv2.imread('fruits.jpg')
vetorizado = img.reshape((-1, 3))
vetorizado = np.float32(vetorizado)
criterio = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

ret, label, centros = cv2.kmeans(vetorizado, 4, None, criterio, 10, cv2.KMEANS_RANDOM_CENTERS)
centros = np.uint8(centros)
img_final = centros[label.flatten()]

img_final = img_final.reshape(img.shape)
mostrar(img_final)