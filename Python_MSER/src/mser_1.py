import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

mser = cv2.MSER_create(_delta=8, _min_diversity=0.1)

img = cv2.imread(os.path.join('data','Teste2.png'))
orig = img.copy()

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#Detectar as regioes do MSER 
regions, boxes = mser.detectRegions(gray)

# Criando uma mascara do tamanho da img original
mask = np.zeros((img.shape[0], img.shape[1], 1), dtype=np.uint8)

# Convex hull das regioes do MSER
hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]

# Plotar a imagem original e com a convexhull
img_only = cv2.bitwise_and(img, img, mask=mask)

imagem_total = cv2.hconcat([cv2.cvtColor(img, cv2.COLOR_BGR2RGB), cv2.cvtColor(img_only, cv2.COLOR_BGR2RGB)])
plt.axis('off')
plt.imshow(imagem_total)
plt.show(block=False)
plt.draw()




# https://stackoverflow.com/questions/17647500/exact-meaning-of-the-parameters-given-to-initialize-mser-in-opencv-2-4-x/18614498