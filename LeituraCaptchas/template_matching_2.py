#! python3

# Bibliotecas
import imutils
import cv2
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import random
import time


# Caminho
path = os.getcwd()

# Lendo todos os png existentes na pasta Repositorio captchas Cut
path_repositorio_captcha_cut = path + '\\Captcha_cut'
filelist_captcha_cut = [f for f in os.listdir(path_repositorio_captcha_cut) if f.endswith(".png")]

# Lendo todos os png existentes na pasta templates captchas 
path_repositorio_captcha = path + '\\templates'
filelist_captcha = [f for f in os.listdir(path_repositorio_captcha) if f.endswith(".png")]

# Deletar os arquivos .png da pasta captcha cut
for f in filelist_captcha_cut:
    os.remove(os.path.join(path_repositorio_captcha_cut, f))


# PEGAR O CAPTCHA
captcha_random = random.choice(filelist_captcha)
main_image = cv2.imread(path + f'\\templates\\{captcha_random}')


# Imagem no espaco HSV
hsv = cv2.cvtColor(main_image, cv2.COLOR_BGR2HSV)

# Limites superior e inferior
lim_inf = np.array([15, 0, 0])
lim_sup = np.array([103, 255, 255])

color_mask = cv2.inRange(hsv, lim_inf, lim_sup)

# Filtro Mediano
img_mediano = cv2.medianBlur(color_mask, 3)

# Filtro Bilateral
img_bilateral = cv2.bilateralFilter(img_mediano, 9, 75, 75)

# Img Erodida
elementoEstruturante = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
img_erode = cv2.erode(img_bilateral, elementoEstruturante, iterations=1)

# Threshold
img_th = cv2.adaptiveThreshold(img_erode, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 1)


# Dividir o Captcha em 5 e salvar
img_th1 = img_th[0:71, 0:79]
img_th2 = img_th[0:71, 80:159]
img_th3 = img_th[0:71, 160:238]
img_th4 = img_th[0:71, 239:317]
img_th5 = img_th[0:71, 318:]


cv2.imwrite(path + f'\\Captcha_cut\\captcha_pt_0.png', img_th1)
cv2.imwrite(path + f'\\Captcha_cut\\captcha_pt_1.png', img_th2)
cv2.imwrite(path + f'\\Captcha_cut\\captcha_pt_2.png', img_th3)
cv2.imwrite(path + f'\\Captcha_cut\\captcha_pt_3.png', img_th4)
cv2.imwrite(path + f'\\Captcha_cut\\captcha_pt_4.png', img_th5)

# Lendo todos os png existentes na pasta Repositorio captchas Cut
path_repositorio_captcha_cut = path + '\\Captcha_cut'
filelist_captcha_cut = [f for f in os.listdir(path_repositorio_captcha_cut) if f.endswith(".png")]


# Lendo todos os png existentes na pasta Repositorio dos templates
path_repositorio = path + '\\template_th_bk'
filelist = [f for f in os.listdir(path_repositorio) if f.endswith(".png")]

vetor_letras = []
max_valor_template = []

for j in range(len(filelist_captcha_cut)):

    img_captcha = cv2.imread(path + f'\\Captcha_cut\\{filelist_captcha_cut[j]}')
    

    for i in range(len(filelist)):

        # Ler a imagem template
        template = cv2.imread(path + f'\\template_th_bk\\{filelist[i]}')

        #print(f'Primeiro Print do arquivo {filelist[i]}')
        #template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        #print(f'Segundo Print do arquivo {filelist[i]}')
        #template = cv2.Canny(template, 10, 25)
        #print(f'Terceiro Print do arquivo {filelist[i]}')
        (height, width) = template.shape[:2]
        #print(f'Quarto Print do arquivo {filelist[i]}')


        temp_found = None
        
        match = cv2.matchTemplate(img_captcha, template, cv2.TM_CCOEFF_NORMED)
        (_, val_max, _, loc_max) = cv2.minMaxLoc(match)
        #print([j,{filelist[i]:val_max}])

        # Matriz com os valores dos calculos de valores max
        max_valor_template.append([j,{filelist[i]:val_max}])

            
        #if val_max > 0.88:
        #    print(f'Letra conrrespondente {filelist[i]}')
        #    vetor_letras.append({filelist[i]: val_max})
        #    time.sleep(0.2)


        print(f'O valor de correspondecia e: {val_max}')
        #plt.subplot(1,3,1), plt.imshow(img_captcha)
        #plt.subplot(1,3,2), plt.imshow(template)
        #plt.subplot(1,3,3), plt.imshow(img_th, cmap='gray')
        #plt.show()
    
# Print da imagem do captcha e os vetores candidatos como solucao
#print(captcha_random, vetor_letras) 



# Organizando o vetor com os candidatos a solucao de acordo com a potuacao do vetor max_valor_template
mat = pd.DataFrame(max_valor_template)

list_mat = list(mat.groupby([0]))

campeoes=[] # Matriz com os templates com melhor valor_max calculado

for i in range(len(list_mat)):
    a = list_mat[i][1]
    pontos = [list(valor.values()) for valor in a[1]]
    pontuacao = [ponto[0] for ponto in pontos]
    campea = np.flip(np.argsort(pontuacao))

    campeoes.append(campea[0])

# Vetor dos captchas solucao
letras_captchas = []
for j in range(len(campeoes)):
    letras_captchas.append(filelist[campeoes[j]].split('.')[0])

letras_captchas_total = pd.Series(letras_captchas)

letras_captchas_total = letras_captchas_total.str.replace('C-Dilha', 'Ç')

letras_captchas_total = letras_captchas_total.tolist()
print(letras_captchas_total)
print(captcha_random)
plt.imshow(img_th, cmap='gray')
plt.show()

