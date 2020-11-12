# Bibliotecas
import numpy as np
from skimage import io, img_as_ubyte
from skimage.transform import rotate, AffineTransform, warp
from skimage.util import random_noise
import random
import os
import cv2
import matplotlib.pyplot as plt

images_path = os.getcwd() + '\\BD_Oficial'  # path das imagens originais

augmented_path = os.getcwd() + '\\Aumented_Images'  # path das imagens modificadas

# Imagens Originais
images = []
for im in os.listdir(images_path):
    images.append(os.path.join(images_path, im))

images_to_generate = 3  # qtd de imagens a serem geradas
i = 1                         # variavel para inteirar no images_to_generate

# Funcoes para geracao de imagens


def rotacao_anti(image):
    '''
        Função responsável por fazer a rotação anti-horaria da imagem.
        Entrada: Imagem 
        Saída: Imagem rotacionada entre 0 a 180° no sentindo anti-horario
    '''
    angle = random.randint(0, 180)
    return rotate(image, angle)


def rotacao_horaria(image):
    '''
        Função responsável por fazer a rotação horaria da imagem.
        Entrada: Imagem 
        Saída: Imagem rotacionada entre 0 a 180° no sentindo horario
    '''
    angle = random.randint(0, 180)
    return rotate(image, -angle)


def h_flip(image):
    '''
        Função responsável por fazer a inversão horizontal da imagem.
        Entrada: Imagem 
        Saída: Imagem invertida no sentido horizontal
    '''
    return np.fliplr(image)


def v_flip(image):
    '''
        Função responsável por fazer a inversão vertical da imagem.
        Entrada: Imagem 
        Saída: Imagem invertida no sentido vertical
    '''
    return np.flipud(image)


def ruidos_img(image):
    '''
        Função responsável por inserir ruídos randomincos do tipo sal e pimenta na imagem.
        Entrada: Imagem 
        Saída: Imagem com ruidos do tipo sal e pimenta
    '''
    return random_noise(image)


def warp_shift(image):
    '''
        Função responsável por fazer a transformacao geometrica de rotacao em relacao as linhas paralelas das imagens.
        Entrada: Imagem 
        Saída: Imagem rotacionada
    '''
    transform = AffineTransform(translation=(0, 40))
    warp_image = warp(image, transform, mode="wrap")
    return warp_image

def brightness(image):
    '''
        Função responsável por incrementar brilho a imagem.
        Entrada: Imagem 
        Saída: Imagem com brilho
    '''
    bright = np.ones(image.shape, dtype="uint8") * 70
    brightincrease = cv2.add(image, bright)

    return brightincrease

def blur_img(image):
    '''
        Função responsável por aplicar um filtro mediana na imagem.
        Entrada: Imagem 
        Saída: Imagem com filtro mediana    '''

    k_size = random.randrange(1,10,2)
    img_blur = cv2.medianBlur(image, k_size)
    return img_blur



# Dicionario para ativacao das funcoes
transformations = {'Rotacao anti-horaria': rotacao_anti,
                   'Rotacao horaria': rotacao_horaria,
                   'Horizontal flip': h_flip,
                   'Vertical flip': v_flip,
                   'warp shift': warp_shift,
                   'Ruidos': ruidos_img,
                   'Brilho': brightness,
                   'Blur Image': blur_img
                   }



while i <= images_to_generate:
    image = random.choice(images)
    original_image = io.imread(image)
    transformed_image = []
#     print(i)
    n = 0       # variável para iterar até o número de transformação 
    # escolha um número aleatório de transformação para aplicar na imagem
    transformation_count = random.randint(1, len(transformations))

    while n <= transformation_count:
        # Escolha aleatorio do metodo a ser aplicado
        key = random.choice(list(transformations))
        print(key)
        transformed_image = transformations[key](original_image)
        n += 1

    new_image_path = "%s/augmented_image_%s.jpg" % (augmented_path, i)
    # Converta uma imagem para o formato de byte sem sinal, com valores em [0, 255].
    transformed_image = img_as_ubyte(transformed_image)
    # converter a imagem antes d egravar
    transformed_image = cv2.cvtColor(transformed_image, cv2.COLOR_BGR2RGB)
    # Salvar a imagem ja convertida
    cv2.imwrite(new_image_path, transformed_image)
    i = i+1



