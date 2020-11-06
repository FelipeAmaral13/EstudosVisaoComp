# Bibliotecas

import numpy as np
from skimage import io 
from  .transform import rotate, AffineTransform, warp
import matplotlib.pyplot as plt
import random
from skimage import img_as_ubyte
import os
from skimage.util import random_noise
import cv2

images_path = os.getcwd() + '\\BD_Oficial' #path das imagens originais

augmented_path = os.getcwd() + '\\Aumented_Images' # path das imagens modificadas

images=[] 

for im in os.listdir(images_path):     
    images.append(os.path.join(images_path,im))

images_to_generate=10       #qtd de imagens a serem geradas
i=1                         # variavel para inteirar no images_to_generate

# Funcoes para geracao de imagens
def anticlockwise_rotation(image):
    '''
        Função responsável por fazer a rotação anti-horaria da imagem.
        Entrada: Imagem 
        Saída: Imagem rotacionada entre 0 a 180° no sentindo anti-horario
    '''
    angle= random.randint(0,180)
    return rotate(image, angle)

def clockwise_rotation(image):
    '''
        Função responsável por fazer a rotação horaria da imagem.
        Entrada: Imagem 
        Saída: Imagem rotacionada entre 0 a 180° no sentindo horario
    '''
    angle= random.randint(0,180)
    return rotate(image, -angle)

def h_flip(image):
    '''
        Função responsável por fazer a inversão horizontal da imagem.
        Entrada: Imagem 
        Saída: Imagem invertida no sentido horizontal
    '''
    return  np.fliplr(image)

def v_flip(image):
    '''
        Função responsável por fazer a inversão vertical da imagem.
        Entrada: Imagem 
        Saída: Imagem invertida no sentido vertical
    '''
    return np.flipud(image)

def add_noise(image):
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
    transform = AffineTransform(translation=(0,40))  
    warp_image = warp(image, transform, mode="wrap")
    return warp_image


# Dicionario para ativacao das funcoes
transformations = {'rotate anticlockwise': anticlockwise_rotation,
                      'rotate clockwise': clockwise_rotation,
                      'horizontal flip': h_flip, 
                      'vertical flip': v_flip,
                   'warp shift': warp_shift,
                   'adding noise': add_noise                   
                 }                #use dictionary to store names of functions 

while i<=images_to_generate:    
    image=random.choice(images)
    original_image = io.imread(image)
    transformed_image=None
#     print(i)
    n = 0       # variável para iterar até o número de transformação para aplicar
    transformation_count = random.randint(1, len(transformations)) # escolha um número aleatório de transformação para aplicar na imagem
    
    while n <= transformation_count:
        key = random.choice(list(transformations)) # Escolha aleatorio do metodo a ser aplicado
        transformed_image = transformations[key](original_image)
        n = n + 1
        
    new_image_path= "%s/augmented_image_%s.jpg" %(augmented_path, i)
    transformed_image = img_as_ubyte(transformed_image)  # Converta uma imagem para o formato de byte sem sinal, com valores em [0, 255].
    transformed_image=cv2.cvtColor(transformed_image, cv2.COLOR_BGR2RGB) # converter a imagem antes d egravar
    cv2.imwrite(new_image_path, transformed_image) # Salvar a imagem ja convertida
    i =i+1

