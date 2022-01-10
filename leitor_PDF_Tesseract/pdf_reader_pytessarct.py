# Biliotecas
import cv2
from glob import glob
import pytesseract
import re
from pdf2image import convert_from_path
from pdf2image.exceptions import PDFPageCountError
import os
import pandas as pd

# Caminho
path = os.getcwd()

# Tesseract
pytesseract.pytesseract.tesseract_cmd = os.path.join(
        path, 'Tesseract-OCR', 'tesseract.exe')
tessdata_dir_config = f'--tessdata-dir "{path}\\Tesseract-OCR\\tessdata"'

# Variaveis
INFO_LIMPO = []
HORAS = []
NOME = []
LOGCORROMPIDO = []


# Funcoes Basicas
def verificar_pasta(caminho: str) -> str:
    # Verificar se pasta Repositorio existe

    if os.path.isdir(caminho) is False:
        print(f'A pasta {caminho} não existe. Criando diretório.')
        os.mkdir(caminho)
    else:
        print(f'A pasta {caminho} existe.')


verificar_pasta(path + r'\Repositorio')

pdfs_files = glob(os.path.join(os.getcwd(), path, 'Repositorio', '*.pdf'))


for fn in range(len(pdfs_files)):

    try:
        # Converter pdf para imagem
        images = convert_from_path(
            pdfs_files[fn],
            poppler_path=r'C:\Program Files (x86)\poppler-0.68.0\bin')

    except PDFPageCountError:
        print('Nao foi possivel ler o pdf. Corrompido')
        LOGCORROMPIDO.append(pdfs_files[fn])
        pass

    for i in range(len(images)):
        images[i].save('page.png', 'PNG')

    # Leitura da imagem
    img = cv2.imread('page.png', cv2.IMREAD_UNCHANGED)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    try:
        # Info da imagem(alt, larg)
        hImg, wImg, _ = img.shape

        # Encontrar as palavras na imagem
        boxes = pytesseract.image_to_data(
            img, lang='por', config=tessdata_dir_config)
        INFO = []    # Armazenar as informacoes obtidas da imagem-pdf

        # Encontrar boxes com as infos
        for a, b in enumerate(boxes.splitlines()):
            print(b)
            if a != 0:
                b = b.split()
                if len(b) == 12:
                    INFO.append(b)
                    # x,y,w,h = int(b[6]),int(b[7]),int(b[8]),int(b[9])
                    # cv2.putText(img,b[11],(x,y-5),cv2.FONT_HERSHEY_SIMPLEX,1,(50,50,255),2)
                    # cv2.rectangle(img, (x,y), (x+w, y+h), (50, 50, 255), 2)

        # plt.imshow(img)
        # plt.show()

        # Limpando as informacoes dos boxes
        INFO_LIMPO = []
        for i in range(len(INFO)):
            # print(INFO[i][-1])
            INFO_LIMPO.append(INFO[i][-1])

        INFO_LIMPO = ' '.join(INFO_LIMPO)
        print(INFO_LIMPO)

    except Exception:
        print('Nao foi possivel ler o pdf')

    # Nome
    try:
        # Nome do aluno
        NOME.append(re.search(
            r"certify that (.*[A-Za-z\s]) successfully", INFO_LIMPO).group(1))
    except AttributeError:
        print('Nao rolou')

    # Carga horaria
    try:
        # Carga hiraria do curso
        HORAS.append(re.search(
            r"completed (.*[0-9]) total", INFO_LIMPO).group(1))
    except AttributeError:
        print('Nao rolou')


df = pd.DataFrame({'Nome': NOME, 'Horas': HORAS})
