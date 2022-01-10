# Biblioteca
import glob
import cv2
import os
import pandas as pd

# inicialize o dicionário de índice para armazenar o nome da imagem
# e os histogramas correspondentes
# e o dicionário de imagens para armazenar as próprias imagens


index = {}  # Armazenar o nome da imagem e os histogramas
images = {}  # Armezar as próprias imagens

# Pegar as imagens na pasta
for imagePath in glob.glob(os.getcwd() + "\\*.jpg"):

    # extrair o nome do arquivo da imagem (considerado único) e
    # carregar a imagem, atualizando o dicionário de imagens
    filename = imagePath[imagePath.rfind("/") + 1:]
    image = cv2.imread(imagePath)
    images[filename] = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # extrair um histograma de cores RGB da imagem,
    # usando 8 caixas por canal, normalizar e atualizar o índice
    hist = cv2.calcHist(
        [image], [0, 1], None, [8, 8], [0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    index[filename] = hist


# Metodos para calculo do histograma
OPENCV_METHODS = (
    ("Correlation", cv2.HISTCMP_CORREL),
    ("Chi-Squared", cv2.HISTCMP_CHISQR),
    ("Intersection", cv2.HISTCMP_INTERSECT),
    ("Bhattacharyya", cv2.HISTCMP_BHATTACHARYYA))

imagem_analisada = '\\apple1.jpg'

lista_resultados = []
lista_methodName = []

for (methodName, method) in OPENCV_METHODS:

    results = {}
    reverse = False

    # se estivermos usando a Correlation ou Intersection
    # classificar os resultados na ordem inversa
    if methodName in ("Correlation", "Intersection"):
        reverse = True

    for (k, hist) in index.items():
        # Calcular a distancia entre os dois histogramas usando os metodos
        # Atualizar o dicionario de resultados
        d = cv2.compareHist(
            index[os.getcwd() + imagem_analisada], hist, method)
        results[k] = d

    # Ordenar os resultados
    # print(methodName)
    lista_methodName.append(methodName)
    results = sorted([(v, k) for (k, v) in results.items()], reverse=reverse)
    lista_resultados.append(results)

# Criar o dataframe

for i in range(len(lista_methodName)):
    lista_resultados[i] = pd.DataFrame(lista_resultados[i])
    lista_resultados[i]['Metodo'] = lista_methodName[i]


df = pd.concat(lista_resultados)
df.to_csv('Resultado_compareHist.csv', sep=';', encoding='latin1')


# Analise do DataFrame
# Correlacao
correlation = df.loc[df['Metodo'] == 'Correlation']
correlation.sort_values(0)

# Chi-Quadrado
ChiSquared = df.loc[df['Metodo'] == 'Chi-Squared']
ChiSquared.sort_values(0)

# Intersecao
Intersection = df.loc[df['Metodo'] == 'Intersection']
Intersection.sort_values(0)

# Bhattacharyya
Bhattacharyya = df.loc[df['Metodo'] == 'Bhattacharyya']
Bhattacharyya.sort_values(0)
