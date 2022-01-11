import numpy as np
import cv2
import matplotlib.pyplot as plt

# Ler imagens
waldo = cv2.imread('waldo.jpg')
puzzle = cv2.imread('puzzle.jpg')

# Converter as imagens
waldoRGB2 = cv2.cvtColor(waldo, cv2.COLOR_BGR2RGB)
puzzleRGB2 = cv2.cvtColor(puzzle, cv2.COLOR_BGR2RGB)

# Shape das imagens
(waldoHeight, waldoWidth) = waldo.shape[:2]
(puzzleHeight, puzzleWidth) = puzzle.shape[:2]

# Inicializando a escala
scale = 0.3
highCorr = float('-inf')
bestHeight, bestWidth, bestScale = 0, 0, scale
finalMaxLoc = None

# Calculando a diferenca dos tamanhos das imagens ate o template matches
while scale <= 2.0:
    temp = cv2.resize(puzzle, (0, 0), fx=scale, fy=scale)
    result2 = cv2.matchTemplate(temp, waldo, cv2.TM_CCOEFF)
    (_, newCorr, newMinLoc, newMaxLoc) = cv2.minMaxLoc(result2)
    # se houver uma alta correlação,
    # mantenha o rastreamento do valor e das coordenadas
    if newCorr > highCorr:
        highCorr = newCorr
        bestScale = scale
        finalMaxLoc = newMaxLoc
    scale += 0.05

# use a nova escala encontrada para o puzzle para encontrar Wally
puzzle = cv2.resize(puzzle, (0, 0), fx=bestScale, fy=bestScale)
# mapear as coordenadas de onde ocorreu a correspondência de alta correlação
topLeft = finalMaxLoc
botRight = (topLeft[0] + waldoWidth, topLeft[1] + waldoHeight)
# estabelecer região de interesse
roi = puzzle[topLeft[1]:botRight[1], topLeft[0]:botRight[0]]

# mascare o resto do quebra-cabeça para destacar a localização do Wally
mask = np.zeros(puzzle.shape, dtype="uint8")
puzzle = cv2.addWeighted(puzzle, 0.25, mask, 0.75, 0)
puzzle[topLeft[1]:botRight[1], topLeft[0]:botRight[0]] = roi

# crie um novo arquivo para mostrar onde ele estava
cv2.imwrite("Waldo_encontrado.jpg", puzzle)
result_rgb = cv2.cvtColor(puzzle, cv2.COLOR_RGB2BGR)
plt.figure(figsize=(15, 15))
plt.imshow(result_rgb)
plt.show()
