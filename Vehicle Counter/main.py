import numpy as np
from time import sleep
import argparse

try:
    import cv2
except ImportError:
    raise ImportError ('Não foi possível encontrar o módulo OpenCV Python. Confira as variáveis de ambiente.'
)

#Variaveis globais

largura_min = 80  # Largura minima do retangulo
altura_min = 80  # Altura minima do retangulo
offset = 6  # Erro permitido entre pixel
pos_linha = 550  # Posição da linha de contagem
delay = 60  # FPS do vídeo
detec = []

def pega_centro(x, y, largura, altura):
    """
    Funcao para encontrar o centro do objeto.
    :param x: x do objeto
    :param y: y do objeto
    :param largura: largura do objeto
    :param altura: altura do objeto
    :return: tupla que contém as coordenadas do centro de um objeto
    """

    x1 = largura // 2
    y1 = altura // 2
    cx = x + x1
    cy = y + y1
    return cx, cy


def set_info(detec):
    global carros
    for (x, y) in detec:
        if (pos_linha + offset) > y > (pos_linha - offset):
            carros += 1
            cv2.line(frame1, (25, pos_linha), (1200, pos_linha), (0, 127, 255), 3)
            detec.remove((x, y))
            print("Carros detectados até o momento: " + str(carros))


def show_info(frame1, dilatada):
    text = f'Carros: {carros}'
    cv2.putText(frame1, text, (450, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)
    cv2.imshow("Video Original", frame1)
    #cv2.imshow("Detectar", dilatada)


carros = caminhoes = 0
subtracao = cv2.bgsegm.createBackgroundSubtractorMOG()  # Pega o fundo e subtrai do que está se movendo

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description = 'Script para contar veículos em uma estrada.' 
        'Adaptado https://github.com/gustavogino/Vehicle-Counter' 
    )
    parser.add_argument('--video', help='Caminho do video exemplo. Se vazio, ligar a webcam')
    parser.add_argument('--largura_min', help='Largura minima. Default 80')
    parser.add_argument('--tempoFrame', help='printar os frames')

    args = parser.parse_args()

    if args.video:
        cap = cv2.VideoCapture(args.video)
    else:
        cap = cv2.VideoCapture(0)

    while True:
        ret, frame1 = cap.read()  # Pega cada frame do vídeo

        tempo = float(1 / delay)
        sleep(tempo)  # Dá um delay entre cada processamento
        if args.tempoFrame:
            print(sleep(tempo))
        else:
            sleep(tempo)
        
        # Processamento
        grey = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)  # Pega o frame e transforma para preto e branco
        
        blur = cv2.GaussianBlur(grey, (3, 3), 5)  # Faz um blur para tentar remover as imperfeições da imagem
        
        img_sub = subtracao.apply(blur)  # Faz a subtração da imagem aplicada no blur
        
        dilat = cv2.dilate(img_sub, np.ones((5, 5)))  # "Engrossa" o que sobrou da subtração
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)) 
        dilatada = cv2.morphologyEx(dilat, cv2.MORPH_CLOSE, kernel)  # Tenta preencher todos os "buracos" da imagem

        #Encontrar os contornos
        contorno, img = cv2.findContours(dilatada, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        #Linha de corte 
        cv2.line(frame1, (25, pos_linha), (1200, pos_linha), (255, 127, 0), 3) 
        for (i, c) in enumerate(contorno):
            (x, y, w, h) = cv2.boundingRect(c)
            if args.largura_min:
                largura_min = args.largura_min 
            else:
                validar_contorno = (w >= largura_min) and (h >= altura_min)
                if not validar_contorno:
                    continue

                cv2.rectangle(frame1, (x, y), (x + w, y + h), (0, 255, 0), 2)
                centro = pega_centro(x, y, w, h)
                detec.append(centro)
                cv2.circle(frame1, centro, 4, (0, 0, 255), -1)

        set_info(detec)
        show_info(frame1, dilatada)

        if cv2.waitKey(1) == 27:
            break

    cv2.destroyAllWindows()
    cap.release()
