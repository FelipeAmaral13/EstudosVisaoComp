import cv2
import numpy as np

num_down = 2
num_bilateral = 7


def cartoon_img(img_rgb: np.ndarray):

    # Downsampling da imagem usando Gaussian Pyramid
    for _ in range(num_down):
        img_rgb = cv2.pyrDown(img_rgb)

    # Aplicacao de um filtro bilateral. O filtro bilateral irá diminuir
    # o pallete das cores, necessario para o efeito de cartoon
    for _ in range(num_bilateral):
        img_rgb = cv2.bilateralFilter(img_rgb, d=9, sigmaColor=9, sigmaSpace=7)

    # Upsampling da imagem usando Gaussian Pyramid
    for _ in range(num_down):
        img_rgb = cv2.pyrUp(img_rgb)

    # Conversao da imagem em tons de cinza
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

    # Filtro mediano para realce
    img_blur = cv2.medianBlur(img_gray, 7)

    # Deteccao de bordas
    img_edge = cv2.adaptiveThreshold(
        img_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY, blockSize=9, C=2)

    # Conversao da imagem para RGB
    img_edge = cv2.cvtColor(img_edge, cv2.COLOR_GRAY2RGB)

    # Combinando a imagem colorida com a imagem com bordas destacadas
    # img_cartoon = cv2.bitwise_and(img_rgb, img_edge)
    # stack = np.hstack([img_rgb, img_cartoon])

    return img_rgb


cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()

    img_rgb = cv2.resize(frame, (800, 800))

    img = cartoon_img(img_rgb)
    cv2.imshow("Cartoon", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
