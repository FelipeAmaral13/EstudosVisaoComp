import cv2
import numpy as np


class Cartoonizer:
    def __init__(self, num_down=2, num_bilateral=7):
        self.num_down = num_down
        self.num_bilateral = num_bilateral

    def cartoon_img(self, img_rgb: np.ndarray):

        # Downsampling da imagem usando Gaussian Pyramid
        for _ in range(self.num_down):
            img_rgb = cv2.pyrDown(img_rgb)

        # Aplicacao de um filtro bilateral. O filtro bilateral irá diminuir
        # o pallete das cores, necessario para o efeito de cartoon
        for _ in range(self.num_bilateral):
            img_rgb = cv2.bilateralFilter(img_rgb, d=9, sigmaColor=9, sigmaSpace=7)

        # Upsampling da imagem usando Gaussian Pyramid
        for _ in range(self.num_down):
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

    def cartoonize_webcam(self):
        cap = cv2.VideoCapture(0)

        while True:
            ret, frame = cap.read()

            img_rgb = cv2.resize(frame, (800, 800))

            img = self.cartoon_img(img_rgb)
            cv2.imshow("Cartoon", img)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    cartoonizer = Cartoonizer(num_down=2, num_bilateral=7)
    cartoonizer.cartoonize_webcam()