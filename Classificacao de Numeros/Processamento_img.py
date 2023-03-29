import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
from imutils import contours
import imutils

class ImageProcessing:
    def __init__(self, image_path, save_dir):
        self.image_path = image_path
        self.save_dir = save_dir

        self.gray = None
        self.img_blur = None
        self.edges = None
        self.img_dilation = None
        self.thresh1 = None
        self.img = None
        self.cnts = None

        self.load_image()
        self.pre_process()
        self.find_contours()

    def load_image(self):
        if os.path.exists(self.image_path):
            self.img = cv2.imread(self.image_path)
        else:
            raise ValueError(f"Image path '{self.image_path}' does not exist.")

    def verificar_pasta(self):
        if not os.path.isdir(self.save_dir):
            print(f"A pasta {self.save_dir} não existe. Criando diretório.")
            os.mkdir(self.save_dir)
        else:
            print(f"A pasta {self.save_dir} existe.")

    def deletar_imagens(self):
        filelist = [f for f in os.listdir(self.save_dir) if f.endswith(".png")]
        for f in filelist:
            os.remove(os.path.join(self.save_dir, f))

    def pre_process(self):
        self.verificar_pasta()
        self.deletar_imagens()

        self.gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        self.img_blur = cv2.medianBlur(self.gray, 3)
        self.edges = cv2.Canny(self.img_blur, 50, 255)
        kernel = np.ones((9, 9), np.uint8)
        self.img_dilation = cv2.dilate(self.edges, kernel, iterations=1)
        ret, self.thresh1 = cv2.threshold(self.img_dilation, 127, 255, cv2.THRESH_BINARY)

    def find_contours(self):
        cnts = cv2.findContours(self.img_dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        cnts = imutils.grab_contours(cnts)
        # (self.cnts, _) = imutils.contours.sort_contours(cnts, method="left-to-right")
        (self.cnts, boundingBoxes) = contours.sort_contours(cnts, method="left-to-right")

    def save_ROI_images(self):
        ROI_number = 0
        for c in self.cnts:
            area = cv2.contourArea(c)
            if area > 1100:
                x, y, w, h = cv2.boundingRect(c)
                cv2.rectangle(
                    self.img, (x - 1, y - 1), (x + 1 + w, y + 1 + h), (0, 0, 255), 1
                )

                ROI = self.thresh1[y : y + h, x : x + w]
                ROI = cv2.resize(ROI, (28, 28), interpolation=cv2.INTER_AREA)
                cv2.drawContours(self.img, [c], -1, (0, 255, 0), -1)

                cv2.imwrite(
                    os.path.join(self.save_dir, f"img_{ROI_number}.png"), ROI
                )
                ROI_number += 1

        plt.imshow(self.img, "gray", vmin=0, vmax=255)
        plt.title("Digitos encontrados!")
        plt.show()


if __name__ == "__main__":
    im = ImageProcessing("Teste.png", "Repositorio")
    im.save_ROI_images()