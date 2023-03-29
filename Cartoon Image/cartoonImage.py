import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt


import matplotlib.pyplot as plt

class Cartoonizer:
    def __init__(self, input_path: str):
        self.input_path = Path(input_path)
        self.img_rgb = cv2.imread(str(self.input_path))

    def resize(self, width: int, height: int):
        self.image = cv2.resize(self.img_rgb, (width, height))

    def convert_to_gray(self):
        self.gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

    def apply_gaussian_blur(self, kernel_size: tuple, sigma: float):
        self.gray_image = cv2.GaussianBlur(self.gray_image, kernel_size, sigma)

    def detect_edges(self, ksize: int, threshold: int):
        self.edge_image = cv2.Laplacian(self.gray_image, -1, ksize=ksize)
        self.edge_image = 255 - self.edge_image
        _, self.edge_image = cv2.threshold(self.edge_image, threshold, 255, cv2.THRESH_BINARY)

    def edge_preserving_filter(self, sigma_s: int, sigma_r: float):
        self.edge_preserving_image = cv2.edgePreservingFilter(self.image, flags=2, sigma_s=sigma_s, sigma_r=sigma_r)

    def stylize_image(self, sigma_s: int, sigma_r: float):
        output = np.zeros(self.gray_image.shape)
        output = cv2.bitwise_and(self.edge_preserving_image, self.edge_preserving_image, mask=self.edge_image)
        self.cartoon_image = cv2.stylization(output, sigma_s=sigma_s, sigma_r=sigma_r)

    def show_image(self, img, title):
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title(title)
        plt.axis('off')
        plt.show()

    def show_steps(self):
        self.show_image(self.img_rgb, 'Original')
        self.show_image(self.image, 'Resized')
        self.show_image(self.gray_image, 'Gray')
        self.show_image(self.edge_image, 'Edges')
        self.show_image(self.edge_preserving_image, 'Edge-Preserving Filter')
        self.show_image(self.cartoon_image, 'Cartoon')




if __name__ == "__main__":
    cartoonizer = Cartoonizer('por_sol.png')

    cartoonizer.resize(800, 800)
    cartoonizer.convert_to_gray()
    cartoonizer.apply_gaussian_blur((3, 3), 0)
    cartoonizer.detect_edges(5, 150)
    cartoonizer.edge_preserving_filter(50, 0.4)
    cartoonizer.stylize_image(150, 0.25)

    cartoonizer.show_steps()
