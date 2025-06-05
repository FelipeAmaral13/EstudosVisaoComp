from pydantic import BaseModel, Field, field_validator
from typing import List, Optional
import cv2
import numpy as np
import glob
import os
from matplotlib import pyplot as plt

class FullRegionGrowing(BaseModel):
    path_folder: str = Field(..., description="Caminho da pasta com imagens PNG")
    threshold: int = Field(15, ge=0, le=255, description="Limiar para crescimento de região")
    
    @field_validator("path_folder")
    def validate_path_exists(cls, v):
        if not os.path.exists(v):
            raise ValueError(f"Pasta não encontrada: {v}")
        return v

    def load_image_gray(self) -> List[np.ndarray]:
        image_list = []
        path_image = glob.glob(os.path.join(self.path_folder, '*.png'))
        for img_path in path_image:
            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise FileNotFoundError(f"Erro ao carregar imagem: {img_path}")
            image_list.append(image)
        return image_list

    @staticmethod
    def preprocess_image(image: np.ndarray) -> np.ndarray:
        blurred = cv2.GaussianBlur(image, (5, 5), 0)
        _, thresholded = cv2.threshold(blurred, 180, 255, cv2.THRESH_BINARY_INV)
        dilated = cv2.erode(thresholded, np.ones((5, 5), np.uint8), iterations=1)
        return dilated

    @staticmethod
    def overlay_mask_on_image(original_gray_image: np.ndarray, labels: np.ndarray) -> np.ndarray:
        label_norm = (labels.astype(np.float32) / (labels.max() + 1e-5) * 255).astype(np.uint8)
        label_color = cv2.applyColorMap(label_norm, cv2.COLORMAP_JET)
        original_bgr = cv2.cvtColor(original_gray_image, cv2.COLOR_GRAY2BGR)
        blended = cv2.addWeighted(original_bgr, 0.5, label_color, 0.5, 0)
        return blended

    def full_region_growing(self, image: np.ndarray) -> np.ndarray:
        height, width = image.shape
        visited = np.zeros_like(image, dtype=bool)
        labels = np.zeros_like(image, dtype=np.int32)
        label = 1

        neighbors = [(-1, -1), (-1, 0), (-1, 1),
                     (0, -1),          (0, 1),
                     (1, -1), (1, 0),  (1, 1)]

        for y in range(height):
            for x in range(width):
                if not visited[y, x]:
                    seed_value = int(image[y, x])
                    queue = [(y, x)]
                    visited[y, x] = True

                    while queue:
                        cy, cx = queue.pop(0)
                        labels[cy, cx] = label

                        for dy, dx in neighbors:
                            ny, nx = cy + dy, cx + dx
                            if 0 <= ny < height and 0 <= nx < width:
                                if not visited[ny, nx] and abs(int(image[ny, nx]) - seed_value) <= self.threshold:
                                    visited[ny, nx] = True
                                    queue.append((ny, nx))

                    label += 1

        return labels

if __name__ == "__main__":
    segmenter = FullRegionGrowing(path_folder="images", threshold=15)

    imgs = segmenter.load_image_gray()
    for img in imgs:
        preprocessed = segmenter.preprocess_image(img)
        labels = segmenter.full_region_growing(preprocessed)
        overlay = segmenter.overlay_mask_on_image(img, labels)

        plt.figure(figsize=(12, 8))
        plt.subplot(1, 3, 1)
        plt.imshow(img, cmap='gray')
        plt.title("Original")
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.imshow(labels, cmap='nipy_spectral')
        plt.title("Rótulos")
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        plt.title("Overlay")
        plt.axis('off')

        plt.tight_layout()
        plt.show()