import asyncio
import easyocr
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import logging
import aiofiles
from scipy.stats import linregress
from typing import Optional, Tuple, List, Any

# Configuração do logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler('ocr_processing.log'),
        logging.StreamHandler()
    ]
)

async def process_image(image_path: str) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    try:
        # Usar run_in_executor para operações bloqueantes
        loop = asyncio.get_running_loop()
        
        # Carregamento da imagem de forma assíncrona
        image = await loop.run_in_executor(None, cv2.imread, image_path)
        
        if image is None:
            logging.error(f"Falha ao carregar imagem: {image_path}")
            raise ValueError(f"Não foi possível carregar a imagem: {image_path}")
        
        # OCR em executor separado
        reader = easyocr.Reader(['pt', 'en'])
        result = await loop.run_in_executor(None, reader.readtext, image_path)
        
        logging.info(f"OCR processada para {image_path}: {len(result)} textos detectados")
        
        x_coords: List[int] = []
        y_coords: List[int] = []
        output = image.copy()
        
        for res in result:
            bbox = [(int(x), int(y)) for x, y in res[0]]
            cv2.polylines(output, [np.array(bbox)], isClosed=True, color=(0, 255, 0), thickness=2)
            cv2.putText(output, res[1], (bbox[0][0], bbox[0][1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5,
                        color=(0, 255, 0), thickness=1)
            
            top_left, top_right = bbox[0], bbox[1]
            x_coords.extend([top_left[0], top_right[0]])
            y_coords.extend([top_left[1], top_right[1]])
        
        rotated = None
        if len(x_coords) > 1:
            try:
                slope, _, _, _, _ = linregress(x_coords, y_coords)
                angle = np.degrees(np.arctan(slope))
                (h, w) = image.shape[:2]
                center = (w // 2, h // 2)
                M = cv2.getRotationMatrix2D(center, -angle, 1.0)
                rotated = cv2.warpAffine(output, M, (w, h),
                                         flags=cv2.INTER_CUBIC,
                                         borderMode=cv2.BORDER_REPLICATE)
                logging.info(f"Rotação aplicada para {image_path}. Ângulo: {angle:.2f} graus")
            except Exception as e:
                logging.warning(f"Erro na rotação de {image_path}: {e}")
        
        return output, rotated
    
    except Exception as e:
        logging.error(f"Erro no processamento de {image_path}: {e}")
        raise

async def process_images(input_folder: str):
    if not os.path.exists(input_folder):
        logging.error(f"Pasta {input_folder} não encontrada!")
        return
    
    images = [os.path.join(input_folder, img) for img in os.listdir(input_folder)
              if img.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
    
    if not images:
        logging.warning(f"Nenhuma imagem encontrada na pasta {input_folder}")
        return
    
    logging.info(f"Iniciando processamento de {len(images)} imagens")
    
    # Processamento assíncrono de imagens
    tasks = [process_image(image_path) for image_path in images]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    for image_path, result in zip(images, results):
        if isinstance(result, Exception):
            logging.error(f"Erro ao processar {image_path}: {result}")
            continue
        
        original, rotated = result
        
        # Conversão para RGB
        original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
        
        plt.figure(figsize=(12, 6))
        plt.subplot(121)
        plt.title('Original com OCR')
        plt.imshow(original_rgb)
        plt.axis('off')
        
        if rotated is not None:
            rotated_rgb = cv2.cvtColor(rotated, cv2.COLOR_BGR2RGB)
            plt.subplot(122)
            plt.title('Rotacionada')
            plt.imshow(rotated_rgb)
            plt.axis('off')
        
        plt.tight_layout()
        plt.show()

def main():
    input_folder = 'document'
    asyncio.run(process_images(input_folder))

if __name__ == "__main__":
    main()