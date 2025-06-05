# Full Region Growing Segmenter

Implementação modular em Python de segmentação de imagens baseada em **crescimento de regiões (region growing)**, com validação de parâmetros usando [`Pydantic`](https://docs.pydantic.dev/) e suporte a visualização com `matplotlib`.

Ideal para aplicações didáticas, prototipagem em visão computacional e testes com segmentação espacial baseada em similaridade de intensidade.

---

## 📦 Estrutura da Classe

A classe `FullRegionGrowing` encapsula os seguintes métodos:

- `load_image_gray()`: Carrega todas as imagens PNG em escala de cinza a partir do diretório fornecido.
- `preprocess_image(image)`: Aplica suavização (Gaussiana), limiarização binária e erosão morfológica.
- `full_region_growing(image)`: Executa segmentação por crescimento de região com limiar configurável.
- `overlay_mask_on_image(original, labels)`: Sobrepõe a máscara colorida sobre a imagem original.

---

## ⚙️ Requisitos

- Python 3.8+
- OpenCV (`cv2`)
- NumPy
- Matplotlib
- Pydantic v2+

Instale os pacotes com:

```bash
pip install opencv-python numpy matplotlib pydantic
```

---

## 🧠 Como Funciona

A segmentação ocorre por meio de análise de conectividade 8 (pixels vizinhos em todas as direções). Cada região cresce a partir de um pixel semente, com base na diferença absoluta de intensidade em relação ao valor do pixel inicial.

A máscara final é composta por rótulos inteiros únicos representando diferentes regiões.

---

## 🚀 Exemplo de Uso

```python
from full_region_growing import FullRegionGrowing
from matplotlib import pyplot as plt

segmenter = FullRegionGrowing(path_folder="images", threshold=15)

imgs = segmenter.load_image_gray()

for img in imgs:
    preprocessed = segmenter.preprocess_image(img)
    labels = segmenter.full_region_growing(preprocessed)
    overlay = segmenter.overlay_mask_on_image(img, labels)

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 3, 1)
    plt.imshow(img, cmap='gray')
    plt.title("Imagem Original")
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(labels, cmap='nipy_spectral')
    plt.title("Segmentação")
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    plt.title("Overlay")
    plt.axis('off')

    plt.tight_layout()
    plt.show()
```

---

## 📁 Organização esperada

A pasta `images/` deve conter arquivos `.png` em escala de cinza. Exemplo:

```
images/
├── exemplo1.png
├── exemplo2.png
└── ...
```

---

## 📌 Observações

- O atributo `threshold` controla a sensibilidade à variação de intensidade. Valores mais baixos geram regiões menores e mais homogêneas.
- A validação da pasta é feita automaticamente via `Pydantic`.

---

## 🔒 Licença

Distribuído sob a licença MIT. Livre para uso acadêmico e profissional com atribuição.

---

## ✍️ Autor

Desenvolvido por [Seu Nome] – para uso em aplicações de segmentação clássica com Pydantic e OpenCV.