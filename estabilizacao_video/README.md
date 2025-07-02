# 🌀 Video Stabilizer com Python e OpenCV

Este projeto implementa um pipeline completo de **estabilização de vídeo** utilizando apenas Python e OpenCV. A técnica empregada é baseada em detecção de keypoints (ORB), matching entre frames consecutivos e suavização de trajetória por média móvel — sem uso de deep learning, redes neurais ou bibliotecas externas pesadas.

## ⚙️ Funcionalidades

- Detecção de movimento entre frames usando ORB
- Estimativa de transformações (translação e rotação)
- Suavização da trajetória com média móvel
- Aplicação de transformações suavizadas (warp affine)
- Visualização dos *matches* entre frames
- Output com montagem visual: `Matches | Original + Estabilizado`
- Escrita em `.mp4` com compressão (`mp4v`)

## 📂 Estrutura do Projeto

```bash
video-stabilizer/
├── main.py                # Código-fonte principal com a classe VideoStabilizer
├── teste.mp4              # Vídeo de entrada (exemplo)
├── output_stabilized.mp4  # Vídeo estabilizado gerado
└── README.md              # Este arquivo
```

## 📦 Requisitos

- Python 3.7+
- OpenCV (versão ≥ 4.5)
- NumPy

Instale as dependências com:

```bash
pip install opencv-python numpy
```

## 🚀 Como usar

Basta executar o script passando o vídeo de entrada:

```bash
python main.py
```

O vídeo estabilizado será salvo como `output_stabilized.mp4` e exibido em tempo real com preview dos matches.

## 📈 Exemplo de Resultado

A interface gerada para cada frame possui:

- **Topo**: Matches entre keypoints detectados (ORB)
- **Base**: Original (esquerda) vs. Estabilizado (direita)

```text
┌──────────────────────────────────────────────┐
│      Matches ORB (entre frame t e t+1)       │
├───────────────┬──────────────────────────────┤
│ Original      │ Estabilizado                │
└───────────────┴──────────────────────────────┘
```

## 🧠 Lógica do Pipeline

1. **Extração de Keypoints:** usando ORB
2. **Matching de Descritores:** usando BFMatcher com Hamming
3. **Estimativa de Affine Transform:** `cv2.estimateAffinePartial2D`
4. **Cálculo de trajetória acumulada:** `np.cumsum`
5. **Suavização com média móvel:** `moving_average`
6. **Aplicação da transformação corrigida:** `cv2.warpAffine`
7. **Remoção de bordas pretas:** leve zoom `fix_border()`
8. **Composição visual final e exportação**

## 📌 Observações Técnicas

- O algoritmo atual considera apenas transformações 2D (sem perspectiva).
- A suavização por média móvel pode ser ajustada via `radius`.
- O algoritmo assume vídeos com movimento leve a moderado.
- Ideal para estabilizar vídeos de drones, câmeras manuais ou webcams.

## 📸 Demonstração Visual

![Exemplo de Matches ORB](https://www.avclabs.com/assets/images/blog/fix-shaky-video-easily.jpg)

> *Imagem meramente ilustrativa. Para resultados reais, utilize sua própria filmagem.*

## 🧪 Testado em

- Python 3.11
- OpenCV 4.8.1
- Windows 10 / Ubuntu 22.04

## 📄 Licença

Este projeto é distribuído sob a licença MIT. Veja o arquivo `LICENSE` para mais detalhes.