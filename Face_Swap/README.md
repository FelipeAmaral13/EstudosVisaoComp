# Face Swap Real-Time com OpenCV e MediaPipe

Projeto robusto e modular para **troca de rostos em tempo real** (Face Swap) utilizando Python, OpenCV, MediaPipe FaceMesh e blending avançado via skimage. Aplicação voltada para demonstrações, prototipação de POCs e estudos em visão computacional.

---

## Visão Geral

Este projeto executa a troca do rosto capturado em tempo real da webcam por um rosto-alvo fornecido via imagem, empregando:

- **Detecção e extração de landmarks faciais** com MediaPipe FaceMesh
- **Triangulação de Delaunay** para mapeamento ponto-a-ponto
- **Warping afim** de triângulos do rosto de origem para o destino
- **Correção de cor** com histogram matching (skimage)
- **Blending final** usando máscaras elípticas e `seamlessClone` do OpenCV
- **Visualização comparativa** lado-a-lado com opção de exibir landmarks

O pipeline foi desenhado para ser **simples de customizar** e facilmente adaptável para novos experimentos em processamento facial.

---

## Requisitos

- Python >= 3.9  
- OpenCV >= 4.7  
- mediapipe >= 0.10.0  
- scikit-image >= 0.20  
- numpy

**Instalação rápida (via pip):**
```bash
pip install opencv-python mediapipe scikit-image numpy
```

---

## Como Usar

1. **Clone o repositório**
   ```bash
   git clone https://github.com/seu-usuario/face-swap-realtime.git
   cd face-swap-realtime
   ```

2. **Execute o script principal**
   ```bash
   python face_swap.py --source caminho/para/rosto_origem.jpg --camera 0
   ```
   - `--source`: caminho para a imagem do rosto que será transplantado
   - `--camera`: ID da webcam (padrão: 0)
   - `--show-landmarks`: (opcional) exibe landmarks faciais na visualização

**Exemplo completo:**
```bash
python face_swap.py --source exemplo_face.jpg --camera 0 --show-landmarks
```

Pressione **Q** para encerrar a execução.

---

## Arquitetura do Pipeline

1. **Detecção de Landmarks**  
   MediaPipe FaceMesh localiza e retorna pontos de referência faciais para ambos os rostos (imagem e vídeo).

2. **Triangulação de Delaunay**  
   Cria malha consistente de triângulos para transferência geométrica fiel entre os rostos.

3. **Warp Triângulo-a-Triângulo**  
   Cada triângulo do rosto origem é deformado para coincidir com a malha do rosto destino.

4. **Correção de Cor (Match Histograms)**  
   Ajusta a paleta de cor do rosto transplantado para combinar com a imagem de destino, evitando artefatos.

5. **Blending Avançado**  
   Máscara elíptica com feathering e seamlessClone do OpenCV para integração natural do rosto.

6. **Visualização Comparativa**  
   Tela única exibindo: frame original, imagem fonte e resultado final.

---

## Aplicações e Limitações

- **Aplicações:**  
  - Prototipagem de apps de entretenimento, filtros de vídeo, demonstrações educacionais e testes de pesquisa em morphing facial.
- **Limitações:**  
  - Pipeline sensível a iluminação e posição do rosto
  - Não suporta múltiplos rostos simultaneamente
  - Não realiza realinhamento 3D (apenas 2D)

**Disclaimer:**  
Uso restrito a propósitos acadêmicos e de prototipação. Não recomendado para produção, deepfakes maliciosos ou qualquer uso antiético.

---

## Contato / Dúvidas

Para sugestões, críticas ou colaborações, entre em contato:  
Felipe Meganha  
[LinkedIn](https://www.linkedin.com/in/felipemeganha/) | [Substack](https://felipemeganha.substack.com/)

---

## Licença

Uso livre para fins educacionais e POC.  
Se utilizar, cite este repositório.  
Proibido uso para produção comercial ou distribuição de deepfakes sem autorização.
