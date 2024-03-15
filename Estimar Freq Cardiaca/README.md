# Estimador de Frequência Cardíaca por Webcam

Este aplicativo captura vídeo de uma webcam e estima a frequência cardíaca da pessoa em foco com base em alterações na cor da pele causadas pelo fluxo sanguíneo. Ele detecta rostos na transmissão de vídeo e calcula a frequência cardíaca analisando as variações de cor na região da testa.
Requisitos

- Python 3.x
- OpenCV (cv2)
- NumPy
- SciPy
- PySide6

## Utilização

- Clone este repositório em sua máquina local.
- Instale as dependências necessárias usando pip:

`pip install opencv-python-headless numpy scipy PySide6`

## Execute o aplicativo:

`python webcam_heart_rate.py`

- Clique no botão "Iniciar" para começar a capturar vídeo da sua webcam.
- Assim que a transmissão de vídeo iniciar, o aplicativo detectará rostos e estimará a frequência cardíaca em batimentos por minuto (BPM).
- Clique no botão "Parar" para interromper a captura de vídeo.

## Observação

Certifique-se de que sua webcam esteja corretamente conectada e acessível.

Este aplicativo utiliza o classificador Haar Cascade para detecção de rostos, então pode não ser robusto em condições de iluminação extrema ou oclusões.

A frequência cardíaca estimada pode variar com base em fatores como iluminação, tom de pele e expressões faciais.

Você pode precisar ajustar parâmetros como min_peak_height e min_peak_distance para otimizar a estimativa da frequência cardíaca com base na sua configuração e ambiente.

Sinta-se à vontade para contribuir e aprimorar este projeto!