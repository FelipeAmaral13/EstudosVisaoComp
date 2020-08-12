# Desafio-Mvisia

### Desafio Proposto

Desenvolver uma solução que permita a um usuário a configuração de uma câmera e a aplicação de operações simples nas imagens obtidas da câmera por meio de um pipeline de processamento. A solução constituir-se-á de uma aplicação web local, onde o usuário controla a câmera e o pipeline de processamento interagindo com elementos de uma página carregada pelo browser. Deverão ser implementados os seguintes blocos de processamento:

- camera() -> img: Esse bloco não possui entradas, sua saída é uma imagem capturada pela câmera.

- crop (img, x, y, dx, dy) -> img: Cropa / fatia o ndarray de entrada em um retângulo. Retorna um ndarray img.

- background_subtract(oimg, img) -> img: Subtrai a imagem atual de uma outra anterior. Retorna um ndarray.

- binarize(img, r, g, b, k) -> img: Retorna uma máscara binarizada da seguinte forma:
    * Out[i, j] = ( Input[i, j, 0]*r + Input[i, j, 1]*g + Input[i, j, 1]*2 ) > k

- stream(img) -> None: Esse bloco simplesmente mostra a imagem de entrada na página (praticamente um streaming da câmera).


Deixamos como desafio extra outros blocos para serem implementados:

- lambda(*args, **kwargs) -> ret: O bloco lambda permite a definição de um  trecho de código qualquer escrito pelo usuário dinamicamente.

- detect_faces() -> (img, x, y, dx, dy): Detecta um rosto na imagem. Além de retornar um ndarray com um retângulo desenhado no contorno do rosto, o bloco retorna as posições do rosto (X, Y, dX, dY). Que tal implementar como um bloco Lambda?
