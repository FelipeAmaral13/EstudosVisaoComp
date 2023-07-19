# QrCode Class

A QrCode é uma classe Python que permite a criação, codificação, decodificação e plotagem de imagens de códigos QR.

# Funcionalidades

A classe QrCode possui os seguintes métodos:

* __init__(self, message): Construtor que recebe uma string message como entrada e cria um objeto img_qrcode com a imagem do código QR correspondente.

* encode(self, img_real): Método que recebe o caminho de uma imagem img_real e insere o código QR na imagem oficial. A imagem resultante é armazenada no atributo img_encoded.

* decode(self): Método que decodifica a imagem com o código QR inserido, retornando a imagem decodificada.

* get_text(self): Método que decodifica a imagem e imprime a mensagem original do código QR.

* plot_bitplanes(self): Método que plota as imagens dos diferentes planos de bits da imagem codificada.

# Dependências

A classe QrCode utiliza as seguintes bibliotecas Python:

- qrcode: Para a criação da imagem do código QR.
- numpy: Para manipulação de arrays multidimensionais.
- PIL: Para a manipulação de imagens.
- pyzbar: Para a decodificação do código QR.

# Como usar

Para utilizar a classe QrCode, siga os passos abaixo:

Importe a classe QrCode para o seu código:


`from qrcode_class import QrCode`

Crie uma instância da classe QrCode, passando como parâmetro a mensagem que deseja codificar:



`qrcode = QrCode('exemplo de mensagem')`

Codifique a mensagem em uma imagem oficial, passando o caminho da imagem como parâmetro:



`qrcode.encode('lena.jpg')`

Se desejar, plote as imagens dos diferentes planos de bits da imagem codificada:



`qrcode.plot_bitplanes()`

Para decodificar a imagem e obter a mensagem original, utilize o método get_text():



`qrcode.get_text()`

Exemplo completo



    from qrcode_class import QrCode
    
    qrcode = QrCode('exemplo de mensagem')
    qrcode.encode('lena.jpg')
    qrcode.plot_bitplanes()
    qrcode.get_text()

Este exemplo cria uma instância da classe QrCode, codifica a mensagem em uma imagem oficial, plota as imagens dos diferentes planos de bits e imprime a mensagem original do código QR.
