# Contador de automóveis

### Requirements

* Numpy  '1.18.4'
* OpenCV ''4.2.0'

### Objetivo:

O objetivo desse programa é fazer uma simples contagem de carros. 

### Funcionamento do programa:

É feito uma subtração de background para saber se o frame atual sofreu alguma atualização em relação a ele mesmo em um momento anterior. Depois é aplicado uma morfologia matematica de um elemento estruturante do tipo eliptico para encontrar a área da estrutura que se locomove, no caso um automóvel. É calculado a centroide dessa área que ao passar por uma linha delimitadora é interado a contagem de carro ao sistema, mostrado na parte superior esquerda.

![1](https://user-images.githubusercontent.com/5797933/89351464-78df0700-d688-11ea-91aa-294057e8ef49.PNG)
