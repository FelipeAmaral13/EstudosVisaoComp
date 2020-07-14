# Edge detection

O Objetivo desse programa é fazer uma simples detecção de bordas a partir de Canny.

### Detecção de bordas por Canny

O detector de borda Canny é um operador de detecção de borda que usa um algoritmo de vários estágios para detectar uma grande variedade de bordas nas imagens. 
Foi desenvolvido por John F. Canny em 1986. Canny também produziu uma teoria computacional da detecção de bordas, explicando por que a técnica funciona.

O filtro Canny é um detector de borda de vários estágios. 
Ele usa um filtro baseado na derivada de um gaussiano para calcular a intensidade dos gradientes. O gaussiano reduz o efeito do ruído presente na imagem. 
Em seguida, as arestas em potencial são reduzidas para curvas de 1 pixel removendo pixels não máximos da magnitude do gradiente. 
Finalmente, os pixels das bordas são mantidos ou removidos usando limiar de histerese na magnitude do gradiente.

O Canny possui três parâmetros ajustáveis: 

* a largura do gaussiano (quanto mais ruidosa a imagem, maior a largura) 
* E os limiares, baixo e alto, para o limiar da histerese.

Os critérios gerais para detecção de borda incluem:

    * Detecção de arestas com baixa taxa de erros, o que significa que a detecção deve capturar com precisão o maior número possível de arestas mostradas na imagem
    * O ponto da borda detectado pelo operador deve localizar com precisão no centro da borda.
    * Uma determinada borda da imagem deve ser marcada apenas uma vez e, sempre que possível, o ruído da imagem não deve criar bordas falsas.
    
Fonte: https://medium.com/@ssatyajitmaitra/what-canny-edge-detection-algorithm-is-all-about-103d94553d21


### Exemplo da detecção

![Edge_Canny](https://user-images.githubusercontent.com/5797933/87423701-188d0600-c5b1-11ea-9389-cbd6109160df.png)
