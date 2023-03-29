# Projeção de Histograma


ImageProcessor é uma classe que processa uma imagem, encontrando os retângulos dos objetos presentes na imagem e criando uma linha de projeção vertical para cada objeto encontrado.
# Pré-requisitos

    Python 3.x
    OpenCV
    Numpy
    Matplotlib

# Como usar

    Crie uma instância da classe ImageProcessor, passando como argumento o caminho da imagem a ser processada:



from image_processor import ImageProcessor

image_processor = ImageProcessor('caminho_da_imagem/imagem.png')

    Execute o método process_image() da instância criada:



image_processor.process_image()

Este método processa a imagem, encontra os retângulos dos objetos na imagem, cria uma linha de projeção vertical para cada objeto encontrado e, em seguida, exibe a imagem processada com os retângulos e as linhas de projeção.
Exemplo



from image_processor import ImageProcessor

image_processor = ImageProcessor('exemplo.png')
image_processor.process_image()

Exemplo de imagem processada
# Licença

Este projeto está sob a licença MIT. Consulte o arquivo LICENSE para mais informações.

![1 coXNZvq_WHzlqRGKX3XKWw](https://user-images.githubusercontent.com/5797933/161387327-f8ae5b58-c102-41f6-a148-26f49d4a2d7e.png)
