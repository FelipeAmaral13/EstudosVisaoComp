# LBP Classifier

Este é um projeto que utiliza a técnica Local Binary Pattern (LBP) para classificar imagens em diferentes categorias. A técnica LBP é um método simples e eficiente de descrever a textura de uma imagem e tem sido amplamente utilizada em aplicações de visão computacional.

O projeto é composto por uma classe LBPClassifier que encapsula as funcionalidades de extração de características, treinamento do modelo, avaliação do modelo e teste em novas imagens.
Requisitos

Para rodar o projeto, é necessário ter as seguintes bibliotecas instaladas:

- numpy
- scikit-learn
- opencv-python
- matplotlib

# Uso

Para utilizar o classificador, basta seguir os seguintes passos:

* Importe a classe LBPClassifier do módulo lbp_classifier.py.
* Instancie um objeto da classe, passando como argumentos o diretório com as imagens de treino e o diretório com as imagens de teste.
* Chame o método extract_features() para extrair as características das imagens de treino.
* Chame o método train_test_split() para dividir as imagens de treino em conjunto de treino e conjunto de teste.
* Chame o método train_model() para treinar o modelo utilizando o conjunto de treino.
* Chame o método evaluate_model() para avaliar o modelo utilizando o conjunto de teste.
* Opcionalmente, chame o método test_model() para testar o modelo em novas imagens.
* Opcionalmente, chame o método plot_validation_curve() para plotar a curva de validação do modelo.

Segue abaixo um exemplo de código:

"""
    from lbp_classifier import LBPClassifier

    # instanciando um objeto da classe LBPClassifier
    lbp_classifier = LBPClassifier(train_dir='data/train', test_dir='data/test')

    # extraindo as características das imagens de treino
    lbp_classifier.extract_features()

    # dividindo as imagens de treino em conjunto de treino e conjunto de teste
    lbp_classifier.train_test_split()

    # treinando o modelo
    lbp_classifier.train_model()

    # avaliando o modelo
    lbp_classifier.evaluate_model()

    # testando o modelo em novas imagens
    lbp_classifier.test_model()

    # plotando a curva de validação do modelo
    lbp_classifier.plot_validation_curve()
"""

# Licença

Este projeto é licenciado sob a licença MIT. Consulte o arquivo LICENSE para mais detalhes.