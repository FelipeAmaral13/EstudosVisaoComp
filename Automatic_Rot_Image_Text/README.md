
# OCR e Processamento de Imagens com EasyOCR

Este projeto utiliza Python para realizar OCR (Reconhecimento Óptico de Caracteres) em imagens, aplicando rotações automáticas para melhorar a legibilidade do texto detectado. 

## Funcionalidades

- Leitura de texto em imagens usando [EasyOCR](https://github.com/JaidedAI/EasyOCR).
- Detecção e marcação de texto nas imagens com bounding boxes.
- Rotação automática de imagens baseada na inclinação do texto.
- Suporte a múltiplas línguas (português e inglês).
- Visualização do resultado processado (imagem original e imagem rotacionada).

## Requisitos

Certifique-se de ter as dependências abaixo instaladas antes de rodar o projeto:

- Python 3.8 ou superior
- Bibliotecas Python:
  - `asyncio`
  - `easyocr`
  - `opencv-python`
  - `matplotlib`
  - `numpy`
  - `scipy`

Você pode instalar as dependências com:

```bash
pip install -r requirements.txt
```

## Estrutura do Projeto

- **`main.py`**: Arquivo principal que contém a lógica para leitura, processamento e exibição das imagens processadas.
- **`ocr_processing.log`**: Arquivo gerado automaticamente contendo os logs do processamento.

## Como Usar

1. Coloque as imagens que deseja processar em uma pasta chamada `document` no diretório raiz do projeto.
2. Execute o script principal:

```bash
python main.py
```

3. O programa processará todas as imagens na pasta e exibirá:
   - Imagem original com as caixas delimitadoras do OCR.
   - Imagem rotacionada (caso necessário).

## Logs

Os logs do processamento são salvos no arquivo `ocr_processing.log` e também exibidos no console. Eles incluem informações sobre:

- Textos detectados.
- Rotação aplicada.
- Erros de processamento.

## Exemplos de Imagem

O programa suporta os seguintes formatos de imagem:

- `.png`
- `.jpg`
- `.jpeg`
- `.bmp`
- `.tiff`

## Contribuindo

Contribuições são bem-vindas! Sinta-se à vontade para abrir issues ou enviar pull requests.

## Licença

Este projeto é licenciado sob a [MIT License](LICENSE).
