# Tesseract

## Instalação:

https://tesseract-ocr.github.io/tessdoc/Home.html

### Linux

* sudo apt install tesseract-ocr
* sudo apt install libtesseract-dev


### Funções:

1. **get_tesseract_version** Retorna a versão instalada do Tesseract.
2. **image_to_string Returns** O resultado do Tesseract OCR em uma string que a imagem contém.
3. **image_to_boxes** Retorna resultado contendo caracteres reconhecidos e seus boxes
4. **image_to_data** Retorna as palavras encontras na imagem delitimadas por boxes
5. **image_to_osd Returns** resultado contendo informações sobre orientação e detecção de script.
6. **run_and_get_output** Retorna a saída bruta do Tesseract OCR. Dá um pouco mais de controle sobre os parâmetros enviados ao tesseract.

### Parametros:

image_to_data(image, lang=None, config='', nice=0, output_type=Output.STRING, timeout=0, pandas_config=None)

**image Object or String** - Imagem. É preciso converter para RGB.
**lang String** - Lingua do Tesseract deverá reconhecer. Defaults é eng. É possível multiplas linguas: lang='eng+fra'
**config String** - Qualquer custom adicional configuração: config='--psm 6'
**nice Integer** - modifica a prioridade do processador para a execução do Tesseract. Não suportado no Windows. Nice ajusta a gentileza de processos do tipo unix.
**output_type** Atributo de classe - especifica o tipo de saída, o padrão é string. Para obter a lista completa de todos os tipos suportados, verifique a definição de classe pytesseract.Output.
**timeout Integer or Float** - duração em segundos para o processamento do OCR, após o qual o pytesseract será encerrado e aumentará o RuntimeError.
**pandas_config Dict** - somente para o tipo Output.DATAFRAME. Dicionário com argumentos personalizados para pandas.read_csv. Permite personalizar a saída de image_to_data.


### Exemplo da detecção da letras por boxes





### Exemploda detecção das palavras por boxes
