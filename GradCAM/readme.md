# Detecção de Tuberculose em Raios-X utilizando CNN e Grad-CAM

## Visão Geral
Este projeto utiliza redes neurais convolucionais (CNNs) para classificar imagens de raio-X como normais ou com tuberculose. Foi utilizada a ResNet50 pré-treinada, ajustada para esta tarefa específica. Também foram gerados mapas de calor Grad-CAM para melhor interpretação das decisões do modelo.

## Estrutura do Projeto
```
/
|-- main.py                 # Script principal de treinamento e avaliação
|-- dados/                  # Diretório com imagens de raio-X
|-- best_model.pth          # Melhor modelo treinado
|-- training_curves.png     # Gráfico de perda e acurácia ao longo do treinamento
|-- confusion_matrix.png    # Matriz de confusão do modelo final
|-- roc_curve.png           # Curva ROC da classificação binária
|-- gradcam_results/        # Diretório com as imagens Grad-CAM
|   |-- gradcam_Normal-10.png
|   |-- gradcam_Tuberculosis-10.png
|   |-- gradcam_layers_Normal-10.png
|   |-- gradcam_layers_Tuberculosis-10.png
|-- model_results.txt       # Resultados quantitativos do modelo
```

## Requisitos
Este projeto requer Python 3 e as seguintes bibliotecas:

```bash
pip install torch torchvision matplotlib numpy tqdm scikit-learn pillow pytorch-grad-cam
```

## Como Executar

1. **Preparar os Dados:**
   - Coloque as imagens dentro da pasta `dados/`, separadas em subpastas `Normal/` e `Tuberculosis/`.
   - O Dataset usado é encontrado no [kaggle](https://www.kaggle.com/datasets/tawsifurrahman/tuberculosis-tb-chest-xray-dataset)

2. **Treinar o Modelo:**
   - Execute o script `main.py` para treinar o modelo.
   ```bash
   python main.py
   ```

3. **Avaliar o Modelo:**
   - O modelo será avaliado automaticamente no final do treinamento.
   - Resultados serão salvos em `model_results.txt`, `confusion_matrix.png` e `roc_curve.png`.

4. **Gerar Visualizações Grad-CAM:**
   - Para verificar quais regiões da imagem influenciaram na decisão do modelo, visualize os arquivos dentro do diretório `gradcam_results/`.


## Explicação Grad-CAM
Os mapas de calor Grad-CAM foram gerados para ajudar a entender como o modelo toma decisões. As imagens `gradcam_layers_Normal-10.png` e `gradcam_layers_Tuberculosis-10.png` mostram as ativações ao longo das camadas da ResNet50.

- Regiões avermelhadas indicam as áreas mais relevantes para a classificação.
- Para imagens de tuberculose, observa-se ativação em regiões pulmonares.

  ![Image](https://github.com/user-attachments/assets/f28558f9-dbf1-470a-a97b-2ae7fc04f129)

  

## Conclusão
Este projeto demonstrou uma abordagem eficiente para detectar tuberculose em raios-X usando CNNs. Os resultados são altamente promissores, com acurácia superior a 98%. A incorporação do Grad-CAM melhorou a interpretabilidade do modelo.

## Melhorias Futuras
- Experimentar outras arquiteturas de CNN, como EfficientNet.
- Utilizar técnicas de balanceamento de dados.
- Implementar atenção visual para aprimorar explicações.

