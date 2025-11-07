
# Quantificação de Tecido Pulmonar com K-Means

Este repositório contém um pipeline em Python para **quantificação automática de imagens histológicas de pulmão**, com base na segmentação via **K-Means clustering** no espaço de cor CIE-Lab.

https://www.kaggle.com/datasets/andrewmvd/lung-and-colon-cancer-histopathological-images/data 

---

## Objetivo

O objetivo é calcular a **fração nuclear** (relação entre área de núcleo e área total de tecido) em imagens histológicas de **casos benignos e adenocarcinoma**, comparando estatisticamente os resultados.

---

## Principais Funcionalidades

- Segmentação de imagens histológicas utilizando **K-Means** (`OpenCV`).
- Extração automática das regiões de **núcleo**, **citoplasma** e **fundo**.
- Cálculo da **fração nuclear** para cada imagem.
- Análise estatística comparativa entre grupos de amostras.
- Visualização interativa dos resultados via **Seaborn** e **Matplotlib**.

---

## Estrutura Esperada de Diretórios

```bash
dados/
│
├── lung_aca/        # Imagens de adenocarcinoma (.jpeg)
└── lung_n/          # Imagens benignas (.jpeg)
```

---

## Execução

1. **Instale as dependências:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Execute o script principal:**
   ```bash
   python kmeans_histology.py
   ```

3. O script irá:
   - Processar todas as imagens em `dados/lung_n` e `dados/lung_aca`.
   - Calcular métricas de fração nuclear.
   - Exibir boxplots comparativos e estatísticas descritivas.

---

## Funções Principais

### `quantify_image(image_path, K=3)`
Processa uma única imagem e retorna a fração nuclear com base nos clusters identificados.

### `run_batch_analysis(base_data_path)`
Executa a análise em lote, retornando um `DataFrame` consolidado com todas as imagens.

### `plot_statistical_results(df)`
Gera boxplots e estatísticas descritivas comparando as classes (Benigno x Adenocarcinoma).

### `segment_histology_kmeans_visualize(image_path, K=3)`
Mostra a segmentação K-Means de uma imagem específica, para depuração visual.

---

## Dependências Principais

- `opencv-python`
- `numpy`
- `matplotlib`
- `pandas`
- `seaborn`
- `tqdm`

---

## Exemplo de Uso

```python
from kmeans_histology import run_batch_analysis, plot_statistical_results

DATA_FOLDER = 'dados/lung'
df = run_batch_analysis(DATA_FOLDER)
plot_statistical_results(df)
```

---

## Resultados Esperados

O gráfico final compara a **fração nuclear média** entre imagens benignas e malignas, permitindo observar diferenças quantitativas entre os grupos.

---

## Créditos

Desenvolvido por Felipe — Cientista de Dados e Professor na área de IA.

---

## Licença

Este projeto é distribuído sob a licença **MIT**, livre para uso acadêmico e comercial.
