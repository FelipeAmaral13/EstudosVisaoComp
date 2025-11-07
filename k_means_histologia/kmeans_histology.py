import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os
import glob
from tqdm import tqdm

def quantify_image(image_path, K=3):
    """
    Carrega e processa UMA imagem, identifica os clusters e 
    retorna as métricas quantitativas.    
    """
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        return None

    # 1. Pipeline K-Means
    img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2Lab)
    pixel_data = img_lab.reshape((-1, 3))
    pixel_data = np.float32(pixel_data)
    
    # Critério: 100 iterações ou 1.0 de precisão
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1.0)
    
    # Reduzir 'attempts' para 5 para acelerar o processamento em lote
    compactness, labels, centers = cv2.kmeans(pixel_data, K, None, criteria, 5, cv2.KMEANS_PP_CENTERS)

    mask = labels.reshape((img_bgr.shape[0], img_bgr.shape[1]))

    # 2. Identificação Automática dos Clusters
    # 'centers' está em L*a*b*. O canal L (Lightness) é o índice 0.
    L_values = centers[:, 0]
    
    label_fundo = np.argmax(L_values)     # Cluster mais claro (maior 'L')
    label_nucleo = np.argmin(L_values)    # Cluster mais escuro (menor 'L')
    
    # Identifica o cluster do citoplasma (o que sobrou)
    labels_set = {0, 1, 2}
    labels_set.remove(label_fundo)
    if label_nucleo in labels_set:
        labels_set.remove(label_nucleo)
    
    # Tratar caso K=2 ou K=1 (embora K=3 seja o padrão)
    if not labels_set: 
        label_citoplasma = label_nucleo # Fallback, embora improvável com K=3
    else:
        label_citoplasma = labels_set.pop()


    # 3. Cálculo das Métricas (Contagem de Pixels)
    area_nuclei = np.sum(mask == label_nucleo)
    area_cytoplasm = np.sum(mask == label_citoplasma)
    
    total_tecido = area_nuclei + area_cytoplasm

    # Evitar divisão por zero se a imagem for toda de fundo
    if total_tecido == 0:
        # print(f"Aviso: Pulando {image_path}, nenhum tecido encontrado.")
        return None
        
    # MÉTRICA-CHAVE: Fração Nuclear
    fracao_nuclear = area_nuclei / total_tecido

    # Retorna um dicionário com os resultados
    return {
        'image_path': os.path.basename(image_path),
        'fracao_nuclear': fracao_nuclear
    }


def run_batch_analysis(base_data_path):
    """
    Executa a quantificação em todas as imagens das subpastas
    e retorna um DataFrame do Pandas com os resultados.
    """
    
    # Vamos comparar 'benigno' (lung_n) vs 'adenocarcinoma' (lung_aca)
    paths_benign = glob.glob(os.path.join(base_data_path, 'lung_n', '*.jpeg'))
    paths_aca = glob.glob(os.path.join(base_data_path, 'lung_aca', '*.jpeg'))
    
    # Opcional: Limitar o número de imagens para um teste rápido
    # paths_benign = paths_benign[:1000]
    # paths_aca = paths_aca[:1000]

    print(f"Encontradas {len(paths_benign)} imagens benignas (lung_n).")
    print(f"Encontradas {len(paths_aca)} imagens de adenocarcinoma (lung_aca).")
    
    if len(paths_benign) == 0 or len(paths_aca) == 0:
        print("\n*** ERRO ***")
        print(f"Pastas de imagem não encontradas ou vazias no caminho: {base_data_path}")
        print("Verifique se o caminho está correto e se as subpastas 'lung_n' e 'lung_aca' existem.")
        return None

    all_results = []
    
    # Processar imagens benignas
    for path in tqdm(paths_benign, desc="Processando Benignas (lung_n)"):
        stats = quantify_image(path, K=3)
        if stats:
            stats['classe'] = 'Benigno'
            all_results.append(stats)
            
    # Processar imagens de adenocarcinoma
    for path in tqdm(paths_aca, desc="Processando Adenocarcinoma (lung_aca)"):
        stats = quantify_image(path, K=3)
        if stats:
            stats['classe'] = 'Adenocarcinoma'
            all_results.append(stats)
            
    if not all_results:
        print("Nenhum resultado foi processado. Verifique os arquivos de imagem.")
        return None
        
    # Converter lista de dicionários para um DataFrame
    df = pd.DataFrame(all_results)
    return df


def plot_statistical_results(df):
    """
    Recebe o DataFrame e plota a análise comparativa.
    """
    if df is None or df.empty:
        print("\nO DataFrame está vazio. Não é possível plotar.")
        return
        
    print("\n--- Análise Estatística Concluída ---")
    
    # Estatísticas descritivas agrupadas por classe
    print("\nEstatísticas Descritivas por Classe:")
    print(df.groupby('classe')['fracao_nuclear'].describe().to_markdown()) # .to_markdown() formata bonito
    
    # Visualização
    plt.figure(figsize=(10, 7))
    
    # Boxplot para comparar as distribuições
    sns.boxplot(x='classe', y='fracao_nuclear', data=df, palette="vlag")
    
    # Adicionar os pontos de dados (jitter) para ver a densidade
    sns.stripplot(x='classe', y='fracao_nuclear', data=df, color=".25", size=3, jitter=True, alpha=0.3)
    
    plt.title('Comparação da Fração Nuclear (Benigno vs. Adenocarcinoma)', fontsize=16)
    plt.ylabel('Fração Nuclear (Área Núcleo / Área Tecido)', fontsize=12)
    plt.xlabel('Classe da Imagem', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()


def segment_histology_kmeans_visualize(image_path, K=3):
    """
    Esta é a sua função original, usada APENAS para visualizar
    e depurar a segmentação de UMA imagem.
    """
    print(f"\n--- Executando Visualização de Exemplo para: {image_path} ---")
    img_bgr = cv2.imread(image_path)
    if img_bgr is None: 
        print(f"Erro: Não foi possível carregar imagem de exemplo em {image_path}")
        return
        
    img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2Lab)
    pixel_data = img_lab.reshape((-1, 3))
    pixel_data = np.float32(pixel_data)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1.0)
    compactness, labels, centers = cv2.kmeans(pixel_data, K, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
    mask = labels.reshape((img_bgr.shape[0], img_bgr.shape[1]))
    centers = np.uint8(centers)
    quantized_pixels = centers[labels.flatten()]
    quantized_img_lab = quantized_pixels.reshape((img_lab.shape))
    quantized_img_bgr = cv2.cvtColor(quantized_img_lab, cv2.COLOR_Lab2BGR)
    
    cluster_0_img = img_bgr.copy(); cluster_0_img[mask != 0] = [0, 0, 0]
    cluster_1_img = img_bgr.copy(); cluster_1_img[mask != 1] = [0, 0, 0]
    cluster_2_img = img_bgr.copy(); cluster_2_img[mask != 2] = [0, 0, 0]
    
    plt.figure(figsize=(15, 10))
    plt.subplot(2, 3, 1); plt.imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)); plt.title("Original (RGB)"); plt.axis('off')
    plt.subplot(2, 3, 2); plt.imshow(mask, cmap='viridis'); plt.title(f"Máscara K-Means (K={K})"); plt.axis('off')
    plt.subplot(2, 3, 3); plt.imshow(cv2.cvtColor(quantized_img_bgr, cv2.COLOR_BGR2RGB)); plt.title("Imagem Quantizada"); plt.axis('off')
    plt.subplot(2, 3, 4); plt.imshow(cv2.cvtColor(cluster_0_img, cv2.COLOR_BGR2RGB)); plt.title("Apenas Cluster 0"); plt.axis('off')
    plt.subplot(2, 3, 5); plt.imshow(cv2.cvtColor(cluster_1_img, cv2.COLOR_BGR2RGB)); plt.title("Apenas Cluster 1"); plt.axis('off')
    plt.subplot(2, 3, 6); plt.imshow(cv2.cvtColor(cluster_2_img, cv2.COLOR_BGR2RGB)); plt.title("Apenas Cluster 2"); plt.axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    DATA_FOLDER = 'dados\lung' 
    
    results_df = run_batch_analysis(DATA_FOLDER)
    plot_statistical_results(results_df)

    try:
        example_path_benign = os.path.join(DATA_FOLDER, 'lung_n', 'lungn78.jpeg') # Seu exemplo original
        example_path_aca = os.path.join(DATA_FOLDER, 'lung_aca', 'lungaca1.jpeg') # Um exemplo de câncer
        
        if os.path.exists(example_path_benign):
             segment_histology_kmeans_visualize(example_path_benign, K=3)
        else:
             print(f"Arquivo de exemplo benigno {example_path_benign} não encontrado para visualização.")
             
        if os.path.exists(example_path_aca):
             segment_histology_kmeans_visualize(example_path_aca, K=3)
        else:
            print(f"Arquivo de exemplo de adenocarcinoma {example_path_aca} não encontrado para visualização.")

    except Exception as e:
        print(f"Erro ao visualizar exemplos: {e}")