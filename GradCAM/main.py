from PIL import Image
import numpy as np
import os
import random
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
from torchvision.models import vgg19, resnet50, ResNet50_Weights
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision.transforms.functional import resize
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc


# Configuração para reprodutibilidade
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed()

class TransformedSubset(torch.utils.data.Dataset):
        def __init__(self, dataset, indices, transform=None):
            self.dataset = dataset
            self.indices = indices
            self.transform = transform
            
        def __getitem__(self, idx):
            img, label = self.dataset[self.indices[idx]]
            if self.transform:
                img = self.transform(img)
            return img, label
        
        def __len__(self):
            return len(self.indices)

train_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),  # Converter para RGB
    transforms.Resize((224, 224)),  # Redimensionar para o tamanho esperado pela VGG19
    transforms.RandomHorizontalFlip(p=0.5),  # Flip horizontal aleatório
    transforms.RandomVerticalFlip(p=0.3),  # Adiciona flip vertical
    transforms.RandomRotation(15),  # Rotação aleatória até 15 graus
    transforms.RandomPerspective(distortion_scale=0.2, p=0.5),  # Adiciona distorção de perspectiva
    transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),  # Translação e escala aleatórias
    transforms.ColorJitter(brightness=0.1, contrast=0.1),  # Variação sutil em brilho e contraste
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalização
])

# Transformações para validação (sem aumento de dados)
val_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),  # Converter para RGB
    transforms.Resize((224, 224)),  # Redimensionar para o tamanho esperado pela VGG19
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalização
])

data_dir = 'dados'


# Carregar o dataset
full_dataset = ImageFolder(root=data_dir)

# Verificar balanceamento das classes
class_counts = np.bincount([label for _, label in full_dataset])
print(f"Distribuição de classes: {class_counts}")
print(f"Classes: {full_dataset.classes}")

# Dividir o dataset em treinamento e validação
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset_indices, val_dataset_indices = random_split(range(len(full_dataset)), [train_size, val_size])

# Criar DataLoaders
train_dataset = TransformedSubset(full_dataset, train_dataset_indices, train_transform)
val_dataset = TransformedSubset(full_dataset, val_dataset_indices, val_transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, pin_memory=True)


# Carregar a VGG19 pré-treinada e ajustar a camada final
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = vgg19(pretrained=True)
# model.classifier[6] = nn.Linear(model.classifier[6].in_features, 2)
model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
model.fc = nn.Sequential(nn.Dropout(0.2) )
model = model.to(device)

# Definir loss e otimizador
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)


# Treinamento
num_epochs = 30
train_losses, val_losses = [], []
train_accs, val_accs = [], []
best_val_loss = float('inf')
patience = 7
counter = 0
best_model_path = 'best_model.pth'

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0
    
    train_loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]", leave=False)
    for images, labels in train_loop:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()
        train_loop.set_postfix(loss=loss.item(), acc=correct_train / total_train)
    
    train_loss = running_loss / len(train_loader)
    train_acc = correct_train / total_train
    train_losses.append(train_loss)
    train_accs.append(train_acc)
    
    model.eval()
    val_loss = 0.0
    correct_val = 0
    total_val = 0
    
    val_loop = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]", leave=False)
    with torch.no_grad():
        for images, labels in val_loop:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_val += labels.size(0)
            correct_val += (predicted == labels).sum().item()
            val_loop.set_postfix(loss=loss.item(), acc=correct_val / total_val)
    
    val_loss /= len(val_loader)
    val_acc = correct_val / total_val
    val_losses.append(val_loss)
    val_accs.append(val_acc)

    scheduler.step(val_loss)
       
    print(f"Epoch {epoch+1}/{num_epochs} | "
          f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
          f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
    
    if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"Saved best model with validation loss: {best_val_loss:.4f}")
    else:
        counter += 1
        if counter >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break

model.load_state_dict(torch.load(best_model_path))

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')

plt.subplot(1, 2, 2)
plt.plot(train_accs, label='Train Accuracy')
plt.plot(val_accs, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training and Validation Accuracy')
plt.tight_layout()
plt.savefig('training_curves.png')
plt.show()

# Avaliação no conjunto de validação
model.eval()
all_preds = []
all_labels = []
all_probs = []

with torch.no_grad():
    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        _, predicted = torch.max(outputs.data, 1)
        
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs[:,1].cpu().numpy() if probs.shape[1] > 1 else probs.cpu().numpy())

# Confusion Matrix
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(8, 6))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(len(full_dataset.classes))
plt.xticks(tick_marks, full_dataset.classes, rotation=45)
plt.yticks(tick_marks, full_dataset.classes)
plt.tight_layout()

thresh = cm.max() / 2.
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, format(cm[i, j], 'd'),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black")

plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.savefig('confusion_matrix.png')
plt.show()

# Classification Report
class_report = classification_report(all_labels, all_preds, target_names=full_dataset.classes)
print("Classification Report:")
print(class_report)

# ROC Curve (para classificação binária)
if len(full_dataset.classes) == 2:
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig('roc_curve.png')
    plt.show()




def visualize_gradcam(model, image_path, transform, device, output_dir=None):
    """
    Visualiza um mapa de calor Grad-CAM sobre uma imagem para a última camada convolucional do ResNet50
    """
    # Verificar se o modelo é ResNet50 e configurar a camada alvo
    if isinstance(model, resnet50().__class__):
        target_layer = model.layer4[-1]  # Última camada residual da ResNet50
    else:
        raise ValueError("Modelo não suportado para Grad-CAM")
    
    # Inicializar o GradCAM
    cam = GradCAM(model=model, target_layers=[target_layer])
    
    # Carregar e transformar a imagem
    image = Image.open(image_path).convert('L')
    image = image.convert('RGB')  # Converter para RGB
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    # Gerar o mapa de ativação
    grayscale_cam = cam(input_tensor=input_tensor, targets=None)
    
    # Redimensionar o mapa de calor para o tamanho da imagem original
    grayscale_cam = grayscale_cam[0, :]  # Selecionar o primeiro mapa de calor
    grayscale_cam = resize(torch.from_numpy(grayscale_cam).unsqueeze(0), (image.size[1], image.size[0]))
    grayscale_cam = grayscale_cam.squeeze().numpy()
    
    # Converter a imagem para o formato adequado para visualização
    rgb_img = np.array(image) / 255.0
    
    # Sobrepor o Grad-CAM na imagem
    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
    
    # Predição do modelo
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
        predicted_class = torch.argmax(probabilities).item()
    
    # Extrair o nome base da imagem
    image_basename = os.path.basename(image_path)
    
    # Exibir a imagem
    plt.figure(figsize=(10, 8))
    plt.imshow(visualization)
    plt.title(f'Grad-CAM: {image_basename}\nPredição: {full_dataset.classes[predicted_class]} ({probabilities[predicted_class]:.2f})')
    plt.axis('off')
    
    # Salvar a imagem se um diretório de saída for fornecido
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f'gradcam_{image_basename}')
        plt.savefig(output_path, bbox_inches='tight')
        print(f"Imagem salva em {output_path}")
    
    plt.show()

def visualize_gradcam_layers(model, image_path, transform, device, output_dir=None):
    """
    Visualiza mapas de calor Grad-CAM para diferentes camadas do ResNet50
    """
    # Verificar se o modelo é ResNet50
    if not isinstance(model, resnet50().__class__):
        raise ValueError("Modelo não suportado para esta visualização")
    
    # Definir as camadas a serem visualizadas
    target_layers = [
        model.layer1[-1],  # Última camada do primeiro bloco
        model.layer2[-1],  # Última camada do segundo bloco
        model.layer3[-1],  # Última camada do terceiro bloco 
        model.layer4[-1]   # Última camada do quarto bloco
    ]
    
    layer_names = ['Layer 1', 'Layer 2', 'Layer 3', 'Layer 4']
    
    # Carregar e transformar a imagem
    image = Image.open(image_path).convert('L')
    image = image.convert('RGB')  # Converter para RGB
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    # Extrair o nome base da imagem
    image_basename = os.path.basename(image_path)
    
    # Criar uma figura para os subplots
    fig, axes = plt.subplots(1, len(target_layers), figsize=(20, 5))
    
    # Predição do modelo
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
        predicted_class = torch.argmax(probabilities).item()
        prediction_text = f'Predição: {full_dataset.classes[predicted_class]} ({probabilities[predicted_class]:.2f})'
    
    # Iterar sobre as camadas
    for i, (layer, name) in enumerate(zip(target_layers, layer_names)):
        print(f"Processando Grad-CAM para {name}")
        
        # Criar o objeto GradCAM para a camada atual
        cam = GradCAM(model=model, target_layers=[layer])
        
        # Gerar o mapa de ativação
        grayscale_cam = cam(input_tensor=input_tensor, targets=None)
        
        # Redimensionar o mapa de calor para o tamanho da imagem original
        grayscale_cam = grayscale_cam[0, :]  # Selecionar o primeiro mapa de calor
        grayscale_cam = resize(torch.from_numpy(grayscale_cam).unsqueeze(0), (image.size[1], image.size[0]))
        grayscale_cam = grayscale_cam.squeeze().numpy()
        
        # Converter a imagem para o formato adequado para visualização
        rgb_img = np.array(image) / 255.0
        
        # Sobrepor o Grad-CAM na imagem
        visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
        
        # Exibir a imagem no subplot correspondente
        axes[i].imshow(visualization)
        axes[i].set_title(f'{name}')
        axes[i].axis('off')
    
    # Adicionar um título geral à figura
    fig.suptitle(f'Grad-CAM para diferentes camadas: {image_basename}\n{prediction_text}', fontsize=16)
    
    # Ajustar o layout
    plt.tight_layout()
    
    # Salvar a imagem se um diretório de saída for fornecido
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f'gradcam_layers_{image_basename}')
        plt.savefig(output_path, bbox_inches='tight')
        print(f"Imagem salva em {output_path}")
    
    plt.show()

def batch_visualize_gradcam(model, data_dir, transform, device, num_samples=3, output_dir='gradcam_output'):
    """
    Gera visualizações Grad-CAM para múltiplas imagens de cada classe
    """
    # Criar diretório de saída
    os.makedirs(output_dir, exist_ok=True)
    
    # Iterar sobre as classes
    for class_name in os.listdir(data_dir):
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
        
        # Listar arquivos de imagem na pasta da classe
        image_files = [f for f in os.listdir(class_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        # Selecionar um subconjunto aleatório de imagens (até num_samples)
        if len(image_files) > num_samples:
            image_files = np.random.choice(image_files, num_samples, replace=False)
        
        # Processar cada imagem selecionada
        for image_file in image_files:
            image_path = os.path.join(class_dir, image_file)
            print(f"Processando {image_path}")
            
            try:
                # Gerar e salvar visualização Grad-CAM
                visualize_gradcam(model, image_path, transform, device, output_dir)
                
                # Gerar e salvar visualização de múltiplas camadas
                visualize_gradcam_layers(model, image_path, transform, device, output_dir)
            except Exception as e:
                print(f"Erro ao processar {image_path}: {e}")

# Exemplo de uso após o treinamento
print("Gerando visualizações Grad-CAM para algumas imagens...")
output_dir = 'gradcam_results'
os.makedirs(output_dir, exist_ok=True)

# Visualizar uma imagem normal e uma com tuberculose
normal_example = os.path.join(data_dir, 'Normal', os.listdir(os.path.join(data_dir, 'Normal'))[0])
tb_example = os.path.join(data_dir, 'Tuberculosis', os.listdir(os.path.join(data_dir, 'Tuberculosis'))[0])

print("Visualizando Grad-CAM para uma imagem normal...")
visualize_gradcam(model, normal_example, val_transform, device, output_dir)

print("Visualizando Grad-CAM para uma imagem com tuberculose...")
visualize_gradcam(model, tb_example, val_transform, device, output_dir)

print("Visualizando diferentes camadas para uma imagem normal...")
visualize_gradcam_layers(model, normal_example, val_transform, device, output_dir)

print("Visualizando diferentes camadas para uma imagem com tuberculose...")
visualize_gradcam_layers(model, tb_example, val_transform, device, output_dir)

# Para processar lotes de imagens, descomente abaixo:
# batch_visualize_gradcam(model, data_dir, val_transform, device, num_samples=3, output_dir=output_dir)

# Avaliação final do modelo
model.eval()
correct = 0
total = 0
all_probs = []
all_preds = []
all_labels = []

print("\nAvaliando o modelo no conjunto de validação...")
with torch.no_grad():
    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        _, predicted = torch.max(outputs.data, 1)
        
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs[:,1].cpu().numpy())

accuracy = 100 * correct / total
print(f'Acurácia do modelo nas imagens de validação: {accuracy:.2f}%')

# Métricas específicas para tuberculose (classe 1)
from sklearn.metrics import precision_score, recall_score, f1_score

precision = precision_score(all_labels, all_preds, average='weighted')
recall = recall_score(all_labels, all_preds, average='weighted')
f1 = f1_score(all_labels, all_preds, average='weighted')

print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")

# Salvar os resultados em um arquivo de texto
with open(os.path.join(output_dir, 'model_results.txt'), 'w') as f:
    f.write(f"Acurácia: {accuracy:.2f}%\n")
    f.write(f"Precision: {precision:.4f}\n")
    f.write(f"Recall: {recall:.4f}\n")
    f.write(f"F1-Score: {f1:.4f}\n")
    f.write(f"\nClassification Report:\n")
    f.write(classification_report(all_labels, all_preds, target_names=full_dataset.classes))

