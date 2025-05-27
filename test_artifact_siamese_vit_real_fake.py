import warnings
warnings.filterwarnings("ignore")

import os

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from transformers import ViTModel
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import umap
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from scipy.spatial.distance import cdist
from colorama import Fore, Style

# Parametri
image_size = 224
batch_size = 512
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Cartelle per ciascuna classe (4 classi)
real_dirs = [r"C:\Users\labor\Desktop\Progetto_FVAB-main-MD\archive\imagenet\test1000"]

fake_dirs_stylegan2 = [r"C:\Users\labor\Desktop\Progetto_FVAB-main-MD\archive\stylegan2\test1000"]
fake_dirs_taming = [r"C:\Users\labor\Desktop\Progetto_FVAB-main-MD\archive\taming_transformer\test1000"]
fake_dirs_sfhq = [r"C:\Users\labor\Desktop\Progetto_FVAB-main-MD\archive\sfhq\test1000"]

# Trasformazioni
transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

class ArtifactDataset(Dataset):
    def __init__(self, directories, label, transform=None):
        self.filepaths = []
        self.labels = []
        self.transform = transform
        for directory in directories:
            for root, _, files in os.walk(directory):
                for file in files:
                    if file.endswith((".jpg", ".png")):
                        self.filepaths.append(os.path.join(root, file))
                        self.labels.append(label)

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        img_path = self.filepaths[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = self.labels[idx]
        return image, label

class EmbeddingNetwork(torch.nn.Module):
    def __init__(self, embedding_dim=768, output_dim=256):
        super(EmbeddingNetwork, self).__init__()
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(embedding_dim, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, output_dim)
        )
    def forward(self, x):
        return self.fc(x)

# Cartella per i risultati
save_dir = "RisultatiSiameseDiversi"
os.makedirs(save_dir, exist_ok=True)

# Caricamento dataset di test con 4 classi
dataset_imagenet = ArtifactDataset(real_dirs, label=0, transform=transform)
dataset_stylegan2 = ArtifactDataset(fake_dirs_stylegan2, label=1, transform=transform)
dataset_taming = ArtifactDataset(fake_dirs_taming, label=2, transform=transform)
dataset_sfhq = ArtifactDataset(fake_dirs_sfhq, label=3, transform=transform)

full_dataset = dataset_imagenet + dataset_stylegan2 + dataset_taming + dataset_sfhq
loader = DataLoader(full_dataset, batch_size=batch_size, shuffle=False)

# Caricamento modelli
print(f"{Fore.CYAN}Caricamento modello ViT preaddestrato...{Style.RESET_ALL}")
vit_model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k").to(device)
vit_model.eval()

print(f"{Fore.CYAN}Caricamento modello siamese addestrato...{Style.RESET_ALL}")
siamese_model = EmbeddingNetwork().to(device)
siamese_model.load_state_dict(torch.load("best_model_val_loss.pt"))
siamese_model.eval()

# Estrazione embeddings di test
print(f"{Fore.CYAN}Estrazione embeddings di test...{Style.RESET_ALL}")
embeddings_test = []
labels_test = []
with torch.no_grad():
    for imgs, lbls in tqdm(loader, desc="Estrazione embeddings test"):
        imgs = imgs.to(device)
        outputs = vit_model(pixel_values=imgs)
        cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu()
        refined = siamese_model(cls_embeddings.to(device)).cpu()
        embeddings_test.append(refined)
        labels_test.append(lbls)

embeddings_test = torch.cat(embeddings_test).numpy()
labels_test = torch.cat(labels_test).numpy()

# Caricamento embeddings di training
print(f"{Fore.CYAN}Caricamento embeddings di training...{Style.RESET_ALL}")
embeddings_train = torch.load("embeddings_siamese.pt").numpy()
labels_train = torch.load("labels.pt").numpy()

# Calcolo predizioni multi-classe (4 classi)
k = 5
alpha = 0.5  # Decadimento esponenziale
beta = 3     # Rafforzamento dei k vicini più stretti

print(f"{Fore.CYAN}Calcolo predizioni multi-classe (4 classi)...{Style.RESET_ALL}")
predictions = []
for test_emb in tqdm(embeddings_test, desc="Calcolo predizioni"):
    dists = cdist([test_emb], embeddings_train)[0]
    sorted_indices = np.argsort(dists)
    weights = np.exp(-alpha * (dists ** 2))
    weights[sorted_indices[:k]] *= beta

    class_scores = {}
    for cls in np.unique(labels_train):
        class_scores[cls] = np.sum(weights[labels_train == cls])

    pred = max(class_scores, key=class_scores.get)
    predictions.append(pred)

predictions = np.array(predictions)

# Metriche multi-classe (macro)
accuracy_top1 = accuracy_score(labels_test, predictions)
precision = precision_score(labels_test, predictions, average='macro', zero_division=0)
recall = recall_score(labels_test, predictions, average='macro', zero_division=0)
f1 = f1_score(labels_test, predictions, average='macro', zero_division=0)

print(f"{Fore.GREEN}Accuracy top-1: {accuracy_top1:.4f}{Style.RESET_ALL}")
print(f"{Fore.GREEN}Precision (macro): {precision:.4f}{Style.RESET_ALL}")
print(f"{Fore.GREEN}Recall (macro): {recall:.4f}{Style.RESET_ALL}")
print(f"{Fore.GREEN}F1-score (macro): {f1:.4f}{Style.RESET_ALL}")
print(f"{Fore.CYAN}Testing multi-classe completato con pesi combinati e k-neighbors amplificati!{Style.RESET_ALL}")

# Funzioni di visualizzazione aggiornate per 4 classi
def plot_2d(embeddings_2d, labels, technique):
    filename = os.path.join(save_dir, f"artifact_siamese_{technique}_test_2d.png")
    plt.figure(figsize=(10, 8))
    colors = ['blue', 'orange', 'green', 'red']
    labels_names = ['Imagenet (real)', 'StyleGAN2', 'Taming Transformer', 'SFHQ']
    for i, (color, name) in enumerate(zip(colors, labels_names)):
        plt.scatter(embeddings_2d[labels == i, 0], embeddings_2d[labels == i, 1], c=color, label=name, alpha=0.6, s=10)
    plt.legend()
    plt.title(f"{technique.upper()} 2D Test Embeddings")
    plt.savefig(filename)
    plt.close()
    print(f"{Fore.GREEN}Salvato: {filename}{Style.RESET_ALL}")

def plot_3d(embeddings_3d, labels, technique):
    filename = os.path.join(save_dir, f"artifact_siamese_{technique}_test_3d.png")
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    colors = ['blue', 'orange', 'green', 'red']
    labels_names = ['Imagenet (real)', 'StyleGAN2', 'Taming Transformer', 'SFHQ']
    for i, (color, name) in enumerate(zip(colors, labels_names)):
        ax.scatter(embeddings_3d[labels == i, 0], embeddings_3d[labels == i, 1], embeddings_3d[labels == i, 2], c=color, label=name, alpha=0.6, s=10)
    ax.legend()
    ax.set_title(f"{technique.upper()} 3D Test Embeddings")
    plt.savefig(filename)
    plt.close()
    print(f"{Fore.GREEN}Salvato: {filename}{Style.RESET_ALL}")

print(f"{Fore.CYAN}Calcolo e salvataggio visualizzazioni 2D e 3D multi-classe...{Style.RESET_ALL}")

# t-SNE
tsne_2d = TSNE(n_components=2, random_state=42).fit_transform(embeddings_test)
plot_2d(tsne_2d, labels_test, "tsne")

tsne_3d = TSNE(n_components=3, random_state=42).fit_transform(embeddings_test)
plot_3d(tsne_3d, labels_test, "tsne")

# PCA
pca_2d = PCA(n_components=2).fit_transform(embeddings_test)
plot_2d(pca_2d, labels_test, "pca")

pca_3d = PCA(n_components=3).fit_transform(embeddings_test)
plot_3d(pca_3d, labels_test, "pca")

# UMAP
umap_2d = umap.UMAP(n_components=2, random_state=42).fit_transform(embeddings_test)
plot_2d(umap_2d, labels_test, "umap")

umap_3d = umap.UMAP(n_components=3, random_state=42).fit_transform(embeddings_test)
plot_3d(umap_3d, labels_test, "umap")

print(f"{Fore.GREEN}Tutte le visualizzazioni 2D e 3D multi-classe sono state salvate con i nomi corretti!{Style.RESET_ALL}")
