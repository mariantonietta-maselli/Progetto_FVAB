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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from scipy.spatial.distance import cdist
from colorama import Fore, Style
import seaborn as sns

# ===================== PARAMETRI =====================
image_size = 224
batch_size = 512
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ===================== PERCORSI =====================
root_dir = r"C:\Users\labor\Desktop\Progetto_FVAB-main-MD"
results_subdir = os.path.join(root_dir, "RisultatiSiameseSimili", "Risultati")

model_path = os.path.join(results_subdir, "best_model_val_loss.pt")
embeddings_path = os.path.join(results_subdir, "embeddings_siamese.pt")
labels_path = os.path.join(results_subdir, "labels.pt")

save_dir = os.path.join(root_dir, "RisultatiSiameseSimili")
graphs_dir = os.path.join(save_dir, "Grafici")
os.makedirs(save_dir, exist_ok=True)
os.makedirs(graphs_dir, exist_ok=True)

# ===================== DATASET =====================
fake_dirs = {
    0: [os.path.join(root_dir, "archive", "generative_inpainting", "test1000")],
    1: [os.path.join(root_dir, "archive", "latent_diffusion", "test1000")],
    2: [os.path.join(root_dir, "archive", "gau_gan", "test1000")],
    3: [os.path.join(root_dir, "archive", "pro_gan", "test1000")]
}

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

# ===================== MODELLO =====================
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

# ===================== CARICAMENTO MODELLI =====================
print(f"{Fore.CYAN}Caricamento modello ViT preaddestrato...{Style.RESET_ALL}")
vit_model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k").to(device)
vit_model.eval()

print(f"{Fore.CYAN}Caricamento modello siamese addestrato...{Style.RESET_ALL}")
siamese_model = EmbeddingNetwork().to(device)
siamese_model.load_state_dict(torch.load(model_path))
siamese_model.eval()

# ===================== DATASET =====================
datasets = [ArtifactDataset(dirs, label, transform) for label, dirs in fake_dirs.items()]
full_dataset = sum(datasets, start=datasets[0])  # concatenazione custom
loader = DataLoader(full_dataset, batch_size=batch_size, shuffle=False)

# ===================== CARICAMENTO EMBEDDINGS TRAIN =====================
print(f"{Fore.CYAN}Caricamento embeddings di training...{Style.RESET_ALL}")
embeddings_train = torch.load(embeddings_path).numpy()
labels_train = torch.load(labels_path).numpy()

# ===================== ESTRAZIONE EMBEDDINGS TEST =====================
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

# ===================== CLASSIFICAZIONE =====================
print(f"{Fore.CYAN}Calcolo predizioni multi-classe (4 classi)...{Style.RESET_ALL}")
k = 5
alpha = 0.5
beta = 3
predictions = []
for test_emb in tqdm(embeddings_test, desc="Calcolo predizioni"):
    dists = cdist([test_emb], embeddings_train)[0]
    sorted_indices = np.argsort(dists)
    weights = np.exp(-alpha * (dists ** 2))
    weights[sorted_indices[:k]] *= beta
    class_scores = {cls: np.sum(weights[labels_train == cls]) for cls in np.unique(labels_train)}
    predictions.append(max(class_scores, key=class_scores.get))

predictions = np.array(predictions)

# ===================== METRICHE =====================
accuracy = accuracy_score(labels_test, predictions)
precision = precision_score(labels_test, predictions, average='macro', zero_division=0)
recall = recall_score(labels_test, predictions, average='macro', zero_division=0)
f1 = f1_score(labels_test, predictions, average='macro', zero_division=0)

print(f"{Fore.GREEN}Accuracy: {accuracy:.4f}{Style.RESET_ALL}")
print(f"{Fore.GREEN}Precision (macro): {precision:.4f}{Style.RESET_ALL}")
print(f"{Fore.GREEN}Recall (macro): {recall:.4f}{Style.RESET_ALL}")
print(f"{Fore.GREEN}F1-score (macro): {f1:.4f}{Style.RESET_ALL}")

# ===================== MATRICE DI CONFUSIONE =====================
class_labels = ['Generative Inpainting', 'Latent Diffusion', 'GauGAN', 'ProGAN']
cf = confusion_matrix(labels_test, predictions)
report = classification_report(labels_test, predictions, target_names=class_labels)

with open(os.path.join(save_dir, "log_metrics.txt"), "w") as f:
    f.write(f"Accuracy: {accuracy:.4f}\n")
    f.write(f"Precision (macro): {precision:.4f}\n")
    f.write(f"Recall (macro): {recall:.4f}\n")
    f.write(f"F1-score (macro): {f1:.4f}\n\n")
    f.write("Classification Report:\n")
    f.write(report)
    f.write("\nConfusion Matrix:\n")
    f.write(str(cf))

def plot_confusion_matrix(cf, class_labels, save_path=None):
    group_counts = ["{0:0.0f}".format(value) for value in cf.flatten()]
    group_percentages = ["{0:.2%}".format(value) for value in cf.flatten() / np.sum(cf)]
    labels = [f"{count}\n({percent})" for count, percent in zip(group_counts, group_percentages)]
    labels = np.asarray(labels).reshape(cf.shape)

    plt.figure(figsize=(7, 6))
    sns.set(style="whitegrid")
    ax = sns.heatmap(cf, annot=labels, fmt='', cmap='Blues', cbar=False,
                     xticklabels=class_labels, yticklabels=class_labels)
    ax.set_title("Matrice di Confusione")
    ax.set_xlabel("Classe Predetta")
    ax.set_ylabel("Classe Reale")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.close()

plot_confusion_matrix(cf, class_labels, os.path.join(graphs_dir, "confusion_matrix.png"))
print(f"{Fore.GREEN}Matrice di confusione salvata in: {graphs_dir}{Style.RESET_ALL}")

# ===================== VISUALIZZAZIONI =====================
def plot_2d(embeddings_2d, labels, technique):
    filename = os.path.join(graphs_dir, f"artifact_siamese_{technique}_test_2d.png")
    plt.figure(figsize=(10, 8))
    colors = ['blue', 'orange', 'green', 'red']
    labels_names = class_labels
    for i, (color, name) in enumerate(zip(colors, labels_names)):
        plt.scatter(embeddings_2d[labels == i, 0], embeddings_2d[labels == i, 1], c=color, label=name, alpha=0.6, s=10)
    plt.legend()
    plt.title(f"{technique.upper()} 2D Test Embeddings")
    plt.savefig(filename)
    plt.close()
    print(f"{Fore.GREEN}Salvato: {filename}{Style.RESET_ALL}")

def plot_3d(embeddings_3d, labels, technique):
    filename = os.path.join(graphs_dir, f"artifact_siamese_{technique}_test_3d.png")
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    colors = ['blue', 'orange', 'green', 'red']
    labels_names = class_labels
    for i, (color, name) in enumerate(zip(colors, labels_names)):
        ax.scatter(embeddings_3d[labels == i, 0], embeddings_3d[labels == i, 1], embeddings_3d[labels == i, 2],
                   c=color, label=name, alpha=0.6, s=10)
    ax.legend()
    ax.set_title(f"{technique.upper()} 3D Test Embeddings")
    plt.savefig(filename)
    plt.close()
    print(f"{Fore.GREEN}Salvato: {filename}{Style.RESET_ALL}")

print(f"{Fore.CYAN}Calcolo e salvataggio visualizzazioni 2D e 3D...{Style.RESET_ALL}")

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

print(f"{Fore.GREEN}Tutte le visualizzazioni e metriche sono state salvate correttamente!{Style.RESET_ALL}")
