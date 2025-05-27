import warnings
warnings.filterwarnings("ignore")

import os
import torch
import torch.nn as nn
import random
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from transformers import ViTModel
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from colorama import Fore, Style
import numpy as np
from scipy.spatial.distance import cdist
import csv

# Device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
image_size = 224
batch_size = 512

# Directory per ogni classe (modificate come richiesto)
train_dirs = {
    'generative_inpainting': [r"C:\Users\labor\Desktop\Progetto_FVAB-main-MD\archive\generative_inpainting\train5000"],
    'latent_diffusion': [r"C:\Users\labor\Desktop\Progetto_FVAB-main-MD\archive\latent_diffusion\train5000"],
    'gau_gan': [r"C:\Users\labor\Desktop\Progetto_FVAB-main-MD\archive\gau_gan\train5000"],
    'pro_gan': [r"C:\Users\labor\Desktop\Progetto_FVAB-main-MD\archive\pro_gan\train5000"]
}

val_dirs = {
    'generative_inpainting': [r"C:\Users\labor\Desktop\Progetto_FVAB-main-MD\archive\generative_inpainting\validation1000"],
    'latent_diffusion': [r"C:\Users\labor\Desktop\Progetto_FVAB-main-MD\archive\latent_diffusion\validation1000"],
    'gau_gan': [r"C:\Users\labor\Desktop\Progetto_FVAB-main-MD\archive\gau_gan\validation1000"],
    'pro_gan': [r"C:\Users\labor\Desktop\Progetto_FVAB-main-MD\archive\pro_gan\validation1000"]
}

# Trasformazioni immagini
transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Mappatura delle classi
class_names = ['generative_inpainting', 'latent_diffusion', 'gau_gan', 'pro_gan']
class_to_idx = {name: idx for idx, name in enumerate(class_names)}

# Dataset custom
class CustomImageDataset(Dataset):
    def __init__(self, dirs_dict, transform):
        self.samples = []
        self.paths = []
        self.labels = []
        self.transform = transform
        
        for class_name, dirs in dirs_dict.items():
            label = class_to_idx[class_name]
            for dir_path in dirs:
                for img_name in os.listdir(dir_path):
                    if img_name.endswith(('.jpg', '.png')):
                        path = os.path.join(dir_path, img_name)
                        self.samples.append((path, label))
                        self.paths.append(path)
                        self.labels.append(label)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        return img, label, img_path

# Modello rete fully connected
class EmbeddingNetwork(nn.Module):
    def __init__(self, embedding_dim=768, output_dim=256):
        super(EmbeddingNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(embedding_dim, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim)
        )

    def forward(self, x):
        return self.fc(x)

# Dataset triplet con logica multi-classe
class TripletDataset(Dataset):
    def __init__(self, embeddings, labels, paths):
        self.embeddings = embeddings
        self.labels = labels
        self.paths = paths

        # Gruppi per classe (indice immagini)
        self.groups = {cls_idx: [] for cls_idx in range(len(class_names))}
        for i, label in enumerate(labels):
            self.groups[label.item()].append(i)

        self.triplets = []
        for anchor_idx in range(len(embeddings)):
            anchor_label = labels[anchor_idx].item()
            pos_pool = self.groups[anchor_label].copy()
            pos_pool.remove(anchor_idx)  # escludi anchor da positivo
            if len(pos_pool) == 0:
                continue  # salta se non ci sono positivi
            pos_idx = random.choice(pos_pool)

            # negativi sono tutte le altre classi diverse dall'anchor
            neg_pool = []
            for cls_idx, idx_list in self.groups.items():
                if cls_idx != anchor_label:
                    neg_pool.extend(idx_list)
            if len(neg_pool) == 0:
                continue
            neg_idx = random.choice(neg_pool)

            self.triplets.append((anchor_idx, pos_idx, neg_idx))

    def save_triplets_to_csv(self, filename):
        with open(filename, mode="w", newline="") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(["path_anchor", "path_positive", "path_negative"])
            for anchor_idx, pos_idx, neg_idx in self.triplets:
                writer.writerow([self.paths[anchor_idx], self.paths[pos_idx], self.paths[neg_idx]])

    def __getitem__(self, index):
        anchor_idx, pos_idx, neg_idx = self.triplets[index]
        return self.embeddings[anchor_idx], self.embeddings[pos_idx], self.embeddings[neg_idx]

    def __len__(self):
        return len(self.triplets)

# Funzione training
def train_siamese_network(embeddings_train, labels_train, paths_train, embeddings_val, labels_val, paths_val, epochs=15, lr=1e-4):
    triplet_dataset = TripletDataset(embeddings_train, labels_train, paths_train)
    triplet_loader = DataLoader(triplet_dataset, batch_size=64, shuffle=True)
    
    triplet_dataset.save_triplets_to_csv(r"C:\Users\labor\Desktop\Progetto_FVAB-main-MD\triplets.csv")

    model = EmbeddingNetwork().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    triplet_loss_fn = nn.TripletMarginLoss(margin=1.0, p=2)

    best_f1 = 0
    best_val_loss = float('inf')

    print(f"{Fore.CYAN}Inizio training rete siamese con validazione...{Style.RESET_ALL}")

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for anchor, positive, negative in tqdm(triplet_loader, leave=False):
            anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)
            anchor_out = model(anchor)
            positive_out = model(positive)
            negative_out = model(negative)

            loss = triplet_loss_fn(anchor_out, positive_out, negative_out)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(triplet_loader)

        with torch.no_grad():
            embeddings_siamese_train = model(embeddings_train.to(device)).cpu()
        torch.save(embeddings_siamese_train, "embeddings_siamese_train.pt")

        model.eval()
        with torch.no_grad():
            val_triplet_dataset = TripletDataset(embeddings_val, labels_val, paths_val)
            val_triplet_loader = DataLoader(val_triplet_dataset, batch_size=64, shuffle=False)

            total_val_loss = 0.0
            for anchor, positive, negative in val_triplet_loader:
                anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)
                anchor_out = model(anchor)
                positive_out = model(positive)
                negative_out = model(negative)
                total_val_loss += triplet_loss_fn(anchor_out, positive_out, negative_out).item()

            val_loss = total_val_loss / len(val_triplet_loader)

            embeddings_val_np = model(embeddings_val.to(device)).cpu().numpy()
            embeddings_siamese_train = torch.load("embeddings_siamese_train.pt").numpy()

            dists = cdist(embeddings_val_np, embeddings_siamese_train)
            predictions = []
            k, alpha, beta = 5, 0.5, 3
            for i in range(len(embeddings_val_np)):
                dists_val = dists[i]
                sorted_idx = np.argsort(dists_val)
                weights = np.exp(-alpha * (dists_val ** 2))
                weights[sorted_idx[:k]] *= beta

                # Score per ogni classe
                class_scores = []
                for cls_idx in range(len(class_names)):
                    cls_mask = (labels_train.numpy() == cls_idx)
                    class_score = np.sum(weights[cls_mask])
                    class_scores.append(class_score)
                pred_class = np.argmax(class_scores)
                predictions.append(pred_class)

            predictions = np.array(predictions)
            acc = accuracy_score(labels_val.numpy(), predictions)
            prec = precision_score(labels_val.numpy(), predictions, average='macro', zero_division=0)
            rec = recall_score(labels_val.numpy(), predictions, average='macro', zero_division=0)
            f1 = f1_score(labels_val.numpy(), predictions, average='macro', zero_division=0)

        print("\n" + "─" * 50)
        print(f"▶ Epoch {epoch + 1} / {epochs} completata")
        print(f"  ➤ Training Loss medio: {avg_loss:.4f}")
        print(f"  ➤ Validation Triplet Loss: {val_loss:.4f}")
        print(f"  ➤ Validation Metrics:")
        print(f"     - Accuracy: {acc:.4f}")
        print(f"     - Precision (macro): {prec:.4f}")
        print(f"     - Recall (macro): {rec:.4f}")
        print(f"     - F1-score (macro): {f1:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), "best_model_f1.pt")
            print(f"{Fore.GREEN}✅ Miglior F1-score aggiornato! Modello salvato (F1: {f1:.4f}){Style.RESET_ALL}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_model_val_loss.pt")
            print(f"{Fore.YELLOW}⭐ Miglior validation loss aggiornata! Modello salvato (Loss: {val_loss:.4f}){Style.RESET_ALL}")

        print("─" * 50)

    with torch.no_grad():
        final_embeddings = model(embeddings_train.to(device)).cpu()
    torch.save(final_embeddings, "embeddings_siamese.pt")
    torch.save(labels_train, "labels.pt")
    print(f"{Fore.GREEN}Embeddings e labels trasformati salvati.{Style.RESET_ALL}")
    return model

# Main
if __name__ == "__main__":
    print(f"{Fore.CYAN}Calcolo embeddings ViT per il training...{Style.RESET_ALL}")
    train_dataset = CustomImageDataset(train_dirs, transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    
    vit_model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k").to(device)
    vit_model.eval()

    embeddings_vit_train, labels_train, paths_train = [], [], []
    with torch.no_grad():
        for imgs, lbls, paths in tqdm(train_loader, desc="Estrazione embeddings train"):
            imgs = imgs.to(device)
            outputs = vit_model(pixel_values=imgs)
            cls_emb = outputs.last_hidden_state[:, 0, :].cpu()
            embeddings_vit_train.append(cls_emb)
            labels_train.append(lbls)
            paths_train.extend(paths)
    embeddings_vit_train = torch.cat(embeddings_vit_train)
    labels_train = torch.cat(labels_train)

    print(f"{Fore.CYAN}Calcolo embeddings ViT per la validation...{Style.RESET_ALL}")
    val_dataset = CustomImageDataset(val_dirs, transform)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    embeddings_vit_val, labels_val, paths_val = [], [], []
    with torch.no_grad():
        for imgs, lbls, paths in tqdm(val_loader, desc="Estrazione embeddings val"):
            imgs = imgs.to(device)
            outputs = vit_model(pixel_values=imgs)
            cls_emb = outputs.last_hidden_state[:, 0, :].cpu()
            embeddings_vit_val.append(cls_emb)
            labels_val.append(lbls)
            paths_val.extend(paths)
    embeddings_vit_val = torch.cat(embeddings_vit_val)
    labels_val = torch.cat(labels_val)

    siamese_model = train_siamese_network(
        embeddings_vit_train, labels_train, paths_train,
        embeddings_vit_val, labels_val, paths_val,
        epochs=15, lr=1e-4
    )
