import os
import json
import torch
from glob import glob
import numpy as np
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import time
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ------------------- MODELO CNN + LSTM -------------------


class CNNLSTMModel(nn.Module):
    def __init__(self, cnn_output_size, hidden_size, num_classes):
        super(CNNLSTMModel, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.lstm = nn.LSTM(input_size=cnn_output_size,
                            hidden_size=hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        batch_size, seq_len, C, H, W = x.size()
        c_out = []
        for i in range(seq_len):
            cnn_out = self.cnn(x[:, i])
            cnn_out = cnn_out.view(batch_size, -1)
            c_out.append(cnn_out)
        cnn_out_seq = torch.stack(c_out, dim=1)
        lstm_out, _ = self.lstm(cnn_out_seq)
        out = self.fc(lstm_out[:, -1])
        return out

# ------------------- DATASET POR SEQUÊNCIA -------------------


def sample_frames(frames, num_samples):
    if len(frames) <= num_samples:
        return frames + [frames[-1]] * (num_samples - len(frames))
    idxs = np.linspace(0, len(frames) - 1, num_samples).astype(int)
    return [frames[i] for i in idxs]


class GestureDataset(Dataset):
    """Com train_augment=True aplica variações de cor/brilho (recomendado para poucos vídeos por classe)."""
    def __init__(self, base_path, max_len=30, use_raw=True, use_aug=True, train_augment=False):
        self.sequences = []
        self.labels = []
        self.class_to_idx = {}

        gestures = sorted(os.listdir(base_path))
        for idx, gesture in enumerate(gestures):
            gesture_path = os.path.join(base_path, gesture)
            if not os.path.isdir(gesture_path):
                continue
            self.class_to_idx[gesture] = idx

            for seq_folder in sorted(os.listdir(gesture_path)):
                seq_path = os.path.join(gesture_path, seq_folder)
                if not os.path.isdir(seq_path):
                    continue

                if use_raw:
                    raw_dir = os.path.join(seq_path, 'raw')
                    if os.path.isdir(raw_dir):
                        frames = sorted(
                            glob(os.path.join(raw_dir, '*.[jp][pn]g')))
                        if frames:
                            self.sequences.append(frames)
                            self.labels.append(idx)

                if use_aug:
                    for sub in sorted(os.listdir(seq_path)):
                        if sub.startswith('aug_'):
                            aug_dir = os.path.join(seq_path, sub)
                            if os.path.isdir(aug_dir):
                                frames = sorted(
                                    glob(os.path.join(aug_dir, '*.[jp][pn]g')))
                                if frames:
                                    self.sequences.append(frames)
                                    self.labels.append(idx)

        self.max_len = max_len
        self.train_augment = train_augment
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                [0.485, 0.456, 0.406],  # Média ImageNet
                [0.229, 0.224, 0.225]   # Desvio padrão ImageNet
            )
        ])
        self.train_transform = transforms.Compose([
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225]
            )
        ])

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        try:
            frames = self.sequences[idx]
            label = self.labels[idx]
            selected_frames = sample_frames(frames, self.max_len)
            tensor_seq = []
            t = self.train_transform if self.train_augment else self.transform
            for fpath in selected_frames:
                img = Image.open(fpath).convert("RGB")
                tensor = t(img)
                tensor_seq.append(tensor)
            tensor_seq = torch.stack(tensor_seq)
            return tensor_seq, torch.tensor(label)
        except Exception as e:
            print(f"Erro no __getitem__ do idx {idx}: {e}")
            raise

# ------------------- TREINAMENTO -------------------


if __name__ == "__main__":
    start_time = time.time()

    # Reprodutibilidade do treino
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Treinando em:", device)

    base_path = '.\imagens_tcc'
    # Dataset COM augmentation em tempo de treino (recomendado para V-Librasil: 3 vídeos/palavra)
    dataset_train = GestureDataset(base_path, max_len=100,
                                   use_raw=True, use_aug=True, train_augment=True)
    # Dataset SEM augmentation para avaliação (métricas estáveis)
    dataset_eval = GestureDataset(base_path, max_len=100,
                                 use_raw=True, use_aug=True, train_augment=False)
    dataloader_train = DataLoader(dataset_train, batch_size=8,
                                 shuffle=True, num_workers=2, pin_memory=True)
    dataloader_eval = DataLoader(dataset_eval, batch_size=8,
                                 shuffle=False, num_workers=2, pin_memory=True)

    best_acc = 0.0
    metrics_history = []
    cnn_output_size = 32 * 56 * 56
    hidden_size = 128
    num_classes = len(dataset_train.class_to_idx)
    num_epochs = 15

    model_config = {
        "num_classes": num_classes,
        "max_len": 100,
        "cnn_output_size": cnn_output_size,
        "hidden_size": hidden_size,
    }

    model = CNNLSTMModel(cnn_output_size, hidden_size, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    print(f"Classes: {dataset_train.class_to_idx}")
    print(f"Total sequências (treino): {len(dataset_train)}")

    with open("class_to_idx.json", "w") as f:
        json.dump(dataset_train.class_to_idx, f)

    print("Treinamento iniciado.")
    for epoch in range(num_epochs):
        epoch_start = time.time()
        model.train()
        for sequences, labels in dataloader_train:
            sequences = sequences.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # Avaliação
        model.eval()
        all_labels = []
        all_preds = []
        with torch.no_grad():
            for sequences, labels in dataloader_eval:
                sequences = sequences.to(device)
                labels = labels.to(device)
                outputs = model(sequences)
                _, preds = torch.max(outputs, 1)
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())

        acc = accuracy_score(all_labels, all_preds)
        prec = precision_score(all_labels, all_preds,
                               average='macro', zero_division=0)
        rec = recall_score(all_labels, all_preds,
                           average='macro', zero_division=0)
        f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        epoch_end = time.time()

        metrics_history.append({
            "epoch": epoch + 1,
            "loss": loss.item(),
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "epoch_time": epoch_end - epoch_start
        })

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), 'cnn_lstm_best_model.pth')
            with open("model_config.json", "w") as f:
                json.dump(model_config, f, indent=2)
            print(f"🔖 Novo melhor modelo salvo! Accuracy: {acc:.4f}")

        print(
            f"Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f} | loss: {loss.item()}")
        print(
            f"Epoch {epoch+1}/{num_epochs} - Tempo: {epoch_end - epoch_start:.2f} segundos", end='\n\n')

    # ------------------- SALVAR MODELO -------------------

    torch.save(model.state_dict(), 'cnn_lstm_model.pth')
    with open("model_config.json", "w") as f:
        json.dump(model_config, f, indent=2)
    print("✅ Modelo treinado e salvo como 'cnn_lstm_model.pth'")
    print("✅ Configuração salva em 'model_config.json'")

    df_metrics = pd.DataFrame(metrics_history)
    df_metrics.to_csv("metrics_history.csv", index=False)
    print("📊 Métricas salvas em metrics_history.csv")

    end_time = time.time()
    total_time = end_time - start_time
    print(f"Tempo total de execução: {total_time:.2f} segundos")
    print(f"Tempo total de execução: {total_time/60:.2f} minutos")
