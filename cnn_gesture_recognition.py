import os
import cv2
import json
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import time
import pandas as pd

start_time = time.time()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Treinando em:", device)

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
        self.lstm = nn.LSTM(input_size=cnn_output_size, hidden_size=hidden_size, batch_first=True)
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
    def __init__(self, base_path, max_len=30, use_raw=True, use_aug=True):
        self.sequences = []
        self.labels = []
        self.class_to_idx = {}

        gestures = sorted(os.listdir(base_path))
        for idx, gesture in enumerate(gestures):
            gesture_path = os.path.join(base_path, gesture)
            if not os.path.isdir(gesture_path):
                continue
            self.class_to_idx[gesture] = idx
            for seq_folder in os.listdir(gesture_path):
                seq_path = os.path.join(gesture_path, seq_folder)
                if not os.path.isdir(seq_path):
                    continue

                if use_raw:
                    frames_path = os.path.join(seq_path, 'raw')
                    if os.path.isdir(frames_path):
                        frames = sorted([
                            os.path.join(frames_path, f) for f in os.listdir(frames_path)
                            if f.endswith(('.jpg', '.png'))
                        ])
                        if len(frames) > 0:
                            self.sequences.append(frames)
                            self.labels.append(idx)

                if use_aug:
                    frames_path = os.path.join(seq_path, 'aug')
                    if os.path.isdir(frames_path):
                        frames = sorted([
                            os.path.join(frames_path, f) for f in os.listdir(frames_path)
                            if f.endswith(('.jpg', '.png'))
                        ])
                        if len(frames) > 0:
                            self.sequences.append(frames)
                            self.labels.append(idx)

        self.max_len = max_len
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224, 224)),
            transforms.Normalize([0.485, 0.456, 0.406],  # Média do ImageNet
                                 [0.229, 0.224, 0.225])  # Desvio padrão do ImageNet
        ])

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        frames = self.sequences[idx]
        label = self.labels[idx]

        selected_frames = sample_frames(frames, self.max_len)
        tensor_seq = []

        for fpath in selected_frames:
            img = cv2.imread(fpath)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            tensor = self.transform(img)
            tensor_seq.append(tensor)

        tensor_seq = torch.stack(tensor_seq)
        return tensor_seq, torch.tensor(label)

# ------------------- TREINAMENTO -------------------

base_path = r'G:\\.shortcut-targets-by-id\\1oE-zIqZbRz2ez0t_V-LtSwaX3WOtwg9E\\TCC - Aline e Gabi\\gestures_dataset'
base_path = r'D:\\Everaldo\\Pictures\\temp_gestures_dataset'
dataset = GestureDataset(base_path, max_len=150, use_raw=True, use_aug=True)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

best_acc = 0.0
metrics_history = []
cnn_output_size = 32 * 56 * 56
hidden_size = 128
num_classes = len(dataset.class_to_idx)

model = CNNLSTMModel(cnn_output_size, hidden_size, num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

print(f"Classes: {dataset.class_to_idx}")
print(f"Total sequências: {len(dataset)}")

# Salva o mapeamento de classes
with open("class_to_idx.json", "w") as f:
    json.dump(dataset.class_to_idx, f)
print("📄 class_to_idx.json salvo.")

num_epochs = 10
for epoch in range(num_epochs):
    epoch_start = time.time()
    model.train()
    for sequences, labels in dataloader:
        sequences = sequences.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(sequences)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}/{num_epochs} - Loss: {loss.item():.4f}")
    
    epoch_end = time.time()
    print(f"Epoch {epoch+1}/{num_epochs} - Tempo: {epoch_end - epoch_start:.2f} segundos")

    # Avaliação
    model.eval()
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for sequences, labels in dataloader:
            sequences = sequences.to(device)
            labels = labels.to(device)
            outputs = model(sequences)
            _, preds = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    rec = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    print(f"Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f}")

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
        print(f"🔖 Novo melhor modelo salvo! Accuracy: {acc:.4f}")

    epoch_end = time.time()
    print(f"Epoch {epoch+1}/{num_epochs} - Tempo: {epoch_end - epoch_start:.2f} segundos\\n")

# ------------------- SALVAR MODELO -------------------

torch.save(model.state_dict(), 'cnn_lstm_model.pth')
print("✅ Modelo treinado e salvo como 'cnn_lstm_model.pth'")

df_metrics = pd.DataFrame(metrics_history)
df_metrics.to_csv("metrics_history.csv", index=False)
print("📊 Métricas salvas em metrics_history.csv")

end_time = time.time()
total_time = end_time - start_time
print(f"Tempo total de execução: {total_time:.2f} segundos")
print(f"Tempo total de execução: {total_time/60:.2f} minutos")