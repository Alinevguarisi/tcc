import os
import cv2
import json
import torch
import torch.nn as nn
import torch.optim as optim
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

class GestureDataset(Dataset):
    def __init__(self, base_path, max_len=30):
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
                frames = sorted([
                    os.path.join(seq_path, f) for f in os.listdir(seq_path)
                    if 'roi' in f and f.endswith(('.jpg', '.png'))
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
        tensor_seq = []

        for fpath in frames[:self.max_len]:
            img = cv2.imread(fpath)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            tensor = self.transform(img)
            tensor_seq.append(tensor)

        while len(tensor_seq) < self.max_len:
            tensor_seq.append(torch.zeros_like(tensor_seq[0]))

        tensor_seq = torch.stack(tensor_seq)  # [max_len, 3, 224, 224]
        return tensor_seq, torch.tensor(label)

# ------------------- TREINAMENTO -------------------

base_path = 'C:\\Users\\Aline\\Desktop\\gestures_dataset'
dataset = GestureDataset(base_path, max_len=30)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

cnn_output_size = 32 * 56 * 56
hidden_size = 128
num_classes = len(dataset.class_to_idx)

model = CNNLSTMModel(cnn_output_size, hidden_size, num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print(f"Classes: {dataset.class_to_idx}")
print(f"Total sequências: {len(dataset)}")

# Salva o mapeamento de classes
with open("class_to_idx.json", "w") as f:
    json.dump(dataset.class_to_idx, f)
print("📄 class_to_idx.json salvo.")

num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    for sequences, labels in dataloader:
        optimizer.zero_grad()
        outputs = model(sequences)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}/{num_epochs} - Loss: {loss.item():.4f}")

    # Avaliação
    model.eval()
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for sequences, labels in dataloader:
            outputs = model(sequences)
            _, preds = torch.max(outputs, 1)
            all_labels.extend(labels.numpy())
            all_preds.extend(preds.numpy())

    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    rec = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    print(f"Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f}")

# ------------------- SALVAR MODELO -------------------

torch.save(model.state_dict(), 'cnn_lstm_model.pth')
print("✅ Modelo treinado e salvo como 'cnn_lstm_model.pth'")
