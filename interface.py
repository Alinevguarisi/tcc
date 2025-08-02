import cv2
import os
import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms
from PIL import Image

# ------ Configurações ------
model_path = 'cnn_lstm_best_model.pth'
dataset_path = r'C:\Users\Aline\Desktop\tcc_imagens\tcc_imagens'
max_len = 60
frame_size = (224, 224)
save_captures = True  # Salvar sequências capturadas

# ------ Modelo CNN + LSTM ------
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
        self.lstm = nn.LSTM(input_size=cnn_output_size, hidden_size=128, batch_first=True)
        self.fc = nn.Linear(128, num_classes)

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

# ------ Obter mapeamento de classes ------
def get_class_map(dataset_path):
    gestures = sorted([d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))])
    return {idx: gesture for idx, gesture in enumerate(gestures)}

# ------ Carrega modelo ------
class_map = get_class_map(dataset_path)
num_classes = len(class_map)
model = CNNLSTMModel(cnn_output_size=32 * 56 * 56, hidden_size=128, num_classes=num_classes)
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

# ------ Transformação para imagens ------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ------ Função para salvar sequência ------
def save_sequence(frames, gesture_name):
    gesture_dir = os.path.join(dataset_path, gesture_name)
    os.makedirs(gesture_dir, exist_ok=True)
    seq_num = len([d for d in os.listdir(gesture_dir) if os.path.isdir(os.path.join(gesture_dir, d))]) + 1
    seq_folder = os.path.join(gesture_dir, f"seq_{seq_num:03d}", "raw")
    os.makedirs(seq_folder, exist_ok=True)
    for i, frame in enumerate(frames):
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        img.save(os.path.join(seq_folder, f"frame_{i:03d}.jpg"))
    print(f"Sequência salva em: {seq_folder}")

# ------ Captura da webcam ------
cap = cv2.VideoCapture(2)
frame_buffer = []

print("Pressione ESPAÇO para capturar gesto e prever.")
print("Pressione ESC para sair.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    display_frame = frame.copy()
    cv2.rectangle(display_frame, (50, 50), (550, 550), (0, 255, 0), 2)
    cv2.putText(display_frame, "Pressione ESPACO para capturar", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("Reconhecimento de Gestos - Libras", display_frame)
    key = cv2.waitKey(1) & 0xFF

    if key == 27:  # ESC
        break

    elif key == 32:  # ESPAÇO
        frame_buffer.clear()
        raw_frames = []
        for i in range(max_len):
            ret, frame = cap.read()
            if not ret:
                break
            roi = frame[50:550, 50:550]
            raw_frames.append(roi.copy())
            img = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, frame_size)
            tensor = transform(img)
            frame_buffer.append(tensor)
            cv2.imshow("Capturando...", roi)
            cv2.waitKey(30)

        # Preenchimento
        while len(frame_buffer) < max_len:
            frame_buffer.append(torch.zeros_like(frame_buffer[0]))
            raw_frames.append(raw_frames[-1])

        input_tensor = torch.stack(frame_buffer).unsqueeze(0)  # [1, seq_len, C, H, W]

        with torch.no_grad():
            output = model(input_tensor)
            _, predicted = torch.max(output, 1)
            predicted_class = class_map[predicted.item()]
            print(f"Gesto reconhecido: {predicted_class}")

            if save_captures:
                save_sequence(raw_frames, predicted_class)

            cv2.putText(display_frame, f"Gesto: {predicted_class}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
            cv2.imshow("Reconhecimento de Gestos - Libras", display_frame)
            cv2.waitKey(1500)

cap.release()
cv2.destroyAllWindows()
