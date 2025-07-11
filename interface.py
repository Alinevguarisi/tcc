import cv2
import os
import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms

# ------ Configurações ------
model_path = 'cnn_lstm_model.pth'
dataset_path = 'C:\\Users\\Aline\\Desktop\\gestures_dataset'
max_len = 30
frame_size = (224, 224)

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
    gestures = sorted(os.listdir(dataset_path))
    return {idx: gesture for idx, gesture in enumerate(gestures)}

# ------ Carrega modelo ------
class_map = get_class_map(dataset_path)
num_classes = len(class_map)
model = CNNLSTMModel(cnn_output_size=32 * 56 * 56, hidden_size=128, num_classes=num_classes)
model.load_state_dict(torch.load(model_path))
model.eval()

# ------ Transformação para imagens ------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(frame_size)
])

# ------ Captura da webcam ------
cap = cv2.VideoCapture(0)
frame_buffer = []

print("Pressione ESPAÇO para capturar gesto e prever.")
print("Pressione ESC para sair.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    display_frame = frame.copy()
    cv2.rectangle(display_frame, (100, 100), (400, 400), (0, 255, 0), 2)
    cv2.putText(display_frame, "Gesto: Pressione ESPACO", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("Reconhecimento de Gestos - Libras", display_frame)
    key = cv2.waitKey(1) & 0xFF

    if key == 27:  # ESC
        break

    elif key == 32:  # ESPAÇO
        frame_buffer.clear()
        for i in range(max_len):
            ret, frame = cap.read()
            if not ret:
                break
            roi = frame[100:400, 100:400]
            img = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, frame_size)
            tensor = transform(img)
            frame_buffer.append(tensor)
            cv2.imshow("Capturando...", roi)
            cv2.waitKey(30)

        # Preenchimento, se necessário
        while len(frame_buffer) < max_len:
            frame_buffer.append(torch.zeros_like(frame_buffer[0]))

        input_tensor = torch.stack(frame_buffer).unsqueeze(0)  # [1, seq_len, C, H, W]

        with torch.no_grad():
            output = model(input_tensor)
            _, predicted = torch.max(output, 1)
            predicted_class = class_map[predicted.item()]
            print(f"Gesto reconhecido: {predicted_class}")

            # Mostrar na tela
            cv2.putText(display_frame, f"Gesto: {predicted_class}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
            cv2.imshow("Reconhecimento de Gestos - Libras", display_frame)
            cv2.waitKey(1500)

cap.release()
cv2.destroyAllWindows()
