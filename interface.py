import cv2
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

# ------ Configurações ------
with open("class_to_idx.json", "r", encoding="utf-8") as f:
    class_to_idx = json.load(f)
class_map = {v: k for k, v in class_to_idx.items()}
model_path = 'cnn_lstm_best_model.pth'

buffer_len = 135    # Use o mesmo do treinamento
frame_size = (224, 224)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

# ------ Carrega modelo ------
num_classes = len(class_map)
model = CNNLSTMModel(cnn_output_size=32 * 56 * 56, hidden_size=128, num_classes=num_classes)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# ------ Transformação para imagens ------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(frame_size),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ------ Captura da webcam ------
cap = cv2.VideoCapture(0)
frame_buffer = []
last_predicted_idx = None

print("Pressione ESC para sair.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    display_frame = frame.copy()
    cv2.rectangle(display_frame, (100, 100), (400, 400), (0, 255, 0), 2)
    cv2.putText(display_frame, "Mostre o sinal", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Captura contínua dos frames da ROI
    roi = frame[100:400, 100:400]
    img = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, frame_size)
    tensor = transform(img)
    frame_buffer.append(tensor)
    if len(frame_buffer) > buffer_len:
        frame_buffer.pop(0)

    if len(frame_buffer) == buffer_len:
        input_tensor = torch.stack(frame_buffer).unsqueeze(0).to(device)  # [1, seq_len, C, H, W]
        with torch.no_grad():
            output = model(input_tensor)
            probs = F.softmax(output, dim=1)[0].cpu().numpy()
            sorted_indices = probs.argsort()[::-1]
            predicted_idx = sorted_indices[0]

            # Print apenas quando o gesto reconhecido mudar
            if predicted_idx != last_predicted_idx:
                print("\nClassificação para a captura atual (ordenado):")
                for idx in sorted_indices:
                    print(f"{class_map[idx]}: {probs[idx]:.4f}")
                print("-" * 40)
                last_predicted_idx = predicted_idx

            predicted_class = class_map[predicted_idx]
            cv2.putText(display_frame, f"Gesto: {predicted_class}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

    cv2.imshow("Reconhecimento de Gestos - Libras", display_frame)
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()