import cv2
import os
import json
import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms
from PIL import Image

# Importa ROI dinâmica (MediaPipe) igual ao treino para consistência
from videos import get_dynamic_square_roi
import mediapipe as mp

# ------ Configurações (podem ser sobrescritas por model_config.json) ------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(SCRIPT_DIR, 'cnn_lstm_best_model.pth')
class_to_idx_path = os.path.join(SCRIPT_DIR, 'class_to_idx.json')
model_config_path = os.path.join(SCRIPT_DIR, 'model_config.json')
dataset_path = r'C:\Users\Aline\Desktop\tcc_imagens\tcc_imagens'  # só para salvar capturas
frame_size = (224, 224)
save_captures = True  # Salvar sequências capturadas
camera_id = 2

# Valores padrão (usados se model_config.json não existir)
DEFAULT_MAX_LEN = 100
DEFAULT_CNN_OUTPUT = 32 * 56 * 56
DEFAULT_HIDDEN = 128

# ------ Modelo CNN + LSTM (igual ao treino) ------
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

# ------ Carregar mapeamento de classes do JSON (mesmo do treino) ------
def load_class_map():
    if os.path.isfile(class_to_idx_path):
        with open(class_to_idx_path, "r", encoding="utf-8") as f:
            class_to_idx = json.load(f)
        idx_to_class = {int(idx): name for name, idx in class_to_idx.items()}
        return idx_to_class
    if os.path.isdir(dataset_path):
        gestures = sorted([d for d in os.listdir(dataset_path)
                          if os.path.isdir(os.path.join(dataset_path, d))])
        return {idx: g for idx, g in enumerate(gestures)}
    raise FileNotFoundError(
        f"Não encontrado {class_to_idx_path}. Rode o treino antes (cnn_gesture_recognition.py)."
    )

# ------ Carregar config do modelo (max_len e dimensões) ------
def load_model_config():
    if os.path.isfile(model_config_path):
        with open(model_config_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {
        "max_len": DEFAULT_MAX_LEN,
        "num_classes": None,
        "cnn_output_size": DEFAULT_CNN_OUTPUT,
        "hidden_size": DEFAULT_HIDDEN,
    }

# ------ Inicialização ------
class_map = load_class_map()
config = load_model_config()
max_len = config.get("max_len", DEFAULT_MAX_LEN)
num_classes = len(class_map)
cnn_output_size = config.get("cnn_output_size", DEFAULT_CNN_OUTPUT)
hidden_size = config.get("hidden_size", DEFAULT_HIDDEN)

model = CNNLSTMModel(cnn_output_size=cnn_output_size, hidden_size=hidden_size, num_classes=num_classes)
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ------ Função para salvar sequência ------
def save_sequence(frames, gesture_name):
    if not os.path.isdir(dataset_path):
        return
    gesture_dir = os.path.join(dataset_path, gesture_name)
    os.makedirs(gesture_dir, exist_ok=True)
    existing = [d for d in os.listdir(gesture_dir) if os.path.isdir(os.path.join(gesture_dir, d))]
    seq_num = len([d for d in existing if d.startswith("seq_")]) + 1
    seq_folder = os.path.join(gesture_dir, f"seq_{seq_num:03d}", "raw")
    os.makedirs(seq_folder, exist_ok=True)
    for i, frame in enumerate(frames):
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        img.save(os.path.join(seq_folder, f"frame_{i:03d}.jpg"))
    print(f"Sequência salva em: {seq_folder}")

# ------ Captura com ROI dinâmica (MediaPipe), igual ao treino ------
mp_holistic = mp.solutions.holistic

cap = cv2.VideoCapture(camera_id)
frame_buffer = []

print("Pressione ESPAÇO para capturar gesto e prever.")
print("ESC para sair. Use a mesma ROI (rosto/mãos) do treino.")
print(f"max_len={max_len} frames (igual ao treino)")

with mp_holistic.Holistic(static_image_mode=False,
                          min_detection_confidence=0.5,
                          min_tracking_confidence=0.5) as holistic:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        roi_preview = get_dynamic_square_roi(frame, holistic)
        if roi_preview is None:
            h, w = frame.shape[:2]
            side = min(h, w) // 2
            x0, y0 = (w - side) // 2, (h - side) // 2
            roi_preview = frame[y0:y0+side, x0:x0+side]
            cv2.putText(frame, "Posicione rosto e maos no centro", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
        else:
            cv2.putText(frame, "ROI detectada - ESPACO para capturar", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        display_frame = frame.copy()
        cv2.imshow("Reconhecimento de Gestos - Libras", display_frame)
        key = cv2.waitKey(1) & 0xFF

        if key == 27:
            break

        elif key == 32:
            frame_buffer.clear()
            raw_frames = []
            for i in range(max_len):
                ret, frame = cap.read()
                if not ret:
                    break
                roi = get_dynamic_square_roi(frame, holistic)
                if roi is None:
                    h, w = frame.shape[:2]
                    side = min(h, w) // 2
                    x0, y0 = (w - side) // 2, (h - side) // 2
                    roi = frame[y0:y0+side, x0:x0+side]
                roi_resized = cv2.resize(roi, frame_size, interpolation=cv2.INTER_AREA)
                raw_frames.append(roi_resized.copy())
                img_rgb = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2RGB)
                tensor = transform(img_rgb)
                frame_buffer.append(tensor)
                cv2.imshow("Capturando...", roi_resized)
                cv2.waitKey(30)

            while len(frame_buffer) < max_len:
                frame_buffer.append(torch.zeros_like(frame_buffer[0]))
                raw_frames.append(raw_frames[-1])

            input_tensor = torch.stack(frame_buffer).unsqueeze(0)

            with torch.no_grad():
                output = model(input_tensor)
                _, predicted = torch.max(output, 1)
                predicted_class = class_map[predicted.item()]
                print(f"Gesto reconhecido: {predicted_class}")

                if save_captures and os.path.isdir(dataset_path):
                    save_sequence(raw_frames, predicted_class)

                display_frame = frame.copy() if ret else display_frame
                cv2.putText(display_frame, f"Gesto: {predicted_class}", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
                cv2.imshow("Reconhecimento de Gestos - Libras", display_frame)
                cv2.waitKey(1500)

cap.release()
cv2.destroyAllWindows()
