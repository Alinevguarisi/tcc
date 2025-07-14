import cv2
import os
import numpy as np
import random
from glob import glob
import mediapipe as mp
import shutil

# Inicializa o MediaPipe Holistic
mp_holistic = mp.solutions.holistic

def normalize_frame(frame, size=(224, 224)):
    return cv2.resize(frame, size)

def get_dynamic_roi(frame, holistic):
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = holistic.process(image_rgb)

    landmarks = []
    if results.face_landmarks:
        landmarks.extend([(lm.x, lm.y) for lm in results.face_landmarks.landmark])
    if results.left_hand_landmarks:
        landmarks.extend([(lm.x, lm.y) for lm in results.left_hand_landmarks.landmark])
    if results.right_hand_landmarks:
        landmarks.extend([(lm.x, lm.y) for lm in results.right_hand_landmarks.landmark])
    if results.pose_landmarks:
        landmarks.extend([(lm.x, lm.y) for lm in results.pose_landmarks.landmark])

    if not landmarks:
        return None

    h, w, _ = frame.shape
    coords = np.array([[int(x * w), int(y * h)] for x, y in landmarks])

    x_min = max(np.min(coords[:, 0]) - 20, 0)
    y_min = max(np.min(coords[:, 1]) - 20, 0)
    x_max = min(np.max(coords[:, 0]) + 20, w)
    y_max = min(np.max(coords[:, 1]) + 20, h)

    roi = frame[y_min:y_max, x_min:x_max]
    return roi

def generate_augmentation_params():
    params = {
        'do_flip': random.random() < 0.3,
        'angle': random.uniform(-10, 10),
        'saturation': random.uniform(0.95, 1.05),
        'brightness': random.uniform(0.95, 1.1),
        'do_blur': random.random() < 0.2,
        'noise_std': 2
    }
    return params

def apply_augmentation(image, params):
    if params['do_flip']:
        image = cv2.flip(image, 1)
    h, w = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), params['angle'], 1)
    image = cv2.warpAffine(image, matrix, (w, h), borderMode=cv2.BORDER_REFLECT)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 1] *= params['saturation']
    hsv[..., 2] *= params['brightness']
    hsv = np.clip(hsv, 0, 255).astype(np.uint8)
    image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    if params['do_blur']:
        ksize = 3
        image = cv2.GaussianBlur(image, (ksize, ksize), 0)
    noise = np.random.normal(0, params['noise_std'], image.shape).astype(np.int16)
    image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return image

def create_output_directory(base_path, gesture):
    gesture_path = os.path.join(base_path, gesture)
    os.makedirs(gesture_path, exist_ok=True)
    existing_sequences = glob(os.path.join(gesture_path, "sequence_*"))
    sequence_number = len(existing_sequences)
    out_dir = os.path.join(gesture_path, f'sequence_{sequence_number}')
    raw_dir = os.path.join(out_dir, 'raw')
    aug_dir = os.path.join(out_dir, 'aug')
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(aug_dir, exist_ok=True)
    return raw_dir, aug_dir

# === CAMINHOS ===
local_temp_base_path = r'D:\\Everaldo\\Pictures\\temp_gestures_dataset'
drive_base_path = r'G:\\.shortcut-targets-by-id\\1oE-zIqZbRz2ez0t_V-LtSwaX3WOtwg9E\\TCC - Aline e Gabi\\gestures_dataset'
BASE_DRIVE_PATH = r"G:\\.shortcut-targets-by-id\\1oE-zIqZbRz2ez0t_V-LtSwaX3WOtwg9E\\TCC - Aline e Gabi"

video_files = glob(os.path.join(BASE_DRIVE_PATH, '**', '*.mp4'), recursive=True)

# Inicializa MediaPipe Holistic fora do loop
with mp_holistic.Holistic(static_image_mode=True) as holistic:
    for vid_path in video_files:
        gesture_name = os.path.basename(os.path.dirname(vid_path))
        vid_name = os.path.basename(vid_path)
        print(f"Processando vídeo: {vid_name} para o sinal: {gesture_name}")

        video = cv2.VideoCapture(vid_path)
        if not video.isOpened():
            print(f"Erro ao abrir o vídeo: {vid_name}")
            continue

        raw_dir, aug_dir = create_output_directory(local_temp_base_path, gesture_name)
        i = 1

        augmentation_params = generate_augmentation_params()
        while video.isOpened():
            flag, frame = video.read()
            if not flag:
                break

            normalized_frame = normalize_frame(frame)
            roi_frame = get_dynamic_roi(normalized_frame, holistic)

            if roi_frame is None:
                roi_frame = normalized_frame

            # Salva ROI raw
            roi_original_path = os.path.join(raw_dir, f"frame_{i}_roi_raw.jpg")
            cv2.imwrite(roi_original_path, roi_frame)

            # Salva ROI augmentada
            roi_augmented = apply_augmentation(roi_frame, augmentation_params)
            roi_augmented_path = os.path.join(aug_dir, f"frame_{i}_roi.jpg")
            cv2.imwrite(roi_augmented_path, roi_augmented)

            if i % 50 == 0:
                print(f'{i} frames processados para {vid_name}')
            i += 1

        video.release()
        print(f'✅ Conversão concluída para: {vid_name}', end='\n\n')

print("🏁 Processamento de vídeos local finalizado.")

# print("Movendo arquivos para o Google Drive...")
# shutil.move(local_temp_base_path, drive_base_path)
# print("✅ Arquivos movidos para o Google Drive com sucesso!")