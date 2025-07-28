import cv2
import os
import numpy as np
import random
from glob import glob
import mediapipe as mp
import shutil

# Inicializa o MediaPipe Holistic
mp_holistic = mp.solutions.holistic

def get_dynamic_square_roi(frame, holistic, padding_factor=1.3):
    h_frame, w_frame, _ = frame.shape
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

    coords = np.array([[lm_x * w_frame, lm_y * h_frame] for lm_x, lm_y in landmarks])
    x_min, y_min = np.min(coords, axis=0)
    x_max, y_max = np.max(coords, axis=0)
    
    box_w = x_max - x_min
    box_h = y_max - y_min
    center_x = x_min + box_w / 2
    center_y = y_min + box_h / 2

    side_length = max(box_w, box_h)
    square_size = int(side_length * padding_factor)

    new_x_min = int(center_x - square_size / 2)
    new_y_min = int(center_y - square_size / 2)
    new_x_max = new_x_min + square_size
    new_y_max = new_y_min + square_size
    
    new_x_min = max(0, new_x_min)
    new_y_min = max(0, new_y_min)
    new_x_max = min(w_frame, new_x_max)
    new_y_max = min(h_frame, new_y_max)

    square_roi = frame[new_y_min:new_y_max, new_x_min:new_x_max]
    
    return square_roi

def generate_augmentation_params():
    params = {
        'angle': random.uniform(-10, 10),
        'scale': random.uniform(0.9, 1.1),
        'tx': random.uniform(-0.05, 0.05),
        'ty': random.uniform(-0.05, 0.05),
        'brightness': random.uniform(0.8, 1.2),
        'saturation': random.uniform(0.8, 1.2),
        'contrast': random.uniform(0.9, 1.1),
        'do_blur': random.random() < 0.15,
        'noise_std': random.uniform(0, 3),
    }
    return params

def apply_augmentation(image, params):
    if image is None or image.size == 0:
        return None
        
    h, w = image.shape[:2]

    center = (w / 2, h / 2)
    matrix = cv2.getRotationMatrix2D(center, params['angle'], params['scale'])
    matrix[0, 2] += w * params['tx']
    matrix[1, 2] += h * params['ty']
    image = cv2.warpAffine(image, matrix, (w, h), borderMode=cv2.BORDER_REFLECT)

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 1] *= params['saturation']
    hsv[..., 2] *= params['brightness']
    hsv = np.clip(hsv, 0, 255).astype(np.uint8)
    image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    image = cv2.addWeighted(image, params['contrast'], np.zeros_like(image), 0, 0)

    if params['do_blur']:
        image = cv2.GaussianBlur(image, (3, 3), 0)
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

# ==================== CAMINHOS (AJUSTE AQUI) ====================
caminho_videos_originais = r"G:\Meu Drive\TCC - Aline e Gabi\sinais_treinados\obrigado"
caminho_local_temporario = r'C:\Users\Aline\Desktop\dataset_temporario'
caminho_final_drive = r'G:\Meu Drive\TCC - Aline e Gabi\gestures_dataset_processado'
# =================================================================

# --- ETAPA DE VERIFICAÇÃO ---
print("--- INICIANDO VERIFICAÇÃO ---")
print(f"Procurando por vídeos .mp4 em: '{caminho_videos_originais}'")

video_files = glob(os.path.join(caminho_videos_originais, '**', '*.mp4'), recursive=True)

print(f"--> Foram encontrados {len(video_files)} vídeos.")
print("---------------------------\n")
# -----------------------------


# O processamento só continua se algum vídeo for encontrado
if not video_files:
    print("AVISO: Nenhum vídeo foi encontrado. O script não continuará.")
    print("Por favor, verifique se a variável 'caminho_videos_originais' está correta.")
else:
    with mp_holistic.Holistic(static_image_mode=True, min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        for vid_path in video_files:
            gesture_name = os.path.basename(os.path.dirname(vid_path))
            if not gesture_name or gesture_name == os.path.basename(caminho_videos_originais):
                gesture_name = os.path.basename(caminho_videos_originais)
                
            vid_name = os.path.basename(vid_path)
            print(f"Processando vídeo: {vid_name} para o sinal: {gesture_name}")

            video = cv2.VideoCapture(vid_path)
            if not video.isOpened():
                print(f"  - Erro ao abrir o vídeo: {vid_name}")
                continue

            raw_dir, aug_dir = create_output_directory(caminho_local_temporario, gesture_name)
            frame_count = 1

            augmentation_params = generate_augmentation_params()
            
            while video.isOpened():
                flag, frame = video.read()
                if not flag:
                    break

                roi_frame = get_dynamic_square_roi(frame, holistic)

                if roi_frame is None or roi_frame.size == 0:
                    continue

                roi_frame_resized = cv2.resize(roi_frame, (256, 256), interpolation=cv2.INTER_AREA)

                roi_original_path = os.path.join(raw_dir, f"frame_{frame_count}_raw.jpg")
                cv2.imwrite(roi_original_path, roi_frame_resized)

                roi_augmented = apply_augmentation(roi_frame_resized, augmentation_params)
                if roi_augmented is not None:
                    roi_augmented_path = os.path.join(aug_dir, f"frame_{frame_count}_aug.jpg")
                    cv2.imwrite(roi_augmented_path, roi_augmented)

                frame_count += 1
            
            print(f'-> {frame_count-1} frames válidos processados para {vid_name}')
            video.release()
            print(f'✅ Conversão concluída para: {vid_name}', end='\n\n')

    print("🏁 Processamento de vídeos finalizado.")

    # Descomente a linha abaixo se quiser mover os arquivos para o Drive automaticamente
    # print(f"Movendo arquivos de '{caminho_local_temporario}' para '{caminho_final_drive}'...")
    # shutil.move(caminho_local_temporario, caminho_final_drive)
    # print("✅ Arquivos movidos para o Google Drive com sucesso!")