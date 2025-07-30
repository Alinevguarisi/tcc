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

    # Determina o lado do quadrado usando o maior lado da caixa delimitadora
    side_length = max(box_w, box_h)
    square_size = int(side_length * padding_factor)

    # Calcula as novas coordenadas para o quadrado centrado
    new_x_min = int(center_x - square_size / 2)
    new_y_min = int(center_y - square_size / 2)
    new_x_max = new_x_min + square_size
    new_y_max = new_y_min + square_size
    
    # Garante que as coordenadas do quadrado não saiam dos limites da imagem
    new_x_min = max(0, new_x_min)
    new_y_min = max(0, new_y_min)
    new_x_max = min(w_frame, new_x_max)
    new_y_max = min(h_frame, new_y_max)

    # Recorta a região de interesse (ROI) quadrada
    square_roi = frame[new_y_min:new_y_max, new_x_min:new_x_max]
    
    # Verifica se o recorte resultou em uma imagem válida antes de retornar
    if square_roi.size == 0:
        return None
        
    return square_roi

def generate_augmentation_params():
    params = {
        'angle': random.uniform(-20, 20),
        'scale': random.uniform(0.8, 1.2),
        'shear': random.uniform(-5, 5),
        'brightness': random.uniform(0.6, 1.4),
        'saturation': random.uniform(0.6, 1.4),
        'contrast': random.uniform(0.8, 1.2),
        'do_blur': random.random() < 0.3,
        'blur_ksize': random.choice([3, 5]),
        'noise_std': random.uniform(0, 10),
    }
    return params

def apply_augmentation(image, params):
    if image is None or image.size == 0:
        return None

    h, w = image.shape[:2]
    cx, cy = w / 2, h / 2

    # 1) Shear
    theta_s = np.deg2rad(params['shear'])
    sh = np.tan(theta_s)
    # matriz de shear pivoteada no centro verticalmente
    M_shear = np.array([
        [1,       sh, -sh * cy],
        [0,       1,   0      ]
    ], dtype=np.float32)
    image = cv2.warpAffine(image, M_shear, (w, h), borderMode=cv2.BORDER_REFLECT)

    # 2) Rotação + escala (sempre uniforme)
    M_rs = cv2.getRotationMatrix2D((cx, cy), params['angle'], params['scale'])
    image = cv2.warpAffine(image, M_rs, (w, h), borderMode=cv2.BORDER_REFLECT)

    # 4) Ajustes de cor em HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 1] *= params['saturation']
    hsv[..., 2] *= params['brightness']
    hsv = np.clip(hsv, 0, 255).astype(np.uint8)
    image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    # contraste
    image = cv2.addWeighted(image, params['contrast'], np.zeros_like(image), 0, 0)

    # 5) Blur opcional
    if params['do_blur']:
        k = params['blur_ksize']
        image = cv2.GaussianBlur(image, (k, k), 0)

    # 6) Ruído gaussiano
    noise = np.random.normal(0, params['noise_std'], image.shape).astype(np.int16)
    image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    return image

def create_output_directory(base_path, gesture, n_augs=3):
    # Cria estrutura: sequence_X/raw + sequence_X/aug_1 ... aug_n
    gesture_path = os.path.join(base_path, gesture)
    os.makedirs(gesture_path, exist_ok=True)
    existing = glob(os.path.join(gesture_path, "sequence_*"))
    seq_num = len(existing)
    out_dir = os.path.join(gesture_path, f'sequence_{seq_num}')
    raw_dir = os.path.join(out_dir, 'raw')
    aug_dirs = [os.path.join(out_dir, f'aug_{i+1}') for i in range(n_augs)]
    os.makedirs(raw_dir, exist_ok=True)
    for d in aug_dirs:
        os.makedirs(d, exist_ok=True)
    return raw_dir, aug_dirs

# ==================== CAMINHOS (AJUSTE AQUI) ====================
caminho_videos_originais = r"G:\.shortcut-targets-by-id\1oE-zIqZbRz2ez0t_V-LtSwaX3WOtwg9E\TCC - Aline e Gabi\sinais_treinados"
caminho_local_temporario = r'D:\Everaldo\Pictures\tcc'
caminho_final_drive = r'G:\Meu Drive\TCC - Aline e Gabi\gestures_dataset_processado'
# =================================================================

# --- ETAPA DE VERIFICAÇÃO ---
print("--- INICIANDO VERIFICAÇÃO ---")
print(f"Procurando por vídeos .mp4 em: '{caminho_videos_originais}'")

video_files = glob(os.path.join(caminho_videos_originais, '**', '*.mp4'), recursive=True)

print(f"--> Foram encontrados {len(video_files)} vídeos.")
print("---------------------------\n")
# -----------------------------


# Exemplo de integração no loop de vídeo:
with mp_holistic.Holistic(static_image_mode=False,
                          min_detection_confidence=0.5,
                          min_tracking_confidence=0.5) as holistic:
    for vid_path in video_files:
        gesture = os.path.basename(os.path.dirname(vid_path))
        raw_dir, aug_dirs = create_output_directory(caminho_local_temporario, gesture, n_augs=3)
        # Gera parâmetros de augmentation POR SEQUÊNCIA
        aug_params_list = [generate_augmentation_params() for _ in aug_dirs]

        cap = cv2.VideoCapture(vid_path)
        frame_count = 1
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            roi = get_dynamic_square_roi(frame, holistic)
            if roi is None:
                continue
            roi_resized = cv2.resize(roi, (224, 224), interpolation=cv2.INTER_AREA)
            # Salva imagem raw
            cv2.imwrite(os.path.join(raw_dir, f'frame_{frame_count:04d}_raw.jpg'), roi_resized)
            # Aplica e salva cada augmentation da sequência
            for idx, params in enumerate(aug_params_list):
                aug_img = apply_augmentation(roi_resized, params)
                if aug_img is not None:
                    cv2.imwrite(
                        os.path.join(aug_dirs[idx], f'frame_{frame_count:04d}.jpg'),
                        aug_img
                    )
            frame_count += 1
        cap.release()

    # Descomente a linha abaixo se quiser mover os arquivos para o Drive automaticamente
    # print(f"Movendo arquivos de '{caminho_local_temporario}' para '{caminho_final_drive}'...")
    # shutil.move(caminho_local_temporario, caminho_final_drive)
    # print("✅ Arquivos movidos para o Google Drive com sucesso!")