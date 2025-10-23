# train_model_combo_D_v2.py
import os
import numpy as np
import joblib
import tensorflow as tf
import random as python_random
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tqdm import tqdm
import gc # Garbage Collector

# ─────────────────── 고정 매개변수 및 시드 고정 ─────────────────── #
NUM_FRAMES = 30
PTS = 43 # Raw input: 코(1) + 왼손(21) + 오른손(21) = 43
HAND_SIZE = 21 # 각 손의 키포인트 수 (손목 포함)
FINGER_KPS_COUNT = HAND_SIZE - 1 # 손목 제외한 손가락/손바닥 키포인트 수 (20개)
NUM_COORDS = 3

# 키포인트 인덱스 (43개 기준, raw input data)
# NOSE_IDX_IN_43 = 0 # 함수 내 로컬 변수로 사용
# LEFT_HAND_START_IDX_IN_43 = 1
# RIGHT_HAND_START_IDX_IN_43 = 1 + HAND_SIZE

# 조합 D 특성 수 정의
TOTAL_FEATURES_PER_FRAME = (HAND_SIZE * NUM_COORDS) + \
                           (HAND_SIZE * NUM_COORDS) + \
                           (FINGER_KPS_COUNT * NUM_COORDS) + \
                           (FINGER_KPS_COUNT * NUM_COORDS)  # 63 + 63 + 60 + 60 = 246

# 재현성을 위한 시드 설정
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
python_random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ───────── 시퀀스 → 특성 벡터 변환 (조합 D: 코-손전체 + 손목-손가락) ───────── #
def seq_to_feat_combo_D(sequence_data_flat): # Input: (NUM_FRAMES, PTS * NUM_COORDS) = (30, 129)
    keypoints_reshaped_abs = sequence_data_flat.reshape(NUM_FRAMES, PTS, NUM_COORDS) # (30, 43, 3)
    output_features_list = []

    _NOSE_IDX = 0
    _LHAND_START = 1
    _RHAND_START = 1 + HAND_SIZE
    _HAND_KPS_COUNT = HAND_SIZE
    _FINGER_KPS_COUNT_LOCAL = FINGER_KPS_COUNT

    for t in range(NUM_FRAMES):
        current_frame_kps_abs_3d = keypoints_reshaped_abs[t]

        nose_abs = current_frame_kps_abs_3d[_NOSE_IDX].astype(np.float32) # 타입 지정

        left_hand_all_abs = current_frame_kps_abs_3d[_LHAND_START : _LHAND_START + _HAND_KPS_COUNT].astype(np.float32)
        right_hand_all_abs = current_frame_kps_abs_3d[_RHAND_START : _RHAND_START + _HAND_KPS_COUNT].astype(np.float32)

        left_wrist_abs = left_hand_all_abs[0]
        right_wrist_abs = right_hand_all_abs[0]

        if np.all(np.abs(left_hand_all_abs) < 1e-9):
            relative_left_hand_to_nose = np.zeros_like(left_hand_all_abs, dtype=np.float32)
        elif np.all(np.abs(nose_abs) < 1e-9):
            relative_left_hand_to_nose = np.zeros_like(left_hand_all_abs, dtype=np.float32)
        else:
            relative_left_hand_to_nose = (left_hand_all_abs - nose_abs[None, :]).astype(np.float32)

        if np.all(np.abs(right_hand_all_abs) < 1e-9):
            relative_right_hand_to_nose = np.zeros_like(right_hand_all_abs, dtype=np.float32)
        elif np.all(np.abs(nose_abs) < 1e-9):
            relative_right_hand_to_nose = np.zeros_like(right_hand_all_abs, dtype=np.float32)
        else:
            relative_right_hand_to_nose = (right_hand_all_abs - nose_abs[None, :]).astype(np.float32)

        if np.all(np.abs(left_wrist_abs) < 1e-9) or np.all(np.abs(left_hand_all_abs[1:]) < 1e-9) :
            relative_left_fingers_to_wrist = np.zeros((_FINGER_KPS_COUNT_LOCAL, NUM_COORDS), dtype=np.float32)
        else:
            relative_left_fingers_to_wrist = (left_hand_all_abs[1:] - left_wrist_abs[None, :]).astype(np.float32)

        if np.all(np.abs(right_wrist_abs) < 1e-9) or np.all(np.abs(right_hand_all_abs[1:]) < 1e-9):
            relative_right_fingers_to_wrist = np.zeros((_FINGER_KPS_COUNT_LOCAL, NUM_COORDS), dtype=np.float32)
        else:
            relative_right_fingers_to_wrist = (right_hand_all_abs[1:] - right_wrist_abs[None, :]).astype(np.float32)

        frame_features = np.concatenate([
            relative_left_hand_to_nose.flatten(),
            relative_right_hand_to_nose.flatten(),
            relative_left_fingers_to_wrist.flatten(),
            relative_right_fingers_to_wrist.flatten()
        ]).astype(np.float32) # 최종 결합 후에도 타입 확인

        if frame_features.shape[0] != TOTAL_FEATURES_PER_FRAME:
            correct_len_features = np.zeros(TOTAL_FEATURES_PER_FRAME, dtype=np.float32)
            copy_len = min(frame_features.shape[0], TOTAL_FEATURES_PER_FRAME)
            correct_len_features[:copy_len] = frame_features[:copy_len]
            frame_features = correct_len_features

        output_features_list.append(frame_features)

    return np.array(output_features_list, dtype=np.float32) # 반환 시 타입 지정


# ───────── 데이터셋 로드 (코+양손) ───────── #
def load_dataset(original_data_path, augmented_data_path):
    X_raw_list, y_labels_list = [], []
    data_paths = []
    if os.path.exists(original_data_path): data_paths.append(original_data_path)
    if os.path.exists(augmented_data_path): data_paths.append(augmented_data_path)
    if not data_paths: raise ValueError(f"Data paths not found: {original_data_path}, {augmented_data_path}")

    for root_dir in data_paths:
        print(f"Loading dataset from: {root_dir}")
        labels = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        for label_name in tqdm(labels, desc=f"Labels in {os.path.basename(root_dir)}", unit="label"):
            label_dir_path = os.path.join(root_dir, label_name)
            npy_files = [f for f in os.listdir(label_dir_path) if f.endswith(".npy") and f.startswith("group_")]
            for file_name in npy_files:
                file_path = os.path.join(label_dir_path, file_name)
                try:
                    sequences_in_group = np.load(file_path) # .astype(np.float32) 여기서 해도 됨
                    if sequences_in_group.ndim == 3 and \
                       sequences_in_group.shape[1] == NUM_FRAMES and \
                       sequences_in_group.shape[2] == PTS * NUM_COORDS:
                        # 로드 시 바로 float32로 변환 (선택 사항, X_raw가 커도 괜찮다면)
                        X_raw_list.extend(sequences_in_group.astype(np.float32))
                        y_labels_list.extend([label_name] * len(sequences_in_group))
                    else:
                        print(f"Skipping file with unexpected shape: {file_path}, shape: {sequences_in_group.shape}")
                except Exception as e:
                    print(f"Error loading or processing file {file_path}: {e}")
    if not X_raw_list: raise ValueError("No data loaded. Check dataset paths and file formats.")
    # X_raw_list에 이미 float32 배열들이 들어있음
    return np.array(X_raw_list), np.array(y_labels_list) # y_labels_list는 문자열이므로 그대로


# --- 학습 데이터 경로 설정 ---
ORIG_DATA_ROOT = 'dataset_original_NoseHands_final_v3_newaug'
AUG_DATA_ROOT = 'dataset_augmented_NoseHands_final_v3_newaug'

# The following paths for saving/loading features and raw labels are no longer used
# as per the request to not save these intermediate files.
# FEATURES_SAVE_PATH = f'features_comboD_{TOTAL_FEATURES_PER_FRAME}_float32.npy'
# LABELS_SAVE_PATH = 'labels_comboD_float32.npy'

# Always load raw data and perform feature extraction.
# The conditional loading of pre-extracted features has been removed.
print("Starting raw data loading and feature extraction...")
X_raw, y_raw = load_dataset(ORIG_DATA_ROOT, AUG_DATA_ROOT)
print(f"Raw data loaded: X_raw shape: {X_raw.shape} (dtype: {X_raw.dtype}), y_raw shape: {y_raw.shape}")

print("Converting sequences to feature vectors (Combo D, float32)...")
num_sequences = X_raw.shape[0]
# 미리 float32로 배열 생성
X_feat = np.zeros((num_sequences, NUM_FRAMES, TOTAL_FEATURES_PER_FRAME), dtype=np.float32)
for i, s_raw in enumerate(tqdm(X_raw, desc="Feature extraction", unit="seq")):
    X_feat[i] = seq_to_feat_combo_D(s_raw) # s_raw가 이미 float32라면 더 좋음

# Saving of extracted features (X_feat) and raw labels (y_raw) is skipped as per request.
print("Feature extraction complete. X_feat and y_raw are processed in memory and will not be saved to .npy files.")

print(f"Feature data shape: X_feat: {X_feat.shape}, dtype: {X_feat.dtype}")

# 더 이상 필요 없는 X_raw는 메모리에서 해제 (이미 X_feat로 변환했거나, 파일에서 로드)
if 'X_raw' in locals() or 'X_raw' in globals():
    del X_raw
    print("X_raw deleted from memory.")
gc.collect()


# ───── 스케일링 & 라벨 인코딩 ───── #
num_samples, num_frames_feat, num_features_actual = X_feat.shape
if num_features_actual != TOTAL_FEATURES_PER_FRAME:
    print(f"FATAL Error: Actual number of features ({num_features_actual}) "
          f"does not match expected ({TOTAL_FEATURES_PER_FRAME}). Check feature extraction logic.")
    exit()

# X_feat가 이미 float32인지 확인
print(f"X_feat dtype before scaling: {X_feat.dtype}")

scaler = StandardScaler()
# StandardScaler는 float64로 계산하는 경향이 있음. fit_transform 후 타입을 확인하고 변환 필요.
X_scaled_flat = scaler.fit_transform(X_feat.reshape(-1, num_features_actual))
X_scaled = X_scaled_flat.reshape(num_samples, num_frames_feat, num_features_actual)
X_scaled = X_scaled.astype(np.float32) # 스케일링 후 float32로 확실하게 변환
print(f"X_scaled dtype after scaling and conversion: {X_scaled.dtype}")

# X_feat도 더 이상 필요 없으므로 삭제
del X_feat
del X_scaled_flat
print("X_feat and X_scaled_flat deleted from memory.")
gc.collect()


# 파일명에 특성 조합 정보 반영
SCALER_FILENAME = f"slr_scaler_comboD_{TOTAL_FEATURES_PER_FRAME}_float32.pkl"
MODEL_FILENAME = f"slr_model_comboD_best_{TOTAL_FEATURES_PER_FRAME}_float32.h5"
LABEL_ENCODER_FILENAME = f"slr_label_encoder_comboD_float32.pkl"

joblib.dump(scaler, SCALER_FILENAME)
print(f"Scaler saved to {SCALER_FILENAME}")

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y_raw)
joblib.dump(label_encoder, LABEL_ENCODER_FILENAME)
print(f"Label encoder saved to {LABEL_ENCODER_FILENAME}")

y_one_hot = to_categorical(y_encoded).astype(np.float32) # 원-핫 인코딩 후 float32로 변환
print(f"y_one_hot dtype: {y_one_hot.dtype}")
num_classes = y_one_hot.shape[1]

# y_raw도 더 이상 필요 없음
if 'y_raw' in locals() or 'y_raw' in globals():
    del y_raw
    print("y_raw deleted from memory.")
gc.collect()


X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_one_hot, test_size=0.2, stratify=y_one_hot, random_state=SEED
)
print(f"Train data shape: X_train: {X_train.shape} (dtype: {X_train.dtype}), y_train: {y_train.shape} (dtype: {y_train.dtype})")
print(f"Test data shape: X_test: {X_test.shape} (dtype: {X_test.dtype}), y_test: {y_test.shape} (dtype: {y_test.dtype})")

# X_scaled와 y_one_hot도 분할 후에는 원본이 필요 없을 수 있음
del X_scaled
del y_one_hot
print("X_scaled and y_one_hot deleted from memory.")
gc.collect()


# ───── LSTM 모델 (input_shape의 특성 수가 num_features_actual로 설정됨) ───── #
model = Sequential([
    LSTM(128, return_sequences=True, input_shape=(NUM_FRAMES, num_features_actual)),
    BatchNormalization(),
    Dropout(0.4),
    LSTM(128, return_sequences=True),
    BatchNormalization(),
    Dropout(0.4),
    LSTM(64),
    BatchNormalization(),
    Dropout(0.4),
    Dense(64, activation='relu'),
    Dense(num_classes, activation='softmax')
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=3e-4),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.summary()

model_checkpoint = ModelCheckpoint(MODEL_FILENAME, save_best_only=True, monitor='val_accuracy', mode='max', verbose=1)
early_stopping = EarlyStopping(monitor='val_loss', patience=25, restore_best_weights=True, verbose=1)
reduce_lr_on_plateau = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10, min_lr=1e-7, verbose=1)
callbacks_list = [model_checkpoint, early_stopping, reduce_lr_on_plateau]

print("Starting model training (Combo D features, float32)...")
history = model.fit(
    X_train, y_train, epochs=10, batch_size=32,
    validation_data=(X_test, y_test), callbacks=callbacks_list, shuffle=True
)

print("✅ 학습 완료")
print(f"Best model saved to: {MODEL_FILENAME}")

best_model = tf.keras.models.load_model(MODEL_FILENAME)
loss, accuracy = best_model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest Loss (best model): {loss:.4f}")
print(f"Test Accuracy (best model): {accuracy:.4f}")