import os
import numpy as np
from flask import Flask, request, jsonify
import tensorflow as tf
import joblib
import random
import copy
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation as R
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# 설정
NUM_FRAMES = 30
NUM_KEYPOINTS = 43
NUM_COORDS = 3
TARGET_SHAPE = (NUM_FRAMES, NUM_KEYPOINTS * NUM_COORDS)

HAND_SIZE = 21
FINGER_COUNT = 20
TOTAL_FEATURES_PER_FRAME = (HAND_SIZE * 3) * 2 + (FINGER_COUNT * 3) * 2

NOSE_IDX = 0
LEFT_HAND_START_IDX = 1
RIGHT_HAND_START_IDX = 22
IS_HAND_INACTIVE_THRESHOLD = 1e-5

# 증강 설정
HAND_ROTATION_DEG_RANGE_INDIVIDUAL = (-15, 15)
KEYPOINT_JITTER_SIGMA_RANGE_V2 = (0.003, 0.005)

# 경로 및 모델 로드
ORIG_DATASET_PATH = 'dataset_original_NoseHands_final_v3_newaug'
AUG_DATASET_PATH = 'dataset_augmented_NoseHands_final_v3_newaug'

LABEL_ENCODER_PATH = 'slr_label_encoder_comboD_float32.pkl'
SCALER_PATH = f'slr_scaler_comboD_{TOTAL_FEATURES_PER_FRAME}_float32.pkl'
MODEL_PATH = f'slr_model_comboD_best_{TOTAL_FEATURES_PER_FRAME}_float32.h5'

label_encoder = joblib.load(LABEL_ENCODER_PATH)
scaler = joblib.load(SCALER_PATH)
model = tf.keras.models.load_model(MODEL_PATH)
labels = label_encoder.classes_

# 유틸리티
def interpolate_zeros(seq_flat):
    seq_interpolated = seq_flat.copy()
    for dim_idx in range(seq_interpolated.shape[1]):
        col = seq_interpolated[:, dim_idx].copy()
        nz_indices = np.nonzero(col)[0]
        if len(nz_indices) == 0: continue
        elif len(nz_indices) == 1:
            seq_interpolated[:, dim_idx] = col[nz_indices[0]]
            continue
        interp_func = interp1d(nz_indices, col[nz_indices], kind='linear', bounds_error=False,
                               fill_value=(col[nz_indices[0]], col[nz_indices[-1]]))
        seq_interpolated[:, dim_idx] = interp_func(np.arange(len(col)))
    return seq_interpolated

def valid(seq_flat, threshold=0.7):
    if seq_flat.shape != TARGET_SHAPE: return False
    if (np.abs(seq_flat) < 1e-9).mean() >= threshold: return False
    return True

def save_sequences(label, sequences_list, base_path):
    path = os.path.join(base_path, label)
    os.makedirs(path, exist_ok=True)
    file = os.path.join(path, 'group_000.npy')
    if os.path.exists(file):
        data = np.load(file, allow_pickle=True)
        data = list(data) if isinstance(data, np.ndarray) else []
    else:
        data = []
    data.extend(sequences_list)
    np.save(file, np.array(data, dtype=object))

def generate_augmentations_v2(original_sequence_flat):
    original_kps_3d_abs = original_sequence_flat.reshape(NUM_FRAMES, NUM_KEYPOINTS, NUM_COORDS)
    augmented_sequences_list = []

    l_hand = original_kps_3d_abs[:, LEFT_HAND_START_IDX:LEFT_HAND_START_IDX+HAND_SIZE, :]
    r_hand = original_kps_3d_abs[:, RIGHT_HAND_START_IDX:RIGHT_HAND_START_IDX+HAND_SIZE, :]
    is_left_active = np.mean(np.abs(l_hand)) > IS_HAND_INACTIVE_THRESHOLD
    is_right_active = np.mean(np.abs(r_hand)) > IS_HAND_INACTIVE_THRESHOLD

    for _ in range(20):
        aug = copy.deepcopy(original_kps_3d_abs)

        for offset, is_active in [(LEFT_HAND_START_IDX, is_left_active), (RIGHT_HAND_START_IDX, is_right_active)]:
            if is_active:
                rot = R.from_euler('xyz', np.random.uniform(-15, 15, 3), degrees=True).as_matrix()
                wrist = aug[:, offset, :]
                hand = aug[:, offset:offset+HAND_SIZE, :]
                rel = hand - wrist[:, np.newaxis, :]
                rotated = np.einsum('ij,tkj->tki', rot, rel)
                aug[:, offset:offset+HAND_SIZE, :] = rotated + wrist[:, np.newaxis, :]

        sigma = random.uniform(*KEYPOINT_JITTER_SIGMA_RANGE_V2)
        noise = np.random.normal(0, sigma, aug.shape)
        aug[:, NOSE_IDX, :] += noise[:, NOSE_IDX, :]
        if is_left_active:
            aug[:, LEFT_HAND_START_IDX:LEFT_HAND_START_IDX+HAND_SIZE, :] += noise[:, LEFT_HAND_START_IDX:LEFT_HAND_START_IDX+HAND_SIZE, :]
        if is_right_active:
            aug[:, RIGHT_HAND_START_IDX:RIGHT_HAND_START_IDX+HAND_SIZE, :] += noise[:, RIGHT_HAND_START_IDX:RIGHT_HAND_START_IDX+HAND_SIZE, :]

        if not is_left_active:
            aug[:, LEFT_HAND_START_IDX:LEFT_HAND_START_IDX+HAND_SIZE, :] = 0
        if not is_right_active:
            aug[:, RIGHT_HAND_START_IDX:RIGHT_HAND_START_IDX+HAND_SIZE, :] = 0

        augmented_sequences_list.append(aug.reshape(TARGET_SHAPE))
    return augmented_sequences_list

def seq_to_feat_combo_D(sequence_data_flat):
    seq = sequence_data_flat.reshape(NUM_FRAMES, NUM_KEYPOINTS, NUM_COORDS)
    out = []
    for t in range(NUM_FRAMES):
        kp = seq[t]
        nose = kp[0]
        lhand = kp[1:1+HAND_SIZE]
        rhand = kp[1+HAND_SIZE:]
        lwrist = lhand[0]
        rwrist = rhand[0]
        l_to_nose = lhand - nose if np.any(lhand) and np.any(nose) else np.zeros_like(lhand)
        r_to_nose = rhand - nose if np.any(rhand) and np.any(nose) else np.zeros_like(rhand)
        l_finger = lhand[1:] - lwrist if np.any(lhand[1:]) and np.any(lwrist) else np.zeros((FINGER_COUNT, 3))
        r_finger = rhand[1:] - rwrist if np.any(rhand[1:]) and np.any(rwrist) else np.zeros((FINGER_COUNT, 3))
        feat = np.concatenate([l_to_nose.flatten(), r_to_nose.flatten(), l_finger.flatten(), r_finger.flatten()])
        if len(feat) != TOTAL_FEATURES_PER_FRAME:
            padded = np.zeros(TOTAL_FEATURES_PER_FRAME)
            padded[:len(feat)] = feat
            feat = padded
        out.append(feat.astype(np.float32))
    return np.array(out, dtype=np.float32)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if 'sequence' not in data:
        return jsonify({'error': 'Missing sequence'}), 400
    seq = np.array(data['sequence'])
    if seq.shape != TARGET_SHAPE:
        return jsonify({'error': 'Invalid input shape'}), 400
    feat = seq_to_feat_combo_D(seq)
    feat_scaled = scaler.transform(feat)
    input_tensor = np.expand_dims(feat_scaled, axis=0)
    pred = model.predict(input_tensor, verbose=0)[0]
    idx = int(np.argmax(pred))
    conf = float(np.max(pred))
    return jsonify({'result': labels[idx], 'confidence': conf}), 200

@app.route('/collect', methods=['POST'])
def collect():
    data = request.get_json()
    if 'sequence' not in data or 'label' not in data:
        return jsonify({'error': 'Missing sequence or label'}), 400
    label = data['label']
    seq = np.array(data['sequence'])
    interpolated = interpolate_zeros(seq)
    if not valid(interpolated):
        return jsonify({'error': 'Invalid interpolated sequence'}), 400
    save_sequences(label, [interpolated], ORIG_DATASET_PATH)
    aug_list = generate_augmentations_v2(interpolated)
    save_sequences(label, aug_list, AUG_DATASET_PATH)
    return jsonify({'message': f"Saved original and {len(aug_list)} augmented samples for '{label}'"}), 200

if __name__ == '__main__':
    os.makedirs(ORIG_DATASET_PATH, exist_ok=True)
    os.makedirs(AUG_DATASET_PATH, exist_ok=True)
    app.run(host='0.0.0.0', port=5000, debug=True)
