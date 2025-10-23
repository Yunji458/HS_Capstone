import os
import numpy as np
from flask import Flask, request, jsonify
import tensorflow as tf
import joblib
from sklearn.preprocessing import StandardScaler
from scipy.interpolate import interp1d
import random
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# ── 고정 설정 ──
NUM_FRAMES = 30
NUM_KEYPOINTS = 43
NUM_COORDS = 3
TARGET_SHAPE = (NUM_FRAMES, NUM_KEYPOINTS * NUM_COORDS)
HAND_SIZE = 21
FINGER_COUNT = 20
TOTAL_FEATURES_PER_FRAME = (HAND_SIZE * 3) * 2 + (FINGER_COUNT * 3) * 2  # = 246

ORIG_DATASET_PATH = 'dataset_original_NoseHands_final_v3_newaug'
AUG_DATASET_PATH = 'dataset_augmented_NoseHands_final_v3_newaug'

LABEL_ENCODER_PATH = f'slr_label_encoder_comboD_float32.pkl'
SCALER_PATH = f'slr_scaler_comboD_{TOTAL_FEATURES_PER_FRAME}_float32.pkl'
MODEL_PATH = f'slr_model_comboD_best_{TOTAL_FEATURES_PER_FRAME}_float32.h5'

# ── 로드 ──
label_encoder = joblib.load(LABEL_ENCODER_PATH)
scaler = joblib.load(SCALER_PATH)
model = tf.keras.models.load_model(MODEL_PATH)
labels = label_encoder.classes_

# ── 조합 D 피처 추출 ──
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

        feat = np.concatenate([l_to_nose.flatten(), r_to_nose.flatten(),
                               l_finger.flatten(), r_finger.flatten()])
        if len(feat) != TOTAL_FEATURES_PER_FRAME:
            padded = np.zeros(TOTAL_FEATURES_PER_FRAME)
            padded[:len(feat)] = feat
            feat = padded
        out.append(feat.astype(np.float32))
    return np.array(out, dtype=np.float32)

# ── 유틸 ──
def is_valid(seq):
    return seq.shape == TARGET_SHAPE and np.mean(np.abs(seq)) > 1e-5

def save_sequence(label, sequence):
    path = os.path.join(ORIG_DATASET_PATH, label)
    os.makedirs(path, exist_ok=True)
    file = os.path.join(path, 'group_000.npy')
    if os.path.exists(file):
        data = np.load(file, allow_pickle=True)
        data = list(data) if isinstance(data, np.ndarray) else []
    else:
        data = []
    data.append(sequence)
    np.save(file, np.array(data, dtype=object))

def simple_augment(seq, num=10):
    out = []
    for _ in range(num):
        aug = seq.copy()
        aug += np.random.normal(0, 0.005, size=aug.shape)
        out.append(aug)
    return out

def save_augmented(label, sequences):
    path = os.path.join(AUG_DATASET_PATH, label)
    os.makedirs(path, exist_ok=True)
    file = os.path.join(path, 'group_000.npy')
    if os.path.exists(file):
        data = np.load(file, allow_pickle=True)
        data = list(data) if isinstance(data, np.ndarray) else []
    else:
        data = []
    data.extend(sequences)
    np.save(file, np.array(data, dtype=object))

# ── 예측 엔드포인트 ──
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if 'sequence' not in data:
        return jsonify({'error': 'Missing sequence'}), 400

    seq = np.array(data['sequence'])
    if not is_valid(seq):
        return jsonify({'error': 'Invalid input shape'}), 400

    feat = seq_to_feat_combo_D(seq)
    feat_scaled = scaler.transform(feat)
    input_tensor = np.expand_dims(feat_scaled, axis=0)
    pred = model.predict(input_tensor, verbose=0)[0]
    idx = int(np.argmax(pred))
    conf = float(np.max(pred))
    return jsonify({'result': labels[idx], 'confidence': conf}), 200

# ── 수집 + 증강 ──
@app.route('/collect', methods=['POST'])
def collect():
    data = request.get_json()
    if 'sequence' not in data or 'label' not in data:
        return jsonify({'error': 'Missing sequence or label'}), 400

    label = data['label']
    seq = np.array(data['sequence'])
    if not is_valid(seq):
        return jsonify({'error': 'Invalid input shape'}), 400

    save_sequence(label, seq)
    aug_list = simple_augment(seq, num=10)
    save_augmented(label, aug_list)
    return jsonify({'message': f"Saved for '{label}'"}), 200

# ── 실행 ──
if __name__ == '__main__':
    os.makedirs(ORIG_DATASET_PATH, exist_ok=True)
    os.makedirs(AUG_DATASET_PATH, exist_ok=True)
    app.run(host='0.0.0.0', port=5000, debug=True)
