import os
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# 🔹 벡터 기반 3D 특성 추출
def extract_vector_features_3d(joint):
    # joint는 (21, 3) 형태의 단일 손 랜드마크라고 가정
    joint_3d = joint[:, :3]
    wrist_landmark = joint_3d[0] # 손목 랜드마크 미리 정의

    # --- 1. 기존 뼈대 벡터 및 손가락 마디 간 각도 ---
    v1_indices = [0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19]
    v2_indices = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
    v_orig = joint_3d[v2_indices] - joint_3d[v1_indices]
    v_orig_normalized = v_orig / (np.linalg.norm(v_orig, axis=1, keepdims=True) + 1e-6) # (20, 3)

    angle_idx1 = [0,1,2,4,5,6,8,9,10,12,13,14,16,17,18]
    angle_idx2 = [1,2,3,5,6,7,9,10,11,13,14,15,17,18,19]
    angles_intra_finger = [ # 15개 각도
        np.degrees(np.arccos(np.clip(np.dot(v_orig_normalized[i], v_orig_normalized[j]), -1.0, 1.0)))
        for i, j in zip(angle_idx1, angle_idx2)
    ]

    # inter_angles_mcp_wrist (기존 3개 특성)는 제거

    # --- 2. 새로운 특성: 손가락 벌어짐/붙어있음 관련 ---
    
    # 2.1. 인접한 손가락 끝(TIP) 랜드마크 간의 거리 (정규화) - 4개 특성
    tip_indices = [4, 8, 12, 16, 20] # Thumb, Index, Middle, Ring, Pinky TIPs
    finger_tip_distances = []
    
    reference_length = np.linalg.norm(joint_3d[9] - wrist_landmark) + 1e-6 # Wrist to Middle MCP distance
    if reference_length < 1e-5: 
        reference_length = 1.0 

    for i in range(len(tip_indices) - 1):
        tip1 = joint_3d[tip_indices[i]]
        tip2 = joint_3d[tip_indices[i+1]]
        distance = np.linalg.norm(tip1 - tip2)
        finger_tip_distances.append(distance / reference_length) 

    # 2.2. 손목을 기준으로 한 인접 손가락 끝(TIP) 벡터 간의 각도 - 4개 특성
    wrist_tip_angles = []
    vectors_wrist_to_tips = []
    for tip_idx in tip_indices:
        vec = joint_3d[tip_idx] - wrist_landmark
        vectors_wrist_to_tips.append(vec / (np.linalg.norm(vec) + 1e-6)) 

    for i in range(len(vectors_wrist_to_tips) - 1):
        vec1 = vectors_wrist_to_tips[i]
        vec2 = vectors_wrist_to_tips[i+1]
        angle = np.degrees(np.arccos(np.clip(np.dot(vec1, vec2), -1.0, 1.0)))
        wrist_tip_angles.append(angle)

    # 모든 특성 결합
    # 기존 특성: 60 (뼈대벡터) + 15 (마디간각도) = 75개
    # 새로운 특성: 4 (손가락끝거리) + 4 (손목-손가락끝각도) = 8개
    # 총 특성 수: 75 + 8 = 83개
    return np.concatenate([
        v_orig_normalized.flatten(),    # 60개
        np.array(angles_intra_finger),  # 15개
        np.array(finger_tip_distances), # 4개
        np.array(wrist_tip_angles)      # 4개
    ])

# 🔹 데이터 로딩
def load_vector_feature_data_3d(dataset_path="dataset"):
    X, y = [], []
    for label in os.listdir(dataset_path):
        label_path = os.path.join(dataset_path, label)
        if not os.path.isdir(label_path): continue
        for file in os.listdir(label_path):
            if file.endswith(".npy") and file.startswith("group_"):
                data = np.load(os.path.join(label_path, file), allow_pickle=True)
                for sequence in data:
                    features = np.array([
                        extract_vector_features_3d(frame.reshape(42, 3)) for frame in sequence
                    ])
                    X.append(features)
                    y.append(label)
    return np.array(X), np.array(y)

# 🔹 학습 코드
if __name__ == "__main__":
    X, y = load_vector_feature_data_3d()
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    y_onehot = to_categorical(y_encoded)
    joblib.dump(le, "label_encoder.pkl")

    X_train, X_val, y_train, y_val = train_test_split(X, y_onehot, test_size=0.2, random_state=42)
    feature_dim = X.shape[2]
    num_classes = y_onehot.shape[1]

    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=(10, feature_dim)),
        Dropout(0.3),
        LSTM(64),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=20, batch_size=32)
    model.save("fingerspelling_vector3d_lstm.h5")
