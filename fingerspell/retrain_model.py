# retrain_model.py (dataset_reaugmented_from_frontal_origin 용, 왼손 특성만 사용)

import os
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import tensorflow as tf
from tqdm import tqdm

# 🔹 벡터 기반 3D 특성 추출 (단일 손, 83 features) - 변경 없음
def extract_vector_features_3d_single_hand(joint_21_3):
    if joint_21_3 is None or joint_21_3.shape != (21, 3) or np.all(np.abs(joint_21_3) < 1e-9):
        return np.zeros(83)

    joint_3d = joint_21_3[:, :3]
    wrist_landmark = joint_3d[0]

    v1_indices = [0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19]
    v2_indices = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
    v_orig = joint_3d[v2_indices] - joint_3d[v1_indices]
    
    norm_v_orig = np.linalg.norm(v_orig, axis=1, keepdims=True)
    v_orig_normalized = np.divide(v_orig, norm_v_orig, out=np.zeros_like(v_orig), where=norm_v_orig!=0)

    angle_idx1 = [0,1,2,4,5,6,8,9,10,12,13,14,16,17,18]
    angle_idx2 = [1,2,3,5,6,7,9,10,11,13,14,15,17,18,19]
    angles_intra_finger = []
    for i, j in zip(angle_idx1, angle_idx2):
        vec_i = v_orig_normalized[i]
        vec_j = v_orig_normalized[j]
        if np.allclose(vec_i, 0) or np.allclose(vec_j, 0):
            angles_intra_finger.append(0.0)
            continue
        dot_product = np.clip(np.dot(vec_i, vec_j), -1.0, 1.0)
        angle = np.degrees(np.arccos(dot_product))
        angles_intra_finger.append(angle)
    if len(angles_intra_finger) != 15: angles_intra_finger = np.zeros(15).tolist()

    tip_indices = [4, 8, 12, 16, 20]
    finger_tip_distances = []
    
    reference_length_vec = joint_3d[9] - wrist_landmark 
    reference_length = np.linalg.norm(reference_length_vec)
    if reference_length < 1e-6:
        reference_length = 1.0 

    for i in range(len(tip_indices) - 1):
        tip1 = joint_3d[tip_indices[i]]
        tip2 = joint_3d[tip_indices[i+1]]
        distance = np.linalg.norm(tip1 - tip2)
        finger_tip_distances.append(distance / reference_length)
    if len(finger_tip_distances) != 4: finger_tip_distances = np.zeros(4).tolist()

    wrist_tip_angles = []
    vectors_wrist_to_tips = []
    for tip_idx in tip_indices:
        vec = joint_3d[tip_idx] - wrist_landmark
        norm_vec = np.linalg.norm(vec)
        if norm_vec < 1e-6:
             vectors_wrist_to_tips.append(np.zeros(3))
        else:
            vectors_wrist_to_tips.append(vec / norm_vec)

    for i in range(len(vectors_wrist_to_tips) - 1):
        vec1 = vectors_wrist_to_tips[i]
        vec2 = vectors_wrist_to_tips[i+1]
        if np.allclose(vec1, 0) or np.allclose(vec2, 0):
            wrist_tip_angles.append(0.0)
            continue
        dot_product = np.clip(np.dot(vec1, vec2), -1.0, 1.0)
        angle = np.degrees(np.arccos(dot_product))
        wrist_tip_angles.append(angle)
    if len(wrist_tip_angles) != 4: wrist_tip_angles = np.zeros(4).tolist()

    features = np.concatenate([
        v_orig_normalized.flatten(),
        np.array(angles_intra_finger),
        np.array(finger_tip_distances),
        np.array(wrist_tip_angles)
    ])
    if features.shape[0] != 83:
        correct_features = np.zeros(83)
        copy_len = min(features.shape[0], 83)
        correct_features[:copy_len] = features[:copy_len]
        return correct_features
    return features

# 🔹 데이터 로딩 (왼손 특성만 처리, dataset_reaugmented_from_frontal_origin 경로 사용)
def load_features_from_frontal_origin_reaugmented_data(dataset_path="dataset_reaugmented_from_frontal_origin_random_views"):
    X_features, y_labels = [], []
    action_folders = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
    
    if not action_folders:
        print(f"'{dataset_path}'에 액션 폴더가 없습니다.")
        return np.array([]), np.array([])

    print(f"'{dataset_path}'에서 데이터 로딩 중 (왼손 특성만 사용)...")
    for label in tqdm(action_folders, desc="액션 폴더 로딩"):
        label_path = os.path.join(dataset_path, label)
        # 각 액션 폴더에는 group_001.npy (또는 다음 번호) 파일 하나만 있을 것으로 예상됨 (rebirth_data 스크립트가 그렇게 저장함)
        group_files = [f for f in os.listdir(label_path) if f.endswith(".npy") and f.startswith("group_")]
        
        for file_name in group_files: # 실제로는 파일이 하나일 것
            file_path = os.path.join(label_path, file_name)
            try:
                # data shape: (1000, num_frames, 126 features_flat)
                sequences_in_group_flat = np.load(file_path, allow_pickle=True)
                
                for seq_idx in range(sequences_in_group_flat.shape[0]):
                    sequence_flat = sequences_in_group_flat[seq_idx] # (num_frames, 126)
                    num_frames = sequence_flat.shape[0]
                    
                    if sequence_flat.shape[1] != 126:
                        print(f"Warning: 파일 {file_name}의 시퀀스 {seq_idx} feature 수가 126이 아닙니다 ({sequence_flat.shape[1]}). 건너뜁니다.")
                        continue
                    
                    sequence_3d_both_hands = sequence_flat.reshape(num_frames, 42, 3)
                    
                    current_sequence_features = []
                    for frame_idx in range(num_frames):
                        frame_both_hands_3d = sequence_3d_both_hands[frame_idx] # (42, 3)
                        
                        left_hand_kps_3d = frame_both_hands_3d[:21, :] # (21, 3) 왼손 데이터
                        
                        # 왼손에 대해서만 특성 추출
                        features_left = extract_vector_features_3d_single_hand(left_hand_kps_3d)   # 83 features
                        
                        current_sequence_features.append(features_left) # 왼손 특성만 추가 (83 features)
                    
                    X_features.append(np.array(current_sequence_features)) # (num_frames, 83)
                    y_labels.append(label)
            except Exception as e:
                print(f"파일 {file_path} 처리 중 오류 발생: {e}")
                
    if not X_features:
        print("로딩된 데이터가 없습니다.")
        return np.array([]), np.array([])

    return np.array(X_features), np.array(y_labels)

# 🔹 학습 코드
if __name__ == "__main__":
    SEED = 42
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    
    # 데이터 로드 (경로 명시)
    X, y = load_features_from_frontal_origin_reaugmented_data(dataset_path="dataset_reaugmented_from_frontal_origin_random_views")

    if X.shape[0] == 0:
        print("데이터 로딩 실패 또는 데이터 없음. 학습을 종료합니다.")
        exit()
        
    print(f"총 로드된 시퀀스 수: {X.shape[0]}")
    print(f"시퀀스 당 프레임 수: {X.shape[1]}") 
    print(f"프레임 당 특성 수: {X.shape[2]}")   # 83 이어야 함

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    y_onehot = to_categorical(y_encoded)
    
    # 저장될 모델 및 라벨 인코더 경로 설정
    output_model_dir = "trained_model_frontal_origin_left_hand" 
    os.makedirs(output_model_dir, exist_ok=True)
    label_encoder_path = os.path.join(output_model_dir, "label_encoder_frontal_origin_left_hand.pkl")
    joblib.dump(le, label_encoder_path)
    print(f"라벨 인코더 저장 완료: {label_encoder_path}")

    X_train, X_val, y_train, y_val = train_test_split(
        X, y_onehot, test_size=0.2, random_state=SEED, stratify=y_encoded
    )

    num_frames_per_sequence = X_train.shape[1] 
    feature_dim = X_train.shape[2] # 이제 83이 됨
    num_classes = y_onehot.shape[1]

    print(f"학습 데이터 형태: X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"검증 데이터 형태: X_val: {X_val.shape}, y_val: {y_val.shape}")
    print(f"모델 입력 feature_dim: {feature_dim}")

    # 모델 정의 (이전과 동일)
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=(num_frames_per_sequence, feature_dim)),
        BatchNormalization(), 
        Dropout(0.4),         
        LSTM(128, return_sequences=True), 
        BatchNormalization(),
        Dropout(0.4),
        LSTM(64),             
        BatchNormalization(),
        Dropout(0.4),
        Dense(64, activation='relu'),
        Dropout(0.3),         
        Dense(num_classes, activation='softmax')
    ])

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005) 
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    # 콜백 설정 (모델 파일명 변경)
    model_checkpoint_path = os.path.join(output_model_dir, "model_frontal_origin_left_hand_best.h5") 
    
    callbacks = [
        ModelCheckpoint(model_checkpoint_path, save_best_only=True, monitor='val_accuracy', mode='max', verbose=1),
        EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True, verbose=1), 
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=7, min_lr=1e-6, verbose=1) 
    ]

    print("\n모델 학습 시작 (정면 원본 기반 왼손 특성만 사용)...")
    history = model.fit(
        X_train, y_train, 
        validation_data=(X_val, y_val), 
        epochs=150, 
        batch_size=32, 
        callbacks=callbacks,
        shuffle=True
    )

    print(f"\n✅ 최적 모델 저장 완료: {model_checkpoint_path}")

    final_model_path = os.path.join(output_model_dir, "model_frontal_origin_left_hand_final.h5") 
    model.save(final_model_path)
    print(f"✅ 최종 모델 저장 완료: {final_model_path}")

    best_model = tf.keras.models.load_model(model_checkpoint_path)
    loss, accuracy = best_model.evaluate(X_val, y_val, verbose=0)
    print(f"\n최적 모델 검증 데이터 성능 (정면 원본 기반 왼손 특성): Loss = {loss:.4f}, Accuracy = {accuracy:.4f}")