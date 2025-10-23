# reaugment_existing_data.py
import os
import numpy as np
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation as R_scipy
import random
from tqdm import tqdm
import copy

# ─────────────────────────── 기본 설정 (데이터 형식 관련) ─────────────────────────── #
NUM_FRAMES = 30
NUM_KEYPOINTS = 1 + 21 + 21 # 코(1) + 왼손(21) + 오른손(21) = 43
NUM_COORDS = 3
TARGET_SHAPE = (NUM_FRAMES, NUM_KEYPOINTS * NUM_COORDS) # (30, 129)
IS_HAND_INACTIVE_THRESHOLD = 1e-5

# --- 키포인트 인덱스 (43개 기준) ---
# Mediapipe import는 직접 사용하지 않으므로 값만 정의
NOSE_IDX = 0
LEFT_HAND_START_IDX = 1
RIGHT_HAND_START_IDX = 1 + 21
HAND_KEYPOINTS_COUNT = 21

# ─────────────────── 새로운 증강 파라미터 (v2) - 이전 코드와 동일 ─────────────────── #
HAND_ROTATION_DEG_RANGE_INDIVIDUAL = (-15, 15)
HAND_TRANSLATE_X_RANGE_INDIVIDUAL = (-0.03, 0.03)
HAND_TRANSLATE_Y_RANGE_INDIVIDUAL = (-0.03, 0.03)
HAND_TRANSLATE_Z_RANGE_INDIVIDUAL = (-0.02, 0.02)
HAND_TRANSLATE_X_RANGE_TOGETHER = (-0.1, 0.1)
HAND_TRANSLATE_Y_RANGE_TOGETHER = (-0.1, 0.1)
HAND_TRANSLATE_Z_RANGE_TOGETHER = (-0.02, 0.02)
KEYPOINT_JITTER_SIGMA_RANGE_V2 = (0.003, 0.005) # 실제로는 aug_add_noise_v2에서 사용
HAND_SCALE_RANGE_V2 = (0.8, 1.2)
FINGER_LENGTH_SCALE_RANGE_V2 = (0.8, 1.2)
OVERALL_SCALE_RANGE = (0.5, 1.2)
VIEW_CONFIGS_RANDOM_V2 = {
    "frontal":    { "yaw_range": (-10, 10), "pitch_range": (-10, 10), "num_augs_total": 500 },
    "left_bias":  { "yaw_range": (35, 55),  "pitch_range": (-10, 10), "num_augs_total": 500 },
    "right_bias": { "yaw_range": (-55, -35),"pitch_range": (-10, 10), "num_augs_total": 500 },
    "up_bias":    { "yaw_range": (-10, 10), "pitch_range": (-55, -35),"num_augs_total": 500 },
    "down_bias":  { "yaw_range": (-10, 10), "pitch_range": (35, 55),  "num_augs_total": 500 },
}

# --- 경로 설정 ---
# 중요: 이 경로들을 실제 환경에 맞게 수정하세요!
EXISTING_ORIGINAL_DATASET_PATH = 'dataset_original_NoseHands_final_v3_newaug' # << 기존 원본 데이터가 있는 경로
NEW_AUGMENTED_DATASET_PATH = 'dataset_augmented_NoseHands_final_v3_newaug' # << 새로 증강된 데이터를 저장할 경로

# ─────────────────── 새로운 증강 함수들 (v2) - 이전 코드와 동일 ─────────────────── #
# (generate_augmentations_v2 및 그 하위 함수들 여기에 복사)

def aug_rotate_hands_independently_v2(kps_3d_in, deg_range, is_left_active, is_right_active):
    kps_3d = kps_3d_in.copy()
    hand_info = [(LEFT_HAND_START_IDX, is_left_active), (RIGHT_HAND_START_IDX, is_right_active)]
    for hand_offset, is_active in hand_info:
        if is_active:
            rand_angles_deg = np.random.uniform(deg_range[0], deg_range[1], 3)
            rotation_matrix = R_scipy.from_euler('xyz', rand_angles_deg, degrees=True).as_matrix()
            wrist_kps_seq = kps_3d[:, hand_offset, :].copy()
            hand_kps_seq = kps_3d[:, hand_offset : hand_offset + HAND_KEYPOINTS_COUNT, :].copy()
            hand_kps_rel_to_wrist_seq = hand_kps_seq - wrist_kps_seq[:, np.newaxis, :]
            rotated_hand_kps_rel_seq = np.einsum('ij,tkj -> tki', rotation_matrix, hand_kps_rel_to_wrist_seq)
            kps_3d[:, hand_offset : hand_offset + HAND_KEYPOINTS_COUNT, :] = rotated_hand_kps_rel_seq + wrist_kps_seq[:, np.newaxis, :]
    return kps_3d

def aug_translate_hands_independently_v2(kps_3d_in, x_range, y_range, z_range, is_left_active, is_right_active):
    kps_3d = kps_3d_in.copy()
    hand_info = [(LEFT_HAND_START_IDX, is_left_active), (RIGHT_HAND_START_IDX, is_right_active)]
    for hand_offset, is_active in hand_info:
        if is_active:
            dx = random.uniform(*x_range)
            dy = random.uniform(*y_range)
            dz = random.uniform(*z_range)
            translation_vector = np.array([dx, dy, dz])
            kps_3d[:, hand_offset : hand_offset + HAND_KEYPOINTS_COUNT, :] += translation_vector[None, None, :]
    return kps_3d

def aug_translate_both_hands_together_v2(kps_3d_in, x_range, y_range, z_range, is_left_active, is_right_active):
    kps_3d = kps_3d_in.copy()
    dx = random.uniform(*x_range)
    dy = random.uniform(*y_range)
    dz = random.uniform(*z_range)
    translation_vector = np.array([dx, dy, dz])
    if is_left_active:
        kps_3d[:, LEFT_HAND_START_IDX : LEFT_HAND_START_IDX + HAND_KEYPOINTS_COUNT, :] += translation_vector[None, None, :]
    if is_right_active:
        kps_3d[:, RIGHT_HAND_START_IDX : RIGHT_HAND_START_IDX + HAND_KEYPOINTS_COUNT, :] += translation_vector[None, None, :]
    return kps_3d

def aug_add_noise_v2(kps_3d_in, sigma_range, is_left_active, is_right_active):
    kps_3d = kps_3d_in.copy()
    sigma = random.uniform(*sigma_range)
    noise = np.random.normal(0, sigma, kps_3d.shape)
    kps_3d[:, NOSE_IDX, :] += noise[:, NOSE_IDX, :]
    if is_left_active:
        kps_3d[:, LEFT_HAND_START_IDX : LEFT_HAND_START_IDX + HAND_KEYPOINTS_COUNT, :] += \
            noise[:, LEFT_HAND_START_IDX : LEFT_HAND_START_IDX + HAND_KEYPOINTS_COUNT, :]
    if is_right_active:
        kps_3d[:, RIGHT_HAND_START_IDX : RIGHT_HAND_START_IDX + HAND_KEYPOINTS_COUNT, :] += \
            noise[:, RIGHT_HAND_START_IDX : RIGHT_HAND_START_IDX + HAND_KEYPOINTS_COUNT, :]
    return kps_3d

def aug_scale_hands_v2(kps_3d_in, scale_range, is_left_active, is_right_active):
    kps_3d = kps_3d_in.copy()
    scale_factor = random.uniform(*scale_range)
    hand_info = [(LEFT_HAND_START_IDX, is_left_active), (RIGHT_HAND_START_IDX, is_right_active)]
    for hand_offset, is_active in hand_info:
        if is_active:
            wrist_kps_seq = kps_3d[:, hand_offset, :]
            for i in range(1, HAND_KEYPOINTS_COUNT):
                kp_idx = hand_offset + i
                vec_from_wrist_seq = kps_3d[:, kp_idx, :] - wrist_kps_seq
                kps_3d[:, kp_idx, :] = wrist_kps_seq + vec_from_wrist_seq * scale_factor
    return kps_3d

def aug_scale_finger_lengths_v2(kps_3d_in, scale_range, is_left_active, is_right_active):
    kps_3d = kps_3d_in.copy()
    scale_factor = random.uniform(*scale_range)
    hand_info = [(LEFT_HAND_START_IDX, is_left_active), (RIGHT_HAND_START_IDX, is_right_active)]
    for hand_offset, is_active in hand_info:
        if is_active:
            wrist_kps_seq = kps_3d[:, hand_offset, :]
            for i in range(1, HAND_KEYPOINTS_COUNT):
                finger_kp_idx = hand_offset + i
                vec_from_wrist_seq = kps_3d[:, finger_kp_idx, :] - wrist_kps_seq
                kps_3d[:, finger_kp_idx, :] = wrist_kps_seq + vec_from_wrist_seq * scale_factor
    return kps_3d

def aug_scale_overall_v2(kps_3d_in, scale_range, center_kp_idx=NOSE_IDX):
    kps_3d = kps_3d_in.copy()
    scale_factor = random.uniform(*scale_range)
    center_of_scaling = kps_3d[0, center_kp_idx, :].copy()
    kps_centered_seq = kps_3d - center_of_scaling[None, None, :]
    kps_scaled_seq = kps_centered_seq * scale_factor
    kps_3d_scaled = kps_scaled_seq + center_of_scaling[None, None, :]
    return kps_3d_scaled

def aug_change_viewpoint_v2(kps_3d_in, yaw_deg, pitch_deg, center_kp_idx=NOSE_IDX):
    kps_3d = kps_3d_in.copy()
    center_of_rotation = kps_3d[0, center_kp_idx, :].copy()
    rotation_matrix = R_scipy.from_euler('yx', [yaw_deg, pitch_deg], degrees=True).as_matrix()
    kps_centered_seq = kps_3d - center_of_rotation[None, None, :]
    kps_rotated_seq = np.einsum('ij,tkj -> tki', rotation_matrix, kps_centered_seq)
    kps_3d_view_changed = kps_rotated_seq + center_of_rotation[None, None, :]
    return kps_3d_view_changed

def generate_augmentations_v2(original_sequence_flat, view_configs):
    original_kps_3d_abs = original_sequence_flat.reshape(NUM_FRAMES, NUM_KEYPOINTS, NUM_COORDS)
    augmented_sequences_list = []

    original_l_hand_data = original_kps_3d_abs[:, LEFT_HAND_START_IDX : LEFT_HAND_START_IDX + HAND_KEYPOINTS_COUNT, :]
    original_r_hand_data = original_kps_3d_abs[:, RIGHT_HAND_START_IDX : RIGHT_HAND_START_IDX + HAND_KEYPOINTS_COUNT, :]
    is_left_active_orig = np.mean(np.abs(original_l_hand_data)) > IS_HAND_INACTIVE_THRESHOLD
    is_right_active_orig = np.mean(np.abs(original_r_hand_data)) > IS_HAND_INACTIVE_THRESHOLD
    # print(f"  (Re-Aug) 활성 손: 왼손={is_left_active_orig}, 오른손={is_right_active_orig}") # 로그 줄이기 위해 주석처리 가능

    total_augs_generated = 0
    for view_name, config in view_configs.items():
        num_augs_for_this_view = config.get("num_augs_total", 10)
        if num_augs_for_this_view == 0: continue
        # print(f"    '{view_name}' 시점 증강 생성 중 ({num_augs_for_this_view}개)...") # 로그 줄이기 위해 주석처리 가능

        for _ in range(num_augs_for_this_view): # tqdm 제거 또는 유지
            current_kps_3d = copy.deepcopy(original_kps_3d_abs)

            current_kps_3d = aug_rotate_hands_independently_v2(current_kps_3d, HAND_ROTATION_DEG_RANGE_INDIVIDUAL, is_left_active_orig, is_right_active_orig)
            current_kps_3d = aug_translate_hands_independently_v2(current_kps_3d, 
                                                                HAND_TRANSLATE_X_RANGE_INDIVIDUAL, 
                                                                HAND_TRANSLATE_Y_RANGE_INDIVIDUAL, 
                                                                HAND_TRANSLATE_Z_RANGE_INDIVIDUAL, 
                                                                is_left_active_orig, is_right_active_orig)
            current_kps_3d = aug_translate_both_hands_together_v2(current_kps_3d, 
                                                                HAND_TRANSLATE_X_RANGE_TOGETHER, 
                                                                HAND_TRANSLATE_Y_RANGE_TOGETHER, 
                                                                HAND_TRANSLATE_Z_RANGE_TOGETHER, 
                                                                is_left_active_orig, is_right_active_orig)
            current_kps_3d = aug_add_noise_v2(current_kps_3d, KEYPOINT_JITTER_SIGMA_RANGE_V2, is_left_active_orig, is_right_active_orig)
            current_kps_3d = aug_scale_hands_v2(current_kps_3d, HAND_SCALE_RANGE_V2, is_left_active_orig, is_right_active_orig)
            current_kps_3d = aug_scale_finger_lengths_v2(current_kps_3d, FINGER_LENGTH_SCALE_RANGE_V2, is_left_active_orig, is_right_active_orig)
            current_kps_3d = aug_scale_overall_v2(current_kps_3d, OVERALL_SCALE_RANGE, center_kp_idx=NOSE_IDX)

            yaw_deg_vp = random.uniform(*config["yaw_range"])
            pitch_deg_vp = random.uniform(*config["pitch_range"])
            current_kps_3d = aug_change_viewpoint_v2(current_kps_3d, yaw_deg_vp, pitch_deg_vp, center_kp_idx=NOSE_IDX)

            if not is_left_active_orig:
                current_kps_3d[:, LEFT_HAND_START_IDX : LEFT_HAND_START_IDX + HAND_KEYPOINTS_COUNT, :] = 0.0
            if not is_right_active_orig:
                current_kps_3d[:, RIGHT_HAND_START_IDX : RIGHT_HAND_START_IDX + HAND_KEYPOINTS_COUNT, :] = 0.0
            
            augmented_sequences_list.append(current_kps_3d.reshape(TARGET_SHAPE))
            total_augs_generated +=1
            
    # print(f"  총 {total_augs_generated}개 증강 생성 완료.") # 로그 줄이기 위해 주석처리 가능
    return augmented_sequences_list


# ─────────────────────────── 유틸리티 (interpolate_zeros, valid, save_sequences) ─────────────────────────── #
# interpolate_zeros는 원본 데이터가 이미 보간되었다고 가정하고 여기서는 사용하지 않음
# (만약 원본 .npy가 보간 전 데이터라면 interpolate_zeros 함수도 필요)

def valid(seq_flat, threshold=0.6): # 기존과 동일
    if seq_flat.shape != TARGET_SHAPE: print(f"❌ Invalid shape: {seq_flat.shape}"); return False
    # 증강 과정에서 0이 많아질 수 있으므로, 이 valid 조건은 재검토 필요할 수 있음
    # if (np.abs(seq_flat) < 1e-9).mean() >= threshold: print(f"❌ Too many zeros"); return False
    return True

def save_sequences_for_reaugmentation(label, sequences_list, base_path):
    path = os.path.join(base_path, label)
    os.makedirs(path, exist_ok=True)
    
    # 기존 파일들을 덮어쓰지 않고 새로운 파일명으로 저장 (또는 기존 파일 삭제 후 새로 생성)
    # 여기서는 단순히 group_000.npy 부터 순차적으로 저장 (기존 증강 파일이 있다면 덮어쓸 수 있음)
    # 더 안전하게 하려면, 기존 파일 개수를 세어 다음 번호로 저장하거나, 파일명에 reaug_ 등을 추가
    
    # 가장 간단한 방법: 항상 새로운 group 파일로 저장 (기존 파일과 이름이 겹치지 않도록)
    existing_files = [f for f in os.listdir(path) if f.startswith('group_') and f.endswith('.npy')]
    next_gid = 0
    if existing_files:
        gids = [int(f.split('_')[1].split('.')[0]) for f in existing_files]
        next_gid = max(gids) + 1 if gids else 0
        
    file_path = os.path.join(path, f'group_{next_gid:03d}.npy')

    try:
        np.save(file_path, np.stack([np.asarray(s) for s in sequences_list]))
        print(f"✅ Saved {len(sequences_list)} re-augmented sequences to {file_path}")
    except Exception as e:
        print(f"❌ Save error for {file_path}: {e}")

# ─────────────────────────── 메인 재증강 로직 ─────────────────────────── #
def reaugment_dataset():
    if not os.path.exists(EXISTING_ORIGINAL_DATASET_PATH):
        print(f"❌ 원본 데이터셋 경로를 찾을 수 없습니다: {EXISTING_ORIGINAL_DATASET_PATH}")
        return

    if not os.path.exists(NEW_AUGMENTED_DATASET_PATH):
        os.makedirs(NEW_AUGMENTED_DATASET_PATH)
        print(f"새로운 증강 데이터 저장 폴더 생성: {NEW_AUGMENTED_DATASET_PATH}")

    labels = [d for d in os.listdir(EXISTING_ORIGINAL_DATASET_PATH) if os.path.isdir(os.path.join(EXISTING_ORIGINAL_DATASET_PATH, d))]

    if not labels:
        print(f"❌ 원본 데이터셋 경로에 라벨 폴더가 없습니다: {EXISTING_ORIGINAL_DATASET_PATH}")
        return

    print(f"총 {len(labels)}개의 라벨에 대해 재증강을 시작합니다.")

    for label in tqdm(labels, desc="Processing Labels"):
        label_path_original = os.path.join(EXISTING_ORIGINAL_DATASET_PATH, label)
        original_files = [f for f in os.listdir(label_path_original) if f.endswith('.npy') and f.startswith('group_')]

        if not original_files:
            print(f"  라벨 '{label}'에 원본 .npy 파일이 없습니다. 건너<0xEB><0x9B><0x84>니다.")
            continue
        
        print(f"\n라벨 '{label}' 처리 중 ({len(original_files)}개 원본 파일)...")

        all_augmented_for_label = [] # 이 라벨에 대한 모든 증강 결과를 모음 (선택사항)

        for original_file_name in tqdm(original_files, desc=f"  Files in '{label}'", leave=False):
            original_file_path = os.path.join(label_path_original, original_file_name)
            try:
                # 원본 .npy 파일은 여러 시퀀스를 포함할 수 있음 (group_XXX.npy)
                original_sequences_stack = np.load(original_file_path)
                if original_sequences_stack.ndim == 2: # 단일 시퀀스 파일인 경우
                    original_sequences_stack = original_sequences_stack[np.newaxis, :, :]
                
                # 각 원본 시퀀스에 대해 증강 수행
                for i in range(original_sequences_stack.shape[0]):
                    original_sequence_flat = original_sequences_stack[i] # (NUM_FRAMES, NUM_KEYPOINTS * NUM_COORDS)

                    # 원본 데이터가 이미 보간되었다고 가정. 필요시 여기서 interpolate_zeros 호출
                    # if not valid(original_sequence_flat): # 원본 자체의 유효성 검사 (선택)
                    #     print(f"    Skipping invalid original sequence in {original_file_name}, index {i}")
                    #     continue

                    augmented_sequences = generate_augmentations_v2(
                        original_sequence_flat, VIEW_CONFIGS_RANDOM_V2
                    )
                    
                    if augmented_sequences:
                        # 방법 1: 각 원본 파일에 대한 증강 결과를 별도의 파일로 저장
                        # (원본 파일 하나당 증강 파일 하나 생성. 원본 파일이 group_000.npy 이고 내부에 1개 시퀀스만 있었다면)
                        # 이 경우 save_sequences_for_reaugmentation 함수는 한 번만 호출됨.
                        # save_sequences_for_reaugmentation(label, augmented_sequences, NEW_AUGMENTED_DATASET_PATH)
                        
                        # 방법 2: 한 라벨에 대한 모든 증강 결과를 모아서 한 번에 저장하거나,
                        #         원본 group 파일 구조를 유지하며 각 group에 대한 증강을 저장.
                        #         여기서는 각 원본 시퀀스에서 파생된 증강들을 모아서 저장.
                        all_augmented_for_label.extend(augmented_sequences)

            except Exception as e:
                print(f"❌ 파일 {original_file_path} 처리 중 오류 발생: {e}")
                import traceback
                traceback.print_exc()
        
        if all_augmented_for_label:
             # 한 라벨에 대한 모든 증강을 하나의 (또는 여러 개의) group 파일로 저장
             # 예: 500개씩 묶어서 저장
            chunk_size = 500 # 한 파일에 저장할 최대 증강 샘플 수
            for i in range(0, len(all_augmented_for_label), chunk_size):
                chunk = all_augmented_for_label[i:i+chunk_size]
                save_sequences_for_reaugmentation(label, chunk, NEW_AUGMENTED_DATASET_PATH)
        else:
            print(f"  라벨 '{label}'에 대해 생성된 증강 데이터가 없습니다.")


    print("\n🎉 모든 라벨에 대한 재증강 작업 완료.")

if __name__ == '__main__':
    reaugment_dataset()