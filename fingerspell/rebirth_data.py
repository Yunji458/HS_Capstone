# re_augment_frontal_origin_only_random_views.py
import cv2
# import mediapipe as mp
import numpy as np
import os
# import time
import random
from scipy.spatial.transform import Rotation as R_scipy
from tqdm import tqdm

# --- diverse_augment_sequence, load_original_sequence, apply_global_view_rotation, save_group_data 함수는 이전과 동일 ---
# (이전 답변의 함수 정의를 그대로 사용한다고 가정합니다. 필요하면 여기에 다시 붙여넣을 수 있습니다.)
# --- diverse_augment_sequence 함수 (이전과 동일, 파라미터 추가) ---
def diverse_augment_sequence(sequence_3d_kps, num_augments=200, angle_range=(-10, 10), noise_std=0.005, finger_scale_range=(0.95, 1.05), shear_range=(-3,3)):
    augmented_sequences_flat = []
    num_frames, num_keypoints, _ = sequence_3d_kps.shape

    for i in range(num_augments):
        current_axes = random.sample(['x', 'y', 'z'], k=random.randint(1, 3))
        current_angles = {
            'x': random.uniform(*angle_range) if 'x' in current_axes else 0,
            'y': random.uniform(*angle_range) if 'y' in current_axes else 0,
            'z': random.uniform(*angle_range) if 'z' in current_axes else 0,
        }
        frame_specific_rot = R_scipy.from_euler('xyz', [current_angles['x'], current_angles['y'], current_angles['z']], degrees=True)
        consistent_noise_pattern = np.random.normal(0, noise_std, (num_keypoints, 3))
        apply_shear = random.random() < 0.5
        shear_rot_matrix = None
        if apply_shear:
            shear_angle_val = random.uniform(*shear_range)
            shear_rot_matrix = R_scipy.from_euler('y', shear_angle_val, degrees=True) # Example shear on y-axis
        
        current_finger_scales = {base: np.random.uniform(*finger_scale_range) for base in [1, 5, 9, 13, 17]} # For left hand

        new_sequence_frames_flat = []
        for frame_idx in range(num_frames):
            current_frame_kps = sequence_3d_kps[frame_idx].copy()
            center_point = current_frame_kps[0].copy() # Assuming first keypoint is the left wrist
            kps_centered = current_frame_kps - center_point
            
            kps_rotated_small = frame_specific_rot.apply(kps_centered)
            kps_noisy = kps_rotated_small + consistent_noise_pattern
            
            # Finger scaling (assuming first 21 keypoints are left hand)
            for base_idx_in_hand, scale_factor in current_finger_scales.items():
                if base_idx_in_hand < (num_keypoints / 2) and (base_idx_in_hand + 3) < (num_keypoints / 2) : 
                    finger_root_kp = kps_noisy[base_idx_in_hand].copy()
                    for j in range(4): 
                        kp_to_scale_idx = base_idx_in_hand + j
                        if kp_to_scale_idx < (num_keypoints / 2) : 
                            vec_from_finger_root = kps_noisy[kp_to_scale_idx] - finger_root_kp
                            kps_noisy[kp_to_scale_idx] = finger_root_kp + vec_from_finger_root * scale_factor
            
            kps_sheared = kps_noisy
            if shear_rot_matrix is not None:
                kps_sheared = shear_rot_matrix.apply(kps_noisy)
            
            final_frame_kps = kps_sheared + center_point
            new_sequence_frames_flat.append(final_frame_kps.flatten())
        augmented_sequences_flat.append(np.array(new_sequence_frames_flat))
    return augmented_sequences_flat

# --- load_original_sequence 함수 (이전과 동일) ---
def load_original_sequence(npy_file_path):
    try:
        data_group = np.load(npy_file_path)
        if data_group.ndim == 3 and data_group.shape[0] >= 1:
            original_sequence_flat = data_group[0] 
            num_frames = original_sequence_flat.shape[0]
            expected_features = 126 
            if original_sequence_flat.shape[1] != expected_features:
                # print(f"⚠️ 경고: {npy_file_path} 파일의 feature 수가 {expected_features}이 아닙니다 (현재: {original_sequence_flat.shape[1]}).")
                if original_sequence_flat.shape[1] % 3 != 0:
                    # print(f"❌ 에러: {npy_file_path} 파일의 feature 수가 3으로 나누어 떨어지지 않아 3D 키포인트로 변환 불가.")
                    return None
                num_keypoints = original_sequence_flat.shape[1] // 3
                # print(f"    feature 수 기반으로 {num_keypoints}개 키포인트로 처리 시도.")
            else:
                num_keypoints = expected_features // 3
            original_sequence_3d = original_sequence_flat.reshape(num_frames, num_keypoints, 3)
            return original_sequence_3d
        else:
            # print(f"❌ 에러: {npy_file_path} 파일 형식이 올바르지 않거나 시퀀스가 없습니다. Shape: {data_group.shape if isinstance(data_group, np.ndarray) else 'Unknown'}")
            return None
    except Exception as e:
        # print(f"❌ 에러: {npy_file_path} 파일 로드 중 오류 발생 - {e}")
        return None

# --- apply_global_view_rotation 함수 (이전과 동일) ---
def apply_global_view_rotation(sequence_3d_kps, yaw_deg=0, pitch_deg=0, roll_deg=0):
    rotated_sequence_frames = []
    global_rotation = R_scipy.from_euler('yxz', [yaw_deg, pitch_deg, roll_deg], degrees=True)
    for frame_kps in sequence_3d_kps:
        center_point = frame_kps[0].copy() 
        kps_centered = frame_kps - center_point
        kps_rotated = global_rotation.apply(kps_centered)
        final_frame_kps = kps_rotated + center_point
        rotated_sequence_frames.append(final_frame_kps)
    return np.array(rotated_sequence_frames)

# --- save_group_data 함수 (이전과 동일) ---
def save_group_data(action, sequences_flat_list, base_dataset_path="dataset_reaugmented"):
    action_path = os.path.join(base_dataset_path, action)
    os.makedirs(action_path, exist_ok=True)
    existing_groups = [f for f in os.listdir(action_path) if f.startswith("group_") and f.endswith(".npy")]
    group_nums = []
    for f in existing_groups:
        try:
            group_nums.append(int(f.split("_")[1].split(".")[0]))
        except ValueError:
            continue
    next_group_num = max(group_nums) + 1 if group_nums else 1
    filename = f"group_{next_group_num:03d}.npy" 
    filepath = os.path.join(action_path, filename)
    try:
        np_sequences = [np.asarray(s) for s in sequences_flat_list]
        np.save(filepath, np.stack(np_sequences))
    except Exception as e:
        print(f"❌ 저장 중 오류 발생: {filepath} - {e}")


if __name__ == '__main__':
    original_dataset_root = "dataset"
    reaugmented_dataset_root = "dataset_reaugmented_from_frontal_origin_random_views" # 저장 폴더명 변경

    if not os.path.exists(original_dataset_root):
        print(f"❌ 원본 데이터셋 폴더를 찾을 수 없습니다: {original_dataset_root}")
        exit()

    os.makedirs(reaugmented_dataset_root, exist_ok=True)
    print(f"🚀 증강된 데이터를 '{reaugmented_dataset_root}' 폴더에 저장합니다.")

    action_folders = [d for d in os.listdir(original_dataset_root) if os.path.isdir(os.path.join(original_dataset_root, d))]
    if not action_folders:
        print(f"🤷 '{original_dataset_root}' 폴더에 액션 폴더가 없습니다.")
        exit()

    print(f"🔎 다음 액션 폴더들을 처리합니다: {action_folders}")

    # === 증강 설정 수정: yaw와 pitch에 범위를 지정 ===
    # 각 시점별로 num_augs_per_base_view 개의 "기본 시점 변환된" 샘플을 만들고,
    # 각 샘플에 대해 diverse_augment_sequence를 한 번씩만 적용 (num_augments=1)
    # 또는, 각 시점별로 num_augs 만큼 반복하면서 매번 새로운 랜덤 yaw/pitch를 뽑고, 그 결과에 diverse_augment_sequence를 num_augments=1로 적용
    # 여기서는 후자의 방식을 따르겠습니다. (더 많은 시점 다양성)

    # 각 "시점 타입"별 생성할 총 증강 샘플 수
    TOTAL_AUGS_PER_VIEW_TYPE = 200 # 예: frontal 타입에서 200개, left 타입에서 200개 등

    view_configs_random = {
        # "시점 타입": { "yaw_range": (min, max), "pitch_range": (min, max), "num_augs_total": 총 생성 수, ... diverse_augment_sequence 파라미터들 ...}
        "frontal":    {
            "yaw_range": (-10, 10),   # 정면이지만 약간의 yaw 변화 허용
            "pitch_range": (-10, 10), # 정면이지만 약간의 pitch 변화 허용
            "num_augs_total": TOTAL_AUGS_PER_VIEW_TYPE, 
            "diverse_angle_range": (-5,5), "noise_std": 0.01, "finger_scale_range":(0.95,1.05), "shear_range":(-3,3)
        },
        "left_bias":  { # "왼쪽"이라는 큰 방향성은 유지하되, 그 안에서 랜덤성 부여
            "yaw_range": (35, 55),    # 45도 기준 +-10도
            "pitch_range": (-10, 10), # 왼쪽 보면서 약간의 상하 변화 허용
            "num_augs_total": TOTAL_AUGS_PER_VIEW_TYPE,
            "diverse_angle_range": (-5,5), "noise_std": 0.01, "finger_scale_range":(0.96,1.04), "shear_range":(-2,2)
        },
        "right_bias": {
            "yaw_range": (-55, -35),  # -45도 기준 +-10도
            "pitch_range": (-10, 10),
            "num_augs_total": TOTAL_AUGS_PER_VIEW_TYPE,
            "diverse_angle_range": (-5,5), "noise_std": 0.01, "finger_scale_range":(0.96,1.04), "shear_range":(-2,2)
        },
        "up_bias":    {
            "yaw_range": (-10, 10),
            "pitch_range": (-55, -35), # -45도 기준 +-10도 (위를 보면 pitch가 음수)
            "num_augs_total": TOTAL_AUGS_PER_VIEW_TYPE,
            "diverse_angle_range": (-5,5), "noise_std": 0.01, "finger_scale_range":(0.96,1.04), "shear_range":(-2,2)
        },
        "down_bias":  {
            "yaw_range": (-10, 10),
            "pitch_range": (35, 55),   # 45도 기준 +-10도 (아래를 보면 pitch가 양수)
            "num_augs_total": TOTAL_AUGS_PER_VIEW_TYPE,
            "diverse_angle_range": (-5,5), "noise_std": 0.01, "finger_scale_range":(0.96,1.04), "shear_range":(-2,2)
        },
    }

    for action_name in tqdm(action_folders, desc="전체 액션 진행"):
        original_action_path = os.path.join(original_dataset_root, action_name)
        frontal_origin_file_name = "group_001.npy"
        original_npy_file_path = os.path.join(original_action_path, frontal_origin_file_name)

        if not os.path.exists(original_npy_file_path):
            tqdm.write(f"🤷 액션 '{action_name}'에 '{frontal_origin_file_name}' 파일(정면 원본)이 없습니다. 건너뜁니다.")
            continue
        
        tqdm.write(f"액션 '{action_name}'의 정면 원본 '{frontal_origin_file_name}' 처리 중...")
        
        frontal_original_sequence_3d_kps = load_original_sequence(original_npy_file_path)
        if frontal_original_sequence_3d_kps is None:
            tqdm.write(f"  '{frontal_origin_file_name}' 로드 실패. 건너뜁니다.")
            continue

        all_newly_augmented_sequences_flat = []

        # === 수정된 증강 루프 ===
        for view_type_name, config in view_configs_random.items():
            tqdm.write(f"  {view_type_name} 시점 타입 증강 중 ({config['num_augs_total']}개 목표)...")
            for _ in range(config['num_augs_total']): # 해당 시점 타입에서 만들 총 증강 샘플 수만큼 반복
                # 1. 랜덤한 전역 시점 값 생성
                rand_yaw = random.uniform(*config['yaw_range'])
                rand_pitch = random.uniform(*config['pitch_range'])
                # rand_roll = 0 # 필요하다면 roll도 추가 가능

                # 2. 전역 시점 변환 적용 (항상 정면 원본 기준)
                view_transformed_sequence_3d = apply_global_view_rotation(
                    frontal_original_sequence_3d_kps, # 항상 정면 원본(group_001)을 기준으로 회전
                    yaw_deg=rand_yaw,
                    pitch_deg=rand_pitch
                    # roll_deg=rand_roll
                )
            
                # 3. 세부 다양성 증강 (여기서는 num_augments=1로 설정하여, 위에서 변환된 시퀀스 하나에 대해 한 번만 미세 증강)
                #    만약 각 랜덤 시점마다 또 여러 개의 미세 증강을 하고 싶다면 이 값을 늘리면 됨.
                #    현재는 총 1000개의 "서로 다른 전역 시점 + 미세 증강"을 만드는 것이 목표.
                augmented_view_and_detail_flat = diverse_augment_sequence(
                    view_transformed_sequence_3d,
                    num_augments=1, # 각 랜덤 시점마다 하나의 미세 증강된 샘플 생성
                    angle_range=config['diverse_angle_range'],
                    noise_std=config['noise_std'],
                    finger_scale_range=config['finger_scale_range'],
                    shear_range=config['shear_range']
                )
                # diverse_augment_sequence는 리스트를 반환하므로, 첫 번째 (그리고 유일한) 요소를 가져옴
                if augmented_view_and_detail_flat: # 비어있지 않은 경우에만 추가
                    all_newly_augmented_sequences_flat.append(augmented_view_and_detail_flat[0])
        
        if all_newly_augmented_sequences_flat:
            save_group_data(action_name, all_newly_augmented_sequences_flat, base_dataset_path=reaugmented_dataset_root)
            tqdm.write(f"  Saved {len(all_newly_augmented_sequences_flat)} augmented sequences for {action_name} (from its frontal origin: {frontal_origin_file_name}) into a new group file in '{reaugmented_dataset_root}'. Total samples: {len(all_newly_augmented_sequences_flat)}")
        else:
            tqdm.write(f"  🤷 {action_name}의 {frontal_origin_file_name}에서 증강된 데이터가 없습니다.")

    print("\n🎉 모든 액션의 정면 원본(group_001.npy)에 대한 랜덤 시점 증강 및 저장 작업 완료!")