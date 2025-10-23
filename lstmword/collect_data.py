import cv2
import os
import time
import random
import numpy as np
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation as R_scipy
import mediapipe as mp
from tqdm import tqdm
import copy

# ─────────────────────────── 설정값 (기본) ─────────────────────────── #
NUM_FRAMES = 30
NUM_KEYPOINTS = 1 + 21 + 21 # 코(1) + 왼손(21) + 오른손(21) = 43
NUM_COORDS = 3
TARGET_SHAPE = (NUM_FRAMES, NUM_KEYPOINTS * NUM_COORDS) # (30, 129)
IS_HAND_INACTIVE_THRESHOLD = 1e-5 # 손 비활성 판단 임계값

# ─────────────────── 새로운 증강 파라미터 (v2) ─────────────────── #
# 1. 각 양손 3D 회전
HAND_ROTATION_DEG_RANGE_INDIVIDUAL = (-15, 15) # 각 축별 회전 범위 (도)

# 2. 각 양손 독립적 이동
HAND_TRANSLATE_X_RANGE_INDIVIDUAL = (-0.03, 0.03)
HAND_TRANSLATE_Y_RANGE_INDIVIDUAL = (-0.03, 0.03)
HAND_TRANSLATE_Z_RANGE_INDIVIDUAL = (-0.02, 0.02) # z는 0.005 부터 -0.01 (min, max 순서로)

# 3. 양손 함께 이동 (코 기준 상대 위치 유지하며)
HAND_TRANSLATE_X_RANGE_TOGETHER = (-0.1, 0.1)
HAND_TRANSLATE_Y_RANGE_TOGETHER = (-0.1, 0.1)
HAND_TRANSLATE_Z_RANGE_TOGETHER = (-0.02, 0.02) # z는 0.01 부터 -0.02 (min, max 순서로)

# 4. 노이즈
KEYPOINT_JITTER_SIGMA_RANGE_V2 = (0.003, 0.005) # 기존 값 재활용 또는 조정

# 5. 손 크기
HAND_SCALE_RANGE_V2 = (0.8, 1.2) # 기존 값 재활용 또는 조정

# 6. 손가락 길이
FINGER_LENGTH_SCALE_RANGE_V2 = (0.8, 1.2) # 기존 값 재활용 또는 조정

# 7. 전체 스케일
OVERALL_SCALE_RANGE = (0.5, 1.2)

# 8. 시점 변화 설정
VIEW_CONFIGS_RANDOM_V2 = {
    "frontal":    { "yaw_range": (-10, 10), "pitch_range": (-10, 10), "num_augs_total": 500 }, # 예시: 정면 증강 수
    "left_bias":  { "yaw_range": (35, 55),  "pitch_range": (-10, 10), "num_augs_total": 500 }, # 예시: 각 방향 증강 수
    "right_bias": { "yaw_range": (-55, -35),"pitch_range": (-10, 10), "num_augs_total": 500 },
    "up_bias":    { "yaw_range": (-10, 10), "pitch_range": (-55, -35),"num_augs_total": 500 },
    "down_bias":  { "yaw_range": (-10, 10), "pitch_range": (35, 55),  "num_augs_total": 500 },
}
# 총 증강 목표 수 (예시, 필요시 VIEW_CONFIGS_RANDOM_V2의 num_augs_total 합으로 계산)
# NUM_TOTAL_AUG_TARGET_V2 = 600 # 이 값은 위 config의 합으로 결정됨

# --- 저장 경로 ---
ORIGINAL_DATASET_PATH = 'dataset_original_NoseHands_final_v3_newaug' # 경로 변경
AUGMENTED_DATASET_PATH = 'dataset_augmented_NoseHands_final_v3_newaug' # 경로 변경

# --- Mediapipe 설정 ---
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
holistic = mp_holistic.Holistic(static_image_mode=False, model_complexity=1,
                               min_detection_confidence=0.5, min_tracking_confidence=0.5)

# --- 키포인트 인덱스 (43개 기준) ---
POSE_ID_NOSE = mp_holistic.PoseLandmark.NOSE.value
NOSE_IDX = 0
LEFT_HAND_START_IDX = 1
RIGHT_HAND_START_IDX = 1 + 21
HAND_KEYPOINTS_COUNT = 21

WINDOW_NAME = "수어 데이터 수집 (코+양손, 신규증강 v3)"

# ───────────────────── 키포인트 추출 및 보간 (기존과 동일) ─────────────────── #
def extract_keypoints(res):
    def get_pose_landmark_nose():
        if res.pose_landmarks and res.pose_landmarks.landmark:
            try: lm = res.pose_landmarks.landmark[POSE_ID_NOSE]; return np.array([lm.x, lm.y, lm.z])
            except IndexError: return np.zeros(NUM_COORDS)
        return np.zeros(NUM_COORDS)
    def get_hand_landmarks(hand_landmarks_mp):
        if hand_landmarks_mp and hand_landmarks_mp.landmark:
            return np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks_mp.landmark])
        return np.zeros((HAND_KEYPOINTS_COUNT, NUM_COORDS))
    nose = get_pose_landmark_nose()
    l_hand = get_hand_landmarks(res.left_hand_landmarks)
    r_hand = get_hand_landmarks(res.right_hand_landmarks)
    return np.concatenate([[nose], l_hand, r_hand]).flatten()

def interpolate_zeros(seq_flat):
    seq_interpolated = seq_flat.copy()
    for dim_idx in range(seq_interpolated.shape[1]):
        col = seq_interpolated[:, dim_idx].copy()
        nz_indices = np.nonzero(col)[0]
        if len(nz_indices) == 0: continue
        elif len(nz_indices) == 1:
            seq_interpolated[:, dim_idx] = col[nz_indices[0]]
            continue
        first_nz_idx, last_nz_idx = nz_indices[0], nz_indices[-1]
        interp_func = interp1d(
            nz_indices, col[nz_indices],
            kind='linear', bounds_error=False,
            fill_value=(col[first_nz_idx], col[last_nz_idx])
        )
        all_indices = np.arange(len(col))
        seq_interpolated[:, dim_idx] = interp_func(all_indices)
    return seq_interpolated

# ─────────────────── 새로운 증강 함수들 (v2) ─────────────────── #

def aug_rotate_hands_independently_v2(kps_3d_in, deg_range, is_left_active, is_right_active):
    kps_3d = kps_3d_in.copy()
    hand_info = [(LEFT_HAND_START_IDX, is_left_active), (RIGHT_HAND_START_IDX, is_right_active)]
    for hand_offset, is_active in hand_info:
        if is_active:
            # 각 손에 대해 독립적인 랜덤 회전 각도 생성
            rand_angles_deg = np.random.uniform(deg_range[0], deg_range[1], 3) # x, y, z 축 회전
            rotation_matrix = R_scipy.from_euler('xyz', rand_angles_deg, degrees=True).as_matrix()
            
            wrist_kps_seq = kps_3d[:, hand_offset, :].copy() # (T, 3)
            hand_kps_seq = kps_3d[:, hand_offset : hand_offset + HAND_KEYPOINTS_COUNT, :].copy() # (T, 21, 3)
            
            # 손목을 중심으로 회전
            hand_kps_rel_to_wrist_seq = hand_kps_seq - wrist_kps_seq[:, np.newaxis, :] # (T, 21, 3)
            rotated_hand_kps_rel_seq = np.einsum('ij,tkj -> tki', rotation_matrix, hand_kps_rel_to_wrist_seq) # (T, 21, 3)
            
            kps_3d[:, hand_offset : hand_offset + HAND_KEYPOINTS_COUNT, :] = rotated_hand_kps_rel_seq + wrist_kps_seq[:, np.newaxis, :]
    return kps_3d

def aug_translate_hands_independently_v2(kps_3d_in, x_range, y_range, z_range, is_left_active, is_right_active):
    kps_3d = kps_3d_in.copy()
    hand_info = [(LEFT_HAND_START_IDX, is_left_active), (RIGHT_HAND_START_IDX, is_right_active)]
    for hand_offset, is_active in hand_info:
        if is_active:
            # 각 손에 대해 독립적인 랜덤 이동량 생성
            dx = random.uniform(*x_range)
            dy = random.uniform(*y_range)
            dz = random.uniform(*z_range)
            translation_vector = np.array([dx, dy, dz])
            
            kps_3d[:, hand_offset : hand_offset + HAND_KEYPOINTS_COUNT, :] += translation_vector[None, None, :]
    return kps_3d

def aug_translate_both_hands_together_v2(kps_3d_in, x_range, y_range, z_range, is_left_active, is_right_active):
    kps_3d = kps_3d_in.copy()
    # 양손에 동일하게 적용될 하나의 랜덤 이동량 생성
    dx = random.uniform(*x_range)
    dy = random.uniform(*y_range)
    dz = random.uniform(*z_range)
    translation_vector = np.array([dx, dy, dz])
    
    if is_left_active:
        kps_3d[:, LEFT_HAND_START_IDX : LEFT_HAND_START_IDX + HAND_KEYPOINTS_COUNT, :] += translation_vector[None, None, :]
    if is_right_active:
        kps_3d[:, RIGHT_HAND_START_IDX : RIGHT_HAND_START_IDX + HAND_KEYPOINTS_COUNT, :] += translation_vector[None, None, :]
    # 코는 이동하지 않음
    return kps_3d

def aug_add_noise_v2(kps_3d_in, sigma_range, is_left_active, is_right_active):
    kps_3d = kps_3d_in.copy()
    sigma = random.uniform(*sigma_range)
    noise = np.random.normal(0, sigma, kps_3d.shape)
    
    # 코에는 항상 노이즈 적용
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
    scale_factor = random.uniform(*scale_range) # 양손에 동일한 스케일 팩터 적용 (일관성)
    
    hand_info = [(LEFT_HAND_START_IDX, is_left_active), (RIGHT_HAND_START_IDX, is_right_active)]
    for hand_offset, is_active in hand_info:
        if is_active:
            wrist_kps_seq = kps_3d[:, hand_offset, :] # (T, 3)
            for i in range(1, HAND_KEYPOINTS_COUNT): # 손목 제외한 나머지 손가락 관절
                kp_idx = hand_offset + i
                vec_from_wrist_seq = kps_3d[:, kp_idx, :] - wrist_kps_seq # (T, 3)
                kps_3d[:, kp_idx, :] = wrist_kps_seq + vec_from_wrist_seq * scale_factor
    return kps_3d

def aug_scale_finger_lengths_v2(kps_3d_in, scale_range, is_left_active, is_right_active):
    kps_3d = kps_3d_in.copy()
    # 손가락 마디별로 다른 scale factor를 적용할 수도 있으나, 여기서는 손 전체에 하나의 factor 적용
    # 좀 더 복잡하게 하려면, 각 손가락 그룹(예: 엄지, 검지 등)별로 다른 scale_factor를 적용할 수 있음
    # 여기서는 간단하게 손목에서 각 손가락 끝점까지의 벡터를 스케일링
    scale_factor = random.uniform(*scale_range) # 양손에 동일한 스케일 팩터 적용

    hand_info = [(LEFT_HAND_START_IDX, is_left_active), (RIGHT_HAND_START_IDX, is_right_active)]
    for hand_offset, is_active in hand_info:
        if is_active:
            wrist_kps_seq = kps_3d[:, hand_offset, :] # (T, 3)
            # 손가락 끝점들 (MCP 제외, 단순화 위해 모든 점에 적용)
            for i in range(1, HAND_KEYPOINTS_COUNT): # 손목(0) 제외
                finger_kp_idx = hand_offset + i
                vec_from_wrist_seq = kps_3d[:, finger_kp_idx, :] - wrist_kps_seq
                kps_3d[:, finger_kp_idx, :] = wrist_kps_seq + vec_from_wrist_seq * scale_factor
    return kps_3d

def aug_scale_overall_v2(kps_3d_in, scale_range, center_kp_idx=NOSE_IDX):
    kps_3d = kps_3d_in.copy()
    scale_factor = random.uniform(*scale_range)
    
    # 기준점 (예: 첫 프레임의 코 위치)
    # 시퀀스 전체에 걸쳐 일관된 기준점을 사용하거나, 각 프레임별 코를 기준으로 할 수 있음.
    # 여기서는 첫 프레임 코를 기준으로 함 (더 안정적)
    center_of_scaling = kps_3d[0, center_kp_idx, :].copy() # (3,)
    
    # 모든 키포인트를 scaling_center 기준으로 변환
    kps_centered_seq = kps_3d - center_of_scaling[None, None, :] # (T, N_kp, 3)
    
    # 스케일링 적용
    kps_scaled_seq = kps_centered_seq * scale_factor
    
    # 다시 원래 위치로 이동
    kps_3d_scaled = kps_scaled_seq + center_of_scaling[None, None, :]
    return kps_3d_scaled

def aug_change_viewpoint_v2(kps_3d_in, yaw_deg, pitch_deg, center_kp_idx=NOSE_IDX):
    kps_3d = kps_3d_in.copy()
    
    # 회전 중심 (예: 첫 프레임의 코 위치)
    center_of_rotation = kps_3d[0, center_kp_idx, :].copy() # (3,)
    
    # 회전 행렬 생성 (Yaw 먼저, 그 다음 Pitch)
    rotation_matrix = R_scipy.from_euler('yx', [yaw_deg, pitch_deg], degrees=True).as_matrix()
    
    # 모든 키포인트를 회전 중심으로 이동
    kps_centered_seq = kps_3d - center_of_rotation[None, None, :] # (T, N_kp, 3)
    
    # 회전 적용: (T, N_kp, 3) x (3,3) -> (T, N_kp, 3)
    # np.einsum('tkj,ji->tki', kps_centered_seq, rotation_matrix.T) 와 동일
    # 또는 np.einsum('ij,tkj->tki', rotation_matrix, kps_centered_seq)
    kps_rotated_seq = np.einsum('ij,tkj->tki', rotation_matrix, kps_centered_seq)

    # 다시 원래 위치(회전 후 기준점)로 이동
    kps_3d_view_changed = kps_rotated_seq + center_of_rotation[None, None, :]
    return kps_3d_view_changed

# ───────────────────── 메인 증강 파이프라인 (v2) ────────────────────── #
def generate_augmentations_v2(original_sequence_flat, view_configs):
    original_kps_3d_abs = original_sequence_flat.reshape(NUM_FRAMES, NUM_KEYPOINTS, NUM_COORDS)
    augmented_sequences_list = []

    # --- 원본 데이터 기준 "활성 손" 판단 ---
    original_l_hand_data = original_kps_3d_abs[:, LEFT_HAND_START_IDX : LEFT_HAND_START_IDX + HAND_KEYPOINTS_COUNT, :]
    original_r_hand_data = original_kps_3d_abs[:, RIGHT_HAND_START_IDX : RIGHT_HAND_START_IDX + HAND_KEYPOINTS_COUNT, :]
    is_left_active_orig = np.mean(np.abs(original_l_hand_data)) > IS_HAND_INACTIVE_THRESHOLD
    is_right_active_orig = np.mean(np.abs(original_r_hand_data)) > IS_HAND_INACTIVE_THRESHOLD
    print(f"  원본 활성 손: 왼손={is_left_active_orig}, 오른손={is_right_active_orig}")

    total_augs_generated = 0
    for view_name, config in view_configs.items():
        num_augs_for_this_view = config.get("num_augs_total", 10) # 각 view type별 생성 수
        if num_augs_for_this_view == 0: continue
        print(f"  '{view_name}' 시점 증강 생성 중 ({num_augs_for_this_view}개)...")

        for _ in tqdm(range(num_augs_for_this_view), desc=f"{view_name} 증강", unit="샘플", leave=False):
            current_kps_3d = copy.deepcopy(original_kps_3d_abs)

            # 순차적 증강 적용
            # 1. 각 양손 3D 회전
            current_kps_3d = aug_rotate_hands_independently_v2(current_kps_3d, HAND_ROTATION_DEG_RANGE_INDIVIDUAL, is_left_active_orig, is_right_active_orig)
            
            # 2. 각 양손 독립적 이동
            current_kps_3d = aug_translate_hands_independently_v2(current_kps_3d, 
                                                                HAND_TRANSLATE_X_RANGE_INDIVIDUAL, 
                                                                HAND_TRANSLATE_Y_RANGE_INDIVIDUAL, 
                                                                HAND_TRANSLATE_Z_RANGE_INDIVIDUAL, 
                                                                is_left_active_orig, is_right_active_orig)
            
            # 3. 양손 함께 이동
            current_kps_3d = aug_translate_both_hands_together_v2(current_kps_3d, 
                                                                HAND_TRANSLATE_X_RANGE_TOGETHER, 
                                                                HAND_TRANSLATE_Y_RANGE_TOGETHER, 
                                                                HAND_TRANSLATE_Z_RANGE_TOGETHER, 
                                                                is_left_active_orig, is_right_active_orig)
            
            # 4. 노이즈
            current_kps_3d = aug_add_noise_v2(current_kps_3d, KEYPOINT_JITTER_SIGMA_RANGE_V2, is_left_active_orig, is_right_active_orig)
            
            # 5. 손 크기
            current_kps_3d = aug_scale_hands_v2(current_kps_3d, HAND_SCALE_RANGE_V2, is_left_active_orig, is_right_active_orig)

            # 6. 손가락 길이
            current_kps_3d = aug_scale_finger_lengths_v2(current_kps_3d, FINGER_LENGTH_SCALE_RANGE_V2, is_left_active_orig, is_right_active_orig)

            # 7. 전체 스케일
            current_kps_3d = aug_scale_overall_v2(current_kps_3d, OVERALL_SCALE_RANGE, center_kp_idx=NOSE_IDX)

            # 8. 시점 변화 (view_configs에서 yaw, pitch 범위를 가져와 랜덤 적용)
            yaw_deg_vp = random.uniform(*config["yaw_range"])
            pitch_deg_vp = random.uniform(*config["pitch_range"])
            current_kps_3d = aug_change_viewpoint_v2(current_kps_3d, yaw_deg_vp, pitch_deg_vp, center_kp_idx=NOSE_IDX)

            # --- 유령손 최종 처리: 원본에서 비활성이었던 손 데이터 0으로 리셋 ---
            if not is_left_active_orig:
                current_kps_3d[:, LEFT_HAND_START_IDX : LEFT_HAND_START_IDX + HAND_KEYPOINTS_COUNT, :] = 0.0
            if not is_right_active_orig:
                current_kps_3d[:, RIGHT_HAND_START_IDX : RIGHT_HAND_START_IDX + HAND_KEYPOINTS_COUNT, :] = 0.0
            
            augmented_sequences_list.append(current_kps_3d.reshape(TARGET_SHAPE))
            total_augs_generated +=1
            
    print(f"모든 증강 생성 완료 (총 {total_augs_generated}개).")
    return augmented_sequences_list

# ─────────────────────────── 유틸리티 (기존과 동일) ─────────────────────────── #
def valid(seq_flat, threshold=0.6):
    if seq_flat.shape != TARGET_SHAPE: print(f"❌ Invalid shape: {seq_flat.shape}"); return False
    if (np.abs(seq_flat) < 1e-9).mean() >= threshold: print(f"❌ Too many zeros"); return False
    return True

def save_sequences(label, sequences_list, base_path):
    path = os.path.join(base_path, label); os.makedirs(path, exist_ok=True)
    existing = [f for f in os.listdir(path) if f.startswith('group_') and f.endswith('.npy')]
    gid = max([int(f.split('_')[1][:-4]) for f in existing]) + 1 if existing else 0
    file_path = os.path.join(path, f'group_{gid:03d}.npy')
    try: np.save(file_path, np.stack([np.asarray(s) for s in sequences_list])); print(f"✅ Saved {len(sequences_list)} to {file_path}")
    except Exception as e: print(f"❌ Save error {file_path}: {e}")

def draw_landmarks_on_frame(frame, results):
    if results.pose_landmarks and results.pose_landmarks.landmark[POSE_ID_NOSE].visibility > 0.1:
        mp_drawing.draw_landmarks(image=frame, landmark_list=results.pose_landmarks,
            connections=[(mp_holistic.PoseLandmark.NOSE, mp_holistic.PoseLandmark.NOSE)],
            landmark_drawing_spec=mp_drawing.DrawingSpec(color=(80,110,10), thickness=2, circle_radius=2))
    if results.left_hand_landmarks: mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    if results.right_hand_landmarks: mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

def put_text_on_frame(frame, text, y_offset=0, color=(0,255,0)):
    cv2.putText(frame, text, (10, 30 + y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

# ───────────────────────── 캡처 및 처리 메인 로직 (수정됨) ──────────────── #
def capture_and_process():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened(): raise RuntimeError("웹캠 Open 실패")
    cv2.namedWindow(WINDOW_NAME)

    while True:
        label = input("\n라벨 입력 (or 'exit'): ").strip()
        if not label: print("라벨 미입력"); continue
        if label.lower() == 'exit': break

        print(f"\n'{label}' 라벨. 웹캠 확인. Enter: 녹화 | ESC: 취소")
        action_cancelled, start_countdown = False, False
        while not start_countdown:
            ok, frame = cap.read()
            if not ok: print("❌ 웹캠 프레임 Read 실패"); action_cancelled=True; break
            frame = cv2.flip(frame,1); results = holistic.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            draw_landmarks_on_frame(frame, results)
            put_text_on_frame(frame, f"Label: {label}",0,(255,255,0))
            put_text_on_frame(frame, "ENTER: Start", 30); put_text_on_frame(frame, "ESC: Cancel", 60, (0,0,255))
            cv2.imshow(WINDOW_NAME, frame)
            key = cv2.waitKey(1)&0xFF
            if key==13: start_countdown=True
            elif key==27: action_cancelled=True; break
        if action_cancelled: continue

        print("캡처 준비...");
        for i in range(3,0,-1):
            ok,frame=cap.read();
            if not ok: print("❌ 웹캠 프레임 Read 실패(카운트다운)"); action_cancelled=True; break
            frame = cv2.flip(frame,1); results = holistic.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            draw_landmarks_on_frame(frame, results)
            put_text_on_frame(frame, f"Starting in {i}...",0,(0,165,255))
            cv2.imshow(WINDOW_NAME, frame)
            if cv2.waitKey(1000)&0xFF==27: action_cancelled=True; break
        if action_cancelled: print(f"'{label}' 캡처 취소"); continue

        print("캡처 시작!");
        captured_frames_flat_data, interrupted = [], False
        target_wait_ms = max(1, int(1000 / (NUM_FRAMES / 2.5))) # ~2.5초 동안 NUM_FRAMES 캡처

        for f_num in range(NUM_FRAMES):
            ok,frame=cap.read()
            if not ok: print("❌ 웹캠 프레임 Read 실패(캡처)"); interrupted=True; break
            results = holistic.process(cv2.cvtColor(cv2.flip(frame.copy(),1), cv2.COLOR_BGR2RGB))
            captured_frames_flat_data.append(extract_keypoints(results))
            display_frame = cv2.flip(frame.copy(),1); draw_landmarks_on_frame(display_frame, results)
            put_text_on_frame(display_frame,f"REC: {label} [{f_num+1}/{NUM_FRAMES}]",0,(0,0,255))
            cv2.imshow(WINDOW_NAME, display_frame)
            if cv2.waitKey(target_wait_ms)&0xFF==27: interrupted=True; print("캡처 중단(ESC)"); break
        print(f"캡처 완료 ({len(captured_frames_flat_data)} 프레임).")

        if interrupted or len(captured_frames_flat_data) < NUM_FRAMES: print("캡처 미완료"); continue

        print("처리 및 증강 시작...");
        original_sequence_flat = np.array(captured_frames_flat_data)
        interpolated_sequence_flat = interpolate_zeros(original_sequence_flat)
        if not valid(interpolated_sequence_flat, threshold=0.7): print("❌ 보간 후 유효성 실패"); continue
        print("원본 유효성 통과.")

        save_sequences(label, [interpolated_sequence_flat.copy()], ORIGINAL_DATASET_PATH)

        try:
            # 새로운 증강 함수 호출
            augmented_sequences = generate_augmentations_v2(
                interpolated_sequence_flat, VIEW_CONFIGS_RANDOM_V2
            )
            if augmented_sequences: save_sequences(label, augmented_sequences, AUGMENTED_DATASET_PATH)
            else: print("증강 데이터 없음.")
        except Exception as e: print(f"❌ 증강 오류: {e}"); import traceback; traceback.print_exc(); continue

    cap.release(); cv2.destroyAllWindows(); holistic.close(); print("\n🎉 프로그램 종료")

if __name__ == '__main__':
    for path in [ORIGINAL_DATASET_PATH, AUGMENTED_DATASET_PATH]:
        if not os.path.exists(path): os.makedirs(path); print(f"폴더 생성: {path}")
    capture_and_process()