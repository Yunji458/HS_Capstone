import cv2
import mediapipe as mp
import numpy as np
import os
import time
import random
from scipy.spatial.transform import Rotation as R

# Mediapipe 설정
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(static_image_mode=False, model_complexity=1)

# ✅ 증강 함수
def diverse_augment_sequence(sequence, num_augments=200, angle_range=(-10, 10)):
    augmented = []
    for _ in range(num_augments):
        new_seq = []

        axes = random.sample(['x', 'y', 'z'], k=random.randint(1, 3))
        angles = {
            'x': random.uniform(*angle_range) if 'x' in axes else 0,
            'y': random.uniform(*angle_range) if 'y' in axes else 0,
            'z': random.uniform(*angle_range) if 'z' in axes else 0,
        }
        rot = R.from_euler('xyz', [angles['x'], angles['y'], angles['z']], degrees=True)

        noise = np.random.normal(0, 0.01, (42, 3))
        shear_angle = random.uniform(-5, 5)
        shear_rot = R.from_euler('y', shear_angle, degrees=True) if random.random() < 0.5 else None
        finger_scales = {base: np.random.uniform(0.95, 1.05) for base in [1, 5, 9, 13, 17]}

        for frame in sequence:
            joints = frame.reshape(-1, 3)
            center = joints[0]
            moved = joints - center
            rotated = rot.apply(moved)
            rotated += noise
            for base in finger_scales:
                scale = finger_scales[base]
                rotated[base:base+4] = (rotated[base:base+4] - rotated[base]) * scale + rotated[base]
            if shear_rot is not None:
                rotated = shear_rot.apply(rotated)
            final = rotated + center
            new_seq.append(final.flatten())
        augmented.append(np.array(new_seq))
    return augmented

# ✅ 키포인트 추출
def extract_keypoints(results):
    def get_landmarks(landmarks):
        if landmarks:
            return np.array([[lm.x, lm.y, lm.z] for lm in landmarks.landmark])
        return np.zeros((21, 3))
    lh = get_landmarks(results.left_hand_landmarks)
    rh = get_landmarks(results.right_hand_landmarks)
    return np.concatenate([lh, rh], axis=0).flatten()

# ✅ 유효성 검사
def is_valid_sequence(sequence, threshold=0.6):
    if sequence.shape != (10, 126): return False
    zero_ratio = np.sum(sequence == 0) / sequence.size
    return zero_ratio < threshold

# ✅ group 단위 저장 함수
def save_group_data(action, sequences):
    dataset_path = "dataset"
    action_path = os.path.join(dataset_path, action)
    os.makedirs(action_path, exist_ok=True)

    # group 파일 번호 찾기
    existing_groups = [f for f in os.listdir(action_path) if f.startswith("group_") and f.endswith(".npy")]
    group_nums = [int(f.split("_")[1].split(".")[0]) for f in existing_groups]
    next_group_num = max(group_nums) + 1 if group_nums else 1

    filename = f"group_{next_group_num:03d}.npy"
    filepath = os.path.join(action_path, filename)

    np.save(filepath, np.array(sequences))
    print(f"✅ '{action}' → '{filename}' 저장 완료! (총 {len(sequences)}개)")

# ✅ 웹캠 실행
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ 웹캠 열 수 없음")
    exit()

while True:
    action = input("\n수집할 동작 라벨 입력 ('exit' 입력 시 종료): ").strip()
    if action.lower() == 'exit': break
    print(f"\n'{action}' 수집 준비 중...")

    # 캡처 전 대기 화면
    while True:
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(rgb)

        for hand in [results.left_hand_landmarks, results.right_hand_landmarks]:
            if hand:
                for lm in hand.landmark:
                    cx, cy = int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])
                    cv2.circle(frame, (cx, cy), 5, (255, 255, 255), -1)
        cv2.putText(frame, "ENTER: 수집 시작", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.imshow("Webcam", frame)
        key = cv2.waitKey(1)
        if key == 13: break
        elif key == 27:
            cap.release(); cv2.destroyAllWindows(); exit()

    print("\n⏳ 3초 대기...")
    time.sleep(3)

    sequence = []
    for frame_num in range(10):
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(rgb)
        for hand in [results.left_hand_landmarks, results.right_hand_landmarks]:
            if hand:
                for lm in hand.landmark:
                    cx, cy = int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])
                    cv2.circle(frame, (cx, cy), 5, (255, 255, 255), -1)

        keypoints = extract_keypoints(results)
        sequence.append(keypoints)

        cv2.putText(frame, f"{action} ({frame_num+1}/10)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Webcam", frame)
        if cv2.waitKey(1) == 27:
            cap.release(); cv2.destroyAllWindows(); exit()

    sequence = np.array(sequence)

    if is_valid_sequence(sequence):
        print(f"\n🔄 '{action}' 회전 증강 중...")
        aug_sequences = [sequence] + diverse_augment_sequence(sequence)
        save_group_data(action, aug_sequences)
        print(f"🚀 '{action}' 증강 완료 및 저장!")
    else:
        print("❌ 데이터 유효하지 않음. 저장 안 함.")

cap.release()
cv2.destroyAllWindows()
print("✅ 전체 수집 종료!")