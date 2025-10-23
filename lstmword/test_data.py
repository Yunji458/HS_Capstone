import numpy as np
import matplotlib.pyplot as plt

# --- 설정값 (수집 코드와 동일하게) ---
NUM_FRAMES = 30
NUM_KEYPOINTS = 43 # 코(1) + 왼손(21) + 오른손(21)
NUM_COORDS = 3

# --- 키포인트 인덱스 (수집 코드와 동일하게) ---
NOSE_IDX = 0
LEFT_HAND_START_IDX = 1  # 코 다음이 왼손 시작
# 왼손 손목은 왼손의 첫 번째 키포인트 (인덱스 0)이므로, 전체 43개 중에서는 LEFT_HAND_START_IDX
LEFT_WRIST_IDX_IN_43 = LEFT_HAND_START_IDX

RIGHT_HAND_START_IDX = 1 + 21 # 왼손 21개 다음이 오른손 시작
# 오른손 손목은 오른손의 첫 번째 키포인트 (인덱스 0)이므로, 전체 43개 중에서는 RIGHT_HAND_START_IDX
RIGHT_WRIST_IDX_IN_43 = RIGHT_HAND_START_IDX

# 1. .npy 파일 로드
# 예시: 원본 데이터 파일 경로
# file_path = 'dataset_original_NoseHands_final_v2/hello/group_000.npy'
# 예시: 증강 데이터 파일 경로
file_path = 'dataset_original_NoseHands_final_v3_newaug/얼굴이름/group_000.npy' # 실제 파일 경로로 변경

try:
    all_sequences_flat = np.load(file_path)
    print(f"파일 로드 성공: {file_path}, 형태: {all_sequences_flat.shape}")
except FileNotFoundError:
    print(f"오류: 파일을 찾을 수 없습니다 - {file_path}")
    exit()
except Exception as e:
    print(f"오류: 파일 로드 중 문제 발생 - {e}")
    exit()

# 2. 데이터 형태 변경 (필요시): (N, T, 129) -> (N, T, 43, 3)
# all_sequences_flat의 현재 형태는 (N, T, 129) 입니다.
num_sequences_in_file = all_sequences_flat.shape[0]
all_sequences_3d = all_sequences_flat.reshape(num_sequences_in_file, NUM_FRAMES, NUM_KEYPOINTS, NUM_COORDS)
print(f"3D 형태로 변경 후 형태: {all_sequences_3d.shape}")


# 3. 특정 시퀀스 선택 (예: 첫 번째 시퀀스)
sequence_index_to_view = 0
if sequence_index_to_view >= num_sequences_in_file:
    print(f"오류: 요청한 시퀀스 인덱스({sequence_index_to_view})가 파일 내 시퀀스 개수({num_sequences_in_file})를 벗어납니다.")
    exit()

selected_sequence_3d = all_sequences_3d[sequence_index_to_view] # (T, 43, 3)
print(f"\n선택된 시퀀스 #{sequence_index_to_view} 데이터 (형태: {selected_sequence_3d.shape})")

# 4. 손목 데이터 추출
left_wrist_data_xyz = selected_sequence_3d[:, LEFT_WRIST_IDX_IN_43, :]  # (T, 3)
right_wrist_data_xyz = selected_sequence_3d[:, RIGHT_WRIST_IDX_IN_43, :] # (T, 3)

print("\n왼손 손목 데이터 (x, y, z):")
for frame_num in range(NUM_FRAMES):
    print(f"  프레임 {frame_num:2d}: {left_wrist_data_xyz[frame_num]}")

print("\n오른손 손목 데이터 (x, y, z):")
for frame_num in range(NUM_FRAMES):
    print(f"  프레임 {frame_num:2d}: {right_wrist_data_xyz[frame_num]}")

# 5. (선택적) 손목 데이터 시각화 (예: 시간에 따른 x, y, z 좌표 변화)
frames = np.arange(NUM_FRAMES)

plt.figure(figsize=(15, 10))

# 왼손 손목
plt.subplot(2, 1, 1)
plt.plot(frames, left_wrist_data_xyz[:, 0], label='Left Wrist X', marker='.')
plt.plot(frames, left_wrist_data_xyz[:, 1], label='Left Wrist Y', marker='.')
plt.plot(frames, left_wrist_data_xyz[:, 2], label='Left Wrist Z', marker='.')
plt.title(f'Left Wrist Coordinates Over Time (Sequence #{sequence_index_to_view})')
plt.xlabel('Frame Number')
plt.ylabel('Coordinate Value (Normalized)')
plt.legend()
plt.grid(True)

# 오른손 손목
plt.subplot(2, 1, 2)
plt.plot(frames, right_wrist_data_xyz[:, 0], label='Right Wrist X', marker='x')
plt.plot(frames, right_wrist_data_xyz[:, 1], label='Right Wrist Y', marker='x')
plt.plot(frames, right_wrist_data_xyz[:, 2], label='Right Wrist Z', marker='x')
plt.title(f'Right Wrist Coordinates Over Time (Sequence #{sequence_index_to_view})')
plt.xlabel('Frame Number')
plt.ylabel('Coordinate Value (Normalized)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# 6. (선택적) 3D 공간에서의 손목 궤적 시각화
fig_3d = plt.figure(figsize=(10, 8))
ax_3d = fig_3d.add_subplot(111, projection='3d')

# MediaPipe 좌표계는 일반적으로 화면을 바라볼 때
# X: 오른쪽으로 갈수록 증가
# Y: 아래로 갈수록 증가 (주의: Matplotlib 3D는 위로 갈수록 증가가 기본)
# Z: 카메라에서 멀어질수록 증가 (화면 안쪽으로) -> 깊이. 값이 작을수록 카메라에 가까움.

# 왼손 손목 궤적 (파란색)
ax_3d.plot(left_wrist_data_xyz[:, 0], left_wrist_data_xyz[:, 1], left_wrist_data_xyz[:, 2], label='Left Wrist Trajectory', color='blue', marker='o', markersize=3)
ax_3d.scatter(left_wrist_data_xyz[0, 0], left_wrist_data_xyz[0, 1], left_wrist_data_xyz[0, 2], color='green', s=100, label='Left Start', marker='^') # 시작점

# 오른손 손목 궤적 (빨간색)
ax_3d.plot(right_wrist_data_xyz[:, 0], right_wrist_data_xyz[:, 1], right_wrist_data_xyz[:, 2], label='Right Wrist Trajectory', color='red', marker='x', markersize=3)
ax_3d.scatter(right_wrist_data_xyz[0, 0], right_wrist_data_xyz[0, 1], right_wrist_data_xyz[0, 2], color='purple', s=100, label='Right Start', marker='^') # 시작점

# 코의 평균 위치 (참고용)
nose_data_xyz = selected_sequence_3d[:, NOSE_IDX, :]
mean_nose_pos = np.mean(nose_data_xyz, axis=0)
ax_3d.scatter(mean_nose_pos[0], mean_nose_pos[1], mean_nose_pos[2], color='black', s=150, label='Mean Nose Pos', marker='s')


ax_3d.set_xlabel('X coordinate')
ax_3d.set_ylabel('Y coordinate')
ax_3d.set_zlabel('Z coordinate (Depth)')
# Y축 방향 반전 (MediaPipe의 아래로 갈수록 증가와 맞추기 위해)
# ax_3d.invert_yaxis() # 필요에 따라 주석 해제/활성화
# Z축 방향 반전 (MediaPipe의 카메라에서 멀어질수록 증가와 맞추기 위해)
# ax_3d.invert_zaxis() # 필요에 따라 주석 해제/활성화 (보통 Z는 그대로 둠)

# 시야각 조절
# ax_3d.view_init(elev=20., azim=-35) # 예시 값, 필요에 따라 조절

plt.title(f'3D Wrist Trajectories (Sequence #{sequence_index_to_view})')
plt.legend()
plt.show()