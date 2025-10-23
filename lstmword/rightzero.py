# process_overwrite_specific_label_no_right_hand.py
import numpy as np
import os
from tqdm import tqdm
import shutil # 백업용

# ------------------- 설정 (collect_data.py와 일치시켜야 함) ------------------- #
NUM_FRAMES = 30
NUM_KEYPOINTS = 1 + 21 + 21
NUM_COORDS = 3
TARGET_SHAPE_FLAT = (NUM_FRAMES, NUM_KEYPOINTS * NUM_COORDS)

RIGHT_HAND_START_IDX_IN_RESHAPED = 1 + 21
HAND_KEYPOINTS_COUNT = 21

# ------------------- 경로 설정 ------------------- #
DATASET_ROOTS_TO_MODIFY = [
    'dataset_original_NoseHands_final_v3_newaug',
    'dataset_augmented_NoseHands_final_v3_newaug'
]

ENABLE_BACKUP = True
BACKUP_DIR_SUFFIX = "_backup_before_no_righthand_specific"

# ------------------- 메인 처리 함수 (동일) ------------------- #
def set_right_hand_to_zero(sequence_flat):
    try:
        sequence_reshaped = sequence_flat.reshape((NUM_FRAMES, NUM_KEYPOINTS, NUM_COORDS))
        sequence_reshaped[:15, RIGHT_HAND_START_IDX_IN_RESHAPED : RIGHT_HAND_START_IDX_IN_RESHAPED + HAND_KEYPOINTS_COUNT, :] = 0.0
        return sequence_reshaped.reshape(TARGET_SHAPE_FLAT)
    except Exception as e:
        print(f"Error processing sequence: {e}")
        return None

def process_and_overwrite_specific_label(dataset_root_path, target_label_name):
    target_label_path = os.path.join(dataset_root_path, target_label_name)

    if not os.path.exists(target_label_path) or not os.path.isdir(target_label_path):
        print(f"경고: '{dataset_root_path}' 내에서 라벨 폴더 '{target_label_name}'를 찾을 수 없습니다. 건너<0xEB><0x9B><0x84>니다.")
        return

    print(f"\n'{target_label_path}' 경로의 파일을 직접 수정합니다...")
    
    npy_files = [f for f in os.listdir(target_label_path) if f.endswith('.npy') and f.startswith('group_')]
    if not npy_files:
        print(f"  라벨 '{target_label_name}'에 처리할 .npy 파일이 없습니다.")
        return

    for file_name in tqdm(npy_files, desc=f"파일 처리 중 ({target_label_name} in {os.path.basename(dataset_root_path)})"):
        file_path = os.path.join(target_label_path, file_name)
        try:
            sequences_in_group = np.load(file_path)
            if sequences_in_group.ndim != 3 or sequences_in_group.shape[1:] != TARGET_SHAPE_FLAT:
                print(f"  경고: {file_path} 파일 형태 오류. 건너<0xEB><0x9B><0x84>니다.")
                continue
            
            modified_sequences_list = []
            file_actually_modified = False
            for i in range(sequences_in_group.shape[0]):
                original_sequence_flat = sequences_in_group[i]
                modified_sequence_flat = set_right_hand_to_zero(original_sequence_flat.copy())
                if modified_sequence_flat is not None:
                    if not np.array_equal(original_sequence_flat, modified_sequence_flat):
                        file_actually_modified = True
                    modified_sequences_list.append(modified_sequence_flat)
                else:
                    modified_sequences_list.append(original_sequence_flat)
            
            if file_actually_modified and modified_sequences_list:
                np.save(file_path, np.array(modified_sequences_list))
        except Exception as e:
            print(f"  오류: {file_path} 파일 처리 중 - {e}")

# ------------------- 스크립트 실행 ------------------- #
if __name__ == "__main__":
    target_label = input("오른손 데이터를 0으로 만들 특정 동작 라벨을 입력하세요: ").strip()
    if not target_label:
        print("라벨이 입력되지 않았습니다. 작업을 종료합니다.")
        exit()

    print(f"\n경고: '{target_label}' 동작에 해당하는 원본 데이터 파일들을 직접 수정합니다.")
    print("실행 전 반드시 해당 라벨의 데이터를 백업하세요!")
    
    proceed = input(f"'{target_label}' 동작에 대해 계속 진행하시겠습니까? (yes/no): ").strip().lower()
    if proceed != 'yes':
        print("작업이 취소되었습니다.")
        exit()

    if ENABLE_BACKUP:
        print(f"\n'{target_label}' 라벨에 대한 백업을 시작합니다...")
        backup_parent_dir = "dataset_backups"
        os.makedirs(backup_parent_dir, exist_ok=True)

        for root_path in DATASET_ROOTS_TO_MODIFY:
            specific_label_path_to_backup = os.path.join(root_path, target_label)
            if os.path.exists(specific_label_path_to_backup) and os.path.isdir(specific_label_path_to_backup):
                # 백업 폴더명: dataset_backups/dataset_original_.../target_label_backup_...
                backup_root_specific = os.path.join(backup_parent_dir, os.path.basename(root_path))
                os.makedirs(backup_root_specific, exist_ok=True)
                
                backup_path_name = target_label + BACKUP_DIR_SUFFIX
                full_backup_path = os.path.join(backup_root_specific, backup_path_name)
                
                if os.path.exists(full_backup_path):
                    print(f"  경고: 백업 폴더 '{full_backup_path}'가 이미 존재합니다.")
                    overwrite_backup = input(f"    기존 백업을 덮어쓰시겠습니까? (yes/no): ").strip().lower()
                    if overwrite_backup == 'yes':
                        print(f"    기존 백업 폴더 '{full_backup_path}'를 삭제합니다...")
                        shutil.rmtree(full_backup_path)
                    else:
                        print(f"    기존 백업을 유지하고 '{specific_label_path_to_backup}' 처리를 건너<0xEB><0x9B><0x84>니다.")
                        continue
                try:
                    print(f"  '{specific_label_path_to_backup}'를 '{full_backup_path}'로 백업 중...")
                    shutil.copytree(specific_label_path_to_backup, full_backup_path)
                    print(f"  백업 완료: '{full_backup_path}'")
                except Exception as e:
                    print(f"  오류: '{specific_label_path_to_backup}' 백업 중 - {e}")
                    print(f"  '{specific_label_path_to_backup}' 처리를 중단합니다.")
                    continue
            else:
                print(f"  경고: 백업할 라벨 경로 '{specific_label_path_to_backup}'를 찾을 수 없습니다.")
        print("백업 완료 (또는 시도 완료).")


    for dataset_path in DATASET_ROOTS_TO_MODIFY:
        process_and_overwrite_specific_label(dataset_path, target_label)

    print(f"\n'{target_label}' 동작에 대한 데이터 직접 수정 완료.")
