# test_model_combo_D_live_v2.py
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import joblib
# from collections import deque # 사용되지 않으므로 제거
from PIL import ImageFont, ImageDraw, Image
# from numpy.linalg import norm # 조합 D 특성 추출에서는 직접 사용 안 함
import time

# ==============================================================================
# train_model_combo_D_v2.py 및 collect_data_v2_new_augs.py 와 일치하는 상수 정의
# ==============================================================================
NUM_FRAMES_CONFIG = 30
PTS_CONFIG = 43 # Raw input: 코(1) + 왼손(21) + 오른손(21) = 43
HAND_SIZE_CONFIG = 21 # 각 손의 키포인트 수 (손목 포함)
FINGER_KPS_COUNT_CONFIG = HAND_SIZE_CONFIG - 1 # 손목 제외한 손가락/손바닥 키포인트 수 (20개)
NUM_COORDS_CONFIG = 3

# 키포인트 인덱스 (43개 기준, raw input data) - collect_data.py 기준
# NOSE_IDX_CONFIG = 0 # 함수 내 로컬 변수로 사용
# L_HAND_START_IDX_CONFIG = 1
# R_HAND_START_IDX_CONFIG = 1 + HAND_SIZE_CONFIG

# train_model_combo_D_v2.py의 TOTAL_FEATURES_PER_FRAME
TOTAL_FEATURES_PER_FRAME_CONFIG = (HAND_SIZE_CONFIG * NUM_COORDS_CONFIG) + \
                                  (HAND_SIZE_CONFIG * NUM_COORDS_CONFIG) + \
                                  (FINGER_KPS_COUNT_CONFIG * NUM_COORDS_CONFIG) + \
                                  (FINGER_KPS_COUNT_CONFIG * NUM_COORDS_CONFIG)  # 63 + 63 + 60 + 60 = 246

# Mediapipe Holistic 설정
mp_holistic = mp.solutions.holistic # 여기로 이동
MP_NOSE_IDX = mp_holistic.PoseLandmark.NOSE.value


# ─────────────────── train_model_combo_D_v2.py에서 가져온 특징 추출 함수 (조합 D) ─────────────────── #
def seq_to_feat_oneshot_combo_D(sequence_data_flat): # Input: (NUM_FRAMES_CONFIG, PTS_CONFIG * NUM_COORDS_CONFIG)
    keypoints_reshaped_abs = sequence_data_flat.reshape(NUM_FRAMES_CONFIG, PTS_CONFIG, NUM_COORDS_CONFIG)
    output_features_list = []

    _NOSE_IDX = 0
    _LHAND_START = 1
    _RHAND_START = 1 + HAND_SIZE_CONFIG
    _HAND_KPS_COUNT = HAND_SIZE_CONFIG
    _FINGER_KPS_COUNT_LOCAL = FINGER_KPS_COUNT_CONFIG

    for t in range(NUM_FRAMES_CONFIG):
        current_frame_kps_abs_3d = keypoints_reshaped_abs[t]

        nose_abs = current_frame_kps_abs_3d[_NOSE_IDX].astype(np.float32)

        left_hand_all_abs = current_frame_kps_abs_3d[_LHAND_START : _LHAND_START + _HAND_KPS_COUNT].astype(np.float32)
        right_hand_all_abs = current_frame_kps_abs_3d[_RHAND_START : _RHAND_START + _HAND_KPS_COUNT].astype(np.float32)

        left_wrist_abs = left_hand_all_abs[0]
        right_wrist_abs = right_hand_all_abs[0]

        if np.all(np.abs(left_hand_all_abs) < 1e-9):
            relative_left_hand_to_nose = np.zeros_like(left_hand_all_abs, dtype=np.float32)
        elif np.all(np.abs(nose_abs) < 1e-9):
            relative_left_hand_to_nose = np.zeros_like(left_hand_all_abs, dtype=np.float32)
        else:
            relative_left_hand_to_nose = (left_hand_all_abs - nose_abs[None, :]).astype(np.float32)

        if np.all(np.abs(right_hand_all_abs) < 1e-9):
            relative_right_hand_to_nose = np.zeros_like(right_hand_all_abs, dtype=np.float32)
        elif np.all(np.abs(nose_abs) < 1e-9):
            relative_right_hand_to_nose = np.zeros_like(right_hand_all_abs, dtype=np.float32)
        else:
            relative_right_hand_to_nose = (right_hand_all_abs - nose_abs[None, :]).astype(np.float32)

        if np.all(np.abs(left_wrist_abs) < 1e-9) or np.all(np.abs(left_hand_all_abs[1:]) < 1e-9) :
            relative_left_fingers_to_wrist = np.zeros((_FINGER_KPS_COUNT_LOCAL, NUM_COORDS_CONFIG), dtype=np.float32)
        else:
            relative_left_fingers_to_wrist = (left_hand_all_abs[1:] - left_wrist_abs[None, :]).astype(np.float32)

        if np.all(np.abs(right_wrist_abs) < 1e-9) or np.all(np.abs(right_hand_all_abs[1:]) < 1e-9):
            relative_right_fingers_to_wrist = np.zeros((_FINGER_KPS_COUNT_LOCAL, NUM_COORDS_CONFIG), dtype=np.float32)
        else:
            relative_right_fingers_to_wrist = (right_hand_all_abs[1:] - right_wrist_abs[None, :]).astype(np.float32)

        frame_features = np.concatenate([
            relative_left_hand_to_nose.flatten(),
            relative_right_hand_to_nose.flatten(),
            relative_left_fingers_to_wrist.flatten(),
            relative_right_fingers_to_wrist.flatten()
        ]).astype(np.float32)

        if frame_features.shape[0] != TOTAL_FEATURES_PER_FRAME_CONFIG:
            correct_len_features = np.zeros(TOTAL_FEATURES_PER_FRAME_CONFIG, dtype=np.float32)
            copy_len = min(frame_features.shape[0], TOTAL_FEATURES_PER_FRAME_CONFIG)
            correct_len_features[:copy_len] = frame_features[:copy_len]
            frame_features = correct_len_features

        output_features_list.append(frame_features)
    return np.array(output_features_list, dtype=np.float32)
# ==============================================================================
# 피처 추출 함수 및 상수 정의 끝
# ==============================================================================

# ✅ PIL 폰트
FONT_PATH = "malgun.ttf" # Windows 기준, macOS/Linux에서는 경로 확인 필요
try:
    pil_font = ImageFont.truetype(FONT_PATH, 30)
except IOError:
    print(f"'{FONT_PATH}' 폰트를 찾을 수 없습니다. OpenCV 기본 폰트를 사용합니다.")
    pil_font = None

def draw_text_with_bg(img_bgr, text, position=(10, 60), font=None, text_color=(255, 255, 0), bg_color=(0,0,100)):
    text_x, text_y_baseline = position
    
    if font is None:
        (text_width, text_height_cv), baseline_cv = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
        actual_text_height_above_baseline = text_height_cv
        # OpenCV putText는 baseline 기준으로 y를 입력받음
        text_render_y_opencv = text_y_baseline 
    else:
        try:
            dummy_img = Image.new("RGB", (1,1)); dummy_draw = ImageDraw.Draw(dummy_img)
            if hasattr(font, 'getbbox'): # 최신 PIL (Pillow 9.2.0+)
                 # (left, top, right, bottom) for text drawn at (0,0)
                bbox = font.getbbox(text)
                text_width = bbox[2] - bbox[0] # width
                # ascent is -bbox[1], descent is bbox[3]
                actual_text_height_above_baseline = -bbox[1] # Ascent
                # PIL draw.text는 좌상단 기준, y는 텍스트의 가장 윗부분
                # 베이스라인에 맞추려면: baseline_y - ascent
                text_render_y_pil = text_y_baseline - actual_text_height_above_baseline
            elif hasattr(dummy_draw, 'textbbox'): # 중간 버전 PIL
                bbox = dummy_draw.textbbox((0,0), text, font=font)
                text_width = bbox[2] - bbox[0]
                actual_text_height_above_baseline = bbox[3] - bbox[1] # 전체 높이로 근사
                text_render_y_pil = text_y_baseline - actual_text_height_above_baseline + int(actual_text_height_above_baseline*0.2) # 경험적 보정
            else: # 구형 PIL
                text_width, actual_text_height_above_baseline = dummy_draw.textsize(text, font=font)
                text_render_y_pil = text_y_baseline - actual_text_height_above_baseline + int(actual_text_height_above_baseline*0.2) # 경험적 보정
        except Exception:
            (text_width, text_height_cv), baseline_cv = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
            actual_text_height_above_baseline = text_height_cv
            text_render_y_opencv = text_y_baseline
            font = None

    padding = 5
    bg_rect_pos_start = (text_x - padding, text_y_baseline - actual_text_height_above_baseline - padding)
    bg_rect_pos_end = (text_x + text_width + padding, text_y_baseline + padding + int(actual_text_height_above_baseline*0.2) if font else text_y_baseline + padding ) # descent 고려
    cv2.rectangle(img_bgr, bg_rect_pos_start, bg_rect_pos_end, bg_color, -1)

    if font is None:
        cv2.putText(img_bgr, text, (text_x, text_render_y_opencv), cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2, cv2.LINE_AA)
    else:
        img_pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        draw.text((text_x, text_render_y_pil), text, font=font, fill=text_color)
        img_bgr[:] = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    return img_bgr


# ✅ 모델, 라벨 인코더, 스케일러 로드 (파일명은 train_model_combo_D_v2.py와 일치)
MODEL_PATH = f"slr_model_comboD_best_{TOTAL_FEATURES_PER_FRAME_CONFIG}_float32.h5"
LABEL_ENCODER_PATH = f"slr_label_encoder_comboD_float32.pkl"
SCALER_PATH = f"slr_scaler_comboD_{TOTAL_FEATURES_PER_FRAME_CONFIG}_float32.pkl"

try:
    model = tf.keras.models.load_model(MODEL_PATH)
    le = joblib.load(LABEL_ENCODER_PATH)
    scaler = joblib.load(SCALER_PATH)
    print(f"모델 로드 완료: {MODEL_PATH}")
    print(f"라벨 인코더 클래스: {le.classes_}")
except Exception as e: print(f"모델/파일 로드 오류: {e}"); exit()

# ✅ Mediapipe Holistic 설정 (mp_holistic은 위에서 이미 정의됨)
mp_drawing = mp.solutions.drawing_utils
detector = mp_holistic.Holistic(static_image_mode=False, model_complexity=1,
                                min_detection_confidence=0.5, min_tracking_confidence=0.5)

# ✅ 설정
SEQUENCE_LENGTH = NUM_FRAMES_CONFIG
raw_keypoints_for_capture = []
is_capturing = False
capture_frame_count = 0
COUNTDOWN_SECONDS = 3
PREDICTION_THRESHOLD = 0.7
last_predicted_action = "대기 중..."
last_prediction_confidence = 0.0

# --- Mediapipe 결과에서 코, 왼손, 오른손 키포인트 추출 (43개 키포인트, (129,) flatten) ---
def extract_all_keypoints_from_results_live(results):
    pose_coords_nose = np.zeros((1, NUM_COORDS_CONFIG), dtype=np.float32)
    if results.pose_landmarks and results.pose_landmarks.landmark:
        if len(results.pose_landmarks.landmark) > MP_NOSE_IDX:
            lm = results.pose_landmarks.landmark[MP_NOSE_IDX]
            if lm.visibility > 0.1:
                 pose_coords_nose[0] = [lm.x, lm.y, lm.z]

    lh_coords = np.zeros((HAND_SIZE_CONFIG, NUM_COORDS_CONFIG), dtype=np.float32)
    if results.left_hand_landmarks:
        for i, lm in enumerate(results.left_hand_landmarks.landmark):
            if i < HAND_SIZE_CONFIG: lh_coords[i] = [lm.x, lm.y, lm.z]

    rh_coords = np.zeros((HAND_SIZE_CONFIG, NUM_COORDS_CONFIG), dtype=np.float32)
    if results.right_hand_landmarks:
        for i, lm in enumerate(results.right_hand_landmarks.landmark):
            if i < HAND_SIZE_CONFIG: rh_coords[i] = [lm.x, lm.y, lm.z]
            
    return np.concatenate([pose_coords_nose, lh_coords, rh_coords]).flatten().astype(np.float32)

# ✅ 웹캠
cap = cv2.VideoCapture(0)
if not cap.isOpened(): print("웹캠을 열 수 없습니다."); exit()
print(f"실시간 수어 인식 (조합 D - {TOTAL_FEATURES_PER_FRAME_CONFIG} features). (Enter: 3초 후 캡처 시작, ESC: 종료)")

countdown_active = False
countdown_start_time = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: print("프레임을 받을 수 없습니다."); break
    
    frame_display = frame.copy()
    frame_display = cv2.flip(frame_display, 1)
    image_rgb = cv2.cvtColor(frame_display, cv2.COLOR_BGR2RGB)
    image_rgb.flags.writeable = False
    results = detector.process(image_rgb)
    image_rgb.flags.writeable = True

    current_keypoints_flat = extract_all_keypoints_from_results_live(results) # (129,) float32

    display_status_text = ""
    action_text_position = (10, 70)
    instruction_text_position = (10, frame_display.shape[0] - 40)

    if countdown_active:
        remaining_time = COUNTDOWN_SECONDS - (time.time() - countdown_start_time)
        if remaining_time > 0:
            display_status_text = f"캡처 시작까지: {int(np.ceil(remaining_time))}초"
        else: 
            countdown_active = False
            is_capturing = True
            raw_keypoints_for_capture = [] 
            capture_frame_count = 0
            print("캡처 시작!")
            last_predicted_action = "캡처 중..."
            last_prediction_confidence = 0.0

    if is_capturing:
        raw_keypoints_for_capture.append(current_keypoints_flat)
        capture_frame_count += 1
        display_status_text = f"캡처 중... {capture_frame_count}/{SEQUENCE_LENGTH}"

        if capture_frame_count == SEQUENCE_LENGTH:
            is_capturing = False
            print("캡처 완료. 분석 중...")
            
            input_raw_seq_np = np.array(raw_keypoints_for_capture, dtype=np.float32) # (30, 129) float32

            try:
                input_features_seq = seq_to_feat_oneshot_combo_D(input_raw_seq_np) # (30, 246) float32
                num_f_actual = input_features_seq.shape[1]
                
                if num_f_actual != TOTAL_FEATURES_PER_FRAME_CONFIG:
                    print(f"Warning: 피처 차원 불일치. 예상: {TOTAL_FEATURES_PER_FRAME_CONFIG}, 실제: {num_f_actual}")
                    last_predicted_action = "피처 오류"
                else:
                    # 스케일링: 입력은 (N_samples * N_frames, N_features) 여야 함
                    # 현재 input_features_seq는 (30, 246) 이므로, (30, 246) -> (30*246) -> (30, 246)
                    input_features_seq_reshaped = input_features_seq.reshape(-1, num_f_actual) # (30, 246)
                    input_features_scaled_flat = scaler.transform(input_features_seq_reshaped) # scaler는 2D 입력을 기대
                    input_features_scaled = input_features_scaled_flat.reshape(SEQUENCE_LENGTH, num_f_actual).astype(np.float32)
                    
                    model_input_data = np.expand_dims(input_features_scaled, axis=0) # (1, 30, 246) float32
                    
                    prediction_probs = model.predict(model_input_data, verbose=0)[0]
                    predicted_class_idx = np.argmax(prediction_probs)
                    last_prediction_confidence = prediction_probs[predicted_class_idx]

                    if last_prediction_confidence >= PREDICTION_THRESHOLD:
                        last_predicted_action = le.inverse_transform([predicted_class_idx])[0]
                        print(f"예측 결과: {last_predicted_action} (신뢰도: {last_prediction_confidence:.2f})")
                    else:
                        last_predicted_action = "인식 불가"
                        print(f"인식 불가 (신뢰도 {last_prediction_confidence:.2f} < {PREDICTION_THRESHOLD})")
            
            except Exception as e_pred:
                print(f"예측 중 오류: {e_pred}")
                import traceback
                traceback.print_exc()
                last_predicted_action = "예측 오류"
            
    if is_capturing or countdown_active:
        pass
    else:
        display_status_text = f"결과: {last_predicted_action} ({last_prediction_confidence:.2f})"
        frame_display = draw_text_with_bg(frame_display, "Enter: 3초 후 캡처 시작", 
                                          position=instruction_text_position, font=pil_font, 
                                          text_color=(0,255,255), bg_color=(50,50,50))

    frame_display = draw_text_with_bg(frame_display, display_status_text, position=action_text_position, font=pil_font)
    
    mp_drawing.draw_landmarks(frame_display, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(80,22,10), thickness=1, circle_radius=1),
                              mp_drawing.DrawingSpec(color=(80,44,121), thickness=1, circle_radius=1))
    mp_drawing.draw_landmarks(frame_display, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(121,22,76), thickness=1, circle_radius=1),
                              mp_drawing.DrawingSpec(color=(121,44,250), thickness=1, circle_radius=1))
    mp_drawing.draw_landmarks(frame_display, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(245,117,66), thickness=1, circle_radius=1),
                              mp_drawing.DrawingSpec(color=(245,66,230), thickness=1, circle_radius=1))

    cv2.imshow('실시간 수어 인식 (조합 D)', frame_display)

    key = cv2.waitKey(1) & 0xFF
    if key == 27: # ESC
        break
    elif key == 13 and not is_capturing and not countdown_active: # Enter
        print("\n카운트다운 시작...")
        countdown_active = True
        countdown_start_time = time.time()
        last_predicted_action = "카운트다운..."
        last_prediction_confidence = 0.0

cap.release()
cv2.destroyAllWindows()
if detector: detector.close()
print("프로그램 종료.")