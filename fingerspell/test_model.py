import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import joblib
from collections import deque
from train_model import extract_vector_features_3d
from PIL import ImageFont, ImageDraw, Image

# ✅ PIL 폰트 (한글 출력용)
pil_font = ImageFont.truetype("C:/Windows/Fonts/malgun.ttf", 30)

def draw_korean_text(img, text, position=(10, 30), font=None, color=(0, 255, 0)):
    img_pil = Image.fromarray(img)
    draw = ImageDraw.Draw(img_pil)
    draw.text(position, text, font=font, fill=color)
    return np.array(img_pil)

# ✅ 모델과 인코더 로드
model = tf.keras.models.load_model("fingerspelling_vector3d_lstm.h5")
le = joblib.load("label_encoder.pkl")

# ✅ Mediapipe 설정
mp_holistic = mp.solutions.holistic
detector = mp_holistic.Holistic(static_image_mode=False, model_complexity=1)

# ✅ 설정
seq = deque(maxlen=10)
last_action = ''
threshold = 0.9

# ✅ 웹캠
cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = detector.process(image_rgb)

    if results.left_hand_landmarks:
        joint = np.zeros((21, 3))
        for i, lm in enumerate(results.left_hand_landmarks.landmark):
            joint[i] = [lm.x, lm.y, lm.z]
        d = extract_vector_features_3d(joint)
        seq.append(d)

        # 10프레임이 쌓이면 예측
        if len(seq) == 10:
            input_data = np.expand_dims(np.array(seq, dtype=np.float32), axis=0)
            y_pred = model.predict(input_data)[0]
            i_pred = int(np.argmax(y_pred))
            conf = y_pred[i_pred]

            if conf > threshold:
                action = le.inverse_transform([i_pred])[0]
                if action != last_action:
                    last_action = action

            frame = draw_korean_text(frame, f"{last_action} ({conf:.2f})", position=(10, 50), font=pil_font)

    # ✅ 손 관절 점 찍기
    if results.left_hand_landmarks:
        for lm in results.left_hand_landmarks.landmark:
            cx, cy = int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])
            cv2.circle(frame, (cx, cy), 5, (255, 255, 255), -1)

    cv2.imshow('Real-Time Fingerspelling', frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
