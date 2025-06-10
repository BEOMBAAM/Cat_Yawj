import cv2
import numpy as np
import deeplabcut

CONFIG_PATH = "C:/Users/drago/Desktop/cat_yawn-BBBB-2025-06-11/config.yaml"
cfg = deeplabcut.auxiliaryfunctions.read_config(CONFIG_PATH)
dlc_predictor = deeplabcut.load_predictor(cfg, shuffle=1, trainingsetindex=0)

KEYPOINTS = [
    'eye-up-left', 'eye-down-left',
    'eye-up-right', 'eye-down-right',
    'mouth-up', 'mouth-low',
    'mouth-left', 'mouth-right',
    'nose'
]

SKELETON = [
    ('eye-up-left', 'eye-down-left'),
    ('eye-up-right', 'eye-down-right'),
    ('mouth-up', 'mouth-low'),
    ('mouth-left', 'mouth-right'),
    ('mouth-up', 'mouth-left'),
    ('mouth-up', 'mouth-right'),
    ('mouth-low', 'mouth-left'),
    ('mouth-low', 'mouth-right'),
    ('nose', 'mouth-up')
]

COLORS = {
    'eye-up-left': (0, 255, 0),
    'eye-down-left': (0, 200, 0),
    'eye-up-right': (0, 255, 255),
    'eye-down-right': (0, 200, 200),
    'mouth-up': (255, 255, 0),
    'mouth-low': (0, 0, 255),
    'mouth-left': (200, 100, 255),
    'mouth-right': (100, 100, 255),
    'nose': (255, 0, 0)
}

EYE_THRESHOLD = 15
MOUTH_THRESHOLD = 25

cap = cv2.VideoCapture(0)
print("웹캠 분석 시작 - 'q'를 누르면 종료됩니다.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    try:
        keypoints = dlc_predictor.predict_single_frame(frame)
        points = {}

        for i, name in enumerate(dlc_predictor.bodyparts):
            x, y, p = keypoints[0][i]
            if p > 0.6:
                points[name] = (int(x), int(y))
                cv2.circle(frame, points[name], 5, COLORS.get(name, (255, 255, 255)), -1)

        for a, b in SKELETON:
            if a in points and b in points:
                cv2.line(frame, points[a], points[b], (180, 180, 180), 2)

        eye_open = mouth_open = False

        if all(k in points for k in ['eye-up-left', 'eye-down-left', 'eye-up-right', 'eye-down-right']):
            eye_dist = (
                np.linalg.norm(np.array(points['eye-up-left']) - np.array(points['eye-down-left'])) +
                np.linalg.norm(np.array(points['eye-up-right']) - np.array(points['eye-down-right']))
            ) / 2
            eye_open = eye_dist > EYE_THRESHOLD

        if all(k in points for k in ['mouth-up', 'mouth-low']):
            mouth_dist = np.linalg.norm(np.array(points['mouth-up']) - np.array(points['mouth-low']))
            mouth_open = mouth_dist > MOUTH_THRESHOLD

        if mouth_open and eye_open:
            label, color = "Yawning 😺", (0, 255, 255)
        elif mouth_open:
            label, color = "Mouth Open 🟡", (0, 255, 0)
        else:
            label, color = "Neutral 😐", (100, 100, 255)

        cv2.putText(frame, label, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

    except Exception as e:
        cv2.putText(frame, f"Detection error: {e}", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        print("[ERROR]", e)

    cv2.imshow("Real-time Yawn Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
