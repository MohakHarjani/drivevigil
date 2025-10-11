import cv2
import mediapipe as mp
import numpy as np
import time
import threading
import queue
import speech_recognition as sr
from scipy.signal import butter, lfilter
import warnings

warnings.filterwarnings("ignore", category=UserWarning)  # suppress TFLite warnings

# === Mediapipe setup ===
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)
mp_drawing = mp.solutions.drawing_utils

LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]

# === Filters ===
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low, high = lowcut / nyq, highcut / nyq
    b, a = butter(order, [low, high], btype="band")
    return b, a

def bandpass_filter(data, lowcut=0.7, highcut=4.0, fs=30, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    return lfilter(b, a, data)

# === EAR ===
def eye_aspect_ratio(landmarks, eye_indices, w, h):
    coords = [(int(landmarks.landmark[i].x * w),
               int(landmarks.landmark[i].y * h)) for i in eye_indices]
    A = np.linalg.norm(np.array(coords[1]) - np.array(coords[5]))
    B = np.linalg.norm(np.array(coords[2]) - np.array(coords[4]))
    C = np.linalg.norm(np.array(coords[0]) - np.array(coords[3]))
    return (A + B) / (2.0 * C)

# === Redness detection ===
def detect_face_redness(frame, landmarks, w, h):
    nose = landmarks.landmark[1]
    cx, cy = int(nose.x * w), int(nose.y * h)
    x1, y1 = max(cx - 15, 0), max(cy - 15, 0)
    x2, y2 = min(cx + 15, w), min(cy + 15, h)
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return 0
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    lower_red1, upper_red1 = np.array([0, 70, 50]), np.array([10, 255, 255])
    lower_red2, upper_red2 = np.array([170, 70, 50]), np.array([180, 255, 255])
    mask = cv2.inRange(hsv, lower_red1, upper_red1) + cv2.inRange(hsv, lower_red2, upper_red2)
    return np.sum(mask) / (roi.size / 3)

# === Voice listener ===
def voice_command_listener(command_queue):
    rec = sr.Recognizer()
    mic = sr.Microphone()
    with mic as src:
        rec.adjust_for_ambient_noise(src)
    while True:
        try:
            with mic as src:
                audio = rec.listen(src, phrase_time_limit=3)
            cmd = rec.recognize_google(audio).lower()
            if cmd:
                command_queue.put(cmd)
        except Exception:
            pass

# === Heart-rate computation (rPPG) ===
def calculate_heart_rate(green_buffer, fs=30):
    if len(green_buffer) < fs * 8:
        return None
    filtered = bandpass_filter(np.array(green_buffer), 0.7, 4.0, fs, order=6)
    fft = np.abs(np.fft.rfft(filtered))
    freqs = np.fft.rfftfreq(len(filtered), d=1.0 / fs)
    valid = np.where((freqs >= 0.8) & (freqs <= 3.0))
    valid_freqs = freqs[valid]
    if len(valid_freqs) == 0:
        return None
    peak = np.argmax(fft[valid])
    bpm = int(valid_freqs[peak] * 60)
    return bpm if 40 <= bpm <= 160 else None

# === Alert overlay ===
def draw_alert_box(frame, alerts):
    h, w = frame.shape[:2]
    box_w, box_h = w // 2, 20 + 30 * len(alerts)
    x, y = 10, h - box_h - 10
    cv2.rectangle(frame, (x, y), (x + box_w, y + box_h), (50, 50, 50), -1)
    cv2.rectangle(frame, (x, y), (x + box_w, y + box_h), (0, 0, 0), 2)
    for i, (msg, col) in enumerate(alerts):
        cv2.putText(frame, msg, (x + 10, y + 30 + i * 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, col, 2)

# === Main ===
def main():
    cv2.startWindowThread()  # Fix for Windows GUI in Git Bash / PowerShell
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Cannot access camera. Close other apps using it and retry.")
        return

    cv2.namedWindow("VW Care-Drive AI", cv2.WINDOW_NORMAL)  # ensures window opens
    EAR_THRESH, CLOSED_THRESH = 0.25, 20
    closed_frames = face_missing = 0
    fs, buffer_size = 30, 300
    green_buf, hr_hist = [], []
    HR_SMOOTH = 5

    COOLDOWN_TIME, ALERT_DURATION = 5.0, 2.5
    alert_cooldowns, active_alerts = {}, {}
    def add_alert(key, msg, color):
        active_alerts[key] = (msg, color, time.time())
        alert_cooldowns[key] = time.time()

    command_queue = queue.Queue()
    threading.Thread(target=voice_command_listener, args=(command_queue,), daemon=True).start()
    help_count, awaiting_conf, conf_start = 0, False, None
    abnormal_counter = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)
        now = time.time()

        # --- Face ---
        if results.multi_face_landmarks:
            face_missing = 0
            lm = results.multi_face_landmarks[0]
            mp_drawing.draw_landmarks(frame, lm, mp_face_mesh.FACEMESH_TESSELATION,
                                      mp_drawing.DrawingSpec(color=(0,255,0), thickness=1),
                                      mp_drawing.DrawingSpec(color=(0,255,0), thickness=1))

            # EAR
            ear = (eye_aspect_ratio(lm, LEFT_EYE_IDX, w, h) +
                   eye_aspect_ratio(lm, RIGHT_EYE_IDX, w, h)) / 2
            cv2.putText(frame, f"EAR: {ear:.2f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            closed_frames = closed_frames + 1 if ear < EAR_THRESH else 0
            if closed_frames >= CLOSED_THRESH and now - alert_cooldowns.get("drowsy", 0) > COOLDOWN_TIME:
                add_alert("drowsy", "DROWSINESS ALERT! Focus!", (0, 0, 255))

            # Redness
            if detect_face_redness(frame, lm, w, h) > 0.35 and now - alert_cooldowns.get("injury", 0) > COOLDOWN_TIME:
                add_alert("injury", "Possible Injury Detected!", (0, 0, 255))

            # Heart-rate
            lf, rf = lm.landmark[70], lm.landmark[300]
            x1, y1, x2, y2 = int(lf.x*w), int(lf.y*h), int(rf.x*w), int(rf.y*h)+20
            roi = frame[y1:y2, x1:x2]
            if roi.size > 0:
                g_mean = np.mean(roi[:, :, 1])
                green_buf.append(g_mean)
                if len(green_buf) > buffer_size: green_buf.pop(0)
                bpm = calculate_heart_rate(green_buf, fs)
                if bpm:
                    hr_hist.append(bpm)
                    if len(hr_hist) > HR_SMOOTH: hr_hist.pop(0)
                    avg = int(sum(hr_hist)/len(hr_hist))
                    cv2.putText(frame, f"Heart Rate: {avg} bpm", (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    if avg < 50 or avg > 120: abnormal_counter += 1
                    else: abnormal_counter = 0
                    if abnormal_counter >= 3 and now - alert_cooldowns.get("cardiac", 0) > COOLDOWN_TIME:
                        add_alert("cardiac", "Abnormal Heart Rate!", (0,0,255))

        else:
            face_missing += 1
            if face_missing > 50:
                add_alert("noface", "No Face Detected!", (0,0,255))

        # --- Voice ---
        while not command_queue.empty():
            cmd = command_queue.get().lower()
            # 1️⃣ Check confirmation first
            if awaiting_conf:
                if "yes" in cmd or "ok" in cmd:
                    add_alert("help_yes", "User confirmed OK. Taking action.", (0,255,0))
                    awaiting_conf = False
                elif "no" in cmd:
                    add_alert("help_no", "Help confirmed! Calling emergency.", (0,0,255))
                    awaiting_conf = False
            # 2️⃣ Then check new help
            elif "help" in cmd:
                help_count += 1
                if help_count >= 3:
                    add_alert("help_auto", "Multiple HELP requests! Auto alert.", (0,0,255))
                elif now - alert_cooldowns.get("help", 0) > COOLDOWN_TIME:
                    add_alert("help", "Are you alright? Say Yes or No.", (0,255,255))
                    awaiting_conf, conf_start = True, now
                    alert_cooldowns['help'] = now

        # Confirmation timeout
        if awaiting_conf and conf_start and now - conf_start > 15:
            add_alert("help_timeout", "No confirmation received. Help cancelled.", (0,0,255))
            awaiting_conf = False

        # Draw alerts
        expired_keys = [k for k,(msg,col,t) in active_alerts.items() if now - t > ALERT_DURATION]
        for k in expired_keys: active_alerts.pop(k)
        draw_alert_box(frame, [(msg,col) for msg,col,t in active_alerts.values()])

        cv2.putText(frame, "Press ESC to exit", (w-220,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,200), 1)
        cv2.imshow("VW Care-Drive AI", frame)
        if cv2.waitKey(1) & 0xFF == 27: break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
