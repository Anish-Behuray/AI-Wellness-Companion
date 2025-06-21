import streamlit as st
import cv2
import numpy as np
import sounddevice as sd
import librosa
from tensorflow.keras.models import load_model
from collections import Counter, defaultdict
import time

# Load models
expression_model = load_model("C:/Users/ANISH/Desktop/project 3/emotion_cnn_model.h5")
sleep_model = load_model("C:/Users/ANISH/Desktop/project 3/sleep_cnn_model.h5")
yawn_model = load_model("C:/Users/ANISH/Desktop/project 3/yawn_cnn_model.h5")
voice_model = load_model("C:/Users/ANISH/Desktop/project 3/ravdess_audio_emotion_model.h5")

expression_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
sleep_labels = ['closed', 'open']
yawn_labels = ['no_yawn', 'yawn']
voice_labels = ['angry', 'calm', 'disgust', 'fearful', 'happy', 'neutral', 'sad', 'surprised']

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

st.title("🩺 AI Doctor Assistant")

tab1, tab2 = st.tabs(["Live Monitoring", "Report"])

with tab1:
    run_cam = st.checkbox("Enable Webcam Monitoring")
    FRAME_WINDOW = st.image([])
    results_log = []
    frame_times = []

    if run_cam:
        cap = cv2.VideoCapture(0)
        monitoring_time = 10  # seconds
        fps = 5  # process 5 frames per second approx

        start_time = time.time()
        last_frame_time = start_time

        while time.time() - start_time < monitoring_time:
            ret, frame = cap.read()
            if not ret:
                break

            now = time.time()
            if now - last_frame_time < 1.0 / fps:
                continue  # skip frame to match fps
            last_frame_time = now

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            expression_pred = "Not detected"
            yawn_pred = "No"
            sleep_pred = "No"

            for (x, y, w, h) in faces:
                face_roi = gray[y:y+h, x:x+w]

                # Expression
                exp_roi = cv2.resize(face_roi, (48, 48))
                exp_roi = exp_roi.astype('float32') / 255.0
                exp_roi = np.expand_dims(exp_roi, axis=-1)
                exp_roi = np.expand_dims(exp_roi, axis=0)
                exp_out = expression_model.predict(exp_roi)
                expression_pred = expression_labels[np.argmax(exp_out)]

                # Yawn
                yawn_roi = cv2.resize(face_roi, (64, 64))
                yawn_roi = yawn_roi.astype('float32') / 255.0
                yawn_roi = np.expand_dims(yawn_roi, axis=-1)
                yawn_roi = np.expand_dims(yawn_roi, axis=0)
                yawn_out = yawn_model.predict(yawn_roi)
                yawn_label = yawn_labels[np.argmax(yawn_out)]
                yawn_pred = "Yes" if yawn_label == 'yawn' else "No"

                # Eyes -> Sleep
                eyes = eye_cascade.detectMultiScale(face_roi)
                for (ex, ey, ew, eh) in eyes:
                    eye_roi = face_roi[ey:ey+eh, ex:ex+ew]
                    eye_roi = cv2.resize(eye_roi, (64, 64))
                    eye_roi = eye_roi.astype('float32') / 255.0
                    eye_roi = np.expand_dims(eye_roi, axis=-1)
                    eye_roi = np.expand_dims(eye_roi, axis=0)
                    sleep_out = sleep_model.predict(eye_roi)
                    sleep_label = sleep_labels[np.argmax(sleep_out)]
                    sleep_pred = "Yes" if sleep_label == 'closed' else "No"
                    break

            cv2.putText(frame, f'Expression: {expression_pred}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)
            cv2.putText(frame, f'Sleep: {sleep_pred}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)
            cv2.putText(frame, f'Drowsy: {yawn_pred}', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)

            FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            # Log results + timestamp
            results_log.append({
                "expression": expression_pred,
                "sleep": sleep_pred,
                "yawn": yawn_pred
            })
            frame_times.append(now - start_time)

        cap.release()

        # Calculate duration spent in each state
        frame_duration = 1.0 / fps
        expr_dur = defaultdict(float)
        sleep_dur = defaultdict(float)
        yawn_dur = defaultdict(float)

        for res in results_log:
            expr_dur[res["expression"]] += frame_duration
            sleep_dur[res["sleep"]] += frame_duration
            yawn_dur[res["yawn"]] += frame_duration

        report_lines = ["AI Doctor Assistant Report (Duration based)"]
        report_lines.append("-" * 40)
        report_lines.append("Expression durations:")
        for expr, dur in expr_dur.items():
            report_lines.append(f"  {expr}: {dur:.2f} seconds")

        report_lines.append("\nSleep state durations:")
        for s, dur in sleep_dur.items():
            report_lines.append(f"  Sleep={s}: {dur:.2f} seconds")

        report_lines.append("\nDrowsy (Yawn) durations:")
        for y, dur in yawn_dur.items():
            report_lines.append(f"  Yawn={y}: {dur:.2f} seconds")

        final_report = "\n".join(report_lines)

        st.download_button("📄 Download Detailed Duration Report", final_report, file_name="ai_doctor_duration_report.txt")
