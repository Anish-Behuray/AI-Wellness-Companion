import sounddevice as sd
import numpy as np
import librosa
from tensorflow.keras.models import load_model

# Load your saved model
model = load_model("C:/Users/ANISH/Desktop/project 3/ravdess_audio_emotion_model.h5")

# Emotion labels (same order as training)

emotion_labels = ['angry', 'calm', 'disgust', 'fearful', 'happy', 'neutral', 'sad', 'surprised']

# Parameters
duration = 3  # seconds
fs = 22050    # Sampling rate

def record_audio(duration, fs):
    print("🎙️ Recording... Speak now!")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
    sd.wait()
    print("✅ Recording complete!")
    return np.squeeze(recording)

def extract_mfcc(audio, sr=22050):
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    mfccs_scaled = np.mean(mfccs.T, axis=0)
    return mfccs_scaled

while True:
    # Record audio
    audio_data = record_audio(duration, fs)

    # Extract features
    features = extract_mfcc(audio_data, fs)
    features = np.expand_dims(features, axis=0)

    # Predict
    preds = model.predict(features)
    emotion = emotion_labels[np.argmax(preds)]

    # Show result
    print(f"🗣️ Detected Emotion: {emotion}")

    # Ask if continue
    choice = input("Press [Enter] to record again, or type 'q' to quit: ")
    if choice.lower() == 'q':
        break
