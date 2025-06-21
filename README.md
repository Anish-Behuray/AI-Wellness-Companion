# 🤖 AI Wellness Companion

**AI Wellness Companion** is a real-time AI system that monitors and reports on human wellness indicators through:  
✅ **Facial expression detection** (happy, sad, angry, neutral, etc.)  
✅ **Sleep detection** (eye closure monitoring)  
✅ **Yawn detection** (drowsiness monitoring)  
✅ **Voice emotion analysis**  

It’s designed to support:
- **Employee wellness tracking**
- **Driver alertness systems**
- **Patient monitoring**
- **General human state sensing**

---

## 🚀 Features

- 🎭 **Facial Expression Recognition:** Identifies emotional states like happy, sad, angry, etc.  
  **Accuracy:** 61%  
- 😴 **Sleep Detection:** Monitors eye closure to detect possible sleep.  
  **Accuracy:** 93%  
- 😮 **Yawn Detection:** Detects yawns as a sign of fatigue.  
  **Accuracy:** 43%  
- 🗣️ **Voice Emotion Analysis:** Analyzes voice tone for emotion classification.  
  **Accuracy:** 45%  
- ⏱️ **Duration-based Reporting:** Tracks and reports how long each state occurred.
- 📄 **Downloadable Report:** Generates a detailed session report summarizing all detected states.

---

## ⚙️ Requirements

```bash
tensorflow
opencv-python
streamlit
librosa
sounddevice
numpy
matplotlib
seaborn
```

## 💻 How to Use

1️⃣ **Run the app**

streamlit run streamlit_webiste_code.py

2️⃣ **Start Monitoring**

- Enable webcam monitoring to track expressions, eye closure (sleep detection), and yawns (drowsiness detection).

- Click Record Voice to record a short voice clip (about 3 seconds) for emotion analysis.

3️⃣ **Real-Time Display**

- The app will show on screen:

- Detected facial expression

- Sleep state (Yes/No for eye closure)

- Drowsiness state (Yes/No from yawn detection)

- Voice emotion (after recording)

4️⃣ **Download Detailed Report**

- After the session (e.g., 10+ seconds of monitoring), download a report that includes:

⏱ Time spent in each facial expression

⏱ Time spent with eyes closed (sleep state)

⏱ Time spent yawning (drowsy state)

🎤 Voice emotion result

