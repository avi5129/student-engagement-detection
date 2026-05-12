# 🎓 Student Engagement Detection System

A real-time AI-powered system that detects and analyzes student engagement levels in a classroom setting using computer vision and deep learning. The system uses **YOLOv12** for emotion classification, **YOLOv8** for pose estimation, and **MediaPipe** for lip movement detection — all integrated into a Flask web dashboard.

---

## ✨ Features

| Feature | Description |
|---|---|
| 🎯 **Real-time Engagement Scoring** | Calculates engagement score (0–100%) per student |
| 😊 **Emotion Recognition** | Custom YOLOv12 model classifies: Focused, Bored, Confused, Frustrated, Drowsy, Looking Away |
| 🧍 **Pose & Gaze Estimation** | YOLOv8-Pose tracks head direction (Center, Left, Right, Up, Down) |
| 💬 **Lip Movement Detection** | MediaPipe Face Landmarker detects if a student is talking |
| 📊 **Teacher Dashboard** | Session history, engagement charts, and per-session analytics |
| 🔐 **User Authentication** | Register / Login system for teachers |
| 🎨 **Clean UI** | Color-coded face boxes (Green = High, Yellow = Moderate, Red = Low) |

---

## 🏗️ Architecture

```
┌─────────────┐    ┌──────────────────┐    ┌────────────────┐
│  Webcam Feed │───▶│  YOLOv8-Pose     │───▶│ Head Pose /    │
│  (OpenCV)    │    │  (Person Track)  │    │ Gaze Direction │
└─────────────┘    └──────────────────┘    └────────────────┘
       │                                           │
       ▼                                           ▼
┌──────────────┐   ┌──────────────────┐   ┌─────────────────┐
│ Face Crop    │──▶│ YOLOv12-CLS      │──▶│ Engagement      │
│ Extraction   │   │ (Emotion Model)  │   │ Score Calculator │
└──────────────┘   └──────────────────┘   └─────────────────┘
       │                                           │
       ▼                                           ▼
┌──────────────┐                          ┌─────────────────┐
│ MediaPipe    │─────────────────────────▶│ Flask Web App   │
│ (Lip Motion) │                          │ + Dashboard     │
└──────────────┘                          └─────────────────┘
```

---

## 📋 Prerequisites

- **Python 3.10 – 3.12** (3.12 recommended)
- **Git** (for cloning YOLOv12 repo)
- **Webcam** (built-in or USB)
- **GPU (optional)**: NVIDIA GPU with CUDA for faster inference — the system also works on CPU

---

## 🚀 Quick Start (Any PC)

### 1. Clone this Repository

```bash
git clone https://github.com/YOUR_USERNAME/student-engagement-detection.git
cd student-engagement-detection
```

### 2. Create a Virtual Environment (Recommended)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Run the Automated Setup

```bash
python setup.py
```

This will:
- ✅ Install all Python dependencies from `requirements.txt`
- ✅ Download YOLOv8 Pose model (~6.5 MB)
- ✅ Download YOLOv12 Classification base model (~5.9 MB)
- ✅ Download MediaPipe Face Landmarker (~3.6 MB)
- ✅ Clone the YOLOv12 repository (required for model architecture)

### 4. Add the Custom Emotion Model

The custom-trained emotion model (`yolo_emotion_model_best.pt`) is **not included** in this repo due to size.

**Option A — Download from Releases:**
Check the [Releases](../../releases) page for `yolo_emotion_model_best.pt` and place it in the project root.

**Option B — Train your own:**
See [Training Your Own Model](#-training-your-own-model) below.

> ⚠️ The app will still run without this model, but emotion detection will be disabled (everything shows "Unknown").

### 5. Start the Application

```bash
# Windows (one-click)
run.bat

# Any OS
python web/app.py
```

### 6. Open in Browser

Navigate to **http://127.0.0.1:5000** in your browser.

- **Register** a new account or **Login**
- Click **Start Session** to begin real-time engagement monitoring
- Visit the **Dashboard** to review past sessions

---

## 📁 Project Structure

```
student-engagement-detection/
├── src/                        # Core ML modules
│   ├── capture.py              # Webcam video source (thread-safe)
│   ├── pose.py                 # YOLOv8 pose estimation + gaze detection
│   ├── emotion.py              # YOLOv12 emotion classification
│   ├── engagement.py           # Engagement score & state calculation
│   └── database.py             # SQLite database layer
│
├── web/                        # Flask web application
│   ├── app.py                  # Main Flask server + video pipeline
│   └── templates/
│       ├── index.html          # Landing page
│       ├── login.html          # Login page
│       ├── register.html       # Registration page
│       ├── live.html           # Live monitoring view
│       └── dashboard.html      # Session analytics dashboard
│
├── tests/                      # Unit & integration tests
│   ├── test_engagement.py
│   ├── test_gaze.py
│   ├── test_refinement.py
│   └── ...
│
├── setup.py                    # Automated setup (downloads models + deps)
├── download_yolo_weights.py    # Standalone weight downloader
├── train_emotion.py            # Training script for custom emotion model
├── train_yolo.py               # YOLO training utilities
├── train_yolo_v12.py           # YOLOv12-specific training script
├── prepare_dataset.py          # Dataset preparation utilities
├── requirements.txt            # Python dependencies
├── run.bat                     # Windows one-click launcher
├── .gitignore
└── README.md
```

---

## ⚙️ How It Works

### Engagement Pipeline (per frame)

1. **YOLOv8-Pose** detects people and extracts 17 body keypoints with tracking IDs
2. **Face crop** is extracted from keypoints (nose, eyes, ears)
3. **Head pose / gaze** direction is estimated from keypoint geometry
4. **YOLOv12 emotion model** classifies the face crop into 6 emotions
5. **MediaPipe Face Landmarker** checks lip distance to detect talking
6. **Engagement state** is determined by priority rules:
   - Gaze away → Unattentive / Distracted
   - Talking → Talking (engaged)
   - Emotion → Focused / Confused / Frustrated / Bored / Tired / Happy
7. **Score** is assigned (0–100) and **level** is computed (High / Moderate / Low)
8. Results are smoothed over 20 frames using mode-based voting

### Engagement States & Scores

| State | Score | Level |
|---|---|---|
| Happy | 95 | High |
| Focused | 90 | High |
| Talking | 85 | High |
| Surprise | 85 | High |
| Confused | 70 | Moderate |
| Frustrated | 60 | Moderate |
| Distracted | 50 | Moderate |
| Bored | 40 | Low |
| Tired | 30 | Low |
| Unattentive | 20 | Low |

---

## 🏋️ Training Your Own Model

If you want to train a custom emotion model:

### 1. Prepare Dataset
Organize images into class folders:
```
dataset/
├── train/
│   ├── Focused/
│   ├── Bored/
│   ├── Confused/
│   ├── Frustrated/
│   ├── Drowsy/
│   └── Looking Away/
└── val/
    ├── Focused/
    ├── Bored/
    └── ...
```

### 2. Run Training
```bash
python train_yolo_v12.py
```

### 3. Use Trained Model
Copy the best model to the project root:
```bash
cp runs/classify/train/weights/best.pt yolo_emotion_model_best.pt
```

---

## 🐛 Troubleshooting

| Problem | Solution |
|---|---|
| Camera not detected | Try changing `VideoSource(0)` to `VideoSource(1)` in `web/app.py` |
| TensorFlow warnings | Set environment variable: `TF_ENABLE_ONEDNN_OPTS=0` |
| `yolov12` import error | Ensure `python setup.py` was run (it clones the YOLOv12 repo) |
| Emotion shows "Unknown" | Make sure `yolo_emotion_model_best.pt` is in the project root |
| Slow performance | Use a GPU, or reduce webcam resolution in `capture.py` |
| Port 5000 in use | Change port in `app.py`: `app.run(port=5001)` |
| `face_landmarker.task` missing | Run `python setup.py` to download it |

---

## 📄 License

This project is developed as a **B.E. Final Year Project**.

---

## 👤 Author

**Avishkar Kawade**  
📧 avishkarkawade5@gmail.com
