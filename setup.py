"""
Automated Setup Script for Student Engagement Detection System.
Downloads all required model weights and dependencies.

Usage:
    python setup.py
"""

import os
import sys
import subprocess
import urllib.request
import shutil


# ── Model Downloads ──────────────────────────────────────────────────────────
MODELS = [
    {
        "name": "YOLOv8 Pose Model",
        "filename": "yolov8n-pose.pt",
        "url": "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n-pose.pt",
    },
    {
        "name": "YOLOv12 Classification Base",
        "filename": "yolov12n-cls.pt",
        "url": "https://github.com/sunsmarterjie/yolov12/releases/download/cls/yolov12n-cls.pt",
    },
    {
        "name": "MediaPipe Face Landmarker",
        "filename": "face_landmarker.task",
        "url": "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task",
    },
]

# The custom emotion model (yolo_emotion_model_best.pt) is project-specific
# and must be provided separately. See README for details.
EMOTION_MODEL = "yolo_emotion_model_best.pt"


def download_file(url, dest):
    """Download a file with progress reporting."""
    print(f"  Downloading {os.path.basename(dest)}...")
    try:
        urllib.request.urlretrieve(url, dest)
        size_mb = os.path.getsize(dest) / (1024 * 1024)
        print(f"  ✓ Downloaded ({size_mb:.1f} MB)")
        return True
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        return False


def clone_yolov12():
    """Clone the YOLOv12 repository if not present."""
    if os.path.isdir("yolov12"):
        print("  ✓ yolov12/ already exists")
        return True

    print("  Cloning YOLOv12 repository...")
    try:
        subprocess.run(
            ["git", "clone", "https://github.com/sunsmarterjie/yolov12.git"],
            check=True,
            capture_output=True,
            text=True,
        )
        print("  ✓ Cloned successfully")
        return True
    except FileNotFoundError:
        print("  ✗ Git is not installed. Please install Git and retry.")
        return False
    except subprocess.CalledProcessError as e:
        print(f"  ✗ Clone failed: {e.stderr}")
        return False


def install_requirements():
    """Install Python dependencies."""
    print("  Installing pip requirements...")
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"],
            check=True,
            capture_output=True,
            text=True,
        )
        print("  ✓ All requirements installed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"  ✗ pip install failed:\n{e.stderr}")
        return False


def main():
    print("=" * 60)
    print("  Student Engagement Detection System — Setup")
    print("=" * 60)

    # ── Step 1: Python requirements ──────────────────────────────
    print("\n[1/4] Installing Python dependencies...")
    install_requirements()

    # ── Step 2: Download pre-trained models ──────────────────────
    print("\n[2/4] Downloading pre-trained models...")
    for model in MODELS:
        if os.path.exists(model["filename"]):
            size_mb = os.path.getsize(model["filename"]) / (1024 * 1024)
            print(f"  ✓ {model['name']} already exists ({size_mb:.1f} MB)")
        else:
            print(f"  → {model['name']}")
            download_file(model["url"], model["filename"])

    # ── Step 3: Clone YOLOv12 repo ───────────────────────────────
    print("\n[3/4] Setting up YOLOv12 architecture...")
    clone_yolov12()

    # ── Step 4: Check custom emotion model ───────────────────────
    print("\n[4/4] Checking custom emotion model...")
    if os.path.exists(EMOTION_MODEL):
        size_mb = os.path.getsize(EMOTION_MODEL) / (1024 * 1024)
        print(f"  ✓ {EMOTION_MODEL} found ({size_mb:.1f} MB)")
    else:
        print(f"  ⚠ {EMOTION_MODEL} NOT FOUND!")
        print(f"    This is the custom-trained emotion classification model.")
        print(f"    The application will still run but emotion detection will")
        print(f"    be disabled. See README for how to obtain or train it.")

    # ── Summary ──────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Setup complete!")
    print("=" * 60)
    print("\n  To start the application:")
    print("    Windows:  run.bat")
    print("    Any OS:   python web/app.py")
    print(f"\n  Then open http://127.0.0.1:5000 in your browser.")
    print()


if __name__ == "__main__":
    main()
