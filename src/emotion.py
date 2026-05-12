import sys
import os

# Add local yolov12 path to support YOLOv12 architecture
yolo_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'yolov12')
if yolo_path not in sys.path:
    # Safely insert the local ultralytics directory at the front to override any installed pip version
    sys.path.insert(0, yolo_path)

from ultralytics import YOLO
import cv2
import numpy as np

class EmotionClassifier:
    def __init__(self, model_path='yolo_emotion_model_best.pt'):
        """
        Initialize the Emotion Classifier using a custom trained YOLOv12 model.
        """
        self.labels = ["Bored", "Confused", "Drowsy", "Focused", "Frustrated", "Looking Away"]
        
        # Ensure model_path is relative to the project root
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.model_path = os.path.join(project_root, model_path)
        self.model = None

        try:
            if os.path.exists(self.model_path):
                print(f"Loading custom YOLOv12 emotion model from {self.model_path}...")
                self.model = YOLO(self.model_path)
                print("Model loaded successfully.")
            else:
                print(f"Warning: Custom YOLO model {self.model_path} not found. Emotion classification will be disabled.")
        except Exception as e:
            print(f"Error loading model: {e}")

    def predict(self, image):
        """
        Predict emotion from an image (can be a crop or full frame).
        :param image: Image (numpy array).
        :return: Predicted emotion label (str).
        """
        if self.model is None or image is None or image.size == 0:
            return "Unknown"

        try:
            # YOLO predict wrapper handles the image format directly (BGR from OpenCV is expected)
            results = self.model.predict(image, verbose=False, imgsz=224)
            
            if len(results) > 0:
                # Get index of the top1 prediction
                top1_index = int(results[0].probs.top1)
                # YOLO auto maps index to class name assuming metadata is stored, but we can double check
                class_name = results[0].names[top1_index]
                return class_name
                
        except Exception as e:
            # print(f"Prediction error: {e}")
            pass
            
        return "Unknown"
