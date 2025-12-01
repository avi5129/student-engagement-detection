from deepface import DeepFace
import cv2
import numpy as np

class EmotionClassifier:
    def __init__(self):
        """
        Initialize the Emotion Classifier using DeepFace.
        """
        # DeepFace loads models on first call, but we can pre-build to save time
        try:
            print("Initializing DeepFace Emotion model...")
            # This might download weights if not present
            # We run a dummy prediction to load the model
            dummy_img = np.zeros((48, 48, 3), dtype=np.uint8)
            DeepFace.analyze(dummy_img, actions=['emotion'], enforce_detection=False)
            print("DeepFace initialized.")
        except Exception as e:
            print(f"DeepFace init warning: {e}")

    def predict(self, image):
        """
        Predict emotion from an image (can be a crop or full frame).
        :param image: Image (numpy array).
        :return: Predicted emotion label (str).
        """
        if image is None or image.size == 0:
            return "Unknown"
            
        try:
            # DeepFace expects BGR (OpenCV default) or RGB.
            # enforce_detection=False because we are passing a crop that might be small or partial
            # detector_backend='skip' because we already detected the face/person
            results = DeepFace.analyze(image, actions=['emotion'], 
                                     enforce_detection=False, 
                                     detector_backend='skip',
                                     silent=True)
            
            if results:
                # results is a list of dicts
                return results[0]['dominant_emotion'].capitalize()
            
            return "Unknown"
        except Exception as e:
            # print(f"Prediction error: {e}")
            return "Unknown"
