import sys
import os
import numpy as np
import cv2

sys.path.append(os.getcwd())
from src.emotion import EmotionClassifier

def test():
    print("Testing EmotionClassifier instantiation...")
    clf = EmotionClassifier("yolo_emotion_model_best.pt")
    
    if clf.model is None:
        print("Failed to load model.")
        sys.exit(1)
        
    print("Testing predict function with real image...")
    img_path = os.path.join(os.getcwd(), 'data_prepared_expanded_final', 'val', 'Focused', '0006.jpg')
    img = cv2.imread(img_path)
    if img is None:
        print(f"Could not load {img_path}")
        sys.exit(1)
        
    res = clf.predict(img)
    print(f"Prediction result for real image (Expected Focus): {res}")
    
    if res in clf.labels or res == "Unknown":
        print("Test passed successfully.")
    else:
        print("Test failed: Unexpected output.")
        sys.exit(1)

if __name__ == "__main__":
    test()
