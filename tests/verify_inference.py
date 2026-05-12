import numpy as np
import cv2
import os
import sys

# Add the project root to sys.path
sys.path.append(os.getcwd())

from src.emotion import EmotionClassifier

def test_inference():
    # 1. Initialize Classifier
    print("Testing EmotionClassifier integration...")
    classifier = EmotionClassifier()
    
    if classifier.model is None:
        print("FAILED: Model could not be loaded.")
        return

    # 2. Create a dummy image (simulating a face crop)
    dummy_face = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
    
    # 3. Predict
    print("Running prediction on dummy image...")
    try:
        prediction = classifier.predict(dummy_face)
        print(f"SUCCESS: Predicted emotion: {prediction}")
        
        if prediction in classifier.labels:
            print("Validation Passed: Prediction is within expected labels.")
        else:
            print(f"Validation WARNING: Unexpected prediction result: {prediction}")
            
    except Exception as e:
        print(f"FAILED: Prediction error: {e}")

if __name__ == "__main__":
    test_inference()
