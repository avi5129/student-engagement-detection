import cv2
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))

from emotion import EmotionClassifier

def main():
    classifier = EmotionClassifier('custom_emotion_model.h5')
    
    test_data_dir = os.path.join(os.getcwd(), 'data_prepared', 'val')
    if not os.path.exists(test_data_dir):
        print(f"Test data directory {test_data_dir} not found.")
        return

    print("\nRunning verification on sample images...")
    for label in os.listdir(test_data_dir):
        label_dir = os.path.join(test_data_dir, label)
        if not os.path.isdir(label_dir):
            continue
            
        # Get one image for each label
        images = [f for f in os.listdir(label_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if not images:
            continue
            
        test_img_path = os.path.join(label_dir, images[0])
        img = cv2.imread(test_img_path)
        
        prediction = classifier.predict(img)
        print(f"Label: {label:<15} | Predicted: {prediction:<15} | {'SUCCESS' if label == prediction else 'FAILURE'}")

if __name__ == "__main__":
    main()
