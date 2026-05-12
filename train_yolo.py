import os
import argparse
from ultralytics import YOLO

def train_model(data_dir, epochs=50, batch_size=32, model_type='yolov11n-cls'):
    """
    Train a YOLO classification model for emotion detection.
    """
    # Load the model
    # Note: If yolov12n-cls exists in the ultralytics repository, it will be downloaded automatically.
    # Otherwise, it falls back to yolov11n-cls.
    print(f"Initializing YOLO model: {model_type}")
    try:
        model = YOLO(f"{model_type}.pt")
    except Exception as e:
        print(f"Error loading {model_type}: {e}")
        print("Falling back to yolov11n-cls...")
        model = YOLO("yolov11n-cls.pt")
    
    # Training
    print(f"Starting training on data from: {data_dir}")
    results = model.train(
        data=data_dir,
        epochs=epochs,
        imgsz=224,
        batch=batch_size,
        patience=10, # Early stopping
        save=True,
        # Optimizer: 'AdamW' is default and often best for YOLO
        # We can also add augmentation if needed, though YOLO already includes it.
    )
    
    # Export for deployment
    # Exporting to ONNX or PyTorch
    export_path = model.export(format='onnx')
    print(f"Model exported to: {export_path}")
    
    # Re-save best.pt to a consistent location
    best_path = os.path.join(results.save_dir, 'weights/best.pt')
    shutil_target = 'yolo_emotion_model_best.pt'
    if os.path.exists(best_path):
        import shutil
        shutil.copy(best_path, shutil_target)
        print(f"Best model weight saved to: {shutil_target}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a YOLOv12 classification model.")
    parser.add_argument('--data_dir', type=str, required=True, help='Path to dataset directory (with train/val subfolders)')
    parser.add_argument('--model', type=str, default='yolov12n-cls', help='YOLO model variant (e.g., yolov12n-cls, yolov11n-cls)')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch', type=int, default=32, help='Batch size')
    
    args = parser.parse_args()
    train_model(args.data_dir, args.epochs, args.batch, args.model)
