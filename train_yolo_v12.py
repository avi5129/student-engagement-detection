import os
import sys
import argparse

# Add the cloned yolov12 repo to sys.path to use its custom ultralytics package
yolo_path = os.path.join(os.getcwd(), 'yolov12')
if yolo_path not in sys.path:
    sys.path.insert(0, yolo_path)

from ultralytics import YOLO

def train_yolo_v12(data_dir, epochs=50, batch_size=32):
    """
    Train using the official YOLOv12-cls architecture.
    """
    model_weights = 'yolov12n-cls.pt'
    if not os.path.exists(model_weights):
        print(f"Error: {model_weights} not found. Please run download_yolo_weights.py first.")
        return
    
    # Initialize YOLOv12 model
    print(f"Initializing YOLOv12 with weights: {model_weights}")
    model = YOLO(model_weights)
    
    # Start Training with heavy data augmentation
    print(f"Starting training on: {data_dir} with advanced augmentation...")
    results = model.train(
        data=data_dir,
        epochs=epochs,
        imgsz=224,
        batch=batch_size,
        patience=15, 
        save=True,
        # Advanced augmentation for better accuracy with smaller datasets
        augment=True,
        degrees=15.0, # Rotate
        translate=0.1, # Shift
        scale=0.1, # Zoom
        shear=5.0, # Shear
        flipud=0.0,
        fliplr=0.5, # Horizontal flip is good for faces
        mosaic=1.0, # Mosaic augmentation
        mixup=0.1, # MixUp augmentation
        project='runs/yolov12_emotion',
        name='v12_heavy_aug'
    )
    
    print(f"Training completed. Results saved in: {results.save_dir}")
    
    # Export for deployment - ONNX is standard
    export_path = model.export(format='onnx')
    print(f"Model exported to: {export_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLOv12 Emotion Detection model.")
    parser.add_argument('--data_dir', type=str, default='data_prepared_expanded_final', help='Path to dataset directory')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch', type=int, default=32, help='Batch size')
    
    args = parser.parse_args()
    
    # Ensure dataset exists before starting
    train_dir = os.path.join(args.data_dir, 'train')
    if not os.path.exists(train_dir):
        print(f"Error: Dataset directory '{train_dir}' not found.")
        sys.exit(1)
        
    train_yolo_v12(args.data_dir, args.epochs, args.batch)
