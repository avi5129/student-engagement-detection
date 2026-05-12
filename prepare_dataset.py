import os
import shutil
import random

def prepare_data(source_root, target_root, val_split=0.2):
    """
    Flatten the dataset into a single level of class folders and split into train/val.
    """
    dataset_path = os.path.join(source_root, "Student Dataset", "Student-engagement-dataset")
    classes = {
        "Engaged": ["Confused", "Focused", "Frustrated"],
        "Not Engaged": ["Bored", "Drowsy", "Looking Away"]
    }
    
    # Create target directories
    for split in ['train', 'val']:
        for cat in classes:
            for subcat in classes[cat]:
                os.makedirs(os.path.join(target_root, split, subcat), exist_ok=True)

    for cat, subcats in classes.items():
        for subcat in subcats:
            subcat_path = os.path.join(dataset_path, cat, subcat)
            if not os.path.exists(subcat_path):
                print(f"Warning: Path {subcat_path} does not exist.")
                continue
            
            images = [f for f in os.listdir(subcat_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            random.shuffle(images)
            split_idx = int(len(images) * (1 - val_split))
            
            train_images = images[:split_idx]
            val_images = images[split_idx:]
            
            print(f"Copying {len(train_images)} train and {len(val_images)} val images for {subcat}...")
            
            for img in train_images:
                shutil.copy(os.path.join(subcat_path, img), os.path.join(target_root, 'train', subcat, img))
            for img in val_images:
                shutil.copy(os.path.join(subcat_path, img), os.path.join(target_root, 'val', subcat, img))

if __name__ == "__main__":
    SOURCE = r"C:\Users\Avishkar\.cache\kagglehub\datasets\programmer3\student-concentration-image-dataset\versions\1"
    TARGET = os.path.join(os.getcwd(), "data_prepared")
    prepare_data(SOURCE, TARGET)
    print(f"Dataset prepared at: {TARGET}")
