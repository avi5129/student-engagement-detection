import os
import shutil
import random
import requests
import zipfile

def prepare_expanded_data(target_root, val_split=0.2):
    """
    Download and merge datasets into a robust structure.
    Using Zenodo (16,000 images) as it doesn't require authentication.
    """
    os.makedirs(target_root, exist_ok=True)
    
    # 1. Download Zenodo Dataset (E-learning Student Engagement)
    # DOI: 10.5281/zenodo.16731734
    zenodo_url = "https://zenodo.org/records/16731734/files/Final%20Dataset%20256.zip?download=1"
    zenodo_zip = "zenodo_dataset.zip"
    
    if not os.path.exists(zenodo_zip):
        print(f"Downloading Zenodo dataset (191MB)...")
        r = requests.get(zenodo_url, stream=True)
        with open(zenodo_zip, 'wb') as f:
            for chunk in r.iter_content(chunk_size=1024*1024):
                f.write(chunk)
        print("Download complete.")
    
    zenodo_extract_dir = "zenodo_extracted"
    if not os.path.exists(zenodo_extract_dir):
        print("Extracting Zenodo dataset...")
        with zipfile.ZipFile(zenodo_zip, 'r') as zip_ref:
            zip_ref.extractall(zenodo_extract_dir)
        print("Extraction complete.")

    # Unified classes for training
    classes = ["Bored", "Confused", "Drowsy", "Focused", "Frustrated", "Looking Away"]
    for split in ['train', 'val']:
        for cls in classes:
            os.makedirs(os.path.join(target_root, split, cls), exist_ok=True)

    def copy_split(src_dir, cls_name, limit=None):
        if not os.path.exists(src_dir):
            print(f"Warning: {src_dir} not found.")
            return
        images = [f for f in os.listdir(src_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        random.shuffle(images)
        if limit:
            images = images[:limit]
        
        split_idx = int(len(images) * (1 - val_split))
        train_imgs = images[:split_idx]
        val_imgs = images[split_idx:]
        
        for img in train_imgs:
            shutil.copy(os.path.join(src_dir, img), os.path.join(target_root, 'train', cls_name, img))
        for img in val_imgs:
            shutil.copy(os.path.join(src_dir, img), os.path.join(target_root, 'val', cls_name, img))

    # 1. Existing Student Concentration Dataset (Original labels)
    # Re-using the local cache path I found earlier
    path_sc = r"C:\Users\Avishkar\.cache\kagglehub\datasets\programmer3\student-concentration-image-dataset\versions\1"
    sc_dataset_base = os.path.join(path_sc, "Student Dataset", "Student-engagement-dataset")
    mapping_sc = {
        os.path.join("Engaged", "Confused"): "Confused",
        os.path.join("Engaged", "Focused"): "Focused",
        os.path.join("Engaged", "Frustrated"): "Frustrated",
        os.path.join("Not Engaged", "Bored"): "Bored",
        os.path.join("Not Engaged", "Drowsy"): "Drowsy",
        os.path.join("Not Engaged", "Looking Away"): "Looking Away"
    }
    for src, dest in mapping_sc.items():
        copy_split(os.path.join(sc_dataset_base, src), dest)

    # 2. Zenodo mapping (Engagement -> Focused, Disengagement -> Bored/Looking Away)
    # The structure in Final Dataset 256.zip is Engagement/Disengagement -> Age -> Gender
    # We will flatten and map:
    # Engagement -> Focused
    # Disengagement -> Looking Away
    
    parent_dir = os.path.join(zenodo_extract_dir, "Final Dataset 256")
    if not os.path.exists(parent_dir):
        # Fallback to search for the directory if nested
        for root, dirs, files in os.walk(zenodo_extract_dir):
            if "Engagement" in dirs and "Disengagement" in dirs:
                parent_dir = root
                break

    print(f"Mapping Zenodo data from: {parent_dir}")
    mapping_zenodo = {
        "Engagement": "Focused",
        "Disengagement": "Looking Away"
    }
    
    for state, target_cls in mapping_zenodo.items():
        state_path = os.path.join(parent_dir, state)
        if not os.path.exists(state_path): continue
        
        # Walk through age and gender folders
        for root, dirs, files in os.walk(state_path):
            img_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            if not img_files: continue
            
            random.shuffle(img_files)
            # Take a good amount to augment our classes (e.g., 2000 per class from Zenodo)
            img_files = img_files[:2000]
            
            split_idx = int(len(img_files) * (1 - val_split))
            for i, img in enumerate(img_files):
                split = 'train' if i < split_idx else 'val'
                dest_path = os.path.join(target_root, split, target_cls, f"zen_{i}_{img}")
                shutil.copy(os.path.join(root, img), dest_path)

    # Show counts
    print("\nFinal Dataset Counts:")
    for split in ['train', 'val']:
        print(f"--- {split} ---")
        split_dir = os.path.join(target_root, split)
        for cls in os.listdir(split_dir):
            count = len(os.listdir(os.path.join(split_dir, cls)))
            print(f"{cls}: {count}")

if __name__ == "__main__":
    TARGET = os.path.join(os.getcwd(), "data_prepared_expanded_final")
    prepare_expanded_data(TARGET)
