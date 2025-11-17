import os
import shutil
import random
from pathlib import Path
import math


# Source folders
SOURCE_IMG_DIR = Path('augmented/images')
SOURCE_LBL_DIR = Path('augmented/labels')

# Destination folders
DEST_DIR = Path('HandSigns_v2')

# Split ratio
SPLIT_RATIO = 0.8

# DESTINATION PATHS
TRAIN_IMG_DIR = DEST_DIR / 'train' / 'images'
TRAIN_LBL_DIR = DEST_DIR / 'train' / 'labels'
VALID_IMG_DIR = DEST_DIR / 'valid' / 'images'
VALID_LBL_DIR = DEST_DIR / 'valid' / 'labels'

# CREATE DESTINATION FOLDERS ---
for path in [TRAIN_IMG_DIR, TRAIN_LBL_DIR, VALID_IMG_DIR, VALID_LBL_DIR]:
    path.mkdir(parents=True, exist_ok=True)

# --- Helper function to move a file pair ---
def copy_file_pair(filename, dest_img_dir, dest_lbl_dir):
    """
    Moves an image and its corresponding label file.
    """
    base_name = Path(filename).stem
    label_name = f"{base_name}.txt"

    # Source paths
    img_src = SOURCE_IMG_DIR / filename
    lbl_src = SOURCE_LBL_DIR / label_name

    # Destination paths
    img_dest = dest_img_dir / filename
    lbl_dest = dest_lbl_dir / label_name

    # Move the image
    if img_src.exists():
        shutil.copy(str(img_src), str(img_dest))
    else:
        print(f"Warning: Image file not found {img_src}, skipping.")
        return False

    # Move the label (if it exists)
    if lbl_src.exists():
        shutil.copy(str(lbl_src), str(lbl_dest))
    else:
        print(f"Warning: Label file not found {lbl_src} for image {filename}.")
    
    return True

def main():
    print("Finding all images...")
    # Get all image filenames
    image_files = [f for f in os.listdir(SOURCE_IMG_DIR) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    
    if not image_files:
        print(f"Error: No images found in {SOURCE_IMG_DIR}")
        return

    print(f"Found {len(image_files)} images. Shuffling list...")
    
    # Shuffle the list in-place
    random.shuffle(image_files)

    # Calculate the split point
    split_index = math.ceil(len(image_files) * SPLIT_RATIO)

    # Split the list
    train_files = image_files[:split_index]
    valid_files = image_files[split_index:]

    print(f"Total files: {len(image_files)}")
    print(f"Training files: {len(train_files)}")
    print(f"Validation files: {len(valid_files)}")

    # Move training files
    print("\nMoving training files...")
    for filename in train_files:
        copy_file_pair(filename, TRAIN_IMG_DIR, TRAIN_LBL_DIR)

    # Move validation files
    print("\nMoving validation files...")
    for filename in valid_files:
        copy_file_pair(filename, VALID_IMG_DIR, VALID_LBL_DIR)

    print("\n--- Dataset split complete! ---")
    print(f"The '{SOURCE_IMG_DIR.parent}' folder should now be empty.")
    print(f"Your 'HandSigns_v1' dataset is ready for training.")

if __name__ == '__main__':
    main()