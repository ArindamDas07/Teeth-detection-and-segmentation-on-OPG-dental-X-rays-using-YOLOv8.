import os
import json
import numpy as np
import cv2
from tqdm import tqdm

DATASET_DIR = "data"
IMG_DIR = os.path.join(DATASET_DIR, "img")
ANN_DIR = os.path.join(DATASET_DIR, "ann")
OUTPUT_DIR = "yolo_dataset"
IMG_SIZE = (1024, 512)  # optional, YOLOv8 can handle resizing internally

# Create YOLO folder structure
for split in ["train", "val", "test"]:
    os.makedirs(os.path.join(OUTPUT_DIR, "images", split), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "labels", split), exist_ok=True)

# Split dataset
all_images = [f for f in os.listdir(IMG_DIR) if f.endswith(".jpg")]
np.random.shuffle(all_images)
num_train = int(len(all_images) * 0.7)
num_val = int(len(all_images) * 0.15)

splits = {
    "train": all_images[:num_train],
    "val": all_images[num_train:num_train+num_val],
    "test": all_images[num_train+num_val:]
}

for split_name, images in splits.items():
    for img_file in tqdm(images, desc=f"Processing {split_name}"):
        # Load & resize image (optional)
        img_path = os.path.join(IMG_DIR, img_file)
        img = cv2.imread(img_path)
        h_original, w_original = img.shape[:2]
        img_resized = cv2.resize(img, IMG_SIZE)
        save_img_path = os.path.join(OUTPUT_DIR, "images", split_name, img_file)
        cv2.imwrite(save_img_path, img_resized)

        # Process JSON annotation
        ann_file = os.path.join(ANN_DIR, img_file + ".json")
        yolo_lines = []
        if os.path.exists(ann_file):
            with open(ann_file) as f:
                data = json.load(f)
            for obj in data['objects']:
                poly = obj['points']['exterior']
                # Scale to resized image
                scaled_poly = [[x*IMG_SIZE[0]/data['size']['width'], 
                                y*IMG_SIZE[1]/data['size']['height']] 
                               for [x, y] in poly]
                # Normalize coordinates
                norm_poly = [[x/IMG_SIZE[0], y/IMG_SIZE[1]] for [x, y] in scaled_poly]
                # Flatten for YOLO
                flat_poly = [str(coord) for point in norm_poly for coord in point]
                # Assuming single class 0 (tooth)
                line = "0 " + " ".join(flat_poly)
                yolo_lines.append(line)

        # Save label file
        label_path = os.path.join(OUTPUT_DIR, "labels", split_name, img_file.replace(".jpg", ".txt"))
        with open(label_path, "w") as f:
            f.write("\n".join(yolo_lines))
