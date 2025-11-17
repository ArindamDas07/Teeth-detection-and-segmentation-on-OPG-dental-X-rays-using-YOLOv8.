# train_yolo.py
import os
from ultralytics import YOLO

# --------------------------
# CONFIGURATION
# --------------------------
DATA_YAML = "teeth.yaml"       # must point to your teeth.yaml
IMG_SIZE = 1024                # single integer
BATCH_SIZE = 4
EPOCHS = 30                  # adjust as needed
DEVICE = 0
SAVED_MODEL_PATH = "saved_models/teeth_model.pt"

os.makedirs("saved_models", exist_ok=True)

# --------------------------
# TRAIN YOLOv8-SEG
# --------------------------
def train_yolo():
    print("Starting YOLOv8-seg training...")

    # Load small YOLOv8 segmentation model
    model = YOLO("yolov8s-seg.pt")  

    # Train
    model.train(
        data=DATA_YAML,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        epochs=EPOCHS,
        device=DEVICE,
        project="runs/segment",
        name="teeth_train",
        exist_ok=True
    )

    # Save trained model
    model.save(SAVED_MODEL_PATH)
    print(f"Model saved at {SAVED_MODEL_PATH}")
    return model

if __name__ == "__main__":
    train_yolo()
