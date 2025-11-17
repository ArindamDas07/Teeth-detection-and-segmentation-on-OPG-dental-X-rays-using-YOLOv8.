# eval_yolo.py
from ultralytics import YOLO

SAVED_MODEL_PATH = "saved_models/teeth_model.pt"

def evaluate_model(weights_path=SAVED_MODEL_PATH):
    model = YOLO(weights_path)
    print("Evaluating model on validation set...")
    metrics = model.val()
    print(metrics)

if __name__ == "__main__":
    evaluate_model()
