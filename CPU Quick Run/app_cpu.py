# app.py
import os
import cv2
import numpy as np
import random
from ultralytics import YOLO
import streamlit as st

# --------------------------
# CONFIG
# --------------------------
SAVED_MODEL_PATH = "saved_models/teeth_model.pt"
IMG_SIZE = (1024, 512)
DEVICE = "cpu"  # Use CPU only

# FDI labels per quadrant
FDI_QUADRANTS = {
    "UR": ["18","17","16","15","14","13","12","11"],
    "UL": ["21","22","23","24","25","26","27","28"],
    "LL": ["31","32","33","34","35","36","37","38"],
    "LR": ["41","42","43","44","45","46","47","48"]
}

# Optional: consistent colors for each tooth
COLOR_MAP = {fdi: [random.randint(0,255) for _ in range(3)] 
             for quad in FDI_QUADRANTS.values() for fdi in quad}

# --------------------------
# Helper functions
# --------------------------
def assign_quadrant(cx, cy, img_w, img_h):
    """Assign tooth to a quadrant based on its center coordinates"""
    if cy < img_h / 2:  # Upper
        return "UR" if cx >= img_w / 2 else "UL"
    else:  # Lower
        return "LR" if cx >= img_w / 2 else "LL"

# --------------------------
# Streamlit App
# --------------------------
def demo_app(weights_path=SAVED_MODEL_PATH):
    st.title("Teeth Detection & Segmentation Demo")
    st.write("""
    Upload an OPG dental X-ray to get teeth detection masks and bounding boxes.  
    **Disclaimer:** This project is for demonstration and educational purposes only.  
    It is **NOT a medical diagnostic tool**.
    """)

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        img_path = "temp.jpg"
        with open(img_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Load YOLO model
        model = YOLO(weights_path)
        results = model.predict(img_path, imgsz=IMG_SIZE, conf=0.25, device=DEVICE)

        # Load original image
        annotated_img = cv2.imread(img_path)
        img_h, img_w = annotated_img.shape[:2]

        if len(results[0].masks) > 0:
            masks = results[0].masks.data.cpu().numpy()  # (num_teeth, H_mask, W_mask)
            boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)

            # Assign teeth to quadrants
            quadrant_teeth = {"UR": [], "UL": [], "LL": [], "LR": []}
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = box
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                quadrant = assign_quadrant(cx, cy, img_w, img_h)
                quadrant_teeth[quadrant].append((i, cx))

            # Sort teeth in each quadrant
            for q in quadrant_teeth:
                if q in ["UR", "LR"]:  # sort descending x
                    quadrant_teeth[q].sort(key=lambda x: -x[1])
                else:  # sort ascending x
                    quadrant_teeth[q].sort(key=lambda x: x[1])

            # Overlay masks, boxes, and FDI labels
            for q, teeth in quadrant_teeth.items():
                labels = FDI_QUADRANTS[q]
                for idx, (i, _) in enumerate(teeth):
                    mask = masks[i]
                    mask_resized = cv2.resize(mask, (img_w, img_h))
                    mask_resized = (mask_resized > 0.5).astype(np.uint8)

                    # Use consistent color if available
                    label = labels[idx] if idx < len(labels) else str(i+1)
                    color = COLOR_MAP.get(label, [random.randint(0,255) for _ in range(3)])

                    # Create colored mask
                    colored_mask = np.zeros_like(annotated_img)
                    for c in range(3):
                        colored_mask[:, :, c] = mask_resized * color[c]

                    # Overlay mask
                    annotated_img = cv2.addWeighted(annotated_img, 1.0, colored_mask, 0.5, 0)

                    # Draw thin bounding box
                    x1, y1, x2, y2 = boxes[i]
                    cv2.rectangle(annotated_img, (x1, y1), (x2, y2), color, 1)  # thin line

                    # Draw smaller FDI label
                    cv2.putText(annotated_img, label, (x1, y1-3),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # Convert BGR to RGB for Streamlit
        annotated_img_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
        st.image(annotated_img_rgb, caption="Predicted Teeth Masks + Boxes", use_column_width=True)
        st.success("Inference Complete!")

# --------------------------
# Run
# --------------------------
if __name__ == "__main__":
    demo_app()
