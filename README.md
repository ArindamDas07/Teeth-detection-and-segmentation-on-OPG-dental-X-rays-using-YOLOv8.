# Teeth Detection and Segmentation on OPG Dental X-Rays using YOLOv8

Detecting and segmenting teeth in OPG dental X-rays using YOLOv8 for demonstration and educational purposes.  
**⚠️ Disclaimer:** This project is **not a medical diagnostic tool**.

**GitHub Short Description:**  
Teeth detection & segmentation in dental X-rays (YOLOv8, demo project).

---

## Dataset

Download the dataset from Kaggle:  
[Teeth Segmentation on Dental X-Ray Images](https://www.kaggle.com/datasets/humansintheloop/teeth-segmentation-on-dental-x-ray-images/data)

**Steps to prepare the dataset:**

1. Extract the downloaded dataset.
2. Go inside `Teeth Segmentation JSON` → `d2` folder.
3. Copy `ann` and `img` folders into a new folder named `data`.
4. Place the `data` folder in your project folder.

---

## Project Setup

```bash
# Create project folder
mkdir teeth_segmentation_yolov8
cd teeth_segmentation_yolov8

# Create virtual environment (Python 3.10)
python -m venv venv

# Activate virtual environment
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

Prepare Data for YOLOv8
# In the project folder, run:
python yolo_data.py
This will create the proper YOLOv8 dataset format


Training
# Train YOLOv8 model
python train_yolo.py

# Evaluate the model
python eval_yolo.py


Demo / Inference
# Run the Streamlit demo
streamlit run app.py
Demo video: demo.mp4 is included for reference.

Environment: Ultralytics 8.3.228, Python 3.10.0, torch 2.5.1+cu121, CUDA:0 (NVIDIA GeForce RTX 3050 A Laptop GPU, 4094MiB)
