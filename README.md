# Project SHIELD: Smart Helmet Identification and Enforcement Using Deep Learning

A state-of-the-art computer vision project for real-time monitoring and predicting helmet compliance among two-wheeler riders using Deep Learning. 

The project involves Data Augmentation, Model Optimization, and YOLO-based Object Detection, with the final model deployed as an interactive Streamlit web application.

## Live Demo
👉 [Open the Project SHIELD App](#) *(Add your Streamlit link here)*

## Problem Statement
Road traffic accidents involving two-wheelers frequently result in severe injuries or fatalities due to non-compliance with helmet safety rules. Manual enforcement is labor-intensive and inefficient. 

The objective of this project is to develop a deep learning object detection model that automatically identifies riders, passengers, and their helmet compliance status from images and video feeds.

The project detects:
- Drivers with and without helmets
- Passengers with and without helmets
- Two-wheeler vehicles (Bikes)

Extensive data augmentation and model optimization were performed to ensure robust detection under varying traffic conditions.

## Dataset
The dataset utilized is the `smart-helmet-detection-using-dl` dataset from Kaggle.
- **Total Classes:** 7
- **Training Set:** 368 images
- **Validation Set:** 52 images
- **Resolution:** Resized to 640x640 for training

### Target Classes
| Class Name | Meaning |
|---|---|
| `driver_with_helmet` | Rider wearing a helmet |
| `driver_without_helmet` | Rider NOT wearing a helmet |
| `passenger_with_helemt` | Pillion rider wearing a helmet |
| `passenger_without_helemt` | Pillion rider NOT wearing a helmet |
| `bike` | The two-wheeler vehicle |
| `driver` | Rider (General) |
| `passenger` | Pillion (General) |

## Data Preprocessing & Augmentation
To improve the model's generalization capabilities on real-world traffic cameras, dynamic augmentations were applied during training using Albumentations:
- **Color & Blur Augmentations:** `Blur`, `MedianBlur`, `ToGray`, and `CLAHE` (Contrast Limited Adaptive Histogram Equalization).
- **Geometric Augmentations:** `MixUp (0.15)`, `Copy-Paste (0.2)`.

## Deep Learning Model
Multiple object detection architectures were considered, and the **YOLO (You Only Look Once) architecture** (`yolo26s.pt`) was selected for the final workflow due to its optimal balance between real-time inference speed and accuracy. 
- **Network Size:** 260 layers
- **Parameters:** ~9.95 Million
- **GFLOPs:** 22.8

## Model Performance
The model was trained for **50 epochs** and achieved the following overall validation metrics on the best epoch:

| Metric | Score |
|---|---|
| **Precision (P)** | 79.50% (0.795) |
| **Recall (R)** | 78.20% (0.782) |
| **mAP@50** | 83.90% (0.839) |
| **mAP@50-95** | 40.10% (0.401) |

*Class-specific highlights: Detection of the core `bike` class achieved an mAP@50 of 88.5%, and `driver_without_helmet` achieved 92.1% mAP@50.*

### Visualizing Performance

**Training Curves:**
*(Loss reduction and metric improvements over 50 epochs)*
![Training Curves](image_5e57f9.jpg)

**Confusion Matrix:**
*(Demonstrating classification accuracy across all 7 target and background classes)*
![Confusion Matrix](image_5e57ff.jpg)

## Hyperparameter Tuning & Configuration
Hyperparameter tuning was performed to control model complexity and improve bounding box precision.
The optimized training configuration:
- `optimizer`: AdamW
- `learning_rate (lr0)`: 0.001
- `momentum`: 0.937
- `weight_decay`: 0.0005
- `batch_size`: 16
- `imgsz`: 640

The optimization successfully stabilized the loss functions (box_loss, cls_loss) and prevented overfitting.

## Deployment Workflow
The trained weights (`best.pt`) were exported and integrated into a Python backend.

```text
Traffic Image/Video Input
      ↓
Python Streamlit Backend
      ↓
Headless OpenCV Processing
      ↓
YOLOv8 Inference Engine
      ↓
Compliance Summary & Bounding Boxes Output

Project-SHIELD/
│
├── app.py                     # Main Streamlit application
├── requirements.txt           # Python library dependencies
├── packages.txt               # Linux system dependencies (libGL for Streamlit Cloud)
├── best.pt                    # Fine-tuned YOLO model weights
├── README.md                  # Project documentation
└── assets/                    # Project images and visualizations

git clone [https://github.com/Trinadh555/Project-SHIELD-Smart-Helmet-Identification-and-Enforcement-Using-Deep-Learning.git](https://github.com/Trinadh555/Project-SHIELD-Smart-Helmet-Identification-and-Enforcement-Using-Deep-Learning.git)
cd Project-SHIELD-Smart-Helmet-Identification-and-Enforcement-Using-Deep-Learning

pip install -r requirements.txt

streamlit run app.py

