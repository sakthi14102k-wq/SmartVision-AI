# SmartVision AI - Intelligent Multi-Class Object Recognition System

SmartVision AI is an end-to-end Computer Vision and Deep Learning project designed for intelligent multi-class image classification and real-time object detection.

The project combines Transfer Learning-based CNN architectures with YOLOv8 to build a scalable AI-powered visual recognition system capable of handling multiple real-world object categories.

---

## 🚀 Project Architecture

```text
COCO Dataset
↓
Data Filtering & Preprocessing
↓
EDA & Visualization
↓
Transfer Learning Models
(VGG16 / ResNet50 / MobileNetV2 / EfficientNetB0)
↓
YOLOv8 Object Detection
↓
Model Evaluation & Comparison
↓
Inference Pipeline
↓
Streamlit Web Application
↓
Hugging Face Deployment
```

---

## 🛠️ Tech Stack

- Python
- TensorFlow
- PyTorch
- OpenCV
- YOLOv8
- Streamlit
- Hugging Face

---

## 🤖 Models Used

- VGG16
- ResNet50
- MobileNetV2
- EfficientNetB0
- YOLOv8

---

## 📂 Dataset

COCO 2017 Dataset - 25 Class Subset

---

## ▶️ Run Project

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## 📁 Project Structure

```text
SmartVision-AI/
│
├── models/
│   ├── yolo_training/
│   ├── EfficientNetB0_best.keras
│   ├── MobileNetV2_best.keras
│   ├── ResNet50_best.keras
│   ├── VGG16_best.keras
│   ├── yolo_best.pt
│   └── yolov8n.pt
│
├── notebooks/
│
├── .gitignore
├── README.md
├── Smartvision.ipynb
├── main.py
└── requirements.txt
```
