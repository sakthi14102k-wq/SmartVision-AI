SmartVision AI - Intelligent Multi-Class Object Recognition System
Overview

SmartVision AI is an end-to-end Computer Vision and Deep Learning project designed for intelligent multi-class image classification and real-time object detection. The project combines Transfer Learning-based CNN architectures with YOLOv8 to build a scalable AI-powered visual recognition system capable of handling multiple real-world object categories.

Project Architecture
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


Tech Stack
Programming Language
Python
Deep Learning Frameworks
TensorFlow
PyTorch
Computer Vision
OpenCV
YOLOv8 (Ultralytics)
Deep Learning Models
VGG16
ResNet50
MobileNetV2
EfficientNetB0
Deployment
Streamlit
Hugging Face Spaces
Data Processing & Visualization
NumPy
Pandas
Matplotlib
Seaborn
Scikit-learn


Dataset
Dataset Name

COCO 2017 Dataset - 25 Class Subset

Dataset Source
COCO Official Dataset
Hugging Face COCO Repository

SmartVision-AI/
│
├── main.py
├── requirements.txt
├── README.md
├── Smartvision.ipynb
│
├── notebooks/
│   ├── EDA.ipynb
│   ├── VGG16.ipynb
│   ├── ResNet50.ipynb
│   ├── MobileNetV2.ipynb
│   ├── EfficientNetB0.ipynb
│   └── YOLOv8.ipynb
│
└── models/
   ├── vgg16_model.h5
   ├── resnet50_model.h5
   ├── mobilenet_model.h5
   ├── efficientnet_model.h5
   └── yolov8_model.pt


