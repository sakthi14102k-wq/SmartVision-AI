import time
from pathlib import Path

import cv2
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms


# Basic setup
st.set_page_config(page_title="SmartVision AI", page_icon="🔍", layout="wide")

# ---------------------------------------------------------
# PROJECT PATH
# ---------------------------------------------------------
# The training notebooks save CNN weights inside:
# models/vgg16_best.pth
# models/resnet50_best.pth
# models/mobilenet_best.pth
# models/efficientnet_b0_best.pth
#
# This keeps the code simple and works when app.py is either:
#   Project/app.py
# or:
#   Project/src/app.py

APP_DIR = Path(__file__).resolve().parent

if (APP_DIR / "models").exists():
    PROJECT_DIR = APP_DIR
elif (APP_DIR.parent / "models").exists():
    PROJECT_DIR = APP_DIR.parent
else:
    PROJECT_DIR = APP_DIR

MODELS_DIR = PROJECT_DIR / "models"

VGG_PATH = MODELS_DIR / "vgg16_best.pth"
RESNET_PATH = MODELS_DIR / "resnet50_best.pth"
MOBILENET_PATH = MODELS_DIR / "mobilenet_best.pth"
EFFICIENTNET_PATH = MODELS_DIR / "efficientnet_b0_best.pth"

# Exact path used by the YOLO training notebook
YOLO_PATH = (
    PROJECT_DIR
    / "src"
    / "runs"
    / "detect"
    / "smartvision_yolo-3"
    / "weights"
    / "best.pt"
)

# Simple fallback only if the exact training path is not present
if not YOLO_PATH.exists():
    YOLO_PATH = (
        PROJECT_DIR
        / "runs"
        / "detect"
        / "smartvision_yolo-3"
        / "weights"
        / "best.pt"
    )

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
yolo_device = 0 if torch.cuda.is_available() else "cpu"


# Same class order used by ImageFolder during training
class_names = [
    "airplane", "bed", "bench", "bicycle", "bird", "bottle", "bowl",
    "bus", "cake", "car", "cat", "chair", "couch", "cow", "cup", "dog",
    "elephant", "horse", "motorcycle", "person", "pizza", "potted_plant",
    "stop_sign", "traffic_light", "train", "truck"
]


# Same validation preprocessing used in the CNN notebooks
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


@st.cache_resource
def load_cnn(model_name):
    """Load one trained classification model."""

    if model_name == "VGG16":
        model = models.vgg16(weights=None)
        model.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        model.classifier = nn.Sequential(
            nn.Flatten(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.4),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),
            nn.Linear(128, 26)
        )
        weight_path = VGG_PATH

    elif model_name == "ResNet50":
        model = models.resnet50(weights=None)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(512, 26)
        )
        weight_path = RESNET_PATH

    elif model_name == "MobileNetV2":
        model = models.mobilenet_v2(weights=None)
        model.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(1280, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, 26)
        )
        weight_path = MOBILENET_PATH

    elif model_name == "EfficientNetB0":
        model = models.efficientnet_b0(weights=None)
        model.classifier = nn.Sequential(
            nn.Linear(1280, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 26)
        )
        weight_path = EFFICIENTNET_PATH

    else:
        return None

    if not weight_path.exists():
        return None

    try:
        state_dict = torch.load(weight_path, map_location=device, weights_only=True)
    except TypeError:
        state_dict = torch.load(weight_path, map_location=device)

    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    # Handles weights saved with DataParallel as well.
    state_dict = {
        key.replace("module.", ""): value
        for key, value in state_dict.items()
    }

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return model


@st.cache_resource
def load_yolo():
    """Load the trained YOLOv8 model."""

    if not YOLO_PATH.exists():
        return None

    from ultralytics import YOLO
    return YOLO(str(YOLO_PATH))


# Sidebar
st.sidebar.title("🔍 SmartVision AI")

page = st.sidebar.radio(
    "Navigation",
    ["Home", "Image Classification", "Object Detection", "Model Performance", "About"]
)

st.sidebar.divider()
st.sidebar.write("Device:", "GPU" if device.type == "cuda" else "CPU")

with st.sidebar.expander("Model Paths"):
    st.write("Project folder:")
    st.code(str(PROJECT_DIR))

    st.write("Models folder:")
    st.code(str(MODELS_DIR))

    st.write("VGG16:", "Found" if VGG_PATH.exists() else "Missing")
    st.write("ResNet50:", "Found" if RESNET_PATH.exists() else "Missing")
    st.write("MobileNetV2:", "Found" if MOBILENET_PATH.exists() else "Missing")
    st.write("EfficientNetB0:", "Found" if EFFICIENTNET_PATH.exists() else "Missing")
    st.write("YOLO:", "Found" if YOLO_PATH.exists() else "Missing")


# -------------------------------------------------------------------
# HOME
# -------------------------------------------------------------------

if page == "Home":
    st.title("SmartVision AI")
    st.write("Simple image classification and object detection application.")

    col1, col2, col3 = st.columns(3)
    col1.metric("CNN Models", 4)
    col2.metric("Classes", 26)
    col3.metric("Detector", "YOLOv8n")

    st.subheader("Project Models")

    models_df = pd.DataFrame({
        "Model": ["VGG16", "ResNet50", "MobileNetV2", "EfficientNetB0", "YOLOv8n"],
        "Task": [
            "Classification",
            "Classification",
            "Classification",
            "Classification",
            "Object Detection"
        ]
    })

    st.dataframe(models_df, use_container_width=True, hide_index=True)

    st.subheader("How to use")
    st.write("• Use **Image Classification** for a single main object.")
    st.write("• Use **Object Detection** for multiple objects in one image.")
    st.write("• Use **Model Performance** to view the recorded model results.")


# -------------------------------------------------------------------
# IMAGE CLASSIFICATION
# -------------------------------------------------------------------

elif page == "Image Classification":
    st.title("Image Classification")
    st.write("Upload an image and choose a classification model.")

    selected_model = st.selectbox(
        "Select model",
        ["Compare All", "VGG16", "ResNet50", "MobileNetV2", "EfficientNetB0"]
    )

    uploaded_file = st.file_uploader(
        "Upload image",
        type=["jpg", "jpeg", "png", "webp"]
    )

    if uploaded_file is None:
        st.info("Upload an image to start.")
        st.stop()

    image = Image.open(uploaded_file).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)

    image_col, result_col = st.columns([1, 2])

    with image_col:
        st.image(image, caption="Uploaded Image", use_container_width=True)

    # Compare all four classification models
    if selected_model == "Compare All":
        rows = []

        with result_col:
            st.subheader("Model Comparison")

            for model_name in ["VGG16", "ResNet50", "MobileNetV2", "EfficientNetB0"]:
                model = load_cnn(model_name)

                if model is None:
                    rows.append({
                        "Model": model_name,
                        "Prediction": "Weights not found",
                        "Confidence": "-",
                        "Time (ms)": "-"
                    })
                    continue

                if device.type == "cuda":
                    torch.cuda.synchronize()

                start = time.perf_counter()

                with torch.no_grad():
                    output = model(input_tensor)
                    probabilities = torch.softmax(output, dim=1)[0]

                if device.type == "cuda":
                    torch.cuda.synchronize()

                inference_ms = (time.perf_counter() - start) * 1000

                confidence, class_id = torch.max(probabilities, dim=0)
                prediction = class_names[class_id.item()].replace("_", " ").title()

                rows.append({
                    "Model": model_name,
                    "Prediction": prediction,
                    "Confidence": f"{confidence.item():.2%}",
                    "Time (ms)": f"{inference_ms:.2f}"
                })

            st.dataframe(
                pd.DataFrame(rows),
                use_container_width=True,
                hide_index=True
            )

    # Run one selected model and show top 5
    else:
        model = load_cnn(selected_model)

        with result_col:
            if model is None:
                st.error(f"{selected_model} weights were not found.")
                st.stop()

            if device.type == "cuda":
                torch.cuda.synchronize()

            start = time.perf_counter()

            with torch.no_grad():
                output = model(input_tensor)
                probabilities = torch.softmax(output, dim=1)[0]

            if device.type == "cuda":
                torch.cuda.synchronize()

            inference_ms = (time.perf_counter() - start) * 1000

            top_values, top_indices = torch.topk(probabilities, 5)

            prediction = class_names[top_indices[0].item()].replace("_", " ").title()
            confidence = top_values[0].item()

            st.subheader("Result")
            st.metric("Prediction", prediction, f"{confidence:.2%}")
            st.caption(f"Inference time: {inference_ms:.2f} ms")

            top5 = []

            for score, class_id in zip(
                top_values.cpu().tolist(),
                top_indices.cpu().tolist()
            ):
                top5.append({
                    "Class": class_names[class_id].replace("_", " ").title(),
                    "Confidence": f"{score:.2%}"
                })

            st.subheader("Top 5 Predictions")
            st.dataframe(
                pd.DataFrame(top5),
                use_container_width=True,
                hide_index=True
            )


# -------------------------------------------------------------------
# OBJECT DETECTION
# -------------------------------------------------------------------

elif page == "Object Detection":
    st.title("YOLOv8 Object Detection")
    st.write("Upload an image to detect multiple objects.")

    confidence_threshold = st.slider(
        "Confidence threshold",
        min_value=0.01,
        max_value=0.90,
        value=0.10,
        step=0.01
    )

    uploaded_file = st.file_uploader(
        "Upload image",
        type=["jpg", "jpeg", "png", "webp"],
        key="detection"
    )

    if uploaded_file is None:
        st.info("Upload an image to start detection.")
        st.stop()

    yolo_model = load_yolo()

    if yolo_model is None:
        st.error("YOLO best.pt was not found.")
        st.code(str(YOLO_PATH))
        st.stop()

    st.caption(f"YOLO model: {YOLO_PATH}")

    with st.expander("YOLO model check"):
        st.write("Number of model classes:", len(yolo_model.names))
        st.write("Classes:", list(yolo_model.names.values()))

    image = Image.open(uploaded_file).convert("RGB")

    with st.spinner("Detecting objects..."):
        start = time.perf_counter()

        results = yolo_model.predict(
            source=image,
            conf=confidence_threshold,
            imgsz=640,
            max_det=1,          # Show only the highest-confidence object
            device=yolo_device,
            verbose=False
        )

        inference_ms = (time.perf_counter() - start) * 1000

    result = results[0]

    detected_image = result.plot()
    detected_image = cv2.cvtColor(detected_image, cv2.COLOR_BGR2RGB)

    st.image(
        detected_image,
        caption="Detected Objects",
        use_container_width=True
    )

    detections = []

    if result.boxes is not None:
        for box in result.boxes:
            class_id = int(box.cls[0].item())
            confidence = float(box.conf[0].item())
            detections.append({
                "Object": result.names[class_id].title(),
                "Confidence": f"{confidence:.2%}"
            })

    metric1, metric2 = st.columns(2)
    metric1.metric("Objects Detected", len(detections))
    metric2.metric("Inference Time", f"{inference_ms:.2f} ms")

    if detections:
        st.subheader("Detection Details")
        st.dataframe(
            pd.DataFrame(detections),
            use_container_width=True,
            hide_index=True
        )
    else:
        st.warning(
            "No objects detected above the selected confidence threshold."
        )


# -------------------------------------------------------------------
# MODEL PERFORMANCE
# -------------------------------------------------------------------

elif page == "Model Performance":
    st.title("Model Performance")
    st.write("Recorded validation results from the training notebooks.")

    st.subheader("Classification")

    classification_results = pd.DataFrame({
        "Model": ["VGG16", "ResNet50", "MobileNetV2", "EfficientNetB0"],
        "Best Validation Accuracy": ["69.74%", "77.44%", "74.62%", "81.54%"]
    })

    st.dataframe(
        classification_results,
        use_container_width=True,
        hide_index=True
    )

    st.subheader("YOLOv8 Detection")

    yolo_results = pd.DataFrame({
        "Metric": ["Precision", "Recall", "mAP@0.5", "mAP@0.5:0.95"],
        "Value": ["0.9169", "0.8693", "0.8869", "0.5479"]
    })

    st.dataframe(yolo_results, use_container_width=True, hide_index=True)

    st.caption(
        "These are saved training/validation results. "
        "This page does not retrain or reevaluate the models."
    )


# -------------------------------------------------------------------
# ABOUT
# -------------------------------------------------------------------

elif page == "About":
    st.title("About SmartVision AI")

    st.write(
        "SmartVision AI combines four transfer-learning classification "
        "models with YOLOv8 object detection."
    )

    st.subheader("Classification Models")
    st.write("VGG16, ResNet50, MobileNetV2 and EfficientNetB0")

    st.subheader("Object Detector")
    st.write("YOLOv8n")

    st.subheader("Supported Classes")
    readable_names = [name.replace("_", " ").title() for name in class_names]
    st.write(", ".join(readable_names))

    st.subheader("Libraries")
    st.write(
        "Streamlit, PyTorch, TorchVision, Ultralytics YOLOv8, "
        "OpenCV, Pillow and Pandas"
    )