import streamlit as st
import numpy as np
import cv2
import os
import time
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from PIL import Image

# =============================================================================
# PAGE CONFIG
# =============================================================================

st.set_page_config(
    page_title="SmartVision AI",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# TITLE
# =============================================================================

st.title("🔍 SmartVision AI")
st.caption("Multi-Class Image Classification + YOLOv8 Object Detection")

# =============================================================================
# CLASS LABELS
# =============================================================================

CLASSES = [
    "person","bicycle","car","motorcycle","airplane","bus","train","truck",
    "traffic light","stop sign","bench","bird","cat","dog","horse","cow",
    "elephant","bottle","cup","bowl","pizza","cake","chair","couch",
    "potted plant","bed"
]

# =============================================================================
# LOAD CLASSIFICATION MODELS
# =============================================================================

@st.cache_resource
def load_classification_models():

    import tensorflow as tf

    models = {}

    model_paths = {
        "VGG16": "models/VGG16_best.keras",
        "ResNet50": "models/ResNet50_best.keras",
        "MobileNetV2": "models/MobileNetV2_best.keras",
        "EfficientNetB0": "models/EfficientNetB0_best.keras"
    }

    for model_name, path in model_paths.items():

        if os.path.exists(path):

            try:
                models[model_name] = tf.keras.models.load_model(path)

            except Exception as e:

                st.error(f"Failed loading {model_name}: {e}")

    return models

# =============================================================================
# LOAD YOLO MODEL
# =============================================================================

@st.cache_resource
def load_yolo_model():

    from ultralytics import YOLO

    possible_paths = [
        "models/yolo_best.pt"
    ]

    for path in possible_paths:

        if os.path.exists(path):

            return YOLO(path)

    return None

# =============================================================================
# IMAGE PREPROCESSING
# =============================================================================

def preprocess_image(image):

    image = image.resize((224, 224))

    image = image.convert("RGB")

    image_array = np.array(image).astype("float32") / 255.0

    image_array = np.expand_dims(image_array, axis=0)

    return image_array

# =============================================================================
# CLASSIFICATION
# =============================================================================

def predict_image(image_array, model):

    predictions = model.predict(image_array, verbose=0)[0]

    top5_indices = predictions.argsort()[-5:][::-1]

    results = []

    for idx in top5_indices:

        class_name = CLASSES[idx]

        confidence = float(predictions[idx])

        results.append((class_name, confidence))

    return results

# =============================================================================
# DRAW YOLO DETECTIONS
# =============================================================================

def draw_detections(image, results, threshold=0.3):

    image_np = np.array(image)

    fig, ax = plt.subplots(figsize=(10, 8))

    ax.imshow(image_np)

    detected_objects = []

    cmap = plt.cm.get_cmap("tab20", len(CLASSES))

    if results and len(results) > 0:

        boxes = results[0].boxes

        if boxes is not None:

            for box in boxes:

                conf = float(box.conf[0])

                if conf < threshold:
                    continue

                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

                class_id = int(box.cls[0])

                class_name = (
                    CLASSES[class_id]
                    if class_id < len(CLASSES)
                    else str(class_id)
                )

                color = cmap(class_id / len(CLASSES))

                rect = patches.Rectangle(
                    (x1, y1),
                    x2 - x1,
                    y2 - y1,
                    linewidth=2,
                    edgecolor=color,
                    facecolor="none"
                )

                ax.add_patch(rect)

                ax.text(
                    x1,
                    max(10, y1 - 5),
                    f"{class_name} {conf:.2f}",
                    color="white",
                    fontsize=9,
                    bbox=dict(
                        facecolor=color,
                        alpha=0.8,
                        boxstyle="round,pad=0.2"
                    )
                )

                detected_objects.append({
                    "class": class_name,
                    "confidence": conf
                })

    ax.axis("off")

    return fig, detected_objects

# =============================================================================
# SIDEBAR
# =============================================================================

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Go to",
    [
        "🏠 Home",
        "📷 Image Classification",
        "🎯 Object Detection",
        "ℹ️ About"
    ]
)

# =============================================================================
# HOME PAGE
# =============================================================================

if page == "🏠 Home":

    st.header("Welcome to SmartVision AI")

    st.markdown("""
    SmartVision AI is a deep learning application supporting:

    - CNN Image Classification
    - YOLOv8 Object Detection
    - Real-time Predictions
    - Performance Analysis
    """)

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Classes", "25")
    col2.metric("CNN Models", "4")
    col3.metric("Detection", "YOLOv8")
    col4.metric("Framework", "TensorFlow")

# =============================================================================
# IMAGE CLASSIFICATION PAGE
# =============================================================================

elif page == "📷 Image Classification":

    st.header("📷 Image Classification")

    uploaded_file = st.file_uploader(
        "Upload Image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:

        image = Image.open(uploaded_file).convert("RGB")

        col1, col2 = st.columns([1, 2])

        with col1:

            st.image(image, caption="Uploaded Image")

        with col2:

            models = load_classification_models()

            if len(models) == 0:

                st.warning("No models found.")

            else:

                image_array = preprocess_image(image)

                for model_name, model in models.items():

                    start = time.time()

                    predictions = predict_image(image_array, model)

                    inference_time = (time.time() - start) * 1000

                    top_class, top_conf = predictions[0]

                    with st.expander(
                        f"{model_name} → {top_class} ({top_conf:.1%}) | {inference_time:.0f} ms"
                    ):

                        for class_name, confidence in predictions:

                            st.progress(
                                float(confidence),
                                text=f"{class_name}: {confidence:.1%}"
                            )

# =============================================================================
# OBJECT DETECTION PAGE
# =============================================================================

elif page == "🎯 Object Detection":

    st.header("🎯 YOLOv8 Object Detection")

    threshold = st.slider(
        "Confidence Threshold",
        0.1,
        0.9,
        0.3,
        0.05
    )

    uploaded_file = st.file_uploader(
        "Upload Image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:

        image = Image.open(uploaded_file).convert("RGB")

        yolo_model = load_yolo_model()

        if yolo_model is None:

            st.error("YOLO model not found.")

        else:

            with st.spinner("Running detection..."):

                start = time.time()

                results = yolo_model.predict(
                    image,
                    conf=threshold,
                    verbose=False
                )

                inference_time = (time.time() - start) * 1000

            fig, detected_objects = draw_detections(
                image,
                results,
                threshold
            )

            st.pyplot(fig)

            st.success(f"Inference Time: {inference_time:.0f} ms")

            st.markdown(f"### Detected Objects: {len(detected_objects)}")

            for obj in detected_objects:

                st.write(
                    f"• {obj['class']} ({obj['confidence']:.2%})"
                )


# =============================================================================
# ABOUT PAGE
# =============================================================================

elif page == "ℹ️ About":

    st.header("ℹ️ About")

    st.markdown("""
    ### SmartVision AI

    Deep learning computer vision system built using:

    - TensorFlow / Keras
    - Transfer Learning
    - YOLOv8
    - Streamlit

    ### Models Used

    - VGG16
    - ResNet50
    - MobileNetV2
    - EfficientNetB0

    ### Features

    - Image Classification
    - Object Detection
    - Performance Dashboard
    - Real-time Predictions
    """)