from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn

from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="SmartVision AI",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =============================================================================
# CONSTANTS
# =============================================================================

APP_FILE = Path(__file__).resolve()
APP_DIR = APP_FILE.parent

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

YOLO_DEVICE: int | str = (
    0 if torch.cuda.is_available() else "cpu"
)

CLASS_NAMES = [
    "airplane",
    "bed",
    "bench",
    "bicycle",
    "bird",
    "bottle",
    "bowl",
    "bus",
    "cake",
    "car",
    "cat",
    "chair",
    "couch",
    "cow",
    "cup",
    "dog",
    "elephant",
    "horse",
    "motorcycle",
    "person",
    "pizza",
    "potted plant",
    "stop sign",
    "traffic light",
    "train",
    "truck",
]

MODEL_FILENAMES = {
    "VGG16": [
        "VGG16_best.pth",
    ],
    "ResNet50": [
        "ResNet50_best.pth",
    ],
    "MobileNetV2": [
        "mobilenetv2_final.pth",
        "MobileNetV2_best.pth",
    ],
    "EfficientNetB0": [
        "efficientnetb0_best.pth",
        "efficientnetb0_final.pth",
        "EfficientNetB0_best.pth",
    ],
}


# =============================================================================
# PROJECT PATHS
# =============================================================================

def find_project_root() -> Path:
    candidates = [
        APP_DIR,
        APP_DIR.parent,
        APP_DIR.parent.parent,
        Path.cwd().resolve(),
        Path.cwd().resolve().parent,
    ]

    for candidate in candidates:
        dataset_folder = (
            candidate
            / "smartvision_dataset"
        )

        if dataset_folder.exists():
            return candidate.resolve()

    for candidate in candidates:
        try:
            matches = list(
                candidate.rglob(
                    "smartvision_dataset"
                )
            )
        except PermissionError:
            continue

        for match in matches:
            if match.is_dir():
                return match.parent.resolve()

    return APP_DIR


PROJECT_ROOT = find_project_root()

CLASSIFICATION_DATA_DIR = (
    PROJECT_ROOT
    / "smartvision_dataset"
    / "classification"
)

DETECTION_DATA_DIR = (
    PROJECT_ROOT
    / "smartvision_dataset"
    / "detection"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "streamlit"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


def find_file(filename: str) -> Path | None:
    direct_paths = [
        PROJECT_ROOT / "models" / filename,
        PROJECT_ROOT / "MODEL" / "models" / filename,
        APP_DIR / "models" / filename,
        APP_DIR.parent / "models" / filename,
        Path.cwd().resolve() / "models" / filename,
    ]

    for path in direct_paths:
        if path.exists():
            return path.resolve()

    search_roots = [
        PROJECT_ROOT,
        APP_DIR,
    ]

    for root in search_roots:
        if not root.exists():
            continue

        try:
            for path in root.rglob(filename):
                if path.is_file():
                    return path.resolve()
        except PermissionError:
            continue

    return None


def get_classification_model_paths() -> dict[str, Path | None]:
    model_paths: dict[str, Path | None] = {}

    for model_name, filenames in MODEL_FILENAMES.items():
        selected_path = None

        for filename in filenames:
            selected_path = find_file(filename)

            if selected_path is not None:
                break

        model_paths[model_name] = selected_path

    return model_paths


def find_yolo_model_path() -> Path | None:
    preferred_paths = [
        PROJECT_ROOT
        / "runs"
        / "detect"
        / "smartvision_yolo"
        / "weights"
        / "best.pt",

        PROJECT_ROOT
        / "MODEL"
        / "runs"
        / "detect"
        / "smartvision_yolo"
        / "weights"
        / "best.pt",

        PROJECT_ROOT
        / "models"
        / "YOLO26"
        / "weights"
        / "best.pt",

        PROJECT_ROOT
        / "models"
        / "yolo_best.pt",
    ]

    for path in preferred_paths:
        if path.exists():
            return path.resolve()

    if PROJECT_ROOT.exists():
        try:
            matches = list(
                PROJECT_ROOT.rglob("best.pt")
            )
        except PermissionError:
            matches = []

        for path in matches:
            lower_path = str(path).lower()

            if (
                path.is_file()
                and "weights" in lower_path
                and (
                    "yolo" in lower_path
                    or "detect" in lower_path
                )
            ):
                return path.resolve()

    return find_file("yolo_best.pt")


# =============================================================================
# MODEL ARCHITECTURES
# =============================================================================

def build_vgg16(
    num_classes: int,
) -> nn.Module:
    model = models.vgg16(
        weights=None
    )

    model.avgpool = (
        nn.AdaptiveAvgPool2d((1, 1))
    )

    model.classifier = nn.Sequential(
        nn.Flatten(),
        nn.BatchNorm1d(512),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, num_classes),
    )

    return model


def build_resnet50(
    num_classes: int,
) -> nn.Module:
    model = models.resnet50(
        weights=None
    )

    input_features = (
        model.fc.in_features
    )

    model.fc = nn.Sequential(
        nn.BatchNorm1d(input_features),
        nn.Dropout(0.5),
        nn.Linear(
            input_features,
            256,
        ),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(
            256,
            num_classes,
        ),
    )

    return model


def build_mobilenetv2(
    num_classes: int,
) -> nn.Module:
    model = models.mobilenet_v2(
        weights=None
    )

    input_features = (
        model.classifier[1].in_features
    )

    model.classifier = nn.Sequential(
        nn.Linear(
            input_features,
            256,
        ),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(
            256,
            num_classes,
        ),
    )

    return model


def build_efficientnetb0(
    num_classes: int,
) -> nn.Module:
    model = models.efficientnet_b0(
        weights=None
    )

    input_features = (
        model.classifier[1].in_features
    )

    model.classifier = nn.Sequential(
        nn.BatchNorm1d(input_features),
        nn.Linear(
            input_features,
            256,
        ),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(
            256,
            num_classes,
        ),
    )

    return model


MODEL_BUILDERS = {
    "VGG16": build_vgg16,
    "ResNet50": build_resnet50,
    "MobileNetV2": build_mobilenetv2,
    "EfficientNetB0": build_efficientnetb0,
}


# =============================================================================
# MODEL LOADING
# =============================================================================

def load_state_dictionary(
    model_path: Path,
) -> dict[str, Any]:
    try:
        checkpoint = torch.load(
            model_path,
            map_location=DEVICE,
            weights_only=True,
        )
    except TypeError:
        checkpoint = torch.load(
            model_path,
            map_location=DEVICE,
        )

    if (
        isinstance(checkpoint, dict)
        and "state_dict" in checkpoint
    ):
        checkpoint = checkpoint[
            "state_dict"
        ]

    if not isinstance(checkpoint, dict):
        raise TypeError(
            "Invalid PyTorch state dictionary."
        )

    cleaned_state = {}

    for key, value in checkpoint.items():
        cleaned_key = key.removeprefix(
            "module."
        )
        cleaned_state[
            cleaned_key
        ] = value

    return cleaned_state


@st.cache_resource(
    show_spinner=False
)
def load_classification_models():
    loaded_models: dict[
        str,
        nn.Module
    ] = {}

    load_messages: dict[
        str,
        str
    ] = {}

    model_paths = (
        get_classification_model_paths()
    )

    for model_name, model_path in (
        model_paths.items()
    ):
        if model_path is None:
            load_messages[
                model_name
            ] = "Model file not found."
            continue

        try:
            model = MODEL_BUILDERS[
                model_name
            ](
                len(CLASS_NAMES)
            )

            state_dictionary = (
                load_state_dictionary(
                    model_path
                )
            )

            model.load_state_dict(
                state_dictionary
            )

            model.to(DEVICE)
            model.eval()

            loaded_models[
                model_name
            ] = model

            load_messages[
                model_name
            ] = (
                f"Loaded from {model_path}"
            )

        except Exception as error:
            load_messages[
                model_name
            ] = (
                f"Loading failed: {error}"
            )

    return (
        loaded_models,
        load_messages,
        model_paths,
    )


@st.cache_resource(
    show_spinner=False
)
def load_yolo_model():
    model_path = (
        find_yolo_model_path()
    )

    if model_path is None:
        return (
            None,
            None,
            "YOLO best.pt was not found.",
        )

    try:
        from ultralytics import YOLO

        model = YOLO(
            str(model_path)
        )

        return (
            model,
            model_path,
            "",
        )

    except Exception as error:
        return (
            None,
            model_path,
            str(error),
        )


# =============================================================================
# IMAGE PREPROCESSING AND CLASSIFICATION
# =============================================================================

CLASSIFICATION_TRANSFORM = (
    transforms.Compose([
        transforms.Resize(
            (224, 224)
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[
                0.485,
                0.456,
                0.406,
            ],
            std=[
                0.229,
                0.224,
                0.225,
            ],
        ),
    ])
)


def predict_classification(
    image: Image.Image,
    model: nn.Module,
):
    tensor = (
        CLASSIFICATION_TRANSFORM(
            image.convert("RGB")
        )
        .unsqueeze(0)
        .to(DEVICE)
    )

    if DEVICE.type == "cuda":
        torch.cuda.synchronize()

    start_time = (
        time.perf_counter()
    )

    with torch.inference_mode():
        logits = model(tensor)

        probabilities = torch.softmax(
            logits,
            dim=1,
        )[0]

    if DEVICE.type == "cuda":
        torch.cuda.synchronize()

    inference_ms = (
        time.perf_counter()
        - start_time
    ) * 1000

    top_values, top_indices = (
        torch.topk(
            probabilities,
            k=min(
                5,
                len(CLASS_NAMES),
            ),
        )
    )

    predictions = []

    for score, index in zip(
        top_values.cpu().tolist(),
        top_indices.cpu().tolist(),
    ):
        predictions.append({
            "class": CLASS_NAMES[
                index
            ],
            "confidence": float(
                score
            ),
        })

    return (
        predictions,
        inference_ms,
    )


# =============================================================================
# DATASET UTILITIES
# =============================================================================

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}


def find_sample_images(
    limit: int = 4,
) -> list[Path]:
    candidate_folders = [
        DETECTION_DATA_DIR
        / "images",

        CLASSIFICATION_DATA_DIR
        / "test",
    ]

    sample_images = []

    for folder in candidate_folders:
        if not folder.exists():
            continue

        for path in folder.rglob("*"):
            if (
                path.is_file()
                and path.suffix.lower()
                in IMAGE_EXTENSIONS
            ):
                sample_images.append(
                    path
                )

            if (
                len(sample_images)
                >= limit
            ):
                return sample_images

    return sample_images


def dataset_summary() -> pd.DataFrame:
    rows = []

    for split in [
        "train",
        "val",
        "test",
    ]:
        split_folder = (
            CLASSIFICATION_DATA_DIR
            / split
        )

        if split_folder.exists():
            image_count = sum(
                1
                for path
                in split_folder.rglob("*")
                if (
                    path.is_file()
                    and path.suffix.lower()
                    in IMAGE_EXTENSIONS
                )
            )

            class_count = sum(
                1
                for path
                in split_folder.iterdir()
                if path.is_dir()
            )

        else:
            image_count = 0
            class_count = 0

        rows.append({
            "Dataset": "Classification",
            "Split": split,
            "Images": image_count,
            "Classes": class_count,
            "Label Files": "-",
        })

    detection_images = (
        DETECTION_DATA_DIR
        / "images"
    )

    detection_labels = (
        DETECTION_DATA_DIR
        / "labels"
    )

    detection_image_count = (
        sum(
            1
            for path
            in detection_images.iterdir()
            if (
                path.is_file()
                and path.suffix.lower()
                in IMAGE_EXTENSIONS
            )
        )
        if detection_images.exists()
        else 0
    )

    detection_label_count = (
        len(
            list(
                detection_labels.glob(
                    "*.txt"
                )
            )
        )
        if detection_labels.exists()
        else 0
    )

    rows.append({
        "Dataset": "Detection",
        "Split": "all",
        "Images": detection_image_count,
        "Classes": len(CLASS_NAMES),
        "Label Files": (
            detection_label_count
        ),
    })

    return pd.DataFrame(rows)


# =============================================================================
# MODEL EVALUATION
# =============================================================================

def evaluate_classification_models(
    loaded_models: dict[
        str,
        nn.Module
    ],
    batch_size: int = 32,
):
    test_folder = (
        CLASSIFICATION_DATA_DIR
        / "test"
    )

    if not test_folder.exists():
        raise FileNotFoundError(
            "Classification test dataset "
            f"not found: {test_folder}"
        )

    test_dataset = datasets.ImageFolder(
        test_folder,
        transform=(
            CLASSIFICATION_TRANSFORM
        ),
    )

    if (
        test_dataset.classes
        != CLASS_NAMES
    ):
        raise ValueError(
            "Test-folder class order does "
            "not match the trained models.\n\n"
            f"Expected:\n{CLASS_NAMES}\n\n"
            f"Found:\n{test_dataset.classes}"
        )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    model_paths = (
        get_classification_model_paths()
    )

    summary_rows = []
    confusion_matrices = {}
    class_reports = {}

    for (
        model_name,
        model,
    ) in loaded_models.items():
        all_true = []
        all_predicted = []

        top5_correct = 0
        total_samples = 0

        if DEVICE.type == "cuda":
            torch.cuda.synchronize()

        start_time = (
            time.perf_counter()
        )

        with torch.inference_mode():
            for (
                images,
                labels,
            ) in test_loader:
                images = images.to(
                    DEVICE
                )
                labels = labels.to(
                    DEVICE
                )

                outputs = model(
                    images
                )

                predicted_classes = (
                    outputs.argmax(
                        dim=1
                    )
                )

                top5_indices = (
                    outputs.topk(
                        min(
                            5,
                            len(
                                CLASS_NAMES
                            ),
                        ),
                        dim=1,
                    ).indices
                )

                top5_correct += (
                    (
                        top5_indices
                        == labels.unsqueeze(1)
                    )
                    .any(dim=1)
                    .sum()
                    .item()
                )

                total_samples += (
                    labels.size(0)
                )

                all_true.extend(
                    labels
                    .cpu()
                    .tolist()
                )

                all_predicted.extend(
                    predicted_classes
                    .cpu()
                    .tolist()
                )

        if DEVICE.type == "cuda":
            torch.cuda.synchronize()

        elapsed_seconds = (
            time.perf_counter()
            - start_time
        )

        accuracy = accuracy_score(
            all_true,
            all_predicted,
        )

        precision = precision_score(
            all_true,
            all_predicted,
            average="weighted",
            zero_division=0,
        )

        recall = recall_score(
            all_true,
            all_predicted,
            average="weighted",
            zero_division=0,
        )

        f1 = f1_score(
            all_true,
            all_predicted,
            average="weighted",
            zero_division=0,
        )

        top5_accuracy = (
            top5_correct
            / total_samples
            if total_samples
            else 0.0
        )

        inference_ms = (
            elapsed_seconds
            / total_samples
            * 1000
            if total_samples
            else 0.0
        )

        model_path = model_paths.get(
            model_name
        )

        model_size_mb = (
            model_path.stat().st_size
            / (1024 * 1024)
            if model_path is not None
            else 0.0
        )

        summary_rows.append({
            "Model": model_name,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1-Score": f1,
            "Top-5 Accuracy": (
                top5_accuracy
            ),
            "Inference ms/image": (
                inference_ms
            ),
            "Model Size MB": (
                model_size_mb
            ),
        })

        confusion_matrices[
            model_name
        ] = confusion_matrix(
            all_true,
            all_predicted,
            labels=list(
                range(
                    len(CLASS_NAMES)
                )
            ),
        )

        report = (
            classification_report(
                all_true,
                all_predicted,
                labels=list(
                    range(
                        len(
                            CLASS_NAMES
                        )
                    )
                ),
                target_names=(
                    CLASS_NAMES
                ),
                output_dict=True,
                zero_division=0,
            )
        )

        class_reports[
            model_name
        ] = (
            pd.DataFrame(
                report
            )
            .transpose()
            .loc[CLASS_NAMES]
            .reset_index()
            .rename(
                columns={
                    "index": "Class"
                }
            )
        )

    summary = (
        pd.DataFrame(
            summary_rows
        )
        .sort_values(
            "Accuracy",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )

    performance_folder = (
        OUTPUT_DIR
        / "model_performance"
    )

    performance_folder.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary.to_csv(
        performance_folder
        / "model_summary.csv",
        index=False,
    )

    return {
        "summary": summary,
        "confusion_matrices": (
            confusion_matrices
        ),
        "class_reports": (
            class_reports
        ),
        "test_size": (
            len(test_dataset)
        ),
    }


# =============================================================================
# SIDEBAR NAVIGATION
# =============================================================================

st.sidebar.title(
    "🔍 SmartVision AI"
)

st.sidebar.caption(
    "PyTorch Classification + "
    "YOLOv8 Detection"
)

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Home",
        "📷 Image Classification",
        "🎯 Object Detection",
        "📊 Model Performance",
        "ℹ️ About",
    ],
)

st.sidebar.divider()

st.sidebar.write(
    "Execution device:",
    (
        "GPU"
        if DEVICE.type == "cuda"
        else "CPU"
    ),
)


# =============================================================================
# PAGE 1: HOME
# =============================================================================

if page == "🏠 Home":
    st.title("🔍 SmartVision AI")

    st.caption(
        "Multi-Class Image Classification "
        "and YOLOv8 Object Detection"
    )

    st.header(
        "Project Overview"
    )

    st.write(
        "SmartVision AI is a computer-vision "
        "application that combines four "
        "transfer-learning classification "
        "models with a YOLOv8 object detector."
    )

    metric_col1, metric_col2, metric_col3, metric_col4 = (
        st.columns(4)
    )

    metric_col1.metric(
        "Object Classes",
        len(CLASS_NAMES),
    )

    metric_col2.metric(
        "CNN Models",
        4,
    )

    metric_col3.metric(
        "Object Detector",
        "YOLOv8",
    )

    metric_col4.metric(
        "Framework",
        "PyTorch",
    )

    st.subheader(
        "Key Features"
    )

    feature_col1, feature_col2 = (
        st.columns(2)
    )

    with feature_col1:
        st.markdown(
            """
            - Single-object image classification
            - Four CNN model predictions
            - Top-5 confidence scores
            - Side-by-side comparison
            """
        )

    with feature_col2:
        st.markdown(
            """
            - Multi-object detection
            - Bounding boxes and labels
            - Adjustable confidence threshold
            - Model-performance dashboard
            """
        )

    st.subheader(
        "Instructions"
    )

    st.markdown(
        """
        1. Use **Image Classification** for one main object.
        2. Use **Object Detection** for multiple objects.
        3. Use **Model Performance** to compare CNN models.
        4. Use **About** for project documentation.
        """
    )

    st.subheader(
        "Model Status"
    )

    classification_paths = (
        get_classification_model_paths()
    )

    yolo_path = (
        find_yolo_model_path()
    )

    status_rows = []

    for (
        model_name,
        model_path,
    ) in classification_paths.items():
        status_rows.append({
            "Component": model_name,
            "Status": (
                "Ready"
                if model_path
                else "Missing"
            ),
            "Path": (
                str(model_path)
                if model_path
                else "Not found"
            ),
        })

    status_rows.append({
        "Component": "YOLOv8",
        "Status": (
            "Ready"
            if yolo_path
            else "Missing"
        ),
        "Path": (
            str(yolo_path)
            if yolo_path
            else "Not found"
        ),
    })

    st.dataframe(
        pd.DataFrame(
            status_rows
        ),
        use_container_width=True,
        hide_index=True,
    )



# =============================================================================
# PAGE 2: IMAGE CLASSIFICATION
# =============================================================================

elif page == "📷 Image Classification":
    st.title(
        "📷 Image Classification"
    )

    st.write(
        "Upload an image containing one "
        "main object. Every available CNN "
        "model returns its top-5 predictions."
    )

    uploaded_file = (
        st.file_uploader(
            "Upload an image",
            type=[
                "jpg",
                "jpeg",
                "png",
                "webp",
            ],
            key="classification_upload",
        )
    )

    (
        loaded_models,
        load_messages,
        model_paths,
    ) = load_classification_models()

    with st.expander(
        "Model Loading Status"
    ):
        for (
            model_name,
            message,
        ) in load_messages.items():
            if (
                model_name
                in loaded_models
            ):
                st.success(
                    f"{model_name}: "
                    f"{message}"
                )
            else:
                st.warning(
                    f"{model_name}: "
                    f"{message}"
                )

    if uploaded_file is None:
        st.info(
            "Upload an image to "
            "start classification."
        )

    elif not loaded_models:
        st.error(
            "No PyTorch classification "
            "models were found."
        )

    else:
        uploaded_image = (
            Image.open(
                uploaded_file
            )
            .convert("RGB")
        )

        image_column, result_column = (
            st.columns(
                [1, 2],
                gap="large",
            )
        )

        with image_column:
            st.image(
                uploaded_image,
                caption="Uploaded Image",
                use_container_width=True,
            )

        comparison_rows = []
        all_results = {}

        with result_column:
            with st.spinner(
                "Running all models..."
            ):
                for (
                    model_name,
                    model,
                ) in loaded_models.items():
                    (
                        predictions,
                        inference_ms,
                    ) = (
                        predict_classification(
                            uploaded_image,
                            model,
                        )
                    )

                    all_results[
                        model_name
                    ] = {
                        "predictions": (
                            predictions
                        ),
                        "inference_ms": (
                            inference_ms
                        ),
                    }

                    top_prediction = (
                        predictions[0]
                    )

                    comparison_rows.append({
                        "Model": model_name,
                        "Prediction": (
                            top_prediction[
                                "class"
                            ]
                        ),
                        "Confidence": (
                            top_prediction[
                                "confidence"
                            ]
                        ),
                        "Inference Time (ms)": (
                            inference_ms
                        ),
                    })

            st.subheader(
                "Side-by-Side Comparison"
            )

            comparison_df = (
                pd.DataFrame(
                    comparison_rows
                )
            )

            display_df = (
                comparison_df.copy()
            )

            display_df[
                "Confidence"
            ] = (
                display_df[
                    "Confidence"
                ]
                .map(
                    lambda value:
                    f"{value:.2%}"
                )
            )

            display_df[
                "Inference Time (ms)"
            ] = (
                display_df[
                    "Inference Time (ms)"
                ]
                .map(
                    lambda value:
                    f"{value:.2f}"
                )
            )

            st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=True,
            )

        st.subheader(
            "Top-5 Predictions"
        )

        result_columns = (
            st.columns(
                len(all_results)
            )
        )

        for (
            column,
            (
                model_name,
                model_result,
            ),
        ) in zip(
            result_columns,
            all_results.items(),
        ):
            with column:
                top_prediction = (
                    model_result[
                        "predictions"
                    ][0]
                )

                st.markdown(
                    f"### {model_name}"
                )

                st.metric(
                    "Top Prediction",
                    str(
                        top_prediction[
                            "class"
                        ]
                    ).title(),
                    f"{float(top_prediction['confidence']):.2%}",
                )

                st.caption(
                    "Inference Time: "
                    f"{model_result['inference_ms']:.2f} ms"
                )

                for prediction in (
                    model_result[
                        "predictions"
                    ]
                ):
                    confidence = float(
                        prediction[
                            "confidence"
                        ]
                    )

                    st.progress(
                        confidence,
                        text=(
                            f"{str(prediction['class']).title()}: "
                            f"{confidence:.2%}"
                        ),
                    )

        st.subheader(
            "Top-1 Confidence Comparison"
        )

        confidence_chart = (
            comparison_df
            .set_index("Model")[
                ["Confidence"]
            ]
        )

        st.bar_chart(
            confidence_chart
        )


# =============================================================================
# PAGE 3: OBJECT DETECTION
# =============================================================================

elif page == "🎯 Object Detection":
    st.title(
        "🎯 YOLOv8 Object Detection"
    )

    st.write(
        "Upload an image to detect "
        "and locate multiple objects."
    )

    setting_col1, setting_col2 = (
        st.columns(2)
    )

    with setting_col1:
        confidence_threshold = (
            st.slider(
                "Confidence Threshold",
                min_value=0.05,
                max_value=0.95,
                value=0.25,
                step=0.05,
            )
        )

    with setting_col2:
        iou_threshold = st.slider(
            "IoU Threshold",
            min_value=0.10,
            max_value=0.90,
            value=0.60,
            step=0.05,
        )

    detection_upload = (
        st.file_uploader(
            "Upload an image",
            type=[
                "jpg",
                "jpeg",
                "png",
                "webp",
            ],
            key="detection_upload",
        )
    )

    (
        yolo_model,
        yolo_path,
        yolo_error,
    ) = load_yolo_model()

    if yolo_model is None:
        st.error(
            "YOLO model could not "
            f"be loaded. {yolo_error}"
        )

    elif detection_upload is None:
        st.caption(
            f"YOLO model: {yolo_path}"
        )

        st.info(
            "Upload an image to "
            "start object detection."
        )

    else:
        detection_image = (
            Image.open(
                detection_upload
            )
            .convert("RGB")
        )

        with st.spinner(
            "Detecting objects..."
        ):
            start_time = (
                time.perf_counter()
            )

            detection_results = (
                yolo_model.predict(
                    source=detection_image,
                    conf=(
                        confidence_threshold
                    ),
                    iou=iou_threshold,
                    device=YOLO_DEVICE,
                    verbose=False,
                )
            )

            detection_time_ms = (
                time.perf_counter()
                - start_time
            ) * 1000

        first_result = (
            detection_results[0]
        )

        annotated_image = (
            first_result.plot()
        )

        annotated_image = (
            cv2.cvtColor(
                annotated_image,
                cv2.COLOR_BGR2RGB,
            )
        )

        original_column, detected_column = (
            st.columns(2)
        )

        with original_column:
            st.image(
                detection_image,
                caption="Original Image",
                use_container_width=True,
            )

        with detected_column:
            st.image(
                annotated_image,
                caption="Detected Objects",
                use_container_width=True,
            )

        detected_objects = []

        if (
            first_result.boxes
            is not None
        ):
            yolo_names = (
                first_result.names
            )

            for box in (
                first_result.boxes
            ):
                class_id = int(
                    box.cls[0].item()
                )

                confidence = float(
                    box.conf[0].item()
                )

                (
                    x1,
                    y1,
                    x2,
                    y2,
                ) = (
                    box.xyxy[0]
                    .detach()
                    .cpu()
                    .numpy()
                    .tolist()
                )

                detected_objects.append({
                    "Object": (
                        yolo_names[
                            class_id
                        ]
                    ),
                    "Confidence": (
                        confidence
                    ),
                    "X1": x1,
                    "Y1": y1,
                    "X2": x2,
                    "Y2": y2,
                })

        detection_metric1, detection_metric2, detection_metric3 = (
            st.columns(3)
        )

        detection_metric1.metric(
            "Detected Objects",
            len(detected_objects),
        )

        detection_metric2.metric(
            "Inference Time",
            f"{detection_time_ms:.2f} ms",
        )

        detection_metric3.metric(
            "Confidence Threshold",
            f"{confidence_threshold:.0%}",
        )

        if not detected_objects:
            st.warning(
                "No objects were detected "
                "above the selected threshold."
            )


# =============================================================================
# PAGE 4: MODEL PERFORMANCE
# =============================================================================

elif page == "📊 Model Performance":
    st.title(
        "📊 Model Performance"
    )

    st.write(
        "Evaluate all available PyTorch "
        "classification models on the "
        "classification test dataset."
    )

    (
        performance_models,
        performance_messages,
        performance_paths,
    ) = load_classification_models()

    st.caption(
        "Test dataset: "
        f"{CLASSIFICATION_DATA_DIR / 'test'}"
    )

    if not performance_models:
        st.error(
            "No classification models "
            "are available for evaluation."
        )

    else:
        evaluation_batch_size = (
            st.selectbox(
                "Evaluation Batch Size",
                options=[
                    8,
                    16,
                    32,
                    64,
                ],
                index=2,
            )
        )

        if st.button(
            "Run Model Evaluation",
            type="primary",
        ):
            with st.spinner(
                "Evaluating models..."
            ):
                try:
                    st.session_state[
                        "performance_results"
                    ] = (
                        evaluate_classification_models(
                            performance_models,
                            batch_size=(
                                evaluation_batch_size
                            ),
                        )
                    )

                except Exception as error:
                    st.error(
                        str(error)
                    )

        performance_results = (
            st.session_state.get(
                "performance_results"
            )
        )

        if performance_results is None:
            st.info(
                "Click 'Run Model Evaluation' "
                "to calculate the metrics."
            )

        else:
            performance_summary = (
                performance_results[
                    "summary"
                ]
            )

            st.success(
                "Evaluation completed on "
                f"{performance_results['test_size']} "
                "test images."
            )

            st.subheader(
                "Overall Model Comparison"
            )

            formatted_summary = (
                performance_summary.copy()
            )

            for metric_name in [
                "Accuracy",
                "Precision",
                "Recall",
                "F1-Score",
                "Top-5 Accuracy",
            ]:
                formatted_summary[
                    metric_name
                ] = (
                    formatted_summary[
                        metric_name
                    ]
                    .map(
                        lambda value:
                        f"{value:.4f}"
                    )
                )

            for metric_name in [
                "Inference ms/image",
                "Model Size MB",
            ]:
                formatted_summary[
                    metric_name
                ] = (
                    formatted_summary[
                        metric_name
                    ]
                    .map(
                        lambda value:
                        f"{value:.2f}"
                    )
                )

            st.dataframe(
                formatted_summary,
                use_container_width=True,
                hide_index=True,
            )



# =============================================================================
# PAGE 5: ABOUT
# =============================================================================

elif page == "ℹ️ About":
    st.title(
        "ℹ️ About SmartVision AI"
    )

    st.header(
        "Project Documentation"
    )

    st.write(
        "SmartVision AI is an end-to-end "
        "computer-vision project for image "
        "classification and object detection."
    )

    st.header(
        "Dataset Information"
    )

    st.dataframe(
        dataset_summary(),
        use_container_width=True,
        hide_index=True,
    )

    st.write(
        f"The project supports "
        f"{len(CLASS_NAMES)} classes:"
    )

    st.write(
        ", ".join(
            CLASS_NAMES
        )
    )

    st.header(
        "Model Architectures"
    )

    architecture_rows = [
        {
            "Model": "VGG16",
            "Task": "Classification",
            "Description": (
                "Sequential convolution "
                "blocks using 3×3 filters"
            ),
        },
        {
            "Model": "ResNet50",
            "Task": "Classification",
            "Description": (
                "Residual connections "
                "for deep feature learning"
            ),
        },
        {
            "Model": "MobileNetV2",
            "Task": "Classification",
            "Description": (
                "Lightweight inverted "
                "residual architecture"
            ),
        },
        {
            "Model": "EfficientNetB0",
            "Task": "Classification",
            "Description": (
                "Balanced scaling of depth, "
                "width, and resolution"
            ),
        },
        {
            "Model": "YOLOv8",
            "Task": "Object Detection",
            "Description": (
                "Single-stage object "
                "localization and classification"
            ),
        },
    ]

    st.dataframe(
        pd.DataFrame(
            architecture_rows
        ),
        use_container_width=True,
        hide_index=True,
    )

    st.header(
        "Technical Stack"
    )

    stack_column1, stack_column2 = (
        st.columns(2)
    )

    with stack_column1:
        st.markdown(
            """
            - Python
            - Streamlit
            - PyTorch
            - TorchVision
            - Ultralytics YOLOv8
            """
        )

    with stack_column2:
        st.markdown(
            """
            - OpenCV
            - Pillow
            - NumPy
            - Pandas
            - Matplotlib
            - Scikit-learn
            """
        )
