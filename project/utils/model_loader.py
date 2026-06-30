"""
model_loader.py
----------------
Loads pretrained YOLOv8 weights for inference only (no training).
Models are cached per (path, device) so switching back and forth in the
UI is instant after the first load.
"""

import os
import streamlit as st
from ultralytics import YOLO


@st.cache_resource(show_spinner="Loading model weights...")
def load_yolo_model(model_path: str, device: str):
    """Load a YOLOv8 model from disk onto the given device.

    Returns (model, error_message). error_message is None on success.
    Cached on (model_path, device) so repeat calls are free.
    """
    if not model_path or not os.path.exists(model_path):
        return None, f"Model weights not found at `{model_path}`. " \
                      f"Update the path in `config.yaml`."
    try:
        model = YOLO(model_path)
        model.to(device)
        print("Successfully loaded model weights from", model_path)
        return model, None
    except Exception as exc:
        return None, f"Failed to load model: {exc}"
