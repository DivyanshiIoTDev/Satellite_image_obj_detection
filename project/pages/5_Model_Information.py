"""pages/5_Model_Information.py — Static metadata about the active model."""

import streamlit as st
from utils.config_loader import get_model_info
from utils.ui_components import load_css, page_header, section_card_start, section_card_end

st.set_page_config(page_title="Model Information", page_icon="🧠", layout="wide")
load_css()
page_header("Model Information", "Architecture, dataset, and deployment details.")

model_key = st.session_state.get("model_key", "model_2")
info = get_model_info(model_key)

if not info:
    st.warning("No model information found in `config.yaml`.")
else:
    labels = {
        "model_name": "Model Name", "dataset": "Dataset", "training_images": "Training Images",
        "num_classes": "Number of Classes", "parameters": "Parameters", "model_size": "Model Size",
        "flops": "FLOPs", "framework": "Framework", "training_platform": "Training Platform",
        "inference_device": "Inference Device", "preprocessing_pipeline": "Preprocessing Pipeline",
        "deployment_framework": "Deployment Framework",
    }
    section_card_start(info.get("model_name", "Model"), "Configured in config.yaml → model_info")
    items = list(labels.items())
    for i in range(0, len(items), 2):
        col1, col2 = st.columns(2)
        for col, (key, label) in zip((col1, col2), items[i:i + 2]):
            with col:
                st.markdown(f"**{label}:** {info.get(key, '—')}")
    section_card_end()
