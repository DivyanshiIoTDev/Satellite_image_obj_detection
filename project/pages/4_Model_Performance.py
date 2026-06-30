"""pages/4_Model_Performance.py — Training metrics and curve plots."""

import os
import streamlit as st
from utils.config_loader import get_training_outputs
from utils.ui_components import load_css, page_header, section_card_start, section_card_end, metric_card, placeholder

st.set_page_config(page_title="Model Performance", page_icon="📈", layout="wide")
load_css()
page_header("Model Performance", "Training metrics and diagnostic curves from YOLOv8 training.")

model_key = st.session_state.get("model_key", "model_2")
st.caption(f"Showing results for: **{st.session_state.get('model_name', model_key)}**")

# NOTE: Replace these with values parsed from your training run (e.g. results.csv)
PERFORMANCE_METRICS = {
    "model_1": {"precision": 0.79, "recall": 0.74, "map50": 0.77, "map50_95": 0.48},
    "model_2": {"precision": 0.86, "recall": 0.81, "map50": 0.84, "map50_95": 0.55},
}
metrics = PERFORMANCE_METRICS.get(model_key, {})

c1, c2, c3, c4 = st.columns(4)
with c1: metric_card("Precision", f"{metrics.get('precision', 0):.2f}")
with c2: metric_card("Recall", f"{metrics.get('recall', 0):.2f}")
with c3: metric_card("mAP50", f"{metrics.get('map50', 0):.2f}")
with c4: metric_card("mAP50-95", f"{metrics.get('map50_95', 0):.2f}")

st.markdown("<br>", unsafe_allow_html=True)

outputs = get_training_outputs(model_key)
graph_titles = {
    "confusion_matrix": "Confusion Matrix", "pr_curve": "Precision-Recall Curve",
    "results": "Training Results", "f1_curve": "F1 Curve",
    "p_curve": "Precision Curve", "r_curve": "Recall Curve",
}

cols = st.columns(2)
for i, (key, title) in enumerate(graph_titles.items()):
    with cols[i % 2]:
        section_card_start(title)
        path = outputs.get(key, "")
        if path and os.path.exists(path):
            st.image(path, use_container_width=True)
        else:
            placeholder(f"`{title}` not available. Place the training output at the path "
                        f"configured in `config.yaml` to display it here.")
        section_card_end()
