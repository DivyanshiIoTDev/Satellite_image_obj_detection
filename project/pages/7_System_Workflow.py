"""pages/7_System_Workflow.py — Visual pipeline diagram via Streamlit components."""

import streamlit as st
from utils.ui_components import load_css, page_header, section_card_start, section_card_end

st.set_page_config(page_title="System Workflow", page_icon="🧭", layout="wide")
load_css()
page_header("System Workflow", "End-to-end pipeline from raw capture to actionable analytics.")

section_card_start("Pipeline Overview")

steps = ["Satellite Image", "CLAHE", "Wiener Filter", "YOLOv8", "Bounding Boxes", "Analytics"]
icons = ["🛰️", "🌗", "🌊", "🎯", "📦", "📊"]

cols = st.columns(len(steps) * 2 - 1)
for i, (step, icon) in enumerate(zip(steps, icons)):
    col_idx = i * 2
    with cols[col_idx]:
        st.markdown(
            f'<div class="workflow-step">{icon}<br>{step}</div>', unsafe_allow_html=True
        )
    if col_idx + 1 < len(cols):
        with cols[col_idx + 1]:
            st.markdown('<div class="workflow-arrow">→</div>', unsafe_allow_html=True)

section_card_end()

section_card_start("Stage Descriptions")
descriptions = {
    "Satellite Image": "Raw low-SNR capture, often affected by sensor noise and atmospheric distortion.",
    "CLAHE": "Boosts local contrast adaptively so faint targets become more separable from background.",
    "Wiener Filter": "Frequency-domain denoising that suppresses noise while preserving target edges.",
    "YOLOv8": "Lightweight single-stage detector predicting class, confidence, and bounding box.",
    "Bounding Boxes": "Per-object detections with class label, confidence score, and coordinates.",
    "Analytics": "Aggregated statistics, distributions, and exportable reports for downstream use.",
}
for step, desc in descriptions.items():
    st.markdown(f"**{step}** — {desc}")
section_card_end()
