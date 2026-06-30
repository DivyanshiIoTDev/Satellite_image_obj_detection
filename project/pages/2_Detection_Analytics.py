"""pages/2_Detection_Analytics.py — Plotly analytics over the last run."""

import numpy as np
import streamlit as st
from utils.ui_components import load_css, page_header, section_card_start, section_card_end, metric_card
from utils.plotting import (
    class_distribution_chart, confidence_histogram, detection_count_per_class,
    confidence_distribution_box, inference_time_history, bbox_area_summary,
)

st.set_page_config(page_title="Detection Analytics", page_icon="📊", layout="wide")
load_css()
page_header("Detection Analytics", "Statistical breakdown of detections and inference performance.")

df = st.session_state.get("detections_df")
history = st.session_state.get("history", [])

avg_area = float(df["Area"].mean()) if df is not None and not df.empty else 0.0
avg_conf = float(df["Confidence"].mean()) if df is not None and not df.empty else 0.0

c1, c2 = st.columns(2)
with c1: metric_card("Average Bounding Box Area", f"{avg_area:.1f} px²")
with c2: metric_card("Average Confidence", f"{avg_conf:.2f}")

st.markdown("<br>", unsafe_allow_html=True)

if df is None:
    import pandas as pd
    df = pd.DataFrame(columns=["Class", "Confidence", "Area"])

col1, col2 = st.columns(2)
with col1:
    section_card_start("Class Distribution")
    st.plotly_chart(class_distribution_chart(df), use_container_width=True)
    section_card_end()
with col2:
    section_card_start("Confidence Histogram")
    st.plotly_chart(confidence_histogram(df), use_container_width=True)
    section_card_end()

col3, col4 = st.columns(2)
with col3:
    section_card_start("Detection Count per Class")
    st.plotly_chart(detection_count_per_class(df), use_container_width=True)
    section_card_end()
with col4:
    section_card_start("Confidence Distribution")
    st.plotly_chart(confidence_distribution_box(df), use_container_width=True)
    section_card_end()

col5, col6 = st.columns(2)
with col5:
    section_card_start("Inference Time History")
    st.plotly_chart(inference_time_history(history), use_container_width=True)
    section_card_end()
with col6:
    section_card_start("Bounding Box Area vs. Confidence")
    st.plotly_chart(bbox_area_summary(df), use_container_width=True)
    section_card_end()
