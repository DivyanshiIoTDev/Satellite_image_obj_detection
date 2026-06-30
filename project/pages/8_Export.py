"""pages/8_Export.py — Download annotated image, CSV, JSON, and PDF report."""

import cv2
import numpy as np
import streamlit as st
from utils.ui_components import load_css, page_header, section_card_start, section_card_end
from utils.export_utils import df_to_csv_bytes, detections_to_json_bytes, image_to_png_bytes
from utils.pdf_report import generate_pdf_report

st.set_page_config(page_title="Export", page_icon="⬇️", layout="wide")
load_css()
page_header("Export", "Download results from the most recent detection run.")

result = st.session_state.get("result")
df = st.session_state.get("detections_df")
model_name = st.session_state.get("model_name", "Unknown Model")

if result is None:
    st.info("No results to export yet. Run a detection from the Dashboard page first.")
    st.stop()

section_card_start("Available Exports")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("**Annotated Image**")
    png_bytes = image_to_png_bytes(result["annotated_image"])
    st.download_button("⬇️ Download PNG", data=png_bytes, file_name="annotated_image.png",
                        mime="image/png", use_container_width=True)

with col2:
    st.markdown("**Detections (CSV)**")
    st.download_button("⬇️ Download CSV", data=df_to_csv_bytes(df), file_name="detections.csv",
                        mime="text/csv", use_container_width=True)

with col3:
    st.markdown("**Detections (JSON)**")
    metadata = {
        "model": model_name, "inference_time_ms": result["inference_time_ms"],
        "num_objects": len(result["detections"]),
    }
    st.download_button("⬇️ Download JSON",
                        data=detections_to_json_bytes(result["detections"], metadata),
                        file_name="detections.json", mime="application/json", use_container_width=True)

with col4:
    st.markdown("**PDF Report**")
    confs = [d["Confidence"] for d in result["detections"]]
    stats = {
        "mean": float(np.mean(confs)) if confs else 0.0,
        "min": float(np.min(confs)) if confs else 0.0,
        "max": float(np.max(confs)) if confs else 0.0,
    }
    pdf_bytes = generate_pdf_report(result["annotated_image"], result["detections"],
                                     result["inference_time_ms"], model_name, stats)
    st.download_button("⬇️ Download PDF", data=pdf_bytes, file_name="detection_report.pdf",
                        mime="application/pdf", use_container_width=True)

section_card_end()
