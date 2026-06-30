"""pages/3_Batch_Processing.py — Process a ZIP of images end-to-end."""

import io
import zipfile
import time

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from utils.config_loader import get_model_path, get_class_names
from utils.model_loader import load_yolo_model
from utils.preprocessing import run_preprocessing_pipeline
from utils.inference import run_inference
from utils.ui_components import load_css, page_header, section_card_start, section_card_end, metric_card
from utils.export_utils import df_to_csv_bytes

st.set_page_config(page_title="Batch Processing", page_icon="🗂️", layout="wide")
load_css()
page_header("Batch Processing", "Run detection across every image in an uploaded ZIP archive.")

zip_file = st.session_state.get("zip_upload")
model_key = st.session_state.get("model_key", "model_2")
device = st.session_state.get("active_device", "cpu")

section_card_start("Batch Source", "Upload a ZIP file from the sidebar on any page to enable this.")
if zip_file is None:
    st.info("No ZIP uploaded yet. Use the **Batch Upload (ZIP)** field in the sidebar.")
    section_card_end()
    st.stop()
else:
    st.success(f"ZIP loaded: `{zip_file.name}`")
section_card_end()

run_batch = st.button("🚀 Process Batch", type="primary")

if run_batch:
    try:
        archive = zipfile.ZipFile(io.BytesIO(zip_file.getvalue()))
    except zipfile.BadZipFile:
        st.error("The uploaded file is not a valid ZIP archive.")
        st.stop()

    image_names = [n for n in archive.namelist()
                   if n.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tiff"))]

    if not image_names:
        st.warning("No valid images found inside the ZIP.")
        st.stop()

    model_path = get_model_path(model_key)
    model, error = load_yolo_model(model_path, device)
    if error:
        st.warning(f"⚠️ {error} — running in demo mode (preprocessing only).")

    class_names = get_class_names()
    rows, times, confs, total_objects = [], [], [], 0
    progress = st.progress(0.0, text="Processing batch...")

    for idx, name in enumerate(image_names):
        try:
            raw = archive.read(name)
            pil_img = Image.open(io.BytesIO(raw)).convert("RGB")
            img_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

            stages = run_preprocessing_pipeline(img_bgr, use_clahe=(model_key == "model_2"), use_wiener=True)
            result = run_inference(model, stages["final"], 0.25, 0.45, device, class_names)

            times.append(result["inference_time_ms"])
            det_confs = [d["Confidence"] for d in result["detections"]]
            confs.extend(det_confs)
            total_objects += len(result["detections"])

            rows.append({
                "Image": name, "Objects": len(result["detections"]),
                "Avg Confidence": round(np.mean(det_confs), 3) if det_confs else 0.0,
                "Inference Time (ms)": result["inference_time_ms"],
                "Status": "OK" if result["error"] is None else result["error"],
            })
        except Exception as exc:
            rows.append({"Image": name, "Objects": 0, "Avg Confidence": 0.0,
                          "Inference Time (ms)": 0.0, "Status": f"Failed: {exc}"})
        progress.progress((idx + 1) / len(image_names), text=f"Processed {idx+1}/{len(image_names)}")

    batch_df = pd.DataFrame(rows)
    st.session_state["batch_results"] = batch_df
    progress.empty()

batch_df = st.session_state.get("batch_results")

if batch_df is not None:
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1: metric_card("Total Images", f"{len(batch_df)}")
    with c2: metric_card("Processed", f"{(batch_df['Status'] == 'OK').sum()}")
    with c3: metric_card("Avg Time", f"{batch_df['Inference Time (ms)'].mean():.1f} ms")
    with c4: metric_card("Avg Confidence", f"{batch_df['Avg Confidence'].mean():.2f}")
    with c5: metric_card("Total Objects", f"{int(batch_df['Objects'].sum())}")

    st.markdown("<br>", unsafe_allow_html=True)
    section_card_start("Batch Results")
    st.dataframe(batch_df, use_container_width=True, height=400)
    st.download_button("⬇️ Export All Predictions (CSV)", data=df_to_csv_bytes(batch_df),
                        file_name="batch_predictions.csv", mime="text/csv")
    section_card_end()
