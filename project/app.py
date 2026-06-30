"""
app.py
-------
Entry point for the Low-SNR Satellite Target Detection dashboard.

This file owns:
  * Page configuration / global theme
  * The persistent sidebar (upload, model, device, thresholds, run/reset)
  * The "Dashboard" landing page (KPIs + image comparison)

All other pages live in `pages/` and read shared state from
`st.session_state`, which this file initializes and populates whenever
"Run Detection" is pressed.
"""

import io
import time
import zipfile

import cv2
import numpy as np
import streamlit as st
from PIL import Image

from utils.config_loader import get_model_path, get_class_names
from utils.device_utils import resolve_device, device_display_name
from utils.model_loader import load_yolo_model
from utils.preprocessing import run_preprocessing_pipeline
from utils.inference import run_inference, detections_to_dataframe
from utils.ui_components import load_css, page_header, metric_card, section_card_start, section_card_end

# ------------------------------------------------------------------ #
# Page configuration
# ------------------------------------------------------------------ #
st.set_page_config(
    page_title="Low-SNR Satellite Target Detection",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)
load_css()

# ------------------------------------------------------------------ #
# Session state initialization
# ------------------------------------------------------------------ #
DEFAULT_STATE = {
    "result": None,            # last inference result dict
    "stages": None,            # preprocessing stage images
    "detections_df": None,     # last detections dataframe
    "history": [],             # inference time history (for analytics)
    "model_key": "model_1",
    "active_device": "cpu",
    "model_name": "",
    "batch_results": None,     # results from batch processing
}
for key, val in DEFAULT_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = val


# ------------------------------------------------------------------ #
# Sidebar — global controls shared by every page
# ------------------------------------------------------------------ #
def render_sidebar():
    st.sidebar.markdown("## ⚙️ Detection Controls")

    uploaded_file = st.sidebar.file_uploader(
        "Upload Image", type=["jpg", "jpeg", "png", "bmp", "tiff"],
        help="Upload a single satellite image for detection.",
    )

    zip_file = st.sidebar.file_uploader(
        "Batch Upload (ZIP)", type=["zip"],
        help="Upload a ZIP of images for batch processing on the Batch Processing page.",
    )
    if zip_file is not None:
        st.session_state["zip_upload"] = zip_file

    st.sidebar.markdown("---")

    device_pref = st.sidebar.selectbox("Device", ["Auto", "CPU", "GPU"], index=0)
    resolved_device = resolve_device(device_pref)
    st.session_state["active_device"] = resolved_device

    model_choice = st.sidebar.selectbox(
        "Model", ["Model 1 — Wiener + YOLOv8", "Model 2 — CLAHE + Wiener + YOLOv8"], index=1,
    )
    model_key = "model_1" if model_choice.startswith("Model 1") else "model_2"
    st.session_state["model_key"] = model_key
    st.session_state["model_name"] = model_choice

    st.sidebar.markdown("---")
    conf_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.25, 0.01)
    iou_threshold = st.sidebar.slider("IoU Threshold", 0.0, 1.0, 0.45, 0.01)

    st.sidebar.markdown("---")
    use_clahe = st.sidebar.checkbox("Enable CLAHE", value=(model_key == "model_2"))
    use_wiener = st.sidebar.checkbox("Enable Wiener Filter", value=True)

    st.sidebar.markdown("---")
    col_run, col_reset = st.sidebar.columns(2)
    run_clicked = col_run.button("🚀 Run Detection", use_container_width=True)
    reset_clicked = col_reset.button("♻️ Reset", use_container_width=True)

    if reset_clicked:
        for key, val in DEFAULT_STATE.items():
            st.session_state[key] = val
        st.rerun()

    return {
        "uploaded_file": uploaded_file,
        "device": resolved_device,
        "model_key": model_key,
        "conf": conf_threshold,
        "iou": iou_threshold,
        "use_clahe": use_clahe,
        "use_wiener": use_wiener,
        "run_clicked": run_clicked,
    }


def execute_detection(controls: dict):
    """Run the full preprocessing + inference pipeline and persist
    results into session_state so every page can read them.
    """
    uploaded_file = controls["uploaded_file"]
    if uploaded_file is None:
        st.sidebar.error("Please upload an image first.")
        return

    try:
        pil_image = Image.open(uploaded_file).convert("RGB")
    except Exception as exc:
        st.sidebar.error(f"Could not read image: {exc}")
        return

    image_bgr = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    # Preprocessing
    stages = run_preprocessing_pipeline(
        image_bgr, use_clahe=controls["use_clahe"], use_wiener=controls["use_wiener"],
    )

    # Model loading
    model_path = get_model_path(controls["model_key"])
    model, error = load_yolo_model(model_path, controls["device"])
    if error:
        st.sidebar.warning(f"⚠️ {error}\n\nRunning in **demo mode** — preprocessing only, no live boxes.")

    class_names = get_class_names()
    result = run_inference(model, stages["final"], controls["conf"], controls["iou"],
                            controls["device"], class_names)

    st.session_state["stages"] = stages
    st.session_state["result"] = result
    st.session_state["detections_df"] = detections_to_dataframe(result["detections"])

    if result["error"] is None:
        st.session_state["history"].append({
            "run": len(st.session_state["history"]) + 1,
            "time_ms": result["inference_time_ms"],
        })
    else:
        st.sidebar.error(result["error"])


controls = render_sidebar()
if controls["run_clicked"]:
    execute_detection(controls)

# ------------------------------------------------------------------ #
# Dashboard landing page content
# ------------------------------------------------------------------ #
page_header(
    "Low-SNR Satellite Target Detection",
    "Adaptive Image Enhancement using CLAHE and Wiener Filtering with Lightweight YOLOv8",
)

result = st.session_state["result"]
num_objects = len(result["detections"]) if result else 0
avg_conf = (
    np.mean([d["Confidence"] for d in result["detections"]])
    if result and result["detections"] else 0.0
)
inference_time = result["inference_time_ms"] if result else 0.0
fps = result["fps"] if result else 0.0

c1, c2, c3, c4, c5 = st.columns(5)
with c1: metric_card("Inference Time", f"{inference_time:.1f} ms")
with c2: metric_card("Objects Detected", f"{num_objects}")
with c3: metric_card("Avg. Confidence", f"{avg_conf:.2f}")
with c4: metric_card("FPS", f"{fps:.1f}")
with c5: metric_card("Active Device", device_display_name(st.session_state["active_device"]))

st.markdown("<br>", unsafe_allow_html=True)

# ------------------------------------------------------------------ #
# Image comparison strip
# ------------------------------------------------------------------ #
section_card_start("Image Processing Pipeline", "Original → CLAHE → Wiener → Final Detection")

stages = st.session_state["stages"]
if stages is None:
    st.info("👈 Upload an image and click **Run Detection** in the sidebar to begin.")
else:
    cols = st.columns(4)
    titles = ["Original Image", "CLAHE Output", "Wiener Output", "Final Detection"]
    images = [
        stages["original"],
        stages["clahe"],
        stages["wiener"],
        result["annotated_image"] if result else stages["final"],
    ]
    for col, title, img in zip(cols, titles, images):
        with col:
            st.markdown(f"**{title}**")
            if img is None:
                st.markdown('<div class="placeholder-box">Stage disabled</div>', unsafe_allow_html=True)
            else:
                st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), use_container_width=True)

section_card_end()

st.caption("Use the pages in the left sidebar to explore Detection Results, Analytics, "
           "Batch Processing, Model Performance, and more.")
