"""pages/1_Detection_Results.py — Sortable detection table + CSV export."""

import streamlit as st
from utils.ui_components import load_css, page_header, section_card_start, section_card_end
from utils.export_utils import df_to_csv_bytes

st.set_page_config(page_title="Detection Results", page_icon="📋", layout="wide")
load_css()
page_header("Detection Results", "Per-object detections from the most recent inference run.")

df = st.session_state.get("detections_df")

section_card_start("Detections Table", "Click a column header to sort.")
if df is None or df.empty:
    st.info("No detections yet. Run a detection from the Dashboard page first.")
else:
    st.dataframe(df, use_container_width=True, height=420)
    st.download_button(
        "⬇️ Download CSV", data=df_to_csv_bytes(df),
        file_name="detections.csv", mime="text/csv",
    )
section_card_end()
