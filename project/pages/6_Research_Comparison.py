"""pages/6_Research_Comparison.py — Benchmark table across methods."""

import pandas as pd
import streamlit as st
from utils.config_loader import get_research_comparison
from utils.ui_components import load_css, page_header, section_card_start, section_card_end

st.set_page_config(page_title="Research Comparison", page_icon="🔬", layout="wide")
load_css()
page_header("Research Comparison", "Benchmark comparison across preprocessing strategies.")

rows = get_research_comparison()
section_card_start("Method Comparison Table", "Values are editable in config.yaml → research_comparison")

if not rows:
    st.info("No comparison data found in `config.yaml`.")
else:
    df = pd.DataFrame(rows).rename(columns={
        "method": "Method", "precision": "Precision", "recall": "Recall",
        "map50": "mAP50", "map50_95": "mAP50-95", "fps": "FPS",
        "inference_time_ms": "Inference Time (ms)",
    })
    st.dataframe(
        df.style.highlight_max(
            subset=["Precision", "Recall", "mAP50", "mAP50-95", "FPS"], color="#1e3a8a"
        ),
        use_container_width=True, hide_index=True,
    )
section_card_end()
