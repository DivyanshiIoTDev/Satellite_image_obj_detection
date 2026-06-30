"""
export_utils.py
-----------------
Helpers that turn in-memory results into downloadable bytes for
Streamlit's download_button (CSV, JSON, annotated PNG).
"""

import io
import json
import cv2
import pandas as pd
import numpy as np


def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def detections_to_json_bytes(detections: list, metadata: dict) -> bytes:
    payload = {"metadata": metadata, "detections": detections}
    return json.dumps(payload, indent=2).encode("utf-8")


def image_to_png_bytes(image_bgr: np.ndarray) -> bytes:
    """Encode a BGR numpy image as PNG bytes for download."""
    success, buffer = cv2.imencode(".png", image_bgr)
    if not success:
        return b""
    return io.BytesIO(buffer.tobytes()).getvalue()
