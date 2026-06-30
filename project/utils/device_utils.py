"""
device_utils.py
----------------
Resolves the active inference device ('Auto' / 'CPU' / 'GPU') safely.
Never raises -- always falls back to CPU with a UI warning if CUDA is
unavailable.
"""

import torch
import streamlit as st


def resolve_device(preference: str = "Auto") -> str:
    """Return a torch-compatible device string ('cpu' or 'cuda') based on
    the user's preference and actual hardware availability.
    """
    cuda_available = torch.cuda.is_available()

    if preference == "GPU":
        if cuda_available:
            return "cuda"
        st.sidebar.warning("⚠️ GPU requested but not available. Falling back to CPU.")
        return "cpu"

    if preference == "CPU":
        return "cpu"

    # Auto
    return "cuda" if cuda_available else "cpu"


def device_display_name(device: str) -> str:
    """Human-friendly label for metric cards."""
    if device == "cuda":
        try:
            return f"GPU ({torch.cuda.get_device_name(0)})"
        except Exception:
            return "GPU"
    return "CPU"
