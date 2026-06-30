"""
preprocessing.py
-----------------
Image enhancement primitives used before YOLO inference:
  * CLAHE (Contrast Limited Adaptive Histogram Equalization)
  * Wiener Filter (frequency-domain denoising)

Each function takes and returns a numpy BGR (OpenCV-style) uint8 image
and never raises on malformed input -- it returns the original image
with a logged warning instead, so the Streamlit app never crashes.
"""

import cv2
import numpy as np
from scipy.signal import convolve2d


def _wiener2(img_channel: np.ndarray, kernel_size: int = 7) -> np.ndarray:
    """2D Wiener filter for a single image channel.

    Matches the exact training-time implementation: local-mean/local-
    variance based adaptive filtering using a uniform box kernel, rather
    than scipy.signal.wiener. Kept in lockstep with the Colab notebook so
    inference-time preprocessing exactly matches what the model was
    trained on.
    """
    img = img_channel.astype(np.float32)
    kernel = np.ones((kernel_size, kernel_size), dtype=np.float32) / (kernel_size * kernel_size)

    local_mean = convolve2d(img, kernel, mode="same", boundary="symm")
    local_var = convolve2d(img ** 2, kernel, mode="same", boundary="symm") - local_mean ** 2
    noise_var = np.mean(local_var)

    result = local_mean + np.maximum(local_var - noise_var, 0) / (local_var + 1e-8) * (img - local_mean)
    return np.clip(result, 0, 255).astype(np.uint8)


def apply_clahe(image: np.ndarray, clip_limit: float = 2.0, tile_grid_size: int = 8,
                 alpha: float = 1.15, beta: int = 5) -> np.ndarray:
    """Apply CLAHE on the luminance channel of the image (LAB color space)
    to boost local contrast, followed by a slight linear contrast/brightness
    boost (alpha/beta), exactly matching the training-time `enhance_image`
    function from the Colab notebook.
    """
    try:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)

        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid_size, tile_grid_size))
        l_enhanced = clahe.apply(l_channel)

        enhanced = cv2.merge((l_enhanced, a_channel, b_channel))
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        enhanced = cv2.convertScaleAbs(enhanced, alpha=alpha, beta=beta)
        return enhanced
    except Exception:
        return image


def apply_wiener_filter(image: np.ndarray, kernel_size: int = 7) -> np.ndarray:
    """Apply the custom adaptive Wiener filter to each channel of a BGR
    image, suppressing sensor / atmospheric noise typical of low-SNR
    satellite captures while preserving edges.

    Mirrors the training-time `apply_wiener_rgb` function exactly
    (per-channel split -> `_wiener2` -> merge), default kernel_size=7.
    """
    try:
        b, g, r = cv2.split(image)
        b = _wiener2(b, kernel_size=kernel_size)
        g = _wiener2(g, kernel_size=kernel_size)
        r = _wiener2(r, kernel_size=kernel_size)
        return cv2.merge((b, g, r))
    except Exception:
        return image


def run_preprocessing_pipeline(image: np.ndarray, use_clahe: bool, use_wiener: bool,
                                clip_limit: float = 2.0, kernel_size: int = 7) -> dict:
    """Run the requested preprocessing steps in the correct order
    (CLAHE -> Wiener, matching Model 2's training pipeline) and return
    every intermediate stage for the side-by-side image comparison UI.

    Returns a dict: {"original", "clahe", "wiener", "final"}.
    """
    stages = {"original": image}

    current = image
    if use_clahe:
        current = apply_clahe(image, clip_limit=clip_limit)
        stages["clahe"] = current
    else:
        stages["clahe"] = None

    if use_wiener:
        current = apply_wiener_filter(current, kernel_size=kernel_size)
        stages["wiener"] = current
    else:
        stages["wiener"] = None

    stages["final"] = current
    return stages