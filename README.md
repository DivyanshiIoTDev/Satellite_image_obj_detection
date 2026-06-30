# AI-Based Low-SNR Target Detection in Satellite Images Using Image Enhancement and YOLOv8

An advanced, research-oriented computer vision pipeline developed to accurately detect low-snr targets in low Signal-to-Noise Ratio (SNR) satellite imagery. This repository contains the source code for the image pre-processing pipeline (CLAHE + Wiener Filtering), YOLOv8 neural network configuration, and an interactive Streamlit dashboard for real-time visualization, model analysis, and experimental comparison.

**Associated Institution:** National Institute of Technology, Kurukshetra, Haryana  
**Timeline:** June 2026 – Present  

---

## 📌 Project Overview

Detecting targets in satellite imagery under low-SNR conditions presents a major challenge due to atmospheric interference, sensor noise, cloud cover, and poor illumination. Standard object detectors often fail to recognize targets when the contrast against the background is extremely minimal.

This project addresses these challenges by engineering a hybrid pipeline that couples classical digital image enhancement techniques with modern deep learning:
1. **Contrast Limited Adaptive Histogram Equalization (CLAHE):** Maximizes localized contrast without over-amplifying sensor background noise.
2. **Wiener Filtering:** Executes adaptive pixel-wise noise reduction based on local image variance to suppress high-frequency noise and sensor artifacts.
3. **YOLOv8 Target Detector:** Exploits enhanced structural details to precisely locate and classify faint targets.

---

## 📊 Core Performance Metrics

Through meticulous algorithmic evaluation and enhancement tuning, the pipeline achieved state-of-the-art results for low-SNR target recognition:

| Metric | Value | Description |
| :--- | :---: | :--- |
| **Precision** | **86.0%** | High fidelity in target predictions with minimal false positives. |
| **Recall** | **81.0%** | Superior sensitivity ensuring faint targets are rarely missed. |
| **mAP@50** | **84.0%** | Mean Average Precision evaluated at an Intersection over Union (IoU) threshold of 0.50. |
| **mAP@50–95** | **55.0%** | Strict evaluation reflecting highly accurate bounding box localization across varied scales. |

---

## 🖥️ Streamlit Interactive Dashboard

The repository includes a web-based interactive deployment dashboard designed using **Streamlit**. It provides domain researchers and engineers with a tool to run inference, compare techniques, and analyze performance characteristics.

### 🏠 Dashboard Features & Pages

#### 1. Real-Time Inference & Evaluation
* **Upload Interface:** Supports high-resolution satellite imagery formats (`.png`, `.jpg`, `.tiff`).
* **Interactive Run:** Generates real-time bounding box annotations and confidence scores.
* **Fallback Guard:** If training curves are missing from the designated path, the application gracefully surfaces a clean, stylized dashboard notification instead of throwing unhandled runtime exceptions.

#### 2. Model Information Profile
A dedicated panel summarizing the exact architectural and training metadata:
* **Model Name:** YOLOv8-LowSNR Custom Variant
* **Dataset:** Low-SNR Real-World Satellite Target Dataset
* **Training Images:** 1600 high-resolution images
* **Number of Classes:** Target specific (e.g., Vessels, Aircraft, Vehicles)
* **Parameters:** ~11.2M 
* **Model Size:** ~22.5 MB (.pt weight file)
* **Framework:** PyTorch & Ultralytics YOLOv8
* **Inference Device:** CPU / CUDA Auto-detect
* **Preprocessing Pipeline:** Dual-stage sequential (CLAHE ➔ Wiener Filter)
* **Deployment Framework:** Streamlit Dashboard Engine

#### 3. Research Comparison Page
An experimental matrix enabling side-by-side performance quantification. 

```json
{
  "methods": [
    {
      "name": "Wiener + YOLOv8",
      "precision": 0.73,
      "recall": 0.70,
      "map50": 0.71,
      "map50_95": 0.44,
      "fps": 38.5,
      "inference_time_ms": 26.0
    },
    {
      "name": "CLAHE + Wiener + YOLOv8 (Ours)",
      "precision": 0.86,
      "recall": 0.81,
      "map50": 0.84,
      "map50_95": 0.55,
      "fps": 32.1,
      "inference_time_ms": 31.1
    }
  ]
}
```

#### 4. System Workflow Page
Visualizes the data propagation path natively using custom Streamlit layouts, showing the end-to-end transformation matrix:

```
[Satellite Image Input]
          │
          ▼
    [CLAHE Engine]  ───► (Contrast Enhancement & Local Equalization)
          │
          ▼
   [Wiener Filter]  ───► (Adaptive Pixel-Wise Denoising)
          │
          ▼
   [YOLOv8 Network] ───► (Deep Feature Extraction & Object Detection)
          │
          ▼
  [Bounding Boxes]  ───► (Coordinate Filtering & Confidence NMS)
          │
          ▼
 [Analytics Summary] ───► (Target Count, Density & Regional Profiling)
```

---

## 🛠️ Installation & Getting Started

### Prerequisites
Ensure you have Python 3.9+ and CUDA-compatible drivers (optional but recommended for faster execution) installed.

### Step 1: Clone the Repository
```bash
git clone https://github.com/username/satellite-low-snr-detection.git
cd satellite-low-snr-detection
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```
*Note: Key dependencies include `streamlit`, `ultralytics`, `opencv-python-headless`, `scipy`, and `numpy`.*

### Step 3: Run the Streamlit App
```bash
streamlit run app.py
```

