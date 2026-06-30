# 🛰️ Low-SNR Satellite Target Detection

A production-style Streamlit dashboard for detecting faint, low-signal-to-noise
(low-SNR) targets in satellite imagery, combining classical image enhancement
(CLAHE, Wiener filtering) with a lightweight YOLOv8 detector.

> Built as an internal-tool-style AI portfolio project — inference-only,
> loading pretrained weights trained separately in Google Colab.

---

## Overview

Two pretrained pipelines are supported:

| Model | Pipeline |
|---|---|
| **Model 1** | Wiener Filter → YOLOv8 |
| **Model 2** | CLAHE → Wiener Filter → YOLOv8 |

The app lets you upload an image (or a ZIP for batch mode), choose the model
and device, tune confidence/IoU thresholds, and inspect results across a
multi-page dashboard: detections table, analytics, batch processing, model
performance, model card, research comparison, system workflow, and export
(PNG / CSV / JSON / PDF report).

## Architecture

```
Upload → Preprocessing (CLAHE / Wiener) → YOLOv8 Inference → Postprocessing
                                                  │
                          ┌───────────────────────┼───────────────────────┐
                    Detections Table        Analytics (Plotly)       Export (CSV/JSON/PDF)
```

All business logic lives in `utils/`; pages are thin presentation layers that
read shared state from `st.session_state`.

## Features

- 🌓 Modern dark theme, professional blue accents, card-based layout
- 🧠 Dynamic model loading (Model 1 / Model 2) with `@st.cache_resource`
- ⚙️ Auto / CPU / GPU device selection with safe fallback (never crashes)
- 🖼️ 4-panel image comparison: Original → CLAHE → Wiener → Final Detection
- 📋 Sortable, downloadable detections table (class, confidence, bbox, area, ID)
- 📊 Plotly analytics: class distribution, confidence histogram, per-class
  counts, confidence box plots, inference time history, bbox-area scatter
- 🗂️ Batch processing from a ZIP archive with aggregate metrics + CSV export
- 📈 Model performance page with precision/recall/mAP cards and training
  curve images (graceful placeholders if missing)
- 🧾 Model information / model card page
- 🔬 Research comparison table (YOLO vs Wiener+YOLO vs CLAHE+Wiener+YOLO),
  values editable in `config.yaml`
- 🧭 Visual system workflow diagram
- ⬇️ Export annotated image, CSV, JSON, and an auto-generated PDF report
- 🛡️ Defensive error handling throughout (missing model, corrupt image,
  wrong file type, missing graphs, zero detections)

## Folder Structure

```
project/
├── app.py                     # Entry point: sidebar + Dashboard page
├── pages/
│   ├── 1_Detection_Results.py
│   ├── 2_Detection_Analytics.py
│   ├── 3_Batch_Processing.py
│   ├── 4_Model_Performance.py
│   ├── 5_Model_Information.py
│   ├── 6_Research_Comparison.py
│   ├── 7_System_Workflow.py
│   └── 8_Export.py
├── utils/
│   ├── config_loader.py       # YAML config accessors (cached)
│   ├── device_utils.py        # Auto/CPU/GPU resolution
│   ├── preprocessing.py       # CLAHE + Wiener filter
│   ├── model_loader.py        # Cached YOLOv8 loading
│   ├── inference.py           # Inference + result parsing
│   ├── plotting.py            # Plotly chart builders
│   ├── export_utils.py        # CSV/JSON/PNG export helpers
│   ├── pdf_report.py          # PDF report generator (fpdf2)
│   └── ui_components.py       # Cards, headers, shared CSS loader
├── models/                    # Place / point to your .pt weights here
├── assets/
│   └── style.css
├── sample_images/             # Optional demo images
├── outputs/
│   ├── model_1/                # confusion_matrix.png, PR_curve.png, etc.
│   └── model_2/
├── config.yaml                 # Model paths, metadata, comparison table
├── requirements.txt
└── README.md
```

## Installation

```bash
git clone <your-repo-url>
cd project
python -m venv venv && source venv/bin/activate   # optional
pip install -r requirements.txt
```

## Configuration

Edit `config.yaml`:

```yaml
model_paths:
  model_1: "/path/to/model1_wiener_yolo.pt"
  model_2: "/path/to/model2_clahe_wiener_yolo.pt"
```

If your weights live in Google Drive, mount the drive (Colab) or sync it
locally (e.g. via `rclone` or Google Drive Desktop) and point the paths at
the mounted location — no code changes required.

Training output images (`confusion_matrix.png`, `PR_curve.png`, `results.png`,
`F1_curve.png`, `P_curve.png`, `R_curve.png`) go in `outputs/model_1/` and
`outputs/model_2/`, paths configurable in `config.yaml`. Missing files show a
clean placeholder instead of an error.

## Running Locally

```bash
streamlit run app.py
```

Then open `http://localhost:8501`.

## Screenshots

> _Add screenshots here once the app is running, e.g._
> `assets/screenshot_dashboard.png`, `assets/screenshot_analytics.png`

## Dependencies

See `requirements.txt` — Streamlit, Ultralytics YOLOv8, PyTorch, OpenCV,
NumPy, Pandas, Matplotlib, Plotly, Pillow, PyYAML, fpdf2, SciPy.

## Future Improvements

- Live confusion-matrix / metrics computation from `results.csv` instead of
  static `config.yaml` values
- Video / stream inference mode
- Model A/B comparison view (side-by-side Model 1 vs Model 2 on one image)
- Authentication + multi-user history (e.g. via Supabase/Firebase)
- Dockerfile + CI for one-click deployment

## License

MIT License — free to use, modify, and showcase.
