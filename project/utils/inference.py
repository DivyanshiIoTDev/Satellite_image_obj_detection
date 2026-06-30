"""
inference.py
-------------
Runs YOLOv8 inference on a preprocessed image and converts raw results
into a clean, UI-friendly structure (list of detection dicts + timing).
"""

import time
import numpy as np
import pandas as pd


def run_inference(model, image: np.ndarray, conf: float, iou: float, device: str,
                   class_names: list) -> dict:
    """Run a single-image YOLOv8 forward pass.

    Returns a dict with keys:
      detections (list[dict]), inference_time_ms (float), fps (float),
      annotated_image (np.ndarray), error (str|None)
    """
    if model is None:
        return {"detections": [], "inference_time_ms": 0.0, "fps": 0.0,
                "annotated_image": image, "error": "Model is not loaded."}

    try:
        start = time.perf_counter()
        results = model.predict(source=image, conf=conf, iou=iou, device=device, verbose=False)
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        result = results[0]
        detections = []
        boxes = result.boxes

        if boxes is not None and len(boxes) > 0:
            for i, box in enumerate(boxes):
                xyxy = box.xyxy[0].cpu().numpy().tolist()
                x1, y1, x2, y2 = xyxy
                cls_id = int(box.cls[0].item())
                conf_score = float(box.conf[0].item())
                width, height = x2 - x1, y2 - y1
                label = class_names[cls_id] if class_names and cls_id < len(class_names) else f"class_{cls_id}"

                detections.append({
                    "Detection ID": f"det_{i+1:03d}",
                    "Class": label,
                    "Confidence": round(conf_score, 4),
                    "X1": round(x1, 1), "Y1": round(y1, 1),
                    "X2": round(x2, 1), "Y2": round(y2, 1),
                    "Center X": round(x1 + width / 2, 1),
                    "Center Y": round(y1 + height / 2, 1),
                    "Area": round(width * height, 1),
                })

        annotated = result.plot()  # BGR numpy array with boxes drawn
        fps = 1000.0 / elapsed_ms if elapsed_ms > 0 else 0.0

        return {
            "detections": detections,
            "inference_time_ms": round(elapsed_ms, 2),
            "fps": round(fps, 2),
            "annotated_image": annotated,
            "error": None,
        }
    except Exception as exc:
        return {"detections": [], "inference_time_ms": 0.0, "fps": 0.0,
                "annotated_image": image, "error": f"Inference failed: {exc}"}


def detections_to_dataframe(detections: list) -> pd.DataFrame:
    """Convert the detections list into a pandas DataFrame for display
    and CSV export. Returns an empty (but correctly-columned) frame if
    there are no detections.
    """
    columns = ["Detection ID", "Class", "Confidence", "X1", "Y1", "X2", "Y2",
               "Center X", "Center Y", "Area"]
    if not detections:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(detections)[columns]
