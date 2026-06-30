"""
pdf_report.py
--------------
Generates a professional one-page PDF summary of a detection run using
fpdf2 (pure Python, no system dependencies).
"""

import io
import tempfile
import os
from datetime import datetime
import cv2
import pandas as pd
from fpdf import FPDF


class _ReportPDF(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 16)
        self.set_text_color(30, 64, 175)
        self.cell(0, 10, "Low-SNR Satellite Target Detection - Report", ln=True, align="C")
        self.set_draw_color(59, 130, 246)
        self.line(10, 20, 200, 20)
        self.ln(6)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f"Page {self.page_no()}", align="C")


def generate_pdf_report(annotated_image_bgr, detections: list, inference_time_ms: float,
                         model_name: str, confidence_stats: dict) -> bytes:
    """Build the PDF and return it as raw bytes.

    annotated_image_bgr: numpy BGR image (or None)
    detections: list of detection dicts
    confidence_stats: dict with mean/min/max confidence
    """
    pdf = _ReportPDF()
    pdf.add_page()

    pdf.set_font("Helvetica", "", 11)
    pdf.set_text_color(20, 20, 20)
    pdf.cell(0, 8, f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
    pdf.cell(0, 8, f"Model Used: {model_name}", ln=True)
    pdf.cell(0, 8, f"Inference Time: {inference_time_ms:.2f} ms", ln=True)
    pdf.cell(0, 8, f"Number of Objects Detected: {len(detections)}", ln=True)
    pdf.cell(0, 8, f"Average Confidence: {confidence_stats.get('mean', 0):.3f}", ln=True)
    pdf.cell(0, 8, f"Min / Max Confidence: {confidence_stats.get('min', 0):.3f} / "
                   f"{confidence_stats.get('max', 0):.3f}", ln=True)
    pdf.ln(4)

    # Embed annotated image if provided
    if annotated_image_bgr is not None:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            cv2.imwrite(tmp.name, annotated_image_bgr)
            tmp_path = tmp.name
        try:
            pdf.image(tmp_path, x=10, w=190)
        finally:
            os.unlink(tmp_path)
        pdf.ln(6)

    # Detection table
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 8, "Detection Table", ln=True)
    pdf.set_font("Helvetica", "", 9)

    if detections:
        col_widths = [25, 30, 25, 35, 35, 25]
        headers = ["ID", "Class", "Conf.", "Center (x,y)", "Box (x1,y1,x2,y2)", "Area"]
        pdf.set_fill_color(59, 130, 246)
        pdf.set_text_color(255, 255, 255)
        for w, h in zip(col_widths, headers):
            pdf.cell(w, 7, h, border=1, fill=True, align="C")
        pdf.ln()

        pdf.set_text_color(20, 20, 20)
        for det in detections:
            row = [
                det.get("Detection ID", ""),
                det.get("Class", ""),
                f"{det.get('Confidence', 0):.2f}",
                f"({det.get('Center X', 0)}, {det.get('Center Y', 0)})",
                f"({det.get('X1', 0)},{det.get('Y1', 0)},{det.get('X2', 0)},{det.get('Y2', 0)})",
                f"{det.get('Area', 0):.0f}",
            ]
            for w, val in zip(col_widths, row):
                pdf.cell(w, 7, str(val), border=1, align="C")
            pdf.ln()
    else:
        pdf.cell(0, 8, "No objects detected in this image.", ln=True)

    return bytes(pdf.output(dest="S"))
