"""
plotting.py
------------
All Plotly chart builders for the Detection Analytics page. Kept
separate from app/page code so charts stay consistent and testable.
A shared dark theme is applied to every figure.
"""

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

DARK_TEMPLATE = "plotly_dark"
ACCENT = "#3B82F6"  # professional blue accent


def _empty_figure(message: str) -> go.Figure:
    """Clean placeholder figure used when there is no data to plot."""
    fig = go.Figure()
    fig.add_annotation(text=message, showarrow=False, font=dict(size=16, color="#9CA3AF"))
    fig.update_layout(template=DARK_TEMPLATE, height=320,
                       xaxis=dict(visible=False), yaxis=dict(visible=False))
    return fig


def class_distribution_chart(df: pd.DataFrame) -> go.Figure:
    if df.empty:
        return _empty_figure("No detections to display.")
    counts = df["Class"].value_counts().reset_index()
    counts.columns = ["Class", "Count"]
    fig = px.bar(counts, x="Class", y="Count", color="Class", template=DARK_TEMPLATE,
                 title="Class Distribution")
    fig.update_layout(showlegend=False, height=360)
    return fig


def confidence_histogram(df: pd.DataFrame) -> go.Figure:
    if df.empty:
        return _empty_figure("No confidence scores to display.")
    fig = px.histogram(df, x="Confidence", nbins=20, template=DARK_TEMPLATE,
                        color_discrete_sequence=[ACCENT], title="Confidence Histogram")
    fig.update_layout(height=360)
    return fig


def detection_count_per_class(df: pd.DataFrame) -> go.Figure:
    if df.empty:
        return _empty_figure("No detections to display.")
    counts = df.groupby("Class").size().reset_index(name="Count")
    fig = px.pie(counts, names="Class", values="Count", template=DARK_TEMPLATE,
                 title="Detection Count per Class", hole=0.45)
    fig.update_layout(height=360)
    return fig


def confidence_distribution_box(df: pd.DataFrame) -> go.Figure:
    if df.empty:
        return _empty_figure("No confidence scores to display.")
    fig = px.box(df, x="Class", y="Confidence", template=DARK_TEMPLATE, color="Class",
                 title="Confidence Distribution by Class")
    fig.update_layout(showlegend=False, height=360)
    return fig


def inference_time_history(history: list) -> go.Figure:
    """history: list of {'run': int, 'time_ms': float}"""
    if not history:
        return _empty_figure("Run detection to build inference history.")
    df = pd.DataFrame(history)
    fig = px.line(df, x="run", y="time_ms", markers=True, template=DARK_TEMPLATE,
                   color_discrete_sequence=[ACCENT], title="Inference Time History (ms)")
    fig.update_layout(height=360, xaxis_title="Run #", yaxis_title="Time (ms)")
    return fig


def bbox_area_summary(df: pd.DataFrame) -> go.Figure:
    if df.empty:
        return _empty_figure("No detections to display.")
    fig = px.scatter(df, x="Area", y="Confidence", color="Class", template=DARK_TEMPLATE,
                      title="Bounding Box Area vs. Confidence", size="Area")
    fig.update_layout(height=360)
    return fig
