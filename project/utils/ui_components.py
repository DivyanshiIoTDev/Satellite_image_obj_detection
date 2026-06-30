"""
ui_components.py
------------------
Reusable Streamlit UI building blocks (metric cards, section wrappers,
CSS loader) shared across every page so the dashboard stays visually
consistent.
"""

import os
import streamlit as st

ASSETS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets")


def load_css():
    """Inject the shared dark-theme stylesheet. Safe no-op if missing."""
    css_path = os.path.join(ASSETS_DIR, "style.css")
    if os.path.exists(css_path):
        with open(css_path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


def page_header(title: str, subtitle: str = ""):
    st.markdown(f'<div class="app-title">{title}</div>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<div class="app-subtitle">{subtitle}</div>', unsafe_allow_html=True)


def metric_card(label: str, value: str):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_card_start(title: str, subtitle: str = ""):
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<div class="section-subtitle">{subtitle}</div>', unsafe_allow_html=True)


def section_card_end():
    st.markdown('</div>', unsafe_allow_html=True)


def placeholder(message: str):
    st.markdown(f'<div class="placeholder-box">{message}</div>', unsafe_allow_html=True)
