"""
config_loader.py
-----------------
Loads config.yaml once (cached) and exposes typed accessor helpers so the
rest of the app never touches raw dict/yaml plumbing.
"""

import os
import yaml
import streamlit as st

CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.yaml")


@st.cache_data(show_spinner=False)
def load_config(path: str = CONFIG_PATH) -> dict:
    """Load and cache the YAML configuration file.

    Returns an empty-but-valid structure if the file is missing so the
    app degrades gracefully instead of crashing.
    """
    if not os.path.exists(path):
        st.warning(f"⚠️ Configuration file not found at `{path}`. Using empty defaults.")
        return {
            "model_paths": {}, "training_outputs": {}, "model_info": {},
            "research_comparison": [], "class_names": [],
        }
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def get_model_path(model_key: str) -> str:
    cfg = load_config()
    return cfg.get("model_paths", {}).get(model_key, "")


def get_training_outputs(model_key: str) -> dict:
    cfg = load_config()
    return cfg.get("training_outputs", {}).get(model_key, {})


def get_model_info(model_key: str) -> dict:
    cfg = load_config()
    return cfg.get("model_info", {}).get(model_key, {})


def get_research_comparison() -> list:
    cfg = load_config()
    return cfg.get("research_comparison", [])


def get_class_names() -> list:
    cfg = load_config()
    return cfg.get("class_names", [])
