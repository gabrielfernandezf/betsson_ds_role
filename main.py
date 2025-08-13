# main.py
"""
App entrypoint: sidebar navigation and page orchestration.
"""

from pathlib import Path
import streamlit as st
from streamlit_option_menu import option_menu

from src.utils import load_data
from src.overview import render as render_overview
from src.eda import render as render_eda

# -----------------------
# Page config & styling
# -----------------------
st.set_page_config(page_title="NBA Prototype — Avazu CTR", layout="wide")
st.markdown(
    "<style>.block-container{padding-top:1rem;padding-bottom:2rem;}</style>",
    unsafe_allow_html=True,
)

# -----------------------
# Data path & state
# -----------------------
DATA_CSV = Path("data/avazu_50k_rows.csv")
if "min_support" not in st.session_state:
    st.session_state.min_support = 300

# -----------------------
# Load data
# -----------------------
if not DATA_CSV.exists():
    st.error("Could not find `data/avazu_50k_rows.csv`. Please add the CSV to that path in the repository.")
    st.stop()

df = load_data(DATA_CSV)

# -----------------------
# Sidebar navigation
# -----------------------
with st.sidebar:
    selected = option_menu(
        "Navigation",
        ["Overview", "EDA", "Model", "Interpretability & Business"],
        icons=["house", "bar-chart", "cpu", "lightbulb"],
        menu_icon="cast",
        default_index=0,
        orientation="vertical",
    )

# -----------------------
# Sections
# -----------------------
if selected == "Overview":
    render_overview(df)

elif selected == "EDA":
    render_eda(df)

elif selected == "Model":
    st.header("Model (coming next)")
    st.info("In the next iteration we will load precomputed artifacts: `artifacts/metrics.json`, curves, and `feature_importance.csv`.")

elif selected == "Interpretability & Business":
    st.header("Interpretability & Business (coming next)")
    st.info("We will show SHAP/feature importance and translate insights into NBA decisions.")
