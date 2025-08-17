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
from src.model import render as render_model
from src.interp_business import render as render_interp

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
# with st.sidebar:
#     st.markdown("### About")
#     st.markdown("**Gabriel Fernández** · Senior DS Candidate")
#     st.markdown("[LinkedIn](https://www.linkedin.com/in/gabriel-data-science/)")

# -----------------------
# Sections
# -----------------------
if selected == "Overview":
    render_overview(df)

elif selected == "EDA":
    render_eda(df)

elif selected == "Model":
    render_model(df)
elif selected == "Interpretability & Business":
    render_interp(df)
