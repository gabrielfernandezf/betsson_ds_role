# src/overview.py
"""
Overview: clean presentation with section subtitles + bullet points.
"""

import streamlit as st
import pandas as pd
from .utils import eda_highlights

st.set_page_config(
    page_title="NBA Prototype",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

def render(df: pd.DataFrame):
    st.title("Next Best Action (NBA) — Prototype")

    # Time window / basics
    base_ctr = float(df["click"].mean()) if "click" in df.columns else float("nan")
    start_dt = pd.to_datetime(df["dt"].min()) if "dt" in df.columns else None
    end_dt   = pd.to_datetime(df["dt"].max()) if "dt" in df.columns else None
    days_cov = int(df["date"].nunique()) if "date" in df.columns else None

    # Data-driven highlights
    hi = eda_highlights(df, min_support=300)
    bp = hi.get("best_banner_pos")

    # --- What this is ---
    st.subheader("What this is")
    st.markdown(
        """
- A lightweight, transparent **propensity model** to support a **Next Best Action** decision.
- We estimate **p(click | context)** and use it to **rank actions**, exposing drivers and trade-offs.
- Results and artifacts are shown in this Streamlit app for quick review during the interview.
        """
    )

    # --- Dataset ---
    st.subheader("Dataset")
    st.markdown(
        f"""
- **Source**: public Avazu CTR sample (anonymized).
- **Rows**: **{len(df):,}** · **Columns**: **{df.shape[1]}**
- **Period**: **{start_dt:%Y-%m-%d %H:%M} → {end_dt:%Y-%m-%d %H:%M}** · **Days covered**: **{days_cov}**
- **Target**: `click` (binary)
        """
    )

    # --- Executive summary (EDA-driven bullets) ---
    st.subheader("Executive summary (EDA highlights)")
    bullets = []
    bullets.append(f"- Baseline CTR: **{hi['base_ctr']:.4f}**.")
    if bp:
        bullets.append(
            f"- Placement: **banner_pos={bp['banner_pos']}** → CTR **{bp['ctr']:.4f}** "
            f"(~**{bp['lift']:.2f}×**) with share **{bp['share']:.2%}**."
        )
    bullets.append(
        f"- Time effects: best hour by CTR is **{int(hi['peak_hr_ctr']['hour_of_day']):02d}:00** "
        f"(**{hi['peak_hr_ctr']['ctr']:.4f}**); reach peaks around **{int(hi['peak_hr_impr']['hour_of_day']):02d}:00** "
        f"(N={int(hi['peak_hr_impr']['impressions'])}, CTR={hi['peak_hr_impr']['ctr']:.4f})."
    )
    bullets.append("- Strong segmentation by **device_type** and **device_conn_type**.")
    bullets.append("- **Premium contexts** in specific site/app categories (see EDA).")
    st.markdown("\n".join(bullets))

    st.download_button(
        "⬇️ Download Executive_Summary.md",
        data="\n".join([b.lstrip("- ").strip() for b in bullets]),
        file_name="Executive_Summary.md",
        mime="text/markdown",
        use_container_width=True,
    )

    # --- How we'll evaluate & decide ---
    st.subheader("How we evaluate & decide")
    st.markdown(
        """
- **Validation**: time-based split (hold out the last day) to mimic production latency.
- **Metrics**: ROC-AUC & PR-AUC (ranking), **LogLoss** (probabilistic accuracy), calibration check, and **lift/gain at K%**.
- **From scores to actions (NBA)**: rank by propensity and align with **value − cost** constraints.
        """
    )

    # --- What’s next (for the interview flow) ---
    st.subheader("What’s next")
    st.markdown(
        """
- Walk through **EDA** tabs (temporal, categoricals, interactions, quality).
- Show **baseline model** → **tuned model**, metrics & interpretability.
- Demo **how scores appear in the app** and how we’d wire them into a simple NBA policy.
        """
    )
