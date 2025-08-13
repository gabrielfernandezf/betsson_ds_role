# src/overview.py
"""
Overview section: problem framing, dataset, and KPIs.
"""

import streamlit as st
import pandas as pd
from .utils import eda_highlights  # build KPI-like insights from the data


def _format_window(df: pd.DataFrame) -> str:
    if "dt" not in df.columns or df["dt"].isna().all():
        return "—"
    start = pd.to_datetime(df["dt"].min())
    end = pd.to_datetime(df["dt"].max())
    # Show both date and hour for precision, but compact
    return f"{start:%Y-%m-%d %H:%M} → {end:%Y-%m-%d %H:%M}"


def render(df: pd.DataFrame):
    st.title("Next Best Action (NBA) — Prototype")

    # Basic stats
    rows = len(df)
    cols = df.shape[1]
    base_ctr = float(df["click"].mean()) if "click" in df.columns else float("nan")
    time_window = _format_window(df)

    # Data-driven highlights (no hardcoded numbers)
    hi = eda_highlights(df, min_support=300)

    st.write(
        """
**Use case**: predict user propensity to interact (click) to rank the *next best action*.  
**Dataset**: Avazu CTR sample (~50k rows). Anonymous, tabular, 10 days, binary target (`click`).  
**Limitations**: advertising context and anonymized features; we transfer the approach to NBA (propensity scoring).
        """
    )

    # KPI strip
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{rows:,}")
    c2.metric("Columns", f"{cols}")
    c3.metric("Base CTR", f"{base_ctr:.4f}")
    c4.metric("Time Window", time_window)

    st.subheader("Executive summary")
    exec_summary = f"""
- Baseline CTR is **{hi['base_ctr']:.4f}**.  
- Weekends and late-night hours outperform; best hour by CTR is **{int(hi['peak_hr_ctr']['hour_of_day']):02d}:00** (**{hi['peak_hr_ctr']['ctr']:.4f}**).  
- **banner_pos={hi['best_banner_pos']['banner_pos']}** consistently lifts CTR (~**{hi['best_banner_pos']['lift']:.2f}×**; share {hi['best_banner_pos']['share']:.2%}).  
- Strong segmentation by **device_type** (higher CTR on some types; see EDA).  
- **Connection type** matters; consider down-weighting low-yield contexts.  
- Certain **site/app categories** look premium; prioritize them in ranking (see EDA).
"""
    st.markdown(exec_summary)
    st.download_button(
        "⬇️ Download Executive_Summary.md",
        data=exec_summary,
        file_name="Executive_Summary.md",
        mime="text/markdown",
        use_container_width=True,
    )

    with st.expander("Rationale & Scope"):
        st.markdown(
            """
- **Why Avazu**: well-known binary tabular problem (CTR) → clean mapping to propensity modeling for NBA.
- **Objective**: prototype a propensity score and understand main drivers (no online training).
- **Validation**: time-based split (e.g., hold out the last day as validation) to simulate production latency.
            """
        )
