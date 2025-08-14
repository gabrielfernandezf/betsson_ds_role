# # src/overview.py
# """
# Overview section: problem framing, dataset, and KPIs.
# """

# import streamlit as st
# import pandas as pd
# from .utils import eda_highlights  # build KPI-like insights from the data


# def _format_window(df: pd.DataFrame) -> str:
#     if "dt" not in df.columns or df["dt"].isna().all():
#         return "—"
#     start = pd.to_datetime(df["dt"].min())
#     end = pd.to_datetime(df["dt"].max())
#     # Show both date and hour for precision, but compact
#     return f"{start:%Y-%m-%d %H:%M} → {end:%Y-%m-%d %H:%M}"


# def render(df: pd.DataFrame):
#     st.title("Next Best Action (NBA) — Prototype")

#     # Basic stats
#     rows = len(df)
#     cols = df.shape[1]
#     base_ctr = float(df["click"].mean()) if "click" in df.columns else float("nan")
#     time_window = _format_window(df)

#     # Data-driven highlights (no hardcoded numbers)
#     hi = eda_highlights(df, min_support=300)

#     st.write(
#         """
# **Use case**: predict user propensity to interact (click) to rank the *next best action*.  
# **Dataset**: Avazu CTR sample (~50k rows). Anonymous, tabular, 10 days, binary target (`click`).  
# **Limitations**: advertising context and anonymized features; we transfer the approach to NBA (propensity scoring).
#         """
#     )

#     # KPI strip
#     c1, c2, c3, c4 = st.columns(4)
#     c1.metric("Rows", f"{rows:,}")
#     c2.metric("Columns", f"{cols}")
#     c3.metric("Base CTR", f"{base_ctr:.4f}")
#     c4.metric("Time Window", time_window)

#     st.subheader("Executive summary")
#     exec_summary = f"""
# - Baseline CTR is **{hi['base_ctr']:.4f}**.  
# - Weekends and late-night hours outperform; best hour by CTR is **{int(hi['peak_hr_ctr']['hour_of_day']):02d}:00** (**{hi['peak_hr_ctr']['ctr']:.4f}**).  
# - **banner_pos={hi['best_banner_pos']['banner_pos']}** consistently lifts CTR (~**{hi['best_banner_pos']['lift']:.2f}×**; share {hi['best_banner_pos']['share']:.2%}).  
# - Strong segmentation by **device_type** (higher CTR on some types; see EDA).  
# - **Connection type** matters; consider down-weighting low-yield contexts.  
# - Certain **site/app categories** look premium; prioritize them in ranking (see EDA).
# """
#     st.markdown(exec_summary)
#     st.download_button(
#         "⬇️ Download Executive_Summary.md",
#         data=exec_summary,
#         file_name="Executive_Summary.md",
#         mime="text/markdown",
#         use_container_width=True,
#     )

#     with st.expander("Rationale & Scope"):
#         st.markdown(
#             """
# - **Why Avazu**: well-known binary tabular problem (CTR) → clean mapping to propensity modeling for NBA.
# - **Objective**: prototype a propensity score and understand main drivers (no online training).
# - **Validation**: time-based split (e.g., hold out the last day as validation) to simulate production latency.
#             """
#         )
# src/overview.py
"""
Overview section: problem framing, dataset, and KPIs (clean layout).
"""

import streamlit as st
import pandas as pd
from .utils import eda_highlights  # data-driven KPIs


def _format_period_short(start: pd.Timestamp, end: pd.Timestamp) -> str:
    """Return a short, human-friendly period like 'Oct 21–30, 2014'."""
    if start.year == end.year and start.month == end.month:
        return f"{start:%b} {start.day}–{end.day}, {end.year}"
    if start.year == end.year:
        return f"{start:%b %d} – {end:%b %d}, {end.year}"
    return f"{start:%b %d, %Y} – {end:%b %d, %Y}"


def render(df: pd.DataFrame):
    st.title("Next Best Action (NBA) — Prototype")

    # --- Basic stats ---
    base_ctr = float(df["click"].mean()) if "click" in df.columns else float("nan")
    start_dt = pd.to_datetime(df["dt"].min()) if "dt" in df.columns else None
    end_dt   = pd.to_datetime(df["dt"].max()) if "dt" in df.columns else None
    unique_days = int(df["date"].nunique()) if "date" in df.columns else None
    period_short = _format_period_short(start_dt, end_dt) if start_dt is not None else "—"

    # --- Highlights for exec summary ---
    hi = eda_highlights(df, min_support=300)
    bp = hi.get("best_banner_pos")

    # --- Framing copy (más claro y profesional) ---
    st.markdown(
        """
**What this is:** a lightweight, transparent **propensity model** to support a *Next Best Action* (NBA) decision.  
**What it does:** estimates \\(p(\\text{click}\\mid\\text{context})\\) and uses it to **rank actions**, exposing drivers and trade-offs.  
**Dataset:** public Avazu CTR sample (~50k impressions, 10 days, anonymized).  
**Why this approach:** CTR propensity is a close proxy for commercial intent and easily transfers to NBA policies (probability × value − cost).  
**What we report:** AUC/PR-AUC, LogLoss, calibration, and **lift@groups** for targeting.  
**Limitations:** advertising context; anonymized IDs; no user history. We mitigate with time-based validation and careful encodings.
        """
    )

    # --- KPIs in two neat rows (no truncation, sin 'delta' engañoso) ---
    r1c1, r1c2, r1c3 = st.columns(3)
    with r1c1:
        st.metric("Rows", f"{len(df):,}")
    with r1c2:
        st.metric("Columns", f"{df.shape[1]}")
    with r1c3:
        st.metric("Base CTR", f"{base_ctr:.4f}")

    r2c1, r2c2, r2c3 = st.columns(3)
    with r2c1:
        st.metric("Days covered", f"{unique_days}" if unique_days is not None else "—")
        st.caption(period_short)
    with r2c2:
        st.metric("Start", f"{start_dt:%Y-%m-%d}" if start_dt is not None else "—")
        st.caption(f"{start_dt:%H:%M}" if start_dt is not None else "")
    with r2c3:
        st.metric("End", f"{end_dt:%Y-%m-%d}" if end_dt is not None else "—")
        st.caption(f"{end_dt:%H:%M}" if end_dt is not None else "")

    st.subheader("Executive summary")
    bullets = [
        f"- Baseline CTR: **{hi['base_ctr']:.4f}**.",
        f"- Time effects: late-night hours outperform; best hour by CTR is **{int(hi['peak_hr_ctr']['hour_of_day']):02d}:00** "
        f"(**{hi['peak_hr_ctr']['ctr']:.4f}**); reach peaks around **{int(hi['peak_hr_impr']['hour_of_day']):02d}:00** "
        f"(N={int(hi['peak_hr_impr']['impressions'])}, CTR={hi['peak_hr_impr']['ctr']:.4f}).",
        "- Strong segmentation by **device_type** and **device_conn_type** (see EDA).",
        "- Certain **site/app categories** behave as *premium contexts* (see EDA).",
    ]
    if bp:
        bullets.insert(1, f"- Placement: **banner_pos={bp['banner_pos']}** lifts CTR to **{bp['ctr']:.4f}** "
                          f"(~**{bp['lift']:.2f}×**) with share **{bp['share']:.2%}**.")
    st.markdown("\n".join(bullets))

    st.download_button(
        "⬇️ Download Executive_Summary.md",
        data="\n".join(bullets),
        file_name="Executive_Summary.md",
        mime="text/markdown",
        use_container_width=True,
    )

    with st.expander("Rationale & scope"):
        st.markdown(
            """
- **Objective:** validate a practical path from EDA → propensity model → NBA policy with transparent drivers.  
- **Validation:** hold out the **last day** to simulate production; optional time-based CV for tuning.  
- **Feature plan (high level):** temporal (hour/dow), low-card categoricals (banner_pos, device_type, conn_type),  
  mid-card anonymous blocks (C14–C21), careful treatment of high-card IDs (freq/hashing), and selected interactions.  
- **Success criteria:** solid AUC/PR-AUC & LogLoss, good **lift at business-relevant cutoffs**, and clear explainability.
            """
        )


