# src/overview.py
"""
Overview: clean presentation with subtitles + bullet points.
- Welcome & assessment context
- Scope & approach (dataset, modeling, validation, anti-leakage)
- What you'll find in this app (navigation guide)
- Quick data-driven highlights (from EDA)
"""

import pandas as pd
import streamlit as st
from .utils import eda_highlights

def _divider():
    st.markdown("<hr style='margin: 0.75rem 0 1rem 0; border: none; height: 1px; background:#e5e7eb;'>", unsafe_allow_html=True)

def _fmt_window(df: pd.DataFrame) -> str:
    if "dt" not in df.columns or df["dt"].isna().all():
        return "—"
    start = pd.to_datetime(df["dt"].min())
    end = pd.to_datetime(df["dt"].max())
    return f"{start:%Y-%m-%d %H:%M} → {end:%Y-%m-%d %H:%M}"


def render(df: pd.DataFrame):
    # Title
    st.title("Next Best Action (NBA) — Prototype")
    _divider()

    # Basic facts for bullets
    rows = len(df)
    cols = df.shape[1]
    base_ctr = float(df["click"].mean()) if "click" in df.columns else float("nan")
    window = _fmt_window(df)
    days = df["date"].nunique() if "date" in df.columns else None

    # Data-driven highlights (no hardcoded numbers)
    hi = eda_highlights(df, min_support=300)
    bp = hi.get("best_banner_pos")

    # ---------------------------
    # Welcome & assessment context
    # ---------------------------
    st.subheader("Welcome")
    st.markdown(
        """
- This app packages my **Senior Data Scientist assessment** as a compact, reviewable prototype.
- The objective is to show how I **frame the problem**, **analyze data**, **build models**, and **connect them to business decisions**.
- Everything is reproducible: the app reads versioned **artifacts** (metrics/tables/plots) exported from the training notebook.
        """
    )
    
    _divider()
    
    st.subheader("Assessment summary")
    st.markdown(
        f"""
- **Use case:** prioritize a *Next Best Action* by estimating **p(click | context)** and ranking users/actions.
- **Dataset:** public **Avazu CTR** sample (anonymized), **{rows:,} rows**, **{cols} columns**, **{days} days** ({window}).
- **Target:** `click` (binary). **Baseline CTR:** **{hi['base_ctr']:.4f}**.
- **Constraints:** anonymized features, advertising context; we transfer the method to an NBA setting (propensity scoring).
        """
    )

    # ---------------------------
    # Scope & Approach
    # ---------------------------
    _divider()
    
    st.subheader("Scope & approach")
    st.markdown(
        """
- **Problem framing:** score **propensity to click** to support ranking/policy (contact who, when, where).
- **Feature engineering (EDA-driven):**
  - **Time**: cyclic hour (`sin/cos`) + `is_night` (night stands out).
  - **Interactions**: `hour × banner_pos`, `hour × device_type` (patterns differ by time).
  - **Category stability**: rare-level collapse (min support), then **smoothed Target Encoding** (learned only on train).
  - **Anti-leakage:** remove high-card IDs (`device_ip`, `device_id`, `id`, …) and use **time-based split**.
- **Models compared:**
  - **Logistic Regression (baseline)** — interpretable, fast, great as a reference.
  - **LightGBM (tuned + calibrated)** — non-linearities/interactions + **isotonic calibration** for reliable probabilities.
- **Validation:** hold out the **last day** to emulate production latency; report **AUC/PR-AUC**, **LogLoss**, **Brier**, and **Lift@K**.
- **Decision layer:** translate probabilities into action using **economic threshold** `p★ = CPA / V` (act if `p ≥ p★`).
        """
    )

    # ---------------------------
    # What you'll find in this app
    # ---------------------------
    _divider()
    
    st.subheader("What you’ll find in this app (navigation guide)")
    st.markdown(
        """
- **Overview (this page):** assessment context, scope/approach, and quick highlights.
- **EDA:** temporal patterns (day/hour), key categoricals (e.g., `banner_pos`, `device_type`, `device_conn_type`), interaction matrices, data quality, and **downloadable** tables/summaries.
- **Model:** side-by-side view of **baseline** vs **LightGBM** — metrics, **calibration curve**, **gain table**, **feature importances**, and **partial dependence**.
- **Interpretability & Business:** a small **simulator**:
  - Choose **budget/capacity** (top K%) → see **expected CTR** from the gain table.
  - Set **CPA** and **value V** → compute **`p★ = CPA/V`** and highlight which segments/deciles are **economically viable**.
        """
    )

    # ---------------------------
    # Quick highlights (from EDA)
    # ---------------------------
    _divider()
    
    st.subheader("Quick highlights (from EDA)")
    bullets = []
    bullets.append(f"- **Baseline CTR**: **{hi['base_ctr']:.4f}**.")
    bullets.append(
        f"- **Time effects**: best hour by CTR → **{int(hi['peak_hr_ctr']['hour_of_day']):02d}:00** "
        f"(**{hi['peak_hr_ctr']['ctr']:.4f}**); reach peaks ~**{int(hi['peak_hr_impr']['hour_of_day']):02d}:00** "
        f"(N={int(hi['peak_hr_impr']['impressions'])}, CTR={hi['peak_hr_impr']['ctr']:.4f})."
    )
    if bp:
        bullets.append(
            f"- **Placement**: `banner_pos={bp['banner_pos']}` shows CTR **{bp['ctr']:.4f}** "
            f"(~**{bp['lift']:.2f}×** over base; share **{bp['share']:.2%}**)."
        )
    bullets.append("- **Device signals** matter (`device_type`, `device_conn_type`), and some site/app categories act as **premium contexts**.")
    bullets.append("- These patterns directly motivated the **feature engineering** and **interactions** used in the models.")
    st.markdown("\n".join(bullets))

    # ---------------------------
    # How to review during the interview
    # ---------------------------
    _divider()
    
    st.subheader("How to review (interview flow)")
    st.markdown(
        """
1) Start with **EDA** to align on data shape and drivers.  
2) Move to **Model** to compare baseline vs tuned + calibrated, and validate **calibration** and **lift**.  
3) Close with **Interpretability & Business** to show **how scores become decisions** (budget → CTR; `p★ = CPA/V`).  
        """
    )
