# # src/overview.py
# """
# Overview: clean presentation with section subtitles + bullet points.
# """

# import streamlit as st
# import pandas as pd
# from .utils import eda_highlights

# st.set_page_config(
#     page_title="NBA Prototype",
#     page_icon="📈",
#     layout="wide",
#     initial_sidebar_state="expanded",
# )

# def render(df: pd.DataFrame):
#     st.title("Next Best Action (NBA) — Prototype")

#     # Time window / basics
#     base_ctr = float(df["click"].mean()) if "click" in df.columns else float("nan")
#     start_dt = pd.to_datetime(df["dt"].min()) if "dt" in df.columns else None
#     end_dt   = pd.to_datetime(df["dt"].max()) if "dt" in df.columns else None
#     days_cov = int(df["date"].nunique()) if "date" in df.columns else None

#     # Data-driven highlights
#     hi = eda_highlights(df, min_support=300)
#     bp = hi.get("best_banner_pos")

#     # --- What this is ---
#     st.subheader("What this is")
#     st.markdown(
#         """
# - A lightweight, transparent **propensity model** to support a **Next Best Action** decision.
# - We estimate **p(click | context)** and use it to **rank actions**, exposing drivers and trade-offs.
# - Results and artifacts are shown in this Streamlit app for quick review during the interview.
#         """
#     )

#     # --- Dataset ---
#     st.subheader("Dataset")
#     st.markdown(
#         f"""
# - **Source**: public Avazu CTR sample (anonymized).
# - **Rows**: **{len(df):,}** · **Columns**: **{df.shape[1]}**
# - **Period**: **{start_dt:%Y-%m-%d %H:%M} → {end_dt:%Y-%m-%d %H:%M}** · **Days covered**: **{days_cov}**
# - **Target**: `click` (binary)
#         """
#     )

#     # --- Executive summary (EDA-driven bullets) ---
#     st.subheader("Executive summary (EDA highlights)")
#     bullets = []
#     bullets.append(f"- Baseline CTR: **{hi['base_ctr']:.4f}**.")
#     if bp:
#         bullets.append(
#             f"- Placement: **banner_pos={bp['banner_pos']}** → CTR **{bp['ctr']:.4f}** "
#             f"(~**{bp['lift']:.2f}×**) with share **{bp['share']:.2%}**."
#         )
#     bullets.append(
#         f"- Time effects: best hour by CTR is **{int(hi['peak_hr_ctr']['hour_of_day']):02d}:00** "
#         f"(**{hi['peak_hr_ctr']['ctr']:.4f}**); reach peaks around **{int(hi['peak_hr_impr']['hour_of_day']):02d}:00** "
#         f"(N={int(hi['peak_hr_impr']['impressions'])}, CTR={hi['peak_hr_impr']['ctr']:.4f})."
#     )
#     bullets.append("- Strong segmentation by **device_type** and **device_conn_type**.")
#     bullets.append("- **Premium contexts** in specific site/app categories (see EDA).")
#     st.markdown("\n".join(bullets))

#     st.download_button(
#         "⬇️ Download Executive_Summary.md",
#         data="\n".join([b.lstrip("- ").strip() for b in bullets]),
#         file_name="Executive_Summary.md",
#         mime="text/markdown",
#         use_container_width=True,
#     )

#     # --- How we'll evaluate & decide ---
#     st.subheader("How we evaluate & decide")
#     st.markdown(
#         """
# - **Validation**: time-based split (hold out the last day) to mimic production latency.
# - **Metrics**: ROC-AUC & PR-AUC (ranking), **LogLoss** (probabilistic accuracy), calibration check, and **lift/gain at K%**.
# - **From scores to actions (NBA)**: rank by propensity and align with **value − cost** constraints.
#         """
#     )

#     # --- What’s next (for the interview flow) ---
#     st.subheader("What’s next")
#     st.markdown(
#         """
# - Walk through **EDA** tabs (temporal, categoricals, interactions, quality).
# - Show **baseline model** → **tuned model**, metrics & interpretability.
# - Demo **how scores appear in the app** and how we’d wire them into a simple NBA policy.
#         """
#     )

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


def _fmt_window(df: pd.DataFrame) -> str:
    if "dt" not in df.columns or df["dt"].isna().all():
        return "—"
    start = pd.to_datetime(df["dt"].min())
    end = pd.to_datetime(df["dt"].max())
    return f"{start:%Y-%m-%d %H:%M} → {end:%Y-%m-%d %H:%M}"


def render(df: pd.DataFrame):
    # Title
    st.title("Next Best Action (NBA) — Prototype")

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
    st.subheader("How to review (interview flow)")
    st.markdown(
        """
1) Start with **EDA** to align on data shape and drivers.  
2) Move to **Model** to compare baseline vs tuned + calibrated, and validate **calibration** and **lift**.  
3) Close with **Interpretability & Business** to show **how scores become decisions** (budget → CTR; `p★ = CPA/V`).  
        """
    )
