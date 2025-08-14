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
Executive Overview: polished hero, KPI cards, badges, and summary callouts.
"""

import streamlit as st
import pandas as pd
from .utils import eda_highlights  # data-driven KPIs


# ---------- Small helpers ----------
def _format_period_short(start: pd.Timestamp, end: pd.Timestamp) -> str:
    if start is None or end is None:
        return "—"
    if start.year == end.year and start.month == end.month:
        return f"{start:%b} {start.day}–{end.day}, {end.year}"
    if start.year == end.year:
        return f"{start:%b %d} – {end:%b %d}, {end.year}"
    return f"{start:%b %d, %Y} – {end:%b %d, %Y}"


def _inject_css():
    st.markdown(
        """
<style>
:root{
  --bg: #0b1020;           /* hero bg dark */
  --card: #101728;         /* card bg */
  --muted: #9aa4b2;        /* secondary text */
  --text: #e6ebf2;         /* primary text */
  --line: #1e2a3f;         /* subtle border */
  --accent: #4f46e5;       /* indigo */
  --accent-2: #00b894;     /* teal */
  --warn: #f59e0b;         /* amber */
}
html, body, [data-testid="stAppViewContainer"] {
  font-variant-numeric: lining-nums;
}
.hero {
  background: radial-gradient(1200px 600px at 20% -10%, rgba(79,70,229,.25), transparent),
              radial-gradient(800px 400px at 110% 10%, rgba(0,184,148,.20), transparent),
              var(--bg);
  border-radius: 14px;
  padding: 28px 28px 18px 28px;
  color: var(--text);
  border: 1px solid var(--line);
}
.hero h1{
  margin: 0 0 6px 0;
  font-weight: 700;
  letter-spacing: .2px;
}
.hero p.lead{
  margin: 4px 0 14px 0;
  color: var(--muted);
  font-size: 0.98rem;
}
.pills {
  display: flex; gap: 8px; flex-wrap: wrap; margin-top: 6px;
}
.pill {
  background: linear-gradient(180deg, rgba(255,255,255,.08), rgba(255,255,255,.02));
  border: 1px solid var(--line);
  border-radius: 999px;
  padding: 6px 10px;
  font-size: 0.83rem;
  color: var(--text);
}
.grid3, .grid2 {
  display: grid;
  gap: 12px;
}
.grid3 { grid-template-columns: repeat(3, minmax(0, 1fr)); }
.grid2 { grid-template-columns: repeat(2, minmax(0, 1fr)); }
@media (max-width: 1100px){ .grid3 { grid-template-columns: 1fr 1fr; } }
@media (max-width: 800px){ .grid3, .grid2 { grid-template-columns: 1fr; } }

.kpi {
  background: var(--card);
  border: 1px solid var(--line);
  border-radius: 14px;
  padding: 14px 16px;
  color: var(--text);
}
.kpi .label{
  color: var(--muted);
  font-size: .80rem;
  margin-bottom: 6px;
}
.kpi .value{
  font-size: 1.45rem;
  font-weight: 700;
  letter-spacing: .3px;
}

.card {
  background: var(--card);
  border: 1px solid var(--line);
  border-radius: 14px;
  padding: 16px 18px;
  color: var(--text);
}
.card h4{
  margin: 0 0 6px 0;
  font-size: 1rem;
}
.card p{
  margin: 0; color: var(--muted); font-size: 0.94rem;
}

.callout {
  border-left: 4px solid var(--accent);
  background: rgba(79,70,229,.08);
  border-radius: 10px;
  padding: 12px 14px;
  color: var(--text);
  border: 1px solid var(--line);
}
.callout ul{ margin: 8px 0 0 18px; }
.callout li{ margin: 2px 0; }

.subtle { color: var(--muted); font-size: .90rem; }

hr.sep { border: none; border-top: 1px solid var(--line); margin: 16px 0; }
</style>
        """,
        unsafe_allow_html=True,
    )


def _kpi(label: str, value: str, caption: str | None = None):
    st.markdown(
        f"""
<div class="kpi">
  <div class="label">{label}</div>
  <div class="value">{value}</div>
  {f'<div class="subtle">{caption}</div>' if caption else ''}
</div>
""",
        unsafe_allow_html=True,
    )


# ---------- Page render ----------
def render(df: pd.DataFrame):
    _inject_css()

    # Basic numbers
    base_ctr = float(df["click"].mean()) if "click" in df.columns else float("nan")
    start_dt = pd.to_datetime(df["dt"].min()) if "dt" in df.columns else None
    end_dt   = pd.to_datetime(df["dt"].max()) if "dt" in df.columns else None
    unique_days = int(df["date"].nunique()) if "date" in df.columns else None
    period_short = _format_period_short(start_dt, end_dt)
    hi = eda_highlights(df, min_support=300)
    bp = hi.get("best_banner_pos")

    # ---------- HERO ----------
    st.markdown(
        f"""
<div class="hero">
  <h1>Next Best Action (NBA) — Prototype</h1>
  <p class="lead">
    A transparent propensity model for ranking actions in marketing-like contexts.
    We predict <em>p(click | context)</em>, surface the main drivers, and show how this translates into a practical NBA policy.
  </p>
  <div class="pills">
    <span class="pill">Dataset: Avazu CTR sample (~50k rows)</span>
    <span class="pill">Binary target: click</span>
    <span class="pill">Period: {period_short}</span>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    # ---------- KPI CARDS (two rows) ----------
    col = st.columns(1)[0]
    with col:
        st.markdown('<div class="grid3">', unsafe_allow_html=True)
        _kpi("Rows", f"{len(df):,}")
        _kpi("Columns", f"{df.shape[1]}")
        _kpi("Base CTR", f"{base_ctr:.4f}")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

        st.markdown('<div class="grid3">', unsafe_allow_html=True)
        _kpi("Days covered", f"{unique_days if unique_days is not None else '—'}", period_short)
        _kpi("Start", f"{start_dt:%Y-%m-%d}" if start_dt else "—", f"{start_dt:%H:%M}" if start_dt else "")
        _kpi("End",   f"{end_dt:%Y-%m-%d}"   if end_dt   else "—", f"{end_dt:%H:%M}"   if end_dt   else "")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<hr class='sep'/>", unsafe_allow_html=True)

    # ---------- EXECUTIVE SUMMARY (callout) ----------
    bullets = []
    bullets.append(f"Baseline CTR: **{hi['base_ctr']:.4f}**.")
    if bp:
        bullets.append(
            f"Placement: **banner_pos={bp['banner_pos']}** lifts CTR to **{bp['ctr']:.4f}** "
            f"(~**{bp['lift']:.2f}×**) with share **{bp['share']:.2%}**."
        )
    bullets.append(
        f"Time effects: late-night hours outperform; best hour by CTR is **{int(hi['peak_hr_ctr']['hour_of_day']):02d}:00** "
        f"(**{hi['peak_hr_ctr']['ctr']:.4f}**). Reach peaks around **{int(hi['peak_hr_impr']['hour_of_day']):02d}:00** "
        f"(N={int(hi['peak_hr_impr']['impressions'])}, CTR={hi['peak_hr_impr']['ctr']:.4f})."
    )
    bullets.append("Strong segmentation by **device_type** and **device_conn_type** (see EDA).")
    bullets.append("Certain **site/app categories** behave as *premium contexts* (see EDA).")

    st.subheader("Executive summary")
    st.markdown(
        f"""
<div class="callout">
  <ul>
    {''.join([f'<li>{b}</li>' for b in bullets])}
  </ul>
</div>
""",
        unsafe_allow_html=True,
    )

    st.download_button(
        "⬇️ Download Executive_Summary.md",
        data="\n".join([b.lstrip("- ").strip() for b in bullets]),
        file_name="Executive_Summary.md",
        mime="text/markdown",
        use_container_width=True,
    )

    st.markdown("<hr class='sep'/>", unsafe_allow_html=True)

    # ---------- PROJECT AT A GLANCE (cards) ----------
    st.subheader("Project at a glance")
    st.markdown('<div class="grid2">', unsafe_allow_html=True)

    st.markdown(
        """
<div class="card">
  <h4>Use case</h4>
  <p>Rank the next best action by estimated propensity to click. Start with CTR as a proxy for intent; extend to NBA by combining probability × value − cost under constraints.</p>
</div>
""", unsafe_allow_html=True)

    st.markdown(
        """
<div class="card">
  <h4>Validation</h4>
  <p>Time-based split: hold out the last day as validation to mimic production latency. Optionally add time-based CV for hyperparameter tuning.</p>
</div>
""", unsafe_allow_html=True)

    st.markdown(
        """
<div class="card">
  <h4>Features</h4>
  <p>Temporal (hour, DOW), low-card categoricals (banner_pos, device_type, conn_type), anonymous blocks (C14–C21), careful treatment of high-card IDs (frequency/hashing), plus selective interactions.</p>
</div>
""", unsafe_allow_html=True)

    st.markdown(
        """
<div class="card">
  <h4>Metrics</h4>
  <p>ROC-AUC & PR-AUC for ranking quality, LogLoss for probabilistic accuracy, calibration check, and lift/gain at business-relevant cutoffs.</p>
</div>
""", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    with st.expander("Rationale & scope"):
        st.markdown(
            """
- **Objective:** show an auditable path from EDA → propensity model → NBA policy with clear drivers.  
- **Assumptions:** anonymized ads data; we focus on structure, not domain-specific IDs.  
- **Risks & mitigations:** leakage from high-card IDs → use frequency/hashing; validate only with forward-time splits.  
- **Success criteria:** solid metrics + explainability + reproducibility (artifacts & code).
            """
        )

