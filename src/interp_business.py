# src/interp_business.py
from pathlib import Path
import math
import pandas as pd
import streamlit as st

def _load_gain():
    p = Path("artifacts/reports/gain_table_val.csv")
    if not p.exists():
        return None
    g = pd.read_csv(p)
    # decile 9 is the top; ensure sorted
    return g.sort_values("decile", ascending=False).reset_index(drop=True)

def _expected_ctr_for_budget(gain: pd.DataFrame, budget_share: float) -> float:
    """
    Approximate CTR if we contact the top K% by score using deciles with rates.
    """
    if gain is None or gain.empty:
        return float("nan")
    # compute cumulative coverage by decile (each decile has n rows)
    total = gain["n"].sum()
    target = total * budget_share
    taken = 0
    weighted = 0.0
    for _, row in gain.iterrows():
        take = min(row["n"], max(0, target - taken))
        if take <= 0: 
            break
        weighted += (take * row["rate"])
        taken += take
    return weighted / max(taken, 1)

def render(df=None):
    st.header("Interpretability & Business")

    gain = _load_gain()
    if gain is None:
        st.error("Missing artifacts/reports/gain_table_val.csv")
        return

    st.subheader("Decision helpers")

    c1, c2 = st.columns(2)
    with c1:
        budget = st.slider("Budget / capacity (Top % of users to contact)", 1, 50, 10, step=1)
    with c2:
        cpa = st.number_input("CPA (cost per action)", min_value=0.0, value=1.0, step=0.1)
        value = st.number_input("Value per click/conversion (V)", min_value=0.0, value=5.0, step=0.1)

    # Expected CTR for top-K
    exp_ctr = _expected_ctr_for_budget(gain, budget/100.0)
    st.metric(f"Expected CTR at top {budget}%", f"{exp_ctr:.3f}")

    # Economic threshold
    p_star = (cpa / value) if value > 0 else float("inf")
    st.metric("Economic threshold p★ (act if p ≥ p★)", f"{p_star:.3f}")

    # Which deciles satisfy p ≥ p★ ?
    eligible = gain[gain["avg_p"] >= p_star]
    st.write("Deciles meeting p ≥ p★ (by avg_p):")
    st.dataframe(eligible[["decile", "n", "avg_p", "rate"]], use_container_width=True)

    st.divider()
    st.subheader("Practical guidance")
    st.markdown(
        "- If the **budget** covers ~top 10–20%, expect **~1.8–2.1×** the base CTR (per your validation).\n"
        "- Use **p★ = CPA/V** to choose the operating point; contact users with `avg_p` ≥ `p★`.\n"
        "- Down-weight/suppress segments consistently below p★ (e.g., low-yield `device_conn_type=2`) unless required for coverage.\n"
        "- Recalibrate periodically and re-tune thresholds if CPA or V change."
    )
