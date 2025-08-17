# src/interp_business.py
from pathlib import Path
import math
import pandas as pd
import streamlit as st
import altair as alt

# ------------ Loaders & helpers ------------
def _load_gain():
    """
    Expects artifacts/reports/gain_table_val.csv with columns:
    decile (0..9 bottom->top), n, positives, avg_p (or rate).
    """
    p = Path("artifacts/reports/gain_table_val.csv")
    if not p.exists():
        return None
    g = pd.read_csv(p)
    # Ensure we have consistent columns
    if "rate" not in g.columns and "avg_p" in g.columns:
        g["rate"] = g["avg_p"]
    if "avg_p" not in g.columns and "rate" in g.columns:
        g["avg_p"] = g["rate"]
    # decile 9 is top; sort top->bottom
    g = g.sort_values("decile", ascending=False).reset_index(drop=True)
    return g[["decile", "n", "positives", "avg_p", "rate"]]

def _expected_ctr_for_budget(gain: pd.DataFrame, budget_share: float) -> float:
    """
    Approximate CTR if we contact the top K% by score using deciles (allow partial take from the next decile).
    """
    if gain is None or gain.empty:
        return float("nan")
    total = gain["n"].sum()
    target = total * budget_share
    taken = 0
    weighted = 0.0
    for _, row in gain.iterrows():
        take = min(row["n"], max(0, target - taken))
        if take <= 0:
            break
        weighted += take * row["rate"]
        taken += take
    return weighted / max(taken, 1)

def _cumulative_by_decile(gain: pd.DataFrame) -> pd.DataFrame:
    """
    Build cumulative table for discrete operating points at 10%, 20%, ... 100%.
    """
    g = gain.copy()
    g["cum_n"] = g["n"].cumsum()
    g["cum_pos"] = g["positives"].cumsum()
    g["cum_rate"] = g["cum_pos"] / g["cum_n"]
    # K = 10% for the first (top) decile, 20% for two deciles, etc.
    g["K"] = (g.index + 1) * 10
    return g

def _econ_for_topK(row, cpa: float, value: float) -> dict:
    """
    Compute economics for a cumulative operating point (discrete deciles).
    """
    n = float(row["cum_n"])
    p = float(row["cum_rate"])
    exp_clicks = n * p
    cost = n * cpa
    rev = exp_clicks * value
    profit = rev - cost
    roi = (rev / cost - 1.0) if cost > 0 else float("inf")
    return {"n": n, "p": p, "exp_clicks": exp_clicks, "cost": cost, "rev": rev, "profit": profit, "roi": roi}

def _argmax_profit(cum_df: pd.DataFrame, cpa: float, value: float) -> dict | None:
    """
    Return the row (dict) of K that maximizes expected profit across discrete deciles.
    """
    if cum_df is None or cum_df.empty:
        return None
    best = None
    for _, r in cum_df.iterrows():
        econ = _econ_for_topK(r, cpa, value)
        row = {**r.to_dict(), **econ}
        if (best is None) or (row["profit"] > best["profit"]):
            best = row
    return best

# ------------ Page render ------------
def render(df=None):
    st.header("Interpretability & Business")

    # --- How to use this page ---
    st.markdown("#### How to use this page")
    st.markdown(
        """
1) **Set budget & economics** (CPA and value per click/conversion **V**).  
2) We show the **expected CTR** if you contact the **top K%** by score (mixing deciles if needed).  
3) We compute the **economic threshold** `p★ = CPA / V` and highlight which **deciles** meet `p ≥ p★`.  
4) We also search the **best discrete operating point** (10%, 20%, …) that **maximizes expected profit**.  
**Goal:** prove the scores are **actionable** for NBA (budget allocation, thresholds, ROI).
"""
    )

    gain = _load_gain()
    if gain is None or gain.empty:
        st.error("Missing artifacts/reports/gain_table_val.csv")
        return

    base_ctr = gain["positives"].sum() / gain["n"].sum()

    # Controls
    st.subheader("Set parameters")
    c1, c2 = st.columns(2)
    with c1:
        budget = st.slider("Budget / capacity (Top % of users to contact)", 1, 50, 10, step=1)
    with c2:
        cpa = st.number_input("CPA (cost per action)", min_value=0.0, value=1.0, step=0.1)
        value = st.number_input("Value per click/conversion (V)", min_value=0.0, value=5.0, step=0.1)

    # Expected CTR for top-K (continuous approximation using deciles)
    exp_ctr = _expected_ctr_for_budget(gain, budget / 100.0)
    p_star = (cpa / value) if value > 0 else float("inf")
    delta_vs_base = (exp_ctr / base_ctr - 1.0) * 100.0 if base_ctr > 0 else float("nan")

    # KPI row
    k1, k2, k3 = st.columns(3)
    k1.metric(f"Expected CTR at top {budget}%", f"{exp_ctr:.3f}", delta=f"{delta_vs_base:+.0f}% vs base ({base_ctr:.3f})")
    k2.metric("Economic threshold p★", f"{p_star:.3f}", help="Act if p ≥ p★ = CPA / V")
    k3.metric("Base CTR (validation)", f"{base_ctr:.3f}")

    # Eligible deciles for p★
    eligible = gain[gain["avg_p"] >= p_star][["decile", "n", "avg_p", "rate", "positives"]]
    st.markdown("**Deciles meeting `p ≥ p★` (by avg_p):**")
    st.dataframe(eligible.reset_index(drop=True), use_container_width=True)

    # Cumulative discrete table (10%,20%,...,100%) and best-K by profit
    cum = _cumulative_by_decile(gain)
    best = _argmax_profit(cum, cpa, value)

    st.divider()
    st.subheader("Recommended operating point (discrete deciles)")
    if best is not None:
        # Display best K with economics
        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Recommended K*", f"{int(best['K'])}%")
        r2.metric("CTR at K*", f"{best['cum_rate']:.3f}")
        r3.metric("Expected profit", f"{best['profit']:.0f}")
        roi_pct = best["roi"] * 100 if math.isfinite(best["roi"]) else float("inf")
        r4.metric("ROI at K*", f"{roi_pct:.0f}%")
        st.caption("K* maximizes expected profit across 10% steps. Use this as a starting operating point.")
    else:
        st.info("Not enough data to compute a recommended K*.")

    # Charts: avg_p by decile (with threshold), and cumulative CTR vs K
    st.subheader("Where the value is")
    ch1, ch2 = st.columns(2)

    # Bar of avg_p by decile (top on the left). Add rule at p★.
    chart_df = gain[["decile", "avg_p"]].copy()
    chart_df["decile_label"] = chart_df["decile"].astype(str)
    bar = (
        alt.Chart(chart_df)
        .mark_bar()
        .encode(x=alt.X("decile_label:O", title="Decile (9=Top)"),
                y=alt.Y("avg_p:Q", title="Avg p in decile"),
                tooltip=["decile", alt.Tooltip("avg_p:Q", format=".3f")])
        .properties(height=260)
    )
    rule = alt.Chart(pd.DataFrame({"pstar": [p_star]})).mark_rule(strokeDash=[6,4]).encode(y="pstar:Q")
    ch1.altair_chart(bar + rule, use_container_width=True)

    # Line of cumulative CTR vs K
    cum_df = cum[["K", "cum_rate"]].copy()
    line = (
        alt.Chart(cum_df)
        .mark_line(point=True)
        .encode(x=alt.X("K:O", title="Top K% (discrete)"),
                y=alt.Y("cum_rate:Q", title="Cumulative CTR"),
                tooltip=["K", alt.Tooltip("cum_rate:Q", format=".3f")])
        .properties(height=260)
    )
    ch2.altair_chart(line, use_container_width=True)

    # Economics at the chosen continuous K (not strictly deciles)
    st.subheader("Economics at your chosen budget")
    total_n = gain["n"].sum()
    selected_n = int(total_n * (budget / 100.0))
    exp_clicks = selected_n * exp_ctr
    cost = selected_n * cpa
    revenue = exp_clicks * value
    profit = revenue - cost
    roi = (revenue / cost - 1.0) if cost > 0 else float("inf")

    e1, e2, e3, e4 = st.columns(4)
    e1.metric("Users contacted", f"{selected_n:,}")
    e2.metric("Expected clicks", f"{int(exp_clicks):,}")
    e3.metric("Expected profit", f"{profit:,.0f}")
    e4.metric("ROI", f"{(roi*100):.0f}%" if math.isfinite(roi) else "∞")

    # Practical guidance
    st.divider()
    st.subheader("Practical guidance")
    st.markdown(
        f"""
- If your **budget** covers ~top 10–20%, expect **~1.8–2.1×** the base CTR (per the validation gain table).
- Use the **economic threshold** `p★ = CPA/V` to decide **who to contact**; prefer deciles with `avg_p ≥ p★`.
- The **recommended K*** above gives a **discrete** operating point that maximizes **expected profit** given your CPA and V.
- Down-weight or suppress segments that systematically fall below `p★`, unless coverage/brand requires them.
- Recalibrate and re-tune thresholds when **CPA** or **V** change or new data shifts the score distribution.
        """
    )

    # Optional: show full gain table
    with st.expander("Show full gain table", expanded=False):
        st.dataframe(gain, use_container_width=True)
