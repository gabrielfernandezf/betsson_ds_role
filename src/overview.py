# src/overview.py
"""
Overview section: problem framing, dataset, and KPIs.
"""

import streamlit as st


def render(df):
    base_ctr = df["click"].mean()
    time_window = f"{df['dt'].min()} → {df['dt'].max()}"

    st.title("Next Best Action (NBA) — Prototype")
    st.write(
        """
**Use case**: predict user propensity to interact (CTR) to prioritize the *next best action*.  
**Dataset**: Avazu CTR sample (50k). Anonymous, tabular, 10 days, binary target (click / no click).  
**Limitations**: advertising context, anonymized features; we transfer the approach to NBA (propensity scoring).
        """
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{len(df):,}")
    c2.metric("Columns", f"{df.shape[1]}")
    c3.metric("Base CTR", f"{base_ctr:.4f}")
    c4.metric("Time Window", time_window)

    with st.expander("Rationale & Scope"):
        st.markdown(
            """
- **Why Avazu**: well-known binary tabular problem (CTR) → clean mapping to propensity modeling for NBA.
- **Objective**: prototype a propensity score and understand main drivers (no online training).
- **Validation**: time-based split (last day as validation) to simulate production usage.
            """
        )
