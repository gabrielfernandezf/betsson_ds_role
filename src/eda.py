
# src/eda.py
"""
EDA section: temporal patterns, categorical coverage & lift,
interaction matrices, and leakage watch.
"""

import streamlit as st
from .utils import eda_tables, alt_bar, df_to_csv_download


def render(df):
    st.header("Exploratory Data Analysis")

    # Local control (remembered with session_state)
    default_support = st.session_state.get("min_support", 300)
    min_support = st.slider("Minimum group size (support)", 50, 2000, default_support, 50)
    st.session_state.min_support = min_support

    by_day, by_hour, ctr_tabs, pivot_ctr, pivot_n, uni = eda_tables(df, min_support)

    c1, c2 = st.columns(2)
    c1.altair_chart(
        alt_bar(by_day, "date:N", "impressions:Q", "Impressions by Day", ["date", "impressions", "ctr"]),
        use_container_width=True,
    )
    c2.altair_chart(
        alt_bar(by_day, "date:N", "ctr:Q", "CTR by Day", ["date", "ctr"]),
        use_container_width=True,
    )

    c3, c4 = st.columns(2)
    c3.altair_chart(
        alt_bar(by_hour, "hour_of_day:N", "impressions:Q", "Impressions by Hour", ["hour_of_day", "impressions", "ctr"]),
        use_container_width=True,
    )
    c4.altair_chart(
        alt_bar(by_hour, "hour_of_day:N", "ctr:Q", "CTR by Hour", ["hour_of_day", "ctr"]),
        use_container_width=True,
    )

    st.subheader("Categoricals — coverage & lift")
    for name, tbl in ctr_tabs.items():
        st.markdown(f"**{name}**")
        st.dataframe(tbl, use_container_width=True)
        df_to_csv_download(tbl, f"eda_ctr_{name}.csv", f"⬇️ Download {name}")

    st.subheader("Interactions — hour_of_day × banner_pos (values)")
    if pivot_ctr is not None:
        st.write("CTR matrix")
        st.dataframe(pivot_ctr, use_container_width=True)
        st.write("Support matrix (N)")
        st.dataframe(pivot_n, use_container_width=True)
        df_to_csv_download(pivot_ctr, "pivot_ctr_hour_banner_pos.csv", "⬇️ Download CTR matrix")
        df_to_csv_download(pivot_n, "pivot_n_hour_banner_pos.csv", "⬇️ Download Support matrix")
    else:
        st.info("Required columns missing (`hour_of_day` and `banner_pos`).")

    st.subheader("Leakage watch — high-cardinality IDs")
    st.dataframe(uni, use_container_width=True)
    df_to_csv_download(uni, "eda_leakage_uniqueness.csv", "⬇️ Download leakage table")
