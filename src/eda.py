# src/eda.py
"""
EDA section: temporal patterns, categorical coverage & lift,
interaction matrices, context mix, C14–C21 overview,
C20 unknown vs known, leakage, and data-quality snapshot.
"""

import streamlit as st
from .utils import (
    eda_tables, alt_bar, df_to_csv_download,
    weekday_table, context_mix, c14_c21_overview, c20_unknown_known,
    nulls_and_duplicates, interaction_matrices
)

# Fixed support threshold to keep UI simple and deterministic
MIN_SUPPORT = 300

def render(df):
    st.header("Exploratory Data Analysis")
    st.caption(f"Filtering small groups with support < **{MIN_SUPPORT}** "
               "(applied to coverage/lift tables and interaction matrices).")

    # Core EDA artifacts (with fixed min support)
    by_day, by_hour, ctr_tabs, pivot_ctr, pivot_n, uni, support_info = eda_tables(df, MIN_SUPPORT)

    # Temporal
    st.subheader("Temporal patterns")
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

    # Day-of-week
    st.subheader("Day-of-week (coverage & lift)")
    dow_tbl = weekday_table(df)
    st.dataframe(dow_tbl, use_container_width=True)
    df_to_csv_download(dow_tbl, "eda_by_dow.csv", "⬇️ Download day-of-week table")

    # Categoricals
    st.subheader("Categoricals — coverage & lift")
    for name, tbl in ctr_tabs.items():
        kept = support_info.get(name, {}).get("kept", None)
        total = support_info.get(name, {}).get("total", None)
        suffix = f"  · kept {kept}/{total} groups ≥ {MIN_SUPPORT}" if kept is not None else ""
        st.markdown(f"**{name}**{suffix}")
        st.dataframe(tbl, use_container_width=True)
        df_to_csv_download(tbl, f"eda_ctr_{name}.csv", f"⬇️ Download {name}")

    # Interactions
    st.subheader("Interactions")
    st.markdown("**hour_of_day × banner_pos**")
    if pivot_ctr is not None:
        st.write("CTR matrix")
        st.dataframe(pivot_ctr, use_container_width=True)
        st.write("Support matrix (N)")
        st.dataframe(pivot_n, use_container_width=True)
        df_to_csv_download(pivot_ctr, "pivot_ctr_hour_banner_pos.csv", "⬇️ Download CTR matrix")
        df_to_csv_download(pivot_n, "pivot_n_hour_banner_pos.csv", "⬇️ Download Support matrix")
    else:
        st.info("Required columns missing (`hour_of_day` and `banner_pos`).")

    # hour × device_type (filtered by the same support)
    st.markdown("**hour_of_day × device_type**")
    dev_ctr, dev_n = interaction_matrices(df, "device_type", min_support=MIN_SUPPORT)
    if dev_ctr is not None:
        st.write("CTR matrix")
        st.dataframe(dev_ctr, use_container_width=True)
        st.write("Support matrix (N)")
        st.dataframe(dev_n, use_container_width=True)
        df_to_csv_download(dev_ctr, "pivot_ctr_hour_device_type.csv", "⬇️ Download CTR matrix (device_type)")
        df_to_csv_download(dev_n, "pivot_n_hour_device_type.csv", "⬇️ Download Support matrix (device_type)")
    else:
        st.info("`device_type` not available.")

    # Context mix
    st.subheader("Context mix (site vs app)")
    ctx = context_mix(df)
    st.dataframe(ctx, use_container_width=True)
    df_to_csv_download(ctx, "eda_context_mix.csv", "⬇️ Download context mix")

    # C14–C21 overview
    st.subheader("Anonymous feature blocks (C14–C21) — quick overview")
    c_over = c14_c21_overview(df)
    st.dataframe(c_over, use_container_width=True)
    df_to_csv_download(c_over, "eda_C14_C21_overview.csv", "⬇️ Download C14–C21 overview")

    # C20 unknown vs known
    st.subheader("C20 — unknown (−1) vs known")
    c20_stats = c20_unknown_known(df)
    if c20_stats is not None:
        st.json(c20_stats)
    else:
        st.info("C20 not present in this dataset.")

    # Leakage
    st.subheader("Leakage watch — high-cardinality IDs")
    st.dataframe(uni, use_container_width=True)
    df_to_csv_download(uni, "eda_leakage_uniqueness.csv", "⬇️ Download leakage table")

    # Data quality
    st.subheader("Data quality")
    nulls, dup_rows = nulls_and_duplicates(df)
    c5, c6 = st.columns([2, 1])
    with c5:
        st.markdown("**Top null rates**")
        st.dataframe(nulls, use_container_width=True)
        df_to_csv_download(nulls, "eda_null_rates.csv", "⬇️ Download null rates")
    with c6:
        st.metric("Exact duplicate rows", f"{dup_rows:,}")
