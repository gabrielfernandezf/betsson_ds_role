# src/eda.py
"""
EDA section: split into tabs (Overview, Temporal, Categoricals, Interactions,
Quality, Insights). Includes optional associations (Cramér's V) and auto insights.
"""

import streamlit as st
import pandas as pd
from .utils import (
    eda_tables, alt_bar, df_to_csv_download,
    weekday_table, context_mix, c14_c21_overview, c20_unknown_known,
    nulls_and_duplicates, interaction_matrices, eda_highlights, cramers_v_matrix
)

MIN_SUPPORT = 300

def _divider():
    st.markdown("<hr style='margin: 0.75rem 0 1rem 0; border: none; height: 1px; background:#e5e7eb;'>", unsafe_allow_html=True)

def render(df: pd.DataFrame):
    st.header("Exploratory Data Analysis")
    st.caption(f"Small groups filtered out with support < **{MIN_SUPPORT}** for lift tables and interaction matrices.")

    # Precompute core artifacts
    by_day, by_hour, ctr_tabs, pivot_ctr, pivot_n, uni, support_info = eda_tables(df, MIN_SUPPORT)
    highlights = eda_highlights(df, MIN_SUPPORT)

    # Tabs
    tab_overview, tab_temporal, tab_cats, tab_inter, tab_quality, tab_insights = st.tabs(
        ["Overview", "Temporal", "Categoricals", "Interactions", "Quality", "Insights"]
    )

    # ---- OVERVIEW ----
    with tab_overview:
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Base CTR", f"{highlights['base_ctr']:.4f}")
        c2.metric("Best Day", f"{highlights['best_day']['date']}", f"{highlights['best_day']['ctr']:.4f} CTR")
        c3.metric("Worst Day", f"{highlights['worst_day']['date']}", f"{highlights['worst_day']['ctr']:.4f} CTR")
        c4.metric("Peak Hour (CTR)", f"{int(highlights['peak_hr_ctr']['hour_of_day'])}:00", f"{highlights['peak_hr_ctr']['ctr']:.4f}")
        if highlights["best_banner_pos"] is not None:
            bp = highlights["best_banner_pos"]
            c5.metric("Best banner_pos (lift)", f"{bp['banner_pos']}", f"{bp['lift']:.2f}×")

        _divider()
        st.subheader("Day-of-week (coverage & lift)")
        dow_tbl = weekday_table(df)
        st.dataframe(dow_tbl, use_container_width=True)
        df_to_csv_download(dow_tbl, "eda_by_dow.csv", "⬇️ Download day-of-week table")

    # ---- TEMPORAL ----
    with tab_temporal:
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

    # ---- CATEGORICALS ----
    with tab_cats:
        st.caption(f"Showing groups with support ≥ {MIN_SUPPORT}.")
        for name, tbl in ctr_tabs.items():
            kept = support_info.get(name, {}).get("kept", None)
            total = support_info.get(name, {}).get("total", None)
            suffix = f" · kept {kept}/{total} groups ≥ {MIN_SUPPORT}" if kept is not None else ""
            st.markdown(f"**{name}**{suffix}")
            st.dataframe(tbl, use_container_width=True)
            df_to_csv_download(tbl, f"eda_ctr_{name}.csv", f"⬇️ Download {name}")
            _divider()

        st.subheader("Context mix (site vs app)")
        ctx = context_mix(df)
        st.dataframe(ctx, use_container_width=True)
        df_to_csv_download(ctx, "eda_context_mix.csv", "⬇️ Download context mix")

        _divider()
        st.subheader("Anonymous feature blocks (C14–C21) — quick overview")
        c_over = c14_c21_overview(df)
        st.dataframe(c_over, use_container_width=True)
        df_to_csv_download(c_over, "eda_C14_C21_overview.csv", "⬇️ Download C14–C21 overview")

        _divider()
        st.subheader("C20 — unknown (−1) vs known")
        c20_stats = c20_unknown_known(df)
        if c20_stats is not None:
            st.json(c20_stats)
        else:
            st.info("C20 not present in this dataset.")

        _divider()
        st.subheader("Associations (Cramér’s V) — low-card categoricals")
        assoc_cols = [c for c in ["banner_pos", "device_type", "device_conn_type", "is_weekend", "dow", "C15", "C16"] if c in df.columns]
        vmat = cramers_v_matrix(df, assoc_cols)
        if vmat is None:
            st.info("Cramér’s V requires `scipy`. Add it to requirements to enable this matrix.")
        else:
            st.dataframe(vmat, use_container_width=True)
            df_to_csv_download(vmat, "eda_cramers_v.csv", "⬇️ Download associations")

    # ---- INTERACTIONS ----
    with tab_inter:
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

        _divider()
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

    # ---- QUALITY ----
    with tab_quality:
        st.subheader("Leakage watch — high-cardinality IDs")
        st.dataframe(uni, use_container_width=True)
        df_to_csv_download(uni, "eda_leakage_uniqueness.csv", "⬇️ Download leakage table")

        _divider()
        st.subheader("Data quality")
        nulls, dup_rows = nulls_and_duplicates(df)
        c5, c6 = st.columns([2, 1])
        with c5:
            st.markdown("**Top null rates**")
            st.dataframe(nulls, use_container_width=True)
            df_to_csv_download(nulls, "eda_null_rates.csv", "⬇️ Download null rates")
        with c6:
            st.metric("Exact duplicate rows", f"{dup_rows:,}")

    # ---- INSIGHTS ----
    with tab_insights:
        st.subheader("Key EDA insights")
        bullets = []
        bullets.append(f"- **Base CTR**: {highlights['base_ctr']:.4f}.")
        bullets.append(f"- **Best day**: {highlights['best_day']['date']} (CTR {highlights['best_day']['ctr']:.4f}; impressions {int(highlights['best_day']['impressions'])}).")
        bullets.append(f"- **Worst day**: {highlights['worst_day']['date']} (CTR {highlights['worst_day']['ctr']:.4f}).")
        bullets.append(f"- **Peak hour by CTR**: {int(highlights['peak_hr_ctr']['hour_of_day'])}:00 (CTR {highlights['peak_hr_ctr']['ctr']:.4f}).")
        bullets.append(f"- **Peak hour by impressions**: {int(highlights['peak_hr_impr']['hour_of_day'])}:00 (N={int(highlights['peak_hr_impr']['impressions'])}; CTR {highlights['peak_hr_impr']['ctr']:.4f}).")
        if highlights["best_banner_pos"]:
            bp = highlights["best_banner_pos"]
            bullets.append(f"- **banner_pos={bp['banner_pos']}** shows CTR {bp['ctr']:.4f} (lift **{bp['lift']:.2f}×**) with share {bp['share']:.2%}.")
        st.markdown("\n".join(bullets))

        st.caption("These insights directly support Slide 3: EDA insights (distributions, correlations/associations, notable patterns).")
