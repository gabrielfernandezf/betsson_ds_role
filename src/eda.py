# src/eda.py
"""
EDA section: split into tabs (Overview, Temporal, Categoricals, Interactions,
Quality, Insights). Adds:
- CTR heatmap Day-of-week × Hour
- Lift tables with 95% Wilson CI and p-values vs base
- Heatmaps for hour × {banner_pos, device_type}
- Segment drilldown: hourly CTR vs global
"""

import streamlit as st
import pandas as pd
import altair as alt

from .utils import (
    # core
    eda_tables, alt_bar, df_to_csv_download,
    weekday_table, context_mix, c14_c21_overview, c20_unknown_known,
    nulls_and_duplicates, interaction_matrices, eda_highlights, cramers_v_matrix,
    # new helpers (ensure these exist in utils.py)
    by_dow_hour, alt_heatmap_hour_dow, ctr_table_stats, alt_lift_ci,
)

MIN_SUPPORT = 300

def _divider():
    st.markdown(
        "<hr style='margin: 0.75rem 0 1rem 0; border: none; height: 1px; background:#e5e7eb;'>",
        unsafe_allow_html=True
    )

def render(df: pd.DataFrame):
    st.header("Exploratory Data Analysis")
    st.caption(f"Small groups filtered out with support < **{MIN_SUPPORT}** for lift tables and interaction matrices.")

    # --- Precompute core artifacts ---
    by_day, by_hour, ctr_tabs, pivot_ctr, pivot_n, uni, support_info = eda_tables(df, MIN_SUPPORT)
    highlights = eda_highlights(df, MIN_SUPPORT)

    # --- Tabs ---
    tab_overview, tab_temporal, tab_cats, tab_inter, tab_quality, tab_insights = st.tabs(
        ["Overview", "Temporal", "Categoricals", "Interactions", "Quality", "Insights"]
    )

    # ========================= OVERVIEW =========================
    with tab_overview:
        c1, c2, c3, c4, c5 = st.columns(5)

        with c1:
            st.metric("Base CTR", f"{highlights['base_ctr']:.4f}")

        with c2:
            st.metric("Best Day", f"{highlights['best_day']['date']}")
            st.caption(f"CTR {highlights['best_day']['ctr']:.4f} · Impr {int(highlights['best_day']['impressions'])}")

        with c3:
            st.metric("Worst Day", f"{highlights['worst_day']['date']}")
            st.caption(f"CTR {highlights['worst_day']['ctr']:.4f}")

        with c4:
            st.metric("Peak Hour (CTR)", f"{int(highlights['peak_hr_ctr']['hour_of_day']):02d}:00")
            st.caption(f"CTR {highlights['peak_hr_ctr']['ctr']:.4f}")

        with c5:
            bp = highlights.get("best_banner_pos")
            if bp:
                st.metric("Best banner_pos", f"{bp['banner_pos']}")
                st.caption(f"CTR {bp['ctr']:.4f} · lift {bp['lift']:.2f}× · share {bp['share']:.2%}")

        _divider()
        st.subheader("Day-of-week (coverage & lift)")
        dow_tbl = weekday_table(df)
        st.dataframe(dow_tbl, use_container_width=True)
        df_to_csv_download(dow_tbl, "eda_by_dow.csv", "⬇️ Download day-of-week table")

    # ========================= TEMPORAL =========================
    with tab_temporal:
        # Ensure datetime for proper x-axis formatting
        by_day_plot = by_day.copy()
        by_day_plot["date"] = pd.to_datetime(by_day_plot["date"])
        bar_size = 30

        # Impressions by Day (x as discrete day, MM-DD labels vertical)
        chart_impr_day = (
            alt.Chart(by_day_plot)
            .mark_bar(size=bar_size)
            .encode(
                x=alt.X(
                    "yearmonthdate(date):O",
                    axis=alt.Axis(format="%m-%d", labelAngle=-90, title="Date"),
                    sort="ascending",
                ),
                y=alt.Y("impressions:Q", title="Impressions"),
                tooltip=[
                    alt.Tooltip("date:T", format="%Y-%m-%d"),
                    alt.Tooltip("impressions:Q", title="Impressions"),
                    alt.Tooltip("ctr:Q", format=".4f", title="CTR"),
                ],
            )
            .properties(title="Impressions by Day", height=280)
        )

        # CTR by Day (same x-axis)
        chart_ctr_day = (
            alt.Chart(by_day_plot)
            .mark_bar(size=bar_size)
            .encode(
                x=alt.X(
                    "yearmonthdate(date):O",
                    axis=alt.Axis(format="%m-%d", labelAngle=-90, title="Date"),
                    sort="ascending",
                ),
                y=alt.Y("ctr:Q", title="CTR"),
                tooltip=[
                    alt.Tooltip("date:T", format="%Y-%m-%d"),
                    alt.Tooltip("ctr:Q", format=".4f", title="CTR"),
                    alt.Tooltip("impressions:Q", title="Impressions"),
                ],
            )
            .properties(title="CTR by Day", height=280)
        )

        c1, c2 = st.columns(2)
        c1.altair_chart(chart_impr_day, use_container_width=True)
        c2.altair_chart(chart_ctr_day, use_container_width=True)

        # Hourly charts
        c3, c4 = st.columns(2)
        c3.altair_chart(
            alt_bar(by_hour, "hour_of_day:N", "impressions:Q", "Impressions by Hour", ["hour_of_day", "impressions", "ctr"]),
            use_container_width=True,
        )
        c4.altair_chart(
            alt_bar(by_hour, "hour_of_day:N", "ctr:Q", "CTR by Hour", ["hour_of_day", "ctr"]),
            use_container_width=True,
        )

        _divider()
        st.subheader("Day-of-week × Hour (CTR heatmap)")
        t = by_dow_hour(df)
        hm = alt_heatmap_hour_dow(t)
        if hm is not None:
            st.altair_chart(hm, use_container_width=True)
        else:
            st.info("Need columns: 'dow', 'hour_of_day', 'click'.")

    # ========================= CATEGORICALS =========================
    with tab_cats:
        st.caption(f"Showing groups with support ≥ {MIN_SUPPORT}. 95% CIs use Wilson; significance vs base: *, **, ***.")
        candidates = list(ctr_tabs.keys())  # keep same detected categoricals

        for name in candidates:
            kept = support_info.get(name, {}).get("kept", None)
            total = support_info.get(name, {}).get("total", None)
            suffix = f" · kept {kept}/{total} groups ≥ {MIN_SUPPORT}" if kept is not None else ""
            st.markdown(f"**{name}**{suffix}")

            # Table with stats: n, clicks, ctr, lift, CI, p-value, sig
            tbl = ctr_table_stats(df, name, min_n=MIN_SUPPORT, top=15)
            st.dataframe(tbl, use_container_width=True)
            df_to_csv_download(tbl, f"eda_ctr_stats_{name}.csv", f"⬇️ Download {name} (with CI)")

            # Optional mini chart with error bars
            if 2 <= len(tbl) <= 12:
                ch = alt_lift_ci(tbl.rename(columns={name: "level"}), "level", f"{name}: CTR ±95% CI (top by support)")
                if ch is not None:
                    st.altair_chart(ch, use_container_width=True)

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

    # ========================= INTERACTIONS =========================
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
        st.subheader("Heatmaps (CTR)")
        # hour × banner_pos heatmap (if pivot available)
        if pivot_ctr is not None:
            hm_bp = (
                alt.Chart(pivot_ctr.reset_index().melt("hour_of_day", var_name="banner_pos", value_name="ctr"))
                .mark_rect()
                .encode(
                    x=alt.X("hour_of_day:O", title="Hour"),
                    y=alt.Y("banner_pos:N", title="banner_pos"),
                    color=alt.Color("ctr:Q", title="CTR", scale=alt.Scale(scheme="viridis")),
                    tooltip=["hour_of_day","banner_pos", alt.Tooltip("ctr:Q", format=".4f")]
                )
                .properties(title="hour × banner_pos — CTR", height=220)
            )
            st.altair_chart(hm_bp, use_container_width=True)

        # hour × device_type heatmap
        dev_ctr, _ = interaction_matrices(df, "device_type", min_support=MIN_SUPPORT)
        if dev_ctr is not None:
            hm_dt = (
                alt.Chart(dev_ctr.reset_index().melt("hour_of_day", var_name="device_type", value_name="ctr"))
                .mark_rect()
                .encode(
                    x=alt.X("hour_of_day:O", title="Hour"),
                    y=alt.Y("device_type:N", title="device_type"),
                    color=alt.Color("ctr:Q", title="CTR", scale=alt.Scale(scheme="magma")),
                    tooltip=["hour_of_day","device_type", alt.Tooltip("ctr:Q", format=".4f")]
                )
                .properties(title="hour × device_type — CTR", height=220)
            )
            st.altair_chart(hm_dt, use_container_width=True)

        _divider()
        st.subheader("Segment drilldown: hourly CTR vs global")
        # Choose categorical and value
        cat_cols = [c for c in ["banner_pos","device_type","device_conn_type","site_category","app_category"] if c in df.columns]
        if cat_cols:
            ccol, vcol = st.columns(2)
            with ccol:
                chosen_col = st.selectbox("Choose a categorical", cat_cols, index=0)
            with vcol:
                top_vals = df[chosen_col].value_counts().head(10).index.tolist()
                chosen_val = st.selectbox("Choose a value (top by support)", top_vals, index=0)

            base = (
                df.groupby("hour_of_day")["click"].mean()
                .reset_index().rename(columns={"click":"ctr"})
            )
            seg  = (
                df[df[chosen_col] == chosen_val]
                .groupby("hour_of_day")["click"].mean()
                .reset_index().rename(columns={"click":"ctr"})
            )
            base["which"] = "Global"
            seg["which"]  = f"{chosen_col}={chosen_val}"
            comb = pd.concat([base, seg], ignore_index=True)

            line = (
                alt.Chart(comb)
                .mark_line(point=True)
                .encode(
                    x=alt.X("hour_of_day:O", title="Hour"),
                    y=alt.Y("ctr:Q", title="CTR"),
                    color="which:N",
                    tooltip=["which","hour_of_day", alt.Tooltip("ctr:Q", format=".4f")]
                )
                .properties(height=280)
            )
            st.altair_chart(line, use_container_width=True)
        else:
            st.info("No categorical columns available for drilldown.")

    # ========================= QUALITY =========================
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

    # ========================= INSIGHTS =========================
    with tab_insights:
        import json
        from .utils import build_eda_summary_md, build_eda_summary_json

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

        _divider()
        st.subheader("Downloadable summary")
        dow_tbl = weekday_table(df)
        summary_md = build_eda_summary_md(highlights, dow_tbl, ctr_tabs)
        summary_json = build_eda_summary_json(highlights, dow_tbl, ctr_tabs)

        edited_md = st.text_area("Edit summary before download (Markdown):", value=summary_md, height=280)

        col_a, col_b = st.columns(2)
        with col_a:
            st.download_button(
                "⬇️ Download EDA_Summary.md",
                data=edited_md,
                file_name="EDA_Summary.md",
                mime="text/markdown",
                use_container_width=True,
            )

        # Safe JSON serialization (handles numpy/pandas/ts)
        def _json_default(o):
            import numpy as np, pandas as pd, datetime as dt
            if isinstance(o, (np.integer, np.floating, np.bool_)):
                return o.item()
            if isinstance(o, (pd.Timestamp, dt.date, dt.datetime)):
                return o.isoformat()
            return str(o)

        with col_b:
            st.download_button(
                "⬇️ Download EDA_Summary.json",
                data=json.dumps(summary_json, indent=2, default=_json_default),
                file_name="EDA_Summary.json",
                mime="application/json",
                use_container_width=True,
            )

# # src/eda.py
# """
# EDA section: split into tabs (Overview, Temporal, Categoricals, Interactions,
# Quality, Insights). Includes optional associations (Cramér's V) and auto insights.
# """

# import streamlit as st
# import pandas as pd
# import altair as alt
# from .utils import (
#     eda_tables, alt_bar, df_to_csv_download,
#     weekday_table, context_mix, c14_c21_overview, c20_unknown_known,
#     nulls_and_duplicates, interaction_matrices, eda_highlights, cramers_v_matrix
# )

# MIN_SUPPORT = 300

# def _divider():
#     st.markdown("<hr style='margin: 0.75rem 0 1rem 0; border: none; height: 1px; background:#e5e7eb;'>", unsafe_allow_html=True)

# def render(df: pd.DataFrame):
#     st.header("Exploratory Data Analysis")
#     st.caption(f"Small groups filtered out with support < **{MIN_SUPPORT}** for lift tables and interaction matrices.")

#     # Precompute core artifacts
#     by_day, by_hour, ctr_tabs, pivot_ctr, pivot_n, uni, support_info = eda_tables(df, MIN_SUPPORT)
#     highlights = eda_highlights(df, MIN_SUPPORT)

#     # Tabs
#     tab_overview, tab_temporal, tab_cats, tab_inter, tab_quality, tab_insights = st.tabs(
#         ["Overview", "Temporal", "Categoricals", "Interactions", "Quality", "Insights"]
#     )

#     # ---- OVERVIEW ----
#     with tab_overview:
        
#         c1, c2, c3, c4, c5 = st.columns(5)
        
#         with c1:
#             st.metric("Base CTR", f"{highlights['base_ctr']:.4f}")
        
#         with c2:
#             st.metric("Best Day", f"{highlights['best_day']['date']}")
#             st.caption(f"CTR {highlights['best_day']['ctr']:.4f} · Impr {int(highlights['best_day']['impressions'])}")
        
#         with c3:
#             st.metric("Worst Day", f"{highlights['worst_day']['date']}")
#             st.caption(f"CTR {highlights['worst_day']['ctr']:.4f}")
        
#         with c4:
#             st.metric("Peak Hour (CTR)", f"{int(highlights['peak_hr_ctr']['hour_of_day']):02d}:00")
#             st.caption(f"CTR {highlights['peak_hr_ctr']['ctr']:.4f}")
        
#         with c5:
#             bp = highlights.get("best_banner_pos")
#             if bp:
#                 st.metric("Best banner_pos", f"{bp['banner_pos']}")
#                 st.caption(f"CTR {bp['ctr']:.4f} · lift {bp['lift']:.2f}× · share {bp['share']:.2%}")
        


#         _divider()
#         st.subheader("Day-of-week (coverage & lift)")
#         dow_tbl = weekday_table(df)
#         st.dataframe(dow_tbl, use_container_width=True)
#         df_to_csv_download(dow_tbl, "eda_by_dow.csv", "⬇️ Download day-of-week table")

#     # ---- TEMPORAL ----
#     with tab_temporal:
#         # Aseguramos tipo datetime
#         by_day_plot = by_day.copy()
#         by_day_plot["date"] = pd.to_datetime(by_day_plot["date"])
#         bar_size = 30
#         # Impressions by Day (eje MM-DD vertical, barras con ancho fijo)
#         chart_impr_day = (
#             alt.Chart(by_day_plot)
#             .mark_bar(size=bar_size)  # controla el ancho de barra en escala discreta
#             .encode(
#                 x=alt.X(
#                     "yearmonthdate(date):O",  # escala discreta por día
#                     axis=alt.Axis(format="%m-%d", labelAngle=-90, title="Date"),
#                     sort="ascending",
#                 ),
#                 y=alt.Y("impressions:Q", title="Impressions"),
#                 tooltip=[
#                     alt.Tooltip("date:T", format="%Y-%m-%d"),
#                     alt.Tooltip("impressions:Q", title="Impressions"),
#                     alt.Tooltip("ctr:Q", format=".4f", title="CTR"),
#                 ],
#             )
#             .properties(title="Impressions by Day", height=280)
#         )
        
#         # CTR by Day (mismo eje)
#         chart_ctr_day = (
#             alt.Chart(by_day_plot)
#             .mark_bar(size=bar_size)
#             .encode(
#                 x=alt.X(
#                     "yearmonthdate(date):O",
#                     axis=alt.Axis(format="%m-%d", labelAngle=-90, title="Date"),
#                     sort="ascending",
#                 ),
#                 y=alt.Y("ctr:Q", title="CTR"),
#                 tooltip=[
#                     alt.Tooltip("date:T", format="%Y-%m-%d"),
#                     alt.Tooltip("ctr:Q", format=".4f", title="CTR"),
#                     alt.Tooltip("impressions:Q", title="Impressions"),
#                 ],
#             )
#             .properties(title="CTR by Day", height=280)
#         )
        
#         c1, c2 = st.columns(2)
#         c1.altair_chart(chart_impr_day, use_container_width=True)
#         c2.altair_chart(chart_ctr_day, use_container_width=True)


#         c3, c4 = st.columns(2)
#         c3.altair_chart(
#             alt_bar(by_hour, "hour_of_day:N", "impressions:Q", "Impressions by Hour", ["hour_of_day", "impressions", "ctr"]),
#             use_container_width=True,
#         )
#         c4.altair_chart(
#             alt_bar(by_hour, "hour_of_day:N", "ctr:Q", "CTR by Hour", ["hour_of_day", "ctr"]),
#             use_container_width=True,
#         )

#     # ---- CATEGORICALS ----
#     with tab_cats:
#         st.caption(f"Showing groups with support ≥ {MIN_SUPPORT}.")
#         for name, tbl in ctr_tabs.items():
#             kept = support_info.get(name, {}).get("kept", None)
#             total = support_info.get(name, {}).get("total", None)
#             suffix = f" · kept {kept}/{total} groups ≥ {MIN_SUPPORT}" if kept is not None else ""
#             st.markdown(f"**{name}**{suffix}")
#             st.dataframe(tbl, use_container_width=True)
#             df_to_csv_download(tbl, f"eda_ctr_{name}.csv", f"⬇️ Download {name}")
#             _divider()

#         st.subheader("Context mix (site vs app)")
#         ctx = context_mix(df)
#         st.dataframe(ctx, use_container_width=True)
#         df_to_csv_download(ctx, "eda_context_mix.csv", "⬇️ Download context mix")

#         _divider()
#         st.subheader("Anonymous feature blocks (C14–C21) — quick overview")
#         c_over = c14_c21_overview(df)
#         st.dataframe(c_over, use_container_width=True)
#         df_to_csv_download(c_over, "eda_C14_C21_overview.csv", "⬇️ Download C14–C21 overview")

#         _divider()
#         st.subheader("C20 — unknown (−1) vs known")
#         c20_stats = c20_unknown_known(df)
#         if c20_stats is not None:
#             st.json(c20_stats)
#         else:
#             st.info("C20 not present in this dataset.")

#         _divider()
#         st.subheader("Associations (Cramér’s V) — low-card categoricals")
#         assoc_cols = [c for c in ["banner_pos", "device_type", "device_conn_type", "is_weekend", "dow", "C15", "C16"] if c in df.columns]
#         vmat = cramers_v_matrix(df, assoc_cols)
#         if vmat is None:
#             st.info("Cramér’s V requires `scipy`. Add it to requirements to enable this matrix.")
#         else:
#             st.dataframe(vmat, use_container_width=True)
#             df_to_csv_download(vmat, "eda_cramers_v.csv", "⬇️ Download associations")

#     # ---- INTERACTIONS ----
#     with tab_inter:
#         st.markdown("**hour_of_day × banner_pos**")
#         if pivot_ctr is not None:
#             st.write("CTR matrix")
#             st.dataframe(pivot_ctr, use_container_width=True)
#             st.write("Support matrix (N)")
#             st.dataframe(pivot_n, use_container_width=True)
#             df_to_csv_download(pivot_ctr, "pivot_ctr_hour_banner_pos.csv", "⬇️ Download CTR matrix")
#             df_to_csv_download(pivot_n, "pivot_n_hour_banner_pos.csv", "⬇️ Download Support matrix")
#         else:
#             st.info("Required columns missing (`hour_of_day` and `banner_pos`).")

#         _divider()
#         st.markdown("**hour_of_day × device_type**")
#         dev_ctr, dev_n = interaction_matrices(df, "device_type", min_support=MIN_SUPPORT)
#         if dev_ctr is not None:
#             st.write("CTR matrix")
#             st.dataframe(dev_ctr, use_container_width=True)
#             st.write("Support matrix (N)")
#             st.dataframe(dev_n, use_container_width=True)
#             df_to_csv_download(dev_ctr, "pivot_ctr_hour_device_type.csv", "⬇️ Download CTR matrix (device_type)")
#             df_to_csv_download(dev_n, "pivot_n_hour_device_type.csv", "⬇️ Download Support matrix (device_type)")
#         else:
#             st.info("`device_type` not available.")

#     # ---- QUALITY ----
#     with tab_quality:
#         st.subheader("Leakage watch — high-cardinality IDs")
#         st.dataframe(uni, use_container_width=True)
#         df_to_csv_download(uni, "eda_leakage_uniqueness.csv", "⬇️ Download leakage table")

#         _divider()
#         st.subheader("Data quality")
#         nulls, dup_rows = nulls_and_duplicates(df)
#         c5, c6 = st.columns([2, 1])
#         with c5:
#             st.markdown("**Top null rates**")
#             st.dataframe(nulls, use_container_width=True)
#             df_to_csv_download(nulls, "eda_null_rates.csv", "⬇️ Download null rates")
#         with c6:
#             st.metric("Exact duplicate rows", f"{dup_rows:,}")

#     # ---- INSIGHTS ----
#     with tab_insights:
#         import json
#         from .utils import build_eda_summary_md, build_eda_summary_json

#         st.subheader("Key EDA insights")

#         # Bullet points (quick read)
#         bullets = []
#         bullets.append(f"- **Base CTR**: {highlights['base_ctr']:.4f}.")
#         bullets.append(f"- **Best day**: {highlights['best_day']['date']} (CTR {highlights['best_day']['ctr']:.4f}; impressions {int(highlights['best_day']['impressions'])}).")
#         bullets.append(f"- **Worst day**: {highlights['worst_day']['date']} (CTR {highlights['worst_day']['ctr']:.4f}).")
#         bullets.append(f"- **Peak hour by CTR**: {int(highlights['peak_hr_ctr']['hour_of_day'])}:00 (CTR {highlights['peak_hr_ctr']['ctr']:.4f}).")
#         bullets.append(f"- **Peak hour by impressions**: {int(highlights['peak_hr_impr']['hour_of_day'])}:00 (N={int(highlights['peak_hr_impr']['impressions'])}; CTR {highlights['peak_hr_impr']['ctr']:.4f}).")
#         if highlights["best_banner_pos"]:
#             bp = highlights["best_banner_pos"]
#             bullets.append(f"- **banner_pos={bp['banner_pos']}** shows CTR {bp['ctr']:.4f} (lift **{bp['lift']:.2f}×**) with share {bp['share']:.2%}.")
#         st.markdown("\n".join(bullets))

#         # Auto-generated full summary (Markdown + JSON) + editor for last-mile tweaks
#         st.subheader("Downloadable summary")
#         dow_tbl = weekday_table(df)
#         summary_md = build_eda_summary_md(highlights, dow_tbl, ctr_tabs)
#         summary_json = build_eda_summary_json(highlights, dow_tbl, ctr_tabs)

#         # Optional in-app edits before download
#         edited_md = st.text_area("Edit summary before download (Markdown):", value=summary_md, height=280)

#         col_a, col_b = st.columns(2)
#         with col_a:
#             st.download_button(
#                 "⬇️ Download EDA_Summary.md",
#                 data=edited_md,
#                 file_name="EDA_Summary.md",
#                 mime="text/markdown",
#                 use_container_width=True,
#             )
#         def _json_default(o):
#             import numpy as np, pandas as pd, datetime as dt
#             if isinstance(o, (np.integer, np.floating, np.bool_)):
#                 return o.item()
#             if isinstance(o, (pd.Timestamp, dt.date, dt.datetime)):
#                 return o.isoformat()
#             return str(o)
        
#         with col_b:
#             st.download_button(
#                 "⬇️ Download EDA_Summary.json",
#                 data=json.dumps(summary_json, indent=2, default=_json_default),
#                 file_name="EDA_Summary.json",
#                 mime="application/json",
#                 use_container_width=True,
#             )


