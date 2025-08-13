# --- utils.py: replace eda_tables with this version ---
@st.cache_data(show_spinner=False)
def eda_tables(df: pd.DataFrame, min_support: int = 300):
    """
    Build key EDA artifacts with support filtering:
      - by_day (impressions, ctr)
      - by_hour (impressions, ctr)
      - coverage & lift tables for selected categoricals (filtered by min_support)
      - interaction matrices: hour_of_day × banner_pos (filtered by min_support on banner_pos)
      - high-cardinality / leakage watch table
      - support_info: how many categories remain after filtering per column
    """
    # Temporal
    by_day = (
        df.groupby("date")
          .agg(impressions=("click", "size"), ctr=("click", "mean"))
          .reset_index()
    )
    by_hour = (
        df.groupby("hour_of_day")
          .agg(impressions=("click", "size"), ctr=("click", "mean"))
          .reset_index()
    )

    # Categorical coverage & lift
    candidates = [c for c in ["banner_pos","device_type","device_conn_type","site_category","app_category"] if c in df.columns]
    ctr_tabs = {c: ctr_table(df, c, min_support, 15) for c in candidates}

    # Support info (kept/total groups)
    support_info = {}
    for c in candidates:
        if c in df.columns:
            vc = df[c].value_counts()
            kept = int((vc >= min_support).sum())
            total = int(vc.shape[0])
            support_info[c] = {"kept": kept, "total": total}

    # Interactions: hour × banner_pos (filter low-support banner_pos)
    if {"hour_of_day","banner_pos"}.issubset(df.columns):
        keep_vals = df["banner_pos"].value_counts()
        keep_set = set(keep_vals[keep_vals >= min_support].index)
        df_bp = df[df["banner_pos"].isin(keep_set)]
        if len(keep_set) > 0 and not df_bp.empty:
            pivot_ctr = (
                df_bp.groupby(["hour_of_day","banner_pos"])["click"]
                     .mean().unstack("banner_pos").sort_index().round(4)
            )
            pivot_n = (
                df_bp.groupby(["hour_of_day","banner_pos"])["click"]
                     .size().unstack("banner_pos").sort_index().fillna(0).astype(int)
            )
        else:
            pivot_ctr, pivot_n = None, None
    else:
        pivot_ctr, pivot_n = None, None

    # Leakage / uniqueness
    cand = [c for c in ["device_ip","device_id","id","site_id","app_id","device_model"] if c in df.columns]
    uni = pd.DataFrame({"col": cand,
                        "nunique": [df[c].nunique(dropna=True) for c in cand],
                        "rows": len(df)})
    uni["uniqueness_ratio"] = (uni["nunique"]/uni["rows"]).round(5)
    uni = uni.sort_values("uniqueness_ratio", ascending=False)

    return by_day, by_hour, ctr_tabs, pivot_ctr, pivot_n, uni, support_info


# --- utils.py: replace interaction_matrices with this version ---
@st.cache_data(show_spinner=False)
def interaction_matrices(df: pd.DataFrame, col: str, min_support: int | None = None):
    """
    Generic interaction matrices: hour_of_day × <col>, with optional support filtering on <col>.
    """
    if not {"hour_of_day", col}.issubset(df.columns):
        return None, None

    df2 = df
    if min_support is not None:
        vc = df[col].value_counts()
        keep = set(vc[vc >= min_support].index)
        df2 = df[df[col].isin(keep)]

    if df2.empty:
        return None, None

    pivot_ctr = (
        df2.groupby(["hour_of_day", col])["click"]
           .mean().unstack(col).sort_index().round(4)
    )
    pivot_n = (
        df2.groupby(["hour_of_day", col])["click"]
           .size().unstack(col).sort_index().fillna(0).astype(int)
    )
    return pivot_ctr, pivot_n
