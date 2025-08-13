# src/utils.py
"""
Shared utilities for the Streamlit app:
- Data loading & robust time parsing
- EDA aggregation tables (with support filtering)
- Day-of-week, context mix, C14–C21 overview
- Generic interaction matrices
- Simple Altair charts
- CSV download helper
"""

from pathlib import Path
import io
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt


# -----------------------------
# Loading & time parsing
# -----------------------------
@st.cache_data(show_spinner=False)
def parse_hour(series: pd.Series) -> pd.Series:
    """Robustly parse Avazu 'hour' column to pandas datetime."""
    s = series.astype(str)
    h0 = s.iloc[0]
    if any(sep in h0 for sep in ["-", ":", " ", "/"]):
        return pd.to_datetime(s, errors="coerce")
    dt = pd.to_datetime(s, format="%Y%m%d%H", errors="coerce")
    if dt.isna().mean() > 0.5:
        dt = pd.to_datetime(s, format="%y%m%d%H", errors="coerce")
    return dt


@st.cache_data(show_spinner=False)
def load_data(path: Path) -> pd.DataFrame:
    """Load CSV and derive time features."""
    df = pd.read_csv(path, low_memory=False)
    dt = parse_hour(df["hour"])
    df["dt"] = dt
    df["date"] = df["dt"].dt.date
    df["hour_of_day"] = df["dt"].dt.hour
    df["dow"] = df["dt"].dt.dayofweek
    df["is_weekend"] = (df["dow"] >= 5).astype("int8")
    return df


# -----------------------------
# EDA core tables (with support filtering)
# -----------------------------
@st.cache_data(show_spinner=False)
def ctr_table(df: pd.DataFrame, col: str, min_n: int = 300, top: int = 15) -> pd.DataFrame:
    """Coverage & lift table for a single categorical column."""
    base = df["click"].mean()
    t = (
        df.groupby(col)
          .agg(n=("click", "size"), ctr=("click", "mean"))
          .reset_index()
    )
    t = t[t["n"] >= min_n].copy()
    t["share"] = t["n"] / len(df)
    t["lift"] = t["ctr"] / base
    return t.sort_values(["n", "ctr"], ascending=[False, False]).head(top)


@st.cache_data(show_spinner=False)
def eda_tables(df: pd.DataFrame, min_support: int = 300):
    """
    Build key EDA artifacts with support filtering:
      - by_day (impressions, ctr)
      - by_hour (impressions, ctr)
      - coverage & lift tables for selected categoricals (filtered by min_support)
      - interaction matrices: hour_of_day × banner_pos (filtered by min_support on banner_pos)
      - high-cardinality / leakage watch table
      - support_info: kept/total groups per categorical
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
    candidates = [c for c in ["banner_pos", "device_type", "device_conn_type", "site_category", "app_category"]
                  if c in df.columns]
    ctr_tabs = {c: ctr_table(df, c, min_support, 15) for c in candidates}

    # Support info
    support_info = {}
    for c in candidates:
        vc = df[c].value_counts()
        kept = int((vc >= min_support).sum())
        total = int(vc.shape[0])
        support_info[c] = {"kept": kept, "total": total}

    # Interactions: hour × banner_pos (filter low-support banner_pos)
    if {"hour_of_day", "banner_pos"}.issubset(df.columns):
        keep_vals = df["banner_pos"].value_counts()
        keep_set = set(keep_vals[keep_vals >= min_support].index)
        df_bp = df[df["banner_pos"].isin(keep_set)]
        if keep_set and not df_bp.empty:
            pivot_ctr = (
                df_bp.groupby(["hour_of_day", "banner_pos"])["click"]
                     .mean().unstack("banner_pos").sort_index().round(4)
            )
            pivot_n = (
                df_bp.groupby(["hour_of_day", "banner_pos"])["click"]
                     .size().unstack("banner_pos").sort_index().fillna(0).astype(int)
            )
        else:
            pivot_ctr, pivot_n = None, None
    else:
        pivot_ctr, pivot_n = None, None

    # Leakage / uniqueness
    leak_cols = [c for c in ["device_ip", "device_id", "id", "site_id", "app_id", "device_model"] if c in df.columns]
    uni = pd.DataFrame({
        "col": leak_cols,
        "nunique": [df[c].nunique(dropna=True) for c in leak_cols],
        "rows": len(df)
    })
    uni["uniqueness_ratio"] = (uni["nunique"] / uni["rows"]).round(5)
    uni = uni.sort_values("uniqueness_ratio", ascending=False)

    return by_day, by_hour, ctr_tabs, pivot_ctr, pivot_n, uni, support_info


# -----------------------------
# Extra EDA helpers
# -----------------------------
@st.cache_data(show_spinner=False)
def weekday_table(df: pd.DataFrame) -> pd.DataFrame:
    base = df["click"].mean()
    tbl = (
        df.groupby("dow")
          .agg(impressions=("click", "size"), ctr=("click", "mean"))
          .reset_index()
    )
    tbl["dow_name"] = tbl["dow"].map({0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri", 5: "Sat", 6: "Sun"})
    tbl["lift"] = tbl["ctr"] / base
    return tbl.sort_values("dow")


@st.cache_data(show_spinner=False)
def context_mix(df: pd.DataFrame) -> pd.DataFrame:
    def is_nonempty(x: pd.Series) -> pd.Series:
        s = x.astype(str).str.lower()
        return ~s.isin(["", "nan", "none", "null", "unknown"])

    has_app = is_nonempty(df["app_id"]) if "app_id" in df.columns else pd.Series(False, index=df.index)
    has_site = is_nonempty(df["site_id"]) if "site_id" in df.columns else pd.Series(False, index=df.index)

    ctx = np.where(has_app & ~has_site, "app_only",
          np.where(has_site & ~has_app, "site_only",
          np.where(has_app & has_site, "both", "unknown")))

    t = pd.DataFrame({"context_type": ctx, "click": df["click"]})
    g = t.groupby("context_type").agg(n=("click", "size"), ctr=("click", "mean")).reset_index()
    g["share"] = g["n"] / len(df)
    return g.sort_values("n", ascending=False)


@st.cache_data(show_spinner=False)
def c14_c21_overview(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in ["C14", "C15", "C16", "C17", "C18", "C19", "C20", "C21"] if c in df.columns]
    out = []
    for c in cols:
        vc = df[c].value_counts(dropna=False).head(5)
        top = ", ".join([f"{k}({v})" for k, v in vc.items()])
        out.append({"column": c, "nunique": int(df[c].nunique(dropna=True)), "top5": top})
    return pd.DataFrame(out)


@st.cache_data(show_spinner=False)
def c20_unknown_known(df: pd.DataFrame):
    if "C20" not in df.columns:
        return None
    unknown = (df["C20"] == -1)
    base = df["click"].mean()
    return {
        "unknown_share": float(unknown.mean()),
        "ctr_unknown": float(df.loc[unknown, "click"].mean()),
        "ctr_known": float(df.loc[~unknown, "click"].mean()),
        "lift_known_vs_base": float(df.loc[~unknown, "click"].mean() / base) if base > 0 else np.nan,
    }


@st.cache_data(show_spinner=False)
def nulls_and_duplicates(df: pd.DataFrame):
    nulls = df.isna().mean().sort_values(ascending=False).to_frame("null_rate").head(20)
    duplicates = int(df.duplicated().sum())
    return nulls, duplicates


@st.cache_data(show_spinner=False)
def interaction_matrices(df: pd.DataFrame, col: str, min_support: int | None = None):
    """Generic interaction matrices: hour_of_day × <col>, with optional support filtering on <col>."""
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


# -----------------------------
# Charts & downloads
# -----------------------------
def alt_bar(df: pd.DataFrame, x: str, y: str, title: str, tooltip=None):
    """Simple reusable Altair bar chart."""
    return (
        alt.Chart(df)
        .mark_bar()
        .encode(x=alt.X(x, sort=None), y=alt.Y(y), tooltip=tooltip or [x, y])
        .properties(title=title, height=280)
    )


def df_to_csv_download(df: pd.DataFrame, filename: str, label: str):
    """Streamlit download button for a DataFrame as CSV."""
    if df is None or df.empty:
        return
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    st.download_button(label=label, data=buf.getvalue(), file_name=filename, mime="text/csv")
