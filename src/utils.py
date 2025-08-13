# src/utils.py
"""
Shared utilities for the Streamlit app:
- Data loading & robust time parsing
- EDA aggregation tables
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
    """
    Robustly parse Avazu 'hour' column to pandas datetime.
    Handles:
      - ISO-like strings: "2014-10-21 00:00:00"
      - Compact formats:  YYYYMMDDHH or YYMMDDHH
    """
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
    """
    Load CSV and derive time features.
    """
    df = pd.read_csv(path, low_memory=False)
    dt = parse_hour(df["hour"])
    df["dt"] = dt
    df["date"] = df["dt"].dt.date
    df["hour_of_day"] = df["dt"].dt.hour
    df["dow"] = df["dt"].dt.dayofweek
    df["is_weekend"] = (df["dow"] >= 5).astype("int8")
    return df


# -----------------------------
# EDA helpers
# -----------------------------
@st.cache_data(show_spinner=False)
def ctr_table(df: pd.DataFrame, col: str, min_n: int = 300, top: int = 15) -> pd.DataFrame:
    """
    Coverage & lift table for a single categorical column.
    - Filters out groups with support < min_n
    - Sorts by (n desc, ctr desc)
    """
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
    Build key EDA artifacts:
      - by_day (impressions, ctr)
      - by_hour (impressions, ctr)
      - coverage & lift tables for selected categoricals
      - interaction matrices: hour_of_day × banner_pos (CTR and support)
      - high-cardinality / leakage watch table
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
    candidates = [
        c for c in ["banner_pos", "device_type", "device_conn_type", "site_category", "app_category"]
        if c in df.columns
    ]
    ctr_tabs = {c: ctr_table(df, c, min_support, 15) for c in candidates}

    # Interaction matrices
    if {"hour_of_day", "banner_pos"}.issubset(df.columns):
        pivot_ctr = (
            df.groupby(["hour_of_day", "banner_pos"])["click"]
              .mean().unstack("banner_pos").sort_index().round(4)
        )
        pivot_n = (
            df.groupby(["hour_of_day", "banner_pos"])["click"]
              .size().unstack("banner_pos").sort_index().fillna(0).astype(int)
        )
    else:
        pivot_ctr, pivot_n = None, None

    # Leakage / uniqueness
    cand = [c for c in ["device_ip", "device_id", "id", "site_id", "app_id", "device_model"] if c in df.columns]
    uni = pd.DataFrame({
        "col": cand,
        "nunique": [df[c].nunique(dropna=True) for c in cand],
        "rows": len(df)
    })
    uni["uniqueness_ratio"] = (uni["nunique"] / uni["rows"]).round(5)
    uni = uni.sort_values("uniqueness_ratio", ascending=False)

    return by_day, by_hour, ctr_tabs, pivot_ctr, pivot_n, uni


# -----------------------------
# Charts & downloads
# -----------------------------
def alt_bar(df: pd.DataFrame, x: str, y: str, title: str, tooltip=None):
    """
    Simple reusable Altair bar chart.
    """
    return (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X(x, sort=None),
            y=alt.Y(y),
            tooltip=tooltip or [x, y],
        )
        .properties(title=title, height=280)
    )


def df_to_csv_download(df: pd.DataFrame, filename: str, label: str):
    """
    Streamlit download button for a DataFrame as CSV.
    """
    if df is None or df.empty:
        return
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    st.download_button(
        label=label,
        data=buf.getvalue(),
        file_name=filename,
        mime="text/csv",
    )
