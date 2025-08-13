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

# ---------- EDA Highlights ----------
@st.cache_data(show_spinner=False)
def eda_highlights(df: pd.DataFrame, min_support: int = 300) -> dict:
    base_ctr = float(df["click"].mean())

    by_day = (
        df.groupby("date")
          .agg(impressions=("click", "size"), ctr=("click", "mean"))
          .reset_index()
    )
    best_day = by_day.loc[by_day["ctr"].idxmax()].to_dict()
    worst_day = by_day.loc[by_day["ctr"].idxmin()].to_dict()

    by_hour = (
        df.groupby("hour_of_day")
          .agg(impressions=("click", "size"), ctr=("click", "mean"))
          .reset_index()
    )
    peak_hr_ctr = by_hour.loc[by_hour["ctr"].idxmax()].to_dict()
    peak_hr_impr = by_hour.loc[by_hour["impressions"].idxmax()].to_dict()

    # best banner_pos (with support filtering)
    bp = ctr_table(df, "banner_pos", min_n=min_support, top=15) if "banner_pos" in df.columns else None
    best_bp = bp.sort_values("lift", ascending=False).iloc[0].to_dict() if bp is not None and len(bp) else None

    return {
        "base_ctr": base_ctr,
        "best_day": best_day,
        "worst_day": worst_day,
        "peak_hr_ctr": peak_hr_ctr,
        "peak_hr_impr": peak_hr_impr,
        "best_banner_pos": best_bp,
    }

# ---------- Optional: Cramér’s V association matrix ----------
@st.cache_data(show_spinner=False)
def cramers_v_matrix(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame | None:
    """
    Computes a Cramér's V matrix for a list of categorical columns.
    Requires scipy; returns None if scipy is not available.
    Uses bias-corrected formula.
    """
    try:
        import numpy as np
        import pandas as pd
        from scipy.stats import chi2_contingency  # optional dependency
    except Exception:
        return None

    cols = [c for c in cols if c in df.columns]
    if not cols:
        return None

    def cramers_v(a: pd.Series, b: pd.Series) -> float:
        ct = pd.crosstab(a, b)
        chi2 = chi2_contingency(ct, correction=False)[0]
        n = ct.values.sum()
        r, k = ct.shape
        if n == 0:
            return np.nan
        phi2 = chi2 / n
        # bias correction (Bergsma)
        phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1)) if n > 1 else 0
        rcorr = r - ((r - 1) ** 2) / (n - 1) if n > 1 else r
        kcorr = k - ((k - 1) ** 2) / (n - 1) if n > 1 else k
        denom = min((kcorr - 1), (rcorr - 1))
        return np.sqrt(phi2corr / denom) if denom > 0 else np.nan

    mat = pd.DataFrame(index=cols, columns=cols, dtype=float)
    for i, c1 in enumerate(cols):
        for j, c2 in enumerate(cols):
            if j < i:
                mat.loc[c1, c2] = mat.loc[c2, c1]
            else:
                mat.loc[c1, c2] = 1.0 if c1 == c2 else cramers_v(df[c1], df[c2])
    return mat.round(3)

# ---------- Build a human-readable Markdown summary ----------
@st.cache_data(show_spinner=False)
def build_eda_summary_md(high: dict, dow_tbl: pd.DataFrame, ctr_tabs: dict[str, pd.DataFrame]) -> str:
    lines = []
    lines.append("# EDA Summary")
    lines.append("")
    lines.append(f"- **Base CTR**: {high['base_ctr']:.4f}")
    lines.append(f"- **Best day**: {high['best_day']['date']} | CTR={high['best_day']['ctr']:.4f} | Impr={int(high['best_day']['impressions'])}")
    lines.append(f"- **Worst day**: {high['worst_day']['date']} | CTR={high['worst_day']['ctr']:.4f}")
    lines.append(f"- **Peak hour (CTR)**: {int(high['peak_hr_ctr']['hour_of_day'])}:00 | CTR={high['peak_hr_ctr']['ctr']:.4f}")
    lines.append(f"- **Peak hour (Impr)**: {int(high['peak_hr_impr']['hour_of_day'])}:00 | Impr={int(high['peak_hr_impr']['impressions'])} | CTR={high['peak_hr_impr']['ctr']:.4f}")
    if high.get("best_banner_pos"):
        bp = high["best_banner_pos"]
        lines.append(f"- **banner_pos={bp['banner_pos']}** → CTR={bp['ctr']:.4f} | lift={bp['lift']:.2f}× | share={bp['share']:.2%}")
    lines.append("")
    lines.append("## Day-of-week (coverage & lift)")
    lines.append(dow_tbl.to_markdown(index=False))
    lines.append("")
    lines.append("## Categoricals (top groups by support)")
    for name, tbl in ctr_tabs.items():
        lines.append(f"### {name}")
        # show top 5 rows only to keep it short
        head5 = tbl.head(5)
        lines.append(head5.to_markdown(index=False))
        lines.append("")
    return "\n".join(lines)

# ---------- Build a machine-friendly JSON snapshot ----------
@st.cache_data(show_spinner=False)
def build_eda_summary_json(high: dict, dow_tbl: pd.DataFrame, ctr_tabs: dict[str, pd.DataFrame]) -> dict:
    return {
        "highlights": high,
        "by_dow": dow_tbl.to_dict(orient="records"),
        "categoricals": {name: tbl.to_dict(orient="records") for name, tbl in ctr_tabs.items()},
    }
