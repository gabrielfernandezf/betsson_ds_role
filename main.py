# main.py
import io
import os
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

# Top nav (horizontal)
try:
    from streamlit_option_menu import option_menu
    HAS_OPT_MENU = True
except Exception:
    HAS_OPT_MENU = False

# -----------------------
# Page config & styling
# -----------------------
st.set_page_config(page_title="NBA Prototype — Avazu CTR", layout="wide")
st.markdown(
    """
    <style>
    .block-container {padding-top: 1rem; padding-bottom: 2rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------
# State & constants
# -----------------------
DATA_DIR = Path("data")
DEFAULT_PATH = DATA_DIR / "50krecords.csv"

if "dataset_label" not in st.session_state:
    st.session_state.dataset_label = "data/50krecords.csv" if DEFAULT_PATH.exists() else "uploaded_file"

if "min_support" not in st.session_state:
    st.session_state.min_support = 300

# -----------------------
# Helpers
# -----------------------
@st.cache_data(show_spinner=False)
def parse_hour(series: pd.Series) -> pd.Series:
    s = series.astype(str)
    h0 = s.iloc[0]
    if any(sep in h0 for sep in ["-", ":", " ", "/"]):
        return pd.to_datetime(s, errors="coerce")
    dt = pd.to_datetime(s, format="%Y%m%d%H", errors="coerce")
    if dt.isna().mean() > 0.5:
        dt = pd.to_datetime(s, format="%y%m%d%H", errors="coerce")
    return dt

@st.cache_data(show_spinner=False)
def load_data(file_or_path):
    """Load CSV (path or uploaded file), build time features."""
    df = pd.read_csv(file_or_path, low_memory=False)
    dt = parse_hour(df["hour"])
    df["dt"] = dt
    df["date"] = df["dt"].dt.date
    df["hour_of_day"] = df["dt"].dt.hour
    df["dow"] = df["dt"].dt.dayofweek
    df["is_weekend"] = (df["dow"] >= 5).astype("int8")
    return df

@st.cache_data(show_spinner=False)
def ctr_table(df: pd.DataFrame, col: str, min_n: int = 300, top: int = 15):
    base = df["click"].mean()
    t = (df.groupby(col).agg(n=("click","size"), ctr=("click","mean")).reset_index())
    t = t[t["n"] >= min_n].copy()
    t["share"] = t["n"] / len(df)
    t["lift"]  = t["ctr"] / base
    return t.sort_values(["n","ctr"], ascending=[False, False]).head(top)

@st.cache_data(show_spinner=False)
def eda_tables(df: pd.DataFrame, min_support: int = 300):
    # Temporal
    by_day  = df.groupby("date").agg(impressions=("click","size"), ctr=("click","mean")).reset_index()
    by_hour = df.groupby("hour_of_day").agg(impressions=("click","size"), ctr=("click","mean")).reset_index()
    # Categorical
    candidates = [c for c in ["banner_pos","device_type","device_conn_type","site_category","app_category"] if c in df.columns]
    ctr_tabs = {c: ctr_table(df, c, min_support, 15) for c in candidates}
    # Interactions: hour x banner_pos
    pivot_ctr = (df.groupby(["hour_of_day","banner_pos"])["click"].mean()
                   .unstack("banner_pos").sort_index().round(4)) if {"hour_of_day","banner_pos"}.issubset(df.columns) else None
    pivot_n   = (df.groupby(["hour_of_day","banner_pos"])["click"].size()
                   .unstack("banner_pos").sort_index().fillna(0).astype(int)) if {"hour_of_day","banner_pos"}.issubset(df.columns) else None
    # Leakage / uniqueness
    cand = [c for c in ["device_ip","device_id","id","site_id","app_id","device_model"] if c in df.columns]
    uni = pd.DataFrame({"col": cand,
                        "nunique": [df[c].nunique(dropna=True) for c in cand],
                        "rows": len(df)})
    uni["uniqueness_ratio"] = (uni["nunique"]/uni["rows"]).round(5)
    uni = uni.sort_values("uniqueness_ratio", ascending=False)
    return by_day, by_hour, ctr_tabs, pivot_ctr, pivot_n, uni

def alt_bar(df, x, y, title, tooltip=None):
    return (
        alt.Chart(df)
        .mark_bar()
        .encode(x=alt.X(x, sort=None),
                y=alt.Y(y),
                tooltip=tooltip or [x, y])
        .properties(title=title, height=280)
    )

def df_to_csv_download(df: pd.DataFrame, filename: str, label: str):
    if df is None or df.empty:
        return
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    st.download_button(label=label, data=buf.getvalue(), file_name=filename, mime="text/csv")

# -----------------------
# Sidebar
# -----------------------
with st.sidebar:
    st.header("Data")
    st.caption("Default: `data/50krecords.csv` (sample). Podés subir otro CSV compatible.")
    up = st.file_uploader("Subir CSV (opcional)", type=["csv"])
    if up is not None:
        st.session_state.dataset_label = "uploaded_file"
        df = load_data(up)
    else:
        st.session_state.dataset_label = str(DEFAULT_PATH) if DEFAULT_PATH.exists() else "uploaded_file"
        df = load_data(DEFAULT_PATH) if DEFAULT_PATH.exists() else None

    st.session_state.min_support = st.slider("Min group size (support)", 50, 2000, st.session_state.min_support, 50)

# -----------------------
# Top Navigation
# -----------------------
menu_items = ["Overview", "EDA", "Model", "Interpretability & Business"]
icons = ["house", "bar-chart", "cpu", "lightbulb"]

if HAS_OPT_MENU:
    selected = option_menu(
        None, menu_items, icons=icons, default_index=0, orientation="horizontal"
    )
else:
    # Fallback simple tabs if option_menu is not installed
    selected = st.tabs(menu_items)
    # When using tabs fallback, just set selected label to first tab's title for rendering once.
    if isinstance(selected, list):
        selected_label = menu_items[0]
    else:
        selected_label = selected
    # We'll keep the same API below
    selected = selected_label

# -----------------------
# Sections
# -----------------------
if df is None:
    st.error("No encontré `data/50krecords.csv` y no subiste un CSV. Subí un archivo o agregá el sample a la carpeta `data/`.")
    st.stop()

base_ctr = df["click"].mean()
time_window = f"{df['dt'].min()} → {df['dt'].max()}"

if selected == "Overview":
    st.title("Next Best Action (NBA) — Prototype")
    st.write("""
    **Use case**: predecir propensión a interactuar (CTR) para priorizar la *próxima mejor acción*.
    **Dataset**: Avazu CTR sample (50k). Anónimo, tabular, 10 días, binaria (click/no click).
    """)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{len(df):,}")
    c2.metric("Cols", f"{df.shape[1]}")
    c3.metric("Base CTR", f"{base_ctr:.4f}")
    c4.metric("Time window", time_window)

    st.info("Tip: en la sidebar podés subir otro CSV. La app recuerda tu selección en esta sesión.")

elif selected == "EDA":
    st.header("Exploratory Data Analysis")
    by_day, by_hour, ctr_tabs, pivot_ctr, pivot_n, uni = eda_tables(df, st.session_state.min_support)

    c1, c2 = st.columns(2)
    c1.altair_chart(alt_bar(by_day, "date:N", "impressions:Q", "Impressions by Day", ["date","impressions","ctr"]), use_container_width=True)
    c2.altair_chart(alt_bar(by_day, "date:N", "ctr:Q", "CTR by Day", ["date","ctr"]), use_container_width=True)

    c3, c4 = st.columns(2)
    c3.altair_chart(alt_bar(by_hour, "hour_of_day:N", "impressions:Q", "Impressions by Hour", ["hour_of_day","impressions","ctr"]), use_container_width=True)
    c4.altair_chart(alt_bar(by_hour, "hour_of_day:N", "ctr:Q", "CTR by Hour", ["hour_of_day","ctr"]), use_container_width=True)

    st.subheader("Categorical (coverage & lift)")
    for name, tbl in ctr_tabs.items():
        st.markdown(f"**{name}**")
        st.dataframe(tbl, use_container_width=True)
        df_to_csv_download(tbl, f"eda_ctr_{name}.csv", f"⬇️ Descargar {name}")

    st.subheader("Interactions: hour_of_day × banner_pos (values)")
    if pivot_ctr is not None:
        st.write("CTR matrix")
        st.dataframe(pivot_ctr, use_container_width=True)
        st.write("Support matrix (N)")
        st.dataframe(pivot_n, use_container_width=True)
        df_to_csv_download(pivot_ctr, "pivot_ctr_hour_banner_pos.csv", "⬇️ Descargar CTR matrix")
        df_to_csv_download(pivot_n,   "pivot_n_hour_banner_pos.csv",   "⬇️ Descargar Support matrix")
    else:
        st.info("No hay columnas para construir la matriz (se necesitan `hour_of_day` y `banner_pos`).")

    st.subheader("Leakage watch (IDs de alta cardinalidad)")
    st.dataframe(uni, use_container_width=True)
    df_to_csv_download(uni, "eda_leakage_uniqueness.csv", "⬇️ Descargar leakage table")

elif selected == "Model":
    st.header("Model (coming next)")
    st.write("Acá vamos a mostrar: preprocessing, métricas (ROC/PR-AUC), Precision@K/Lift, y comparación Baseline vs Tuned.")
    st.warning("Stub por ahora. En la próxima iteración cargamos artifacts precalculados (`artifacts/metrics.json`, curvas, feature_importance.csv).")

elif selected == "Interpretability & Business":
    st.header("Interpretability & Business (coming next)")
    st.write("Acá vamos a mostrar: Feature importance, SHAP summary y traducción de insights a decisiones NBA.")
    st.warning("Stub por ahora. Agregamos imágenes/artefactos en la siguiente iteración.")
