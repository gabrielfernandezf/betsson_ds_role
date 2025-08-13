# main.py
import io
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
from streamlit_option_menu import option_menu

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
# Constants & State
# -----------------------
DATA_CSV = Path("data/avazu_50k_rows.csv")

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
def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
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
    if {"hour_of_day","banner_pos"}.issubset(df.columns):
        pivot_ctr = (df.groupby(["hour_of_day","banner_pos"])["click"].mean()
                       .unstack("banner_pos").sort_index().round(4))
        pivot_n   = (df.groupby(["hour_of_day","banner_pos"])["click"].size()
                       .unstack("banner_pos").sort_index().fillna(0).astype(int))
    else:
        pivot_ctr, pivot_n = None, None
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
# Load data (fixed path)
# -----------------------
if not DATA_CSV.exists():
    st.error("No encontré `data/avazu_50k_rows.csv`. Subí el CSV a esa ruta en el repo.")
    st.stop()

df = load_data(DATA_CSV)
base_ctr = df["click"].mean()
time_window = f"{df['dt'].min()} → {df['dt'].max()}"

# -----------------------
# Sidebar Navigation
# -----------------------
with st.sidebar:
    selected = option_menu(
        "Navegación",
        ["Overview", "EDA", "Model", "Interpretability & Business"],
        icons=["house", "bar-chart", "cpu", "lightbulb"],
        menu_icon="cast",
        default_index=0,
        orientation="vertical",
    )

# -----------------------
# Sections
# -----------------------
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

elif selected == "EDA":
    st.header("Exploratory Data Analysis")

    # control dentro de la página (no en sidebar)
    min_support = st.slider("Min group size (support)", 50, 2000, st.session_state.min_support, 50)
    st.session_state.min_support = min_support

    by_day, by_hour, ctr_tabs, pivot_ctr, pivot_n, uni = eda_tables(df, min_support)

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
    import json
    from pathlib import Path
    import pandas as pd

    st.header("Model performance & evaluation")

    ART = Path("artifacts")
    met_path = ART / "metrics.json"
    roc_path = ART / "roc_curve.png"
    pr_path  = ART / "pr_curve.png"
    fi_path  = ART / "feature_importance.csv"

    # metrics.json
    if met_path.exists():
        with met_path.open() as f:
            metrics = json.load(f)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("ROC-AUC", f"{metrics.get('roc_auc', float('nan')):.4f}")
        c2.metric("PR-AUC",  f"{metrics.get('pr_auc', float('nan')):.4f}")
        c3.metric("Precision@5%", f"{metrics.get('precision_at_05', float('nan')):.4f}")
        c4.metric("Lift@5%",      f"{metrics.get('lift_at_05', float('nan')):.2f}")
        st.json(metrics)
    else:
        st.warning("No encontré artifacts/metrics.json. Generá los artefactos con el script y subilos al repo.")

    # Curvas
    cols = st.columns(2)
    if roc_path.exists():
        cols[0].image(str(roc_path), caption="ROC curve", use_column_width=True)
    if pr_path.exists():
        cols[1].image(str(pr_path), caption="Precision-Recall curve", use_column_width=True)

    # Feature Importance
    if fi_path.exists():
        st.subheader("Feature importance")
        st.dataframe(pd.read_csv(fi_path), use_container_width=True)
    else:
        st.info("Subí artifacts/feature_importance.csv para mostrar importancia de variables.")

elif selected == "Interpretability & Business":
    st.header("Interpretability & Business (coming next)")
    st.write("Acá vamos a mostrar: Feature importance, SHAP summary y traducción de insights a decisiones NBA.")
    st.warning("Stub por ahora. Agregamos imágenes/artefactos en la siguiente iteración.")
