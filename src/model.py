# src/model.py
import json, pandas as pd, streamlit as st
from pathlib import Path

REPORTS = Path("artifacts/reports")

def render(df):
    st.header("Model — Results & Diagnostics")

    # Load metrics.json
    mpath = REPORTS / "metrics.json"
    with open(mpath, "r") as f:
        M = json.load(f)

    st.subheader("Validation summary")
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Val day", M.get("val_day","—"))
    c2.metric("Base CTR", f"{M['lgbm_calibrated']['base_ctr']:.4f}")
    c3.metric("ROC-AUC", f"{M['lgbm_calibrated']['roc_auc']:.3f}")
    c4.metric("LogLoss", f"{M['lgbm_calibrated']['logloss']:.3f}")
    c1,c2,c3 = st.columns(3)
    c1.metric("Lift@5%", f"{M['lgbm_calibrated']['lift@5%']:.2f}×")
    c2.metric("Lift@10%", f"{M['lgbm_calibrated']['lift@10%']:.2f}×")
    c3.metric("Lift@20%", f"{M['lgbm_calibrated']['lift@20%']:.2f}×")

    st.write("Compare baselines")
    st.json({
        "baseline_lr": M["baseline_lr"],
        "lgbm_tuned_raw": M["lgbm_tuned_raw"],
        "lgbm_calibrated": M["lgbm_calibrated"],
    })

    st.subheader("Gain table (validation)")
    gain = pd.read_csv(REPORTS / "gain_table_val.csv")
    st.dataframe(gain, use_container_width=True)

    st.subheader("Feature importances (LGBM)")
    imps = pd.read_csv(REPORTS / "feature_importances.csv")
    st.dataframe(imps.head(20), use_container_width=True)

    st.subheader("Calibration & PDPs")
    st.image(str(REPORTS / "calibration_curve.png"), caption="Calibration curve — LGBM (isotonic)")
    for png in ["pd_banner_pos.png", "pd_device_type.png", "pd_device_conn_type.png"]:
        p = REPORTS / png
        if p.exists():
            st.image(str(p), caption=png.replace("pd_","PD: ").replace(".png",""))
