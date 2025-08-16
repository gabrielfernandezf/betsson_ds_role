# src/model.py
from pathlib import Path
import json
import pandas as pd
import streamlit as st

def _load_reports(base: Path):
    metrics = json.loads((base / "metrics.json").read_text())
    gain = pd.read_csv(base / "gain_table_val.csv")
    importances = pd.read_csv(base / "feature_importances.csv")
    imgs = {
        "calibration": base / "calibration_curve.png",
        "pd_banner_pos": base / "pd_banner_pos.png",
        "pd_device_type": base / "pd_device_type.png",
        "pd_device_conn_type": base / "pd_device_conn_type.png",
    }
    return metrics, gain, importances, imgs

def render(df=None):
    st.header("Model")

    # where Streamlit finds your artifacts
    reports_dir = Path("artifacts/reports")
    if not reports_dir.exists():
        st.error("artifacts/reports/ not found. Please add exported reports.")
        return

    metrics, gain, importances, imgs = _load_reports(reports_dir)

    st.subheader("Validation metrics (hold-out last day)")
    m = metrics["lgbm_calibrated"]
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("ROC-AUC", f"{m['roc_auc']:.3f}")
    c2.metric("PR-AUC", f"{m['pr_auc']:.3f}")
    c3.metric("LogLoss", f"{m['logloss']:.3f}")
    c4.metric("Brier", f"{m['brier']:.3f}")
    c5.metric("Base CTR", f"{m['base_ctr']:.3f}")

    # Lift@K (calibrated)
    st.caption("Lift@K (calibrated)")
    k_cols = [k for k in m.keys() if k.startswith("lift@")]
    st.write({k: round(m[k], 3) for k in sorted(k_cols)})

    st.divider()
    st.subheader("Calibration")
    if imgs["calibration"].exists():
        st.image(str(imgs["calibration"]), use_container_width=True, caption="Calibration curve — LGBM (calibrated)")
    else:
        st.info("calibration_curve.png not found in artifacts/reports.")

    st.subheader("Gain table (validation deciles)")
    st.dataframe(gain, use_container_width=True)
    st.download_button("⬇️ Download gain table", data=gain.to_csv(index=False),
                       file_name="gain_table_val.csv", mime="text/csv")

    st.divider()
    st.subheader("Top feature importances (LGBM)")
    st.dataframe(importances.head(20), use_container_width=True)
    st.download_button("⬇️ Download importances", data=importances.to_csv(index=False),
                       file_name="feature_importances.csv", mime="text/csv")

    st.divider()
    st.subheader("Partial dependence (sanity checks)")
    imgs_found = False
    for key, p in imgs.items():
        if key == "calibration":
            continue
        if p.exists():
            imgs_found = True
            st.image(str(p), use_container_width=True, caption=p.stem.replace("_", " "))
    if not imgs_found:
        st.info("PD images not found in artifacts/reports.")

    # Short narrative using your numbers
    st.divider()
    st.subheader("What the validation shows")
    st.markdown(
        "- **Calibration:** curve sits on the diagonal (≈0.05–0.35). Scores behave like true probabilities, enabling rules like `p > CPA/V`.\n"
        "- **Concentration:** base CTR ~ **0.166**. Top 10% ⇒ **0.354** (Lift **2.13×**); top 20% ⇒ **0.304** (Lift **1.83×**); top 30% ⇒ **0.277** (Lift **1.67×**).\n"
        "- **PDP sanity-check:** `device_type=1` above `4–5`; `conn_type=2` underperforms; `banner_pos` benefits with hour context.\n"
        "- **Feature story:** engineered interactions (`hour×banner_pos`, `hour×device_type`) dominate importances—EDA→FE→Model alignment."
    )
