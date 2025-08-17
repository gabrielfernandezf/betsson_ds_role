# src/model.py
from pathlib import Path
import json
import re
import pandas as pd
import streamlit as st
import altair as alt

REPORTS_DIR = Path("artifacts/reports")
NB_PATH = Path("artifacts/notebooks/01_train_eval_export.ipynb")

# ------------- Helpers -------------
def _load_reports(base: Path):
    """Load metrics.json, gain table, importances and image paths. Be robust to missing files."""
    metrics = json.loads((base / "metrics.json").read_text()) if (base / "metrics.json").exists() else {}
    gain = pd.read_csv(base / "gain_table_val.csv") if (base / "gain_table_val.csv").exists() else None
    importances = pd.read_csv(base / "feature_importances.csv") if (base / "feature_importances.csv").exists() else None
    imgs = {
        # Typically we only exported LGBM calibration/PDPs, handle missing gracefully
        "calibration": base / "calibration_curve.png",
        "pd_banner_pos": base / "pd_banner_pos.png",
        "pd_device_type": base / "pd_device_type.png",
        "pd_device_conn_type": base / "pd_device_conn_type.png",
    }
    return metrics, gain, importances, imgs


def _lift_items(metric_dict: dict) -> list[tuple[int, float]]:
    """Extract [('5',2.1),...] from keys like 'lift@5%' and sort by K asc."""
    if not metric_dict:
        return []
    out = []
    for k, v in metric_dict.items():
        m = re.search(r"lift@(\d+)%", k)
        if m:
            out.append((int(m.group(1)), float(v)))
    out.sort(key=lambda t: t[0])
    return out


def _metric_row(name: str, d: dict) -> dict:
    if not isinstance(d, dict):
        d = {}
    return {
        "model": name,
        "ROC-AUC": d.get("roc_auc"),
        "PR-AUC": d.get("pr_auc"),
        "LogLoss": d.get("logloss"),
        "Brier": d.get("brier"),
        "Base CTR": d.get("base_ctr"),
    }


def _lift_long_df(all_metrics: dict) -> pd.DataFrame | None:
    """Build long df with columns [model, K, lift]."""
    rows = []
    for name, md in all_metrics.items():
        for K, lift in _lift_items(md):
            rows.append({"model": name, "K": K, "lift": lift})
    if not rows:
        return None
    return pd.DataFrame(rows)


# ------------- Page render -------------
def render(df=None):
    st.header("Model")

    if not REPORTS_DIR.exists():
        st.error("artifacts/reports/ not found. Please add exported reports.")
        return

    metrics, gain, importances, imgs = _load_reports(REPORTS_DIR)

    # Detect available models in metrics.json
    # Expected keys: "lr_baseline", "lgbm_calibrated" (names can vary; we try to guess sensibly)
    model_keys = [k for k in metrics.keys() if isinstance(metrics[k], dict)]
    # Prefer canonical names if present
    lr_key = next((k for k in model_keys if "lr" in k.lower()), None)
    lgbm_key = next((k for k in model_keys if "lgbm" in k.lower()), None)
    # Fallbacks
    if lgbm_key is None and "calibrated" in metrics:
        lgbm_key = "calibrated"

    # -------- Summary: side-by-side KPIs for LR vs LGBM --------
    st.subheader("Validation metrics (hold-out last day)")
    comp = []
    if lr_key: comp.append(_metric_row("LR baseline", metrics[lr_key]))
    if lgbm_key: comp.append(_metric_row("LGBM (tuned + calibrated)", metrics[lgbm_key]))
    if comp:
        df_comp = pd.DataFrame(comp)
        # Show neatly, round numeric cols
        num_cols = [c for c in df_comp.columns if c != "model"]
        df_comp[num_cols] = df_comp[num_cols].astype(float).round(3)
        st.dataframe(df_comp, use_container_width=True, hide_index=True)
    else:
        st.info("No model metrics found in metrics.json.")

    # -------- Lift@K tiles (for LGBM by default, fall back to LR if needed) --------
    st.subheader("Lift@K (calibrated probabilities)")
    chosen_key = lgbm_key or lr_key
    if chosen_key:
        items = _lift_items(metrics[chosen_key])
        if items:
            # KPI tiles, up to 5 per row
            cols = st.columns(min(len(items), 5))
            for i, (K, lift) in enumerate(items[:5]):
                delta = (lift - 1.0) * 100.0
                cols[i].metric(label=f"Lift@{K}%", value=f"{lift:.2f}×", delta=f"{delta:+.0f}%")
        else:
            st.info(f"No Lift@K keys found for '{chosen_key}'.")
    else:
        st.info("No model selected for Lift@K.")

    # -------- Lift@K comparison chart (LR vs LGBM) --------
    all_models = {}
    if lr_key: all_models["LR baseline"] = metrics[lr_key]
    if lgbm_key: all_models["LGBM (tuned + calibrated)"] = metrics[lgbm_key]
    long_df = _lift_long_df(all_models)
    if long_df is not None and not long_df.empty:
        base = alt.Chart(long_df).mark_bar().encode(
            x=alt.X("K:O", title="Top K% (by score)"),
            y=alt.Y("lift:Q", title="Lift vs base CTR"),
            color=alt.Color("model:N", legend=alt.Legend(title=None))
        ).properties(height=280)
        rule = alt.Chart(pd.DataFrame({"y": [1.0]})).mark_rule(strokeDash=[6,4]).encode(y="y:Q")
        st.altair_chart(base + rule, use_container_width=True)

    # -------- Rationale & modeling decisions --------
    st.subheader("Why two models and why LGBM for NBA")
    st.markdown(
        """
- **Two models on purpose**  
  - **Logistic Regression (LR)**: fast, interpretable, great baseline. It validates that our **EDA-driven features** (cyclic hour, night flag, hour×context interactions, smoothed TE) carry the signal we think they do.  
  - **LightGBM (LGBM)**: captures **non-linearities** and higher-order **interactions** automatically. We still feed the engineered features to **bias it in the right direction** (good first-order priors).
- **Calibrated probabilities (isotonic)**  
  Scores behave like **true probabilities** (calibration curve ~ diagonal), which is what we need to apply a **business threshold** `p★ = CPA/V` and reason about **expected value** per user.
- **Observed validation behavior**  
  **Lift@K** clearly concentrates positives at the top (e.g., K=10–20%), and **LogLoss/Brier** confirm the probability quality after calibration.
- **Why LGBM for the NBA app**  
  Better **ranking** in the head of the list, more **stable** under diverse contexts, and **probabilities you can use** for budget/threshold decisions. We keep LR as a **reference** and for **sanity checks**.
        """
    )

    # -------- Calibration & PDPs (two columns layout) --------
    st.subheader("Calibration & Partial Dependence (sanity checks)")

    # Row 1: calibration + a PDP if available
    col1, col2 = st.columns(2)
    if imgs["calibration"].exists():
        col1.image(str(imgs["calibration"]), caption="Calibration — LGBM (isotonic)", use_container_width=True)
    else:
        col1.info("calibration_curve.png not found.")

    # Pick any two PDPs for the first row if present
    pd_list = [("PDP — banner_pos", imgs["pd_banner_pos"]),
               ("PDP — device_type", imgs["pd_device_type"]),
               ("PDP — device_conn_type", imgs["pd_device_conn_type"])]
    first_pdp_shown = False
    for title, p in pd_list:
        if p.exists():
            col2.image(str(p), caption=title, use_container_width=True)
            first_pdp_shown = True
            break
    if not first_pdp_shown:
        col2.info("Partial dependence images not found.")

    # Row 2: the remaining PDPs (if any)
    remaining = [item for item in pd_list if item[1].exists()][1:]
    if remaining:
        r1, r2 = st.columns(2)
        if len(remaining) >= 1:
            r1.image(str(remaining[0][1]), caption=remaining[0][0], use_container_width=True)
        if len(remaining) >= 2:
            r2.image(str(remaining[1][1]), caption=remaining[1][0], use_container_width=True)

    # -------- Gain table & Importances --------
    st.subheader("Gain table (validation deciles)")
    if gain is not None:
        st.dataframe(gain, use_container_width=True)
        st.download_button("⬇️ Download gain table (CSV)",
                           data=gain.to_csv(index=False),
                           file_name="gain_table_val.csv",
                           mime="text/csv",
                           use_container_width=True)
    else:
        st.info("gain_table_val.csv not found.")

    st.subheader("Top feature importances (LGBM)")
    if importances is not None and not importances.empty:
        st.dataframe(importances.head(20), use_container_width=True)
        st.download_button("⬇️ Download importances (CSV)",
                           data=importances.to_csv(index=False),
                           file_name="feature_importances.csv",
                           mime="text/csv",
                           use_container_width=True)
    else:
        st.info("feature_importances.csv not found.")

    # -------- Notebook download --------
    st.divider()
    st.subheader("Reproducibility")
    st.markdown("All artifacts in this page come from the training notebook.")
    if NB_PATH.exists():
        st.download_button(
            "⬇️ Download training notebook (.ipynb)",
            data=NB_PATH.read_bytes(),
            file_name=NB_PATH.name,
            mime="application/x-ipynb+json",
            use_container_width=True,
        )
    else:
        st.info("Notebook file not found in artifacts/notebooks/.")

# # src/model.py
# from pathlib import Path
# import json
# import pandas as pd
# import streamlit as st
# import re, altair as alt

# def _load_reports(base: Path):
#     metrics = json.loads((base / "metrics.json").read_text())
#     gain = pd.read_csv(base / "gain_table_val.csv")
#     importances = pd.read_csv(base / "feature_importances.csv")
#     imgs = {
#         "calibration": base / "calibration_curve.png",
#         "pd_banner_pos": base / "pd_banner_pos.png",
#         "pd_device_type": base / "pd_device_type.png",
#         "pd_device_conn_type": base / "pd_device_conn_type.png",
#     }
#     return metrics, gain, importances, imgs

# def render(df=None):
#     st.header("Model")

#     # where Streamlit finds your artifacts
#     reports_dir = Path("artifacts/reports")
#     if not reports_dir.exists():
#         st.error("artifacts/reports/ not found. Please add exported reports.")
#         return

#     metrics, gain, importances, imgs = _load_reports(reports_dir)

#     st.subheader("Validation metrics (hold-out last day)")
#     m = metrics["lgbm_calibrated"]
#     c1, c2, c3, c4, c5 = st.columns(5)
#     c1.metric("ROC-AUC", f"{m['roc_auc']:.3f}")
#     c2.metric("PR-AUC", f"{m['pr_auc']:.3f}")
#     c3.metric("LogLoss", f"{m['logloss']:.3f}")
#     c4.metric("Brier", f"{m['brier']:.3f}")
#     c5.metric("Base CTR", f"{m['base_ctr']:.3f}")

#     st.subheader("Lift@K (calibrated)")
    
#     # Collect and sort by numeric K
#     lift_items = []
#     for k, v in m.items():
#         if k.startswith("lift@"):
#             # extract the number before '%', e.g., 'lift@10%' -> 10
#             match = re.search(r"lift@(\d+)%", k)
#             if match:
#                 lift_items.append((int(match.group(1)), v))
#     lift_items.sort(key=lambda t: t[0])  # ascending by K
    
#     if not lift_items:
#         st.info("No Lift@K metrics found.")
#     else:
#         # KPI tiles
#         n = len(lift_items)
#         cols = st.columns(n if n <= 5 else 5)  # show up to 5 tiles per row
#         for i, (k, lift) in enumerate(lift_items[:5]):
#             delta = (lift - 1.0) * 100.0
#             cols[i].metric(
#                 label=f"Lift@{k}%",
#                 value=f"{lift:.2f}×",
#                 delta=f"{delta:+.0f}%"
#             )


#     st.divider()
#     st.subheader("What the validation shows")
#     st.markdown(
#         "- **Calibration:** curve sits on the diagonal (≈0.05–0.35). Scores behave like true probabilities, enabling rules like `p > CPA/V`.\n"
#         "- **Concentration:** base CTR ~ **0.166**. Top 10% ⇒ **0.354** (Lift **2.13×**); top 20% ⇒ **0.304** (Lift **1.83×**); top 30% ⇒ **0.277** (Lift **1.67×**).\n"
#         "- **PDP sanity-check:** `device_type=1` above `4–5`; `conn_type=2` underperforms; `banner_pos` benefits with hour context.\n"
#         "- **Feature story:** engineered interactions (`hour×banner_pos`, `hour×device_type`) dominate importances—EDA→FE→Model alignment."
#     )

#     st.divider()
#     st.subheader("Calibration")
#     if imgs["calibration"].exists():
#         st.image(str(imgs["calibration"]), use_container_width=True, caption="Calibration curve — LGBM (calibrated)")
#     else:
#         st.info("calibration_curve.png not found in artifacts/reports.")

#     st.subheader("Gain table (validation deciles)")
#     st.dataframe(gain, use_container_width=True)
#     st.download_button("⬇️ Download gain table", data=gain.to_csv(index=False),
#                        file_name="gain_table_val.csv", mime="text/csv")

#     st.divider()
#     st.subheader("Top feature importances (LGBM)")
#     st.dataframe(importances.head(20), use_container_width=True)
#     st.download_button("⬇️ Download importances", data=importances.to_csv(index=False),
#                        file_name="feature_importances.csv", mime="text/csv")

#     st.divider()
#     st.subheader("Partial dependence (sanity checks)")
#     imgs_found = False
#     for key, p in imgs.items():
#         if key == "calibration":
#             continue
#         if p.exists():
#             imgs_found = True
#             st.image(str(p), use_container_width=True, caption=p.stem.replace("_", " "))
#     if not imgs_found:
#         st.info("PD images not found in artifacts/reports.")

