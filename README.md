
# Next Best Action (NBA) — Senior DS Assessment

A lightweight, transparent **propensity modeling** prototype to support **Next Best Action** decisions.  
The Streamlit app walks through **EDA → Modeling (LR & LGBM) → Interpretability & Business** and shows how model scores become **budgeted actions** with **economic thresholds**.

> **Live app:** (https://gabriel-ds-role.streamlit.app/)
> **Slide-free demo:** the app itself is the presentation.

---

## Table of Contents
- [Quick Start](#quick-start)
- [Repository Structure](#repository-structure)
- [What’s in the App (Navigation)](#whats-in-the-app-navigation)
- [Data](#data)
- [Artifacts the App Reads](#artifacts-the-app-reads)
- [Reproduce / Regenerate Artifacts](#reproduce--regenerate-artifacts)
- [Modeling Approach (Why These Choices)](#modeling-approach-why-these-choices)
- [Using the “Interpretability & Business” Page](#using-the-interpretability--business-page)
- [Theming & UX](#theming--ux)
- [Troubleshooting](#troubleshooting)
- [Notes & Attribution](#notes--attribution)
- [License](#license)

---

## Quick Start

### Local
```bash
git clone https://github.com/gabrielfernandezf/betsson_ds_role.git
cd betsson_ds_role
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scriptsctivate
pip install -r requirements.txt
streamlit run main.py
```

### Streamlit Cloud
1. Deploy this repo.  
2. Ensure the files listed in **Artifacts the App Reads** exist under `artifacts/reports/`.  
3. Open the app.

---

## Repository Structure
```
betsson_ds_role/
├─ .streamlit/
│  └─ config.toml                 # App theme (dark) + brand colors
├─ artifacts/                     # Exported models & reports used by the app
│  ├─ models/                     # (Saved model bundles — optional for this demo)
│  ├─ reports/                    # ← The app reads from here (see list below)
│  └─ notebooks/
│     └─ 01_train_eval_export.ipynb  # Reproducible training notebook
├─ data/
│  └─ avazu_50k_rows.csv          # Public sample used for EDA & demo
├─ src/
│  ├─ utils.py
│  ├─ overview.py                 # Overview page (welcome, scope, approach)
│  ├─ eda.py                      # Exploratory analysis (temporal, cats, interactions)
│  ├─ model.py                    # LR vs LGBM metrics, calibration, gain table, importances
│  └─ interp_business.py          # Decision helpers (budget, p★, profit/ROI, K*)
├─ main.py                        # Top navigation & page routing
├─ requirements.txt
└─ README.md
```

---

## What’s in the App (Navigation)

### Overview
Welcome, **assessment scope**, dataset, modeling approach, and how to read the app.

### EDA
Temporal patterns, categorical lifts, interactions (e.g., **hour × banner_pos**), data quality, and an auto-generated **insights** summary.

### Model
**Logistic Regression (baseline)** vs **LightGBM (tuned + isotonic calibration)**:
- ROC-AUC, PR-AUC, LogLoss, Brier, **Lift@K** tiles & comparison bar  
- **Calibration curve**, **gain table** (deciles), **feature importances**, PDP snapshots  
- Clear rationale for selecting **LGBM calibrated** for the NBA policy

### Interpretability & Business
Turn scores into decisions: set **budget**, **CPA** and **value (V)**.  
See **Expected CTR at top K%**, the **economic threshold** `p★ = CPA/V`, eligible deciles, **recommended K*** (profit-max), and **profit/ROI** at your chosen K.

---

## Data
- **Source:** public **Avazu CTR** sample (~50k rows)  
- **Target:** `click` (0/1)  
- **Window:** ~10 days (hourly)  
- **Path:** `data/avazu_50k_rows.csv`

> The goal is to show an NBA-style **propensity** workflow; the ad context is a proxy for “user action likelihood”.

---

## Artifacts the App Reads
Place these under `artifacts/reports/` (created by the notebook):

- `metrics.json` — metrics for **lr_baseline** and **lgbm_calibrated**  
  (ROC-AUC, PR-AUC, LogLoss, Brier, Base CTR, Lift@K)
- `gain_table_val.csv` — validation deciles: `decile, n, positives, rate, avg_p`
- `feature_importances.csv` — LightGBM importances (sorted)
- `calibration_curve.png` — calibration plot (hold-out day)
- `pd_banner_pos.png`, `pd_device_type.png`, `pd_device_conn_type.png` — PDP snapshots
---

## Reproduce / Regenerate Artifacts
Open and run:
```
artifacts/notebooks/01_train_eval_export.ipynb
```
The notebook:
1. **Loads & splits** (time-based: last day for validation)  
2. **EDA-aligned Feature Engineering:** cyclic hour, `is_night`, **hour×banner_pos**, **hour×device_type**, rare-collapse (support≥300), **smoothed target encoding** (train-only), anti-leakage (remove raw IDs)  
3. Trains **Logistic Regression** (baseline) and **LightGBM** (tuned)  
4. **Calibrates** LGBM (isotonic) and evaluates  
5. **Exports** artifacts to `artifacts/reports/`

---

## Modeling Approach (Why These Choices)
- **Two models by design**
  - **LR baseline** — interpretable, quick, verifies that FE carries signal (explicit interactions)
  - **LGBM tuned + calibrated** — stronger ranking + **probabilities you can trust** (calibration enables economic thresholds)
- **Metrics**
  - Ranking: ROC-AUC, PR-AUC  
  - Probabilities: **LogLoss, Brier**  
  - Business: **Lift@K**, **gain table**
- **Why calibration?**  
  If the model says **p = 0.20**, we want ~20% to click on average ⇒ enables rules like **act if `p ≥ CPA/V`**.

---

## Using the “Interpretability & Business” Page
- Set **Budget** (top K% to contact), **CPA**, and **V**.
- Read the KPIs:
  - **Expected CTR at top K%** (+Δ vs base): the concentration the model provides for your capacity  
  - **Economic threshold** `p★ = CPA/V`: the minimum probability that makes the action profitable on average  
  - **Eligible deciles**: groups with `avg_p ≥ p★`  
  - **Recommended K***: discrete K (10%, 20%, …) that **maximizes expected profit**  
  - **Economics** for your chosen K: users contacted, expected clicks, profit, ROI
- Charts:
  - **Avg p by decile** (+ rule at `p★`)  
  - **Cumulative CTR vs K** (coverage vs quality trade-off)

---

## Theming & UX
- Theme in `.streamlit/config.toml` (dark):
  - Background: `#040404`
  - Primary: `#0eaeb0`
  - Secondary: `#f66c24`
  - Text: white
- Top navigation via `streamlit-option-menu` in `main.py`.

---

## Notes & Attribution
- Dataset inspired by the **Avazu CTR** competition (Kaggle).  
- This codebase is for an interview assessment; not production-grade.

---

## License
_Add your license here (e.g., MIT) or mark as “Assessment only”._
