betsson_ds_role/
├─ .streamlit/
│  └─ config.toml                 # App theme (dark) + brand colors
├─ artifacts/                     # Exported models & reports used by the app
│  ├─ models/                     # (Saved model bundles — optional for this demo)
│  ├─ reports/                    # ← The app reads from here (see list below)
│  └─ notebooks/
│     └─ 01_train_eval_export.ipynb  # Reproducible training notebook (English)
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
