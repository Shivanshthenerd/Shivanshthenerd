👋 Hi, I'm Shivansh and I'm currently exploring the world of computers.

---

# 📈 Indian SIP Churn Rate Prediction

A **production-grade** Python project for predicting SIP (Systematic Investment Plan) churn in India, built on real AMFI mutual fund data.

---

## Project Structure

```
.
├── data/
│   ├── raw/                        # Source CSVs (real Indian MF data)
│   │   ├── india_mf_funds.csv      # 814 AMFI-registered fund schemes
│   │   └── sip_india_monthly.csv   # 87 912 fund × month observations
│   ├── processed/                  # Cleaned & merged artefacts
│   │   ├── funds_clean.csv
│   │   ├── monthly_clean.csv
│   │   └── merged_panel.csv
│   └── features/
│       └── features.csv            # ← final feature table for modelling
│
├── src/
│   ├── ingestion/
│   │   ├── data_loader.py          # Stage 1 – load & validate raw CSVs
│   │   └── data_cleaning.py        # Stage 2 – impute, cap, dedup, merge
│   ├── features/
│   │   └── feature_engineering.py  # Stage 3 – generate all features
│   └── utils/
│       ├── logger.py               # Shared logging configuration
│       └── io_helpers.py           # Safe CSV read/write helpers
│
├── notebooks/                      # Jupyter notebooks for exploration
├── outputs/                        # Model outputs, plots, reports
├── run_pipeline.py                 # ← entry point (runs all 3 stages)
├── sip_churn_prediction.ipynb      # Deep learning notebook (DNN + LSTM)
└── requirements.txt
```

---

## Real Indian Dataset

| File | Source | Rows |
|---|---|---|
| `data/raw/india_mf_funds.csv` | **Kaggle "Mutual Funds India Detailed"** (AMFI) | 814 schemes |
| `data/raw/sip_india_monthly.csv` | Derived from above (814 funds × 120 months) | 87 912 |

> Individual SIP investor records are proprietary to AMCs and are not publicly available in India. This project models attrition at the **fund-month level** using real AMFI fund statistics — the standard approach in academic research on Indian MF churn.

---

## Feature Engineering Pipeline

Run the entire pipeline with one command:

```bash
python run_pipeline.py
```

### Features generated (`data/features/features.csv` — 48 columns)

| Group | Features |
|---|---|
| **Tenure** | `tenure_months`, `tenure_band`, `is_early_stage` |
| **Rolling returns** | `roll_3m_return`, `roll_6m_return`, `roll_12m_return`, `return_momentum`, `return_reversal`, `excess_return_3m`, `excess_return_6m` |
| **Volatility** | `volatility_3m`, `volatility_ratio`, `sharpe_3m` |
| **SIP consistency** | `missed_payment_ratio`, `payment_regularity`, `consec_neg_flag` |
| **Investment / cost** | `avg_sip_amount`, `relative_expense`, `cost_drag` |
| **Market trend** | `drawdown_severity`, `above_benchmark`, `alpha_positive` |
| **Fund quality** | `rating_band`, `risk_adj_return`, `size_band` |
| **Raw fund stats** | `expense_ratio`, `alpha`, `beta`, `sharpe`, `sortino`, `risk_level`, `rating`, … |
| **Target** | **`churn`** (1 = discontinued SIP, 0 = active) |

---

## Getting Started

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the feature pipeline (ingestion → cleaning → feature engineering)
python run_pipeline.py

# 3. Open the deep learning notebook
jupyter notebook sip_churn_prediction.ipynb
```

---

## Module Reference

| Module | Responsibility |
|---|---|
| `src/utils/logger.py` | Shared `get_logger()` — consistent timestamps across all stages |
| `src/utils/io_helpers.py` | `read_csv()` / `save_csv()` with logging and auto-mkdir |
| `src/ingestion/data_loader.py` | Load raw CSVs, standardise column names, validate schema |
| `src/ingestion/data_cleaning.py` | Impute nulls (median/mode), cap IQR outliers, deduplicate, merge |
| `src/features/feature_engineering.py` | Generate all 40+ features, encode categoricals, produce `features.csv` |
| `run_pipeline.py` | Orchestrate all three stages end-to-end |
