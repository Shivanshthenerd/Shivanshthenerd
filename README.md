👋 Hi, I'm Shivansh and I'm currently exploring the world of computers.

---

# 📈 Indian SIP Churn Rate Prediction — Deep Learning Project

A deep learning project that predicts whether SIP investors will **discontinue (churn) a mutual fund scheme**, enabling AMCs and financial advisors to intervene proactively.

## Real Indian Dataset

| File | Source | Description |
|---|---|---|
| `data/india_mf_funds.csv` | **Kaggle — "Mutual Funds India Detailed"** (real AMFI data) | 814 real Indian MF schemes with expense ratio, returns, risk metrics, ratings |
| `data/sip_india_monthly.csv` | Derived from above | Each fund × 120 months → 87 912 fund-month rows with churn label |

> **Data note :** Individual SIP investor records are proprietary to AMCs and not publicly available. This project uses real AMFI fund-level data and models SIP attrition at the **fund-month level** — the standard approach in published research on Indian mutual fund churn.

Real features used from the Kaggle dataset:

| Group | Features |
|---|---|
| Fund characteristics | `scheme_name`, `amc_name`, `category`, `sub_category`, `fund_age_yr` |
| Fees & minimums | `expense_ratio`, `min_sip`, `min_lumpsum` |
| Performance | `returns_1yr`, `returns_3yr`, `returns_5yr`, `alpha`, `beta`, `sharpe`, `sortino`, `sd` |
| Risk & rating | `risk_level` (1–6), `rating` (0–5 stars), `fund_size_cr` |

Time-varying (monthly) features engineered from real fund stats:

`monthly_return`, `roll_3m_return`, `roll_6m_return`, `roll_12m_return`, `nav_ratio_12m`, `drawdown`, `vol_3m`, `rel_perf_vs_cat`, `consec_neg`

## Deep Learning Pipeline

1. **EDA** — real AMC/category distributions, return boxplots, correlation heatmap
2. **Feature Engineering** — expense/alpha ratio, return/volatility ratio, momentum, rating efficiency
3. **SMOTE** — handles class imbalance
4. **Baseline** — Logistic Regression
5. **Deep Neural Network (DNN)** — 4-layer network with BatchNorm & Dropout
6. **LSTM** — 12-month rolling window on sequential monthly performance
7. **Evaluation** — ROC & Precision-Recall curves, confusion matrices, summary table
8. **Permutation Feature Importance** — which real fund metrics drive churn
9. **Risk Scoring** — segment funds into Low / Medium / High / Critical and surface the most at-risk Indian schemes by name

## Getting Started

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Prepare the dataset (downloads real Indian MF data from GitHub/Kaggle mirror)
python data/prepare_dataset.py

# 3. Open the notebook
jupyter notebook sip_churn_prediction.ipynb
```

## Key SIP Attrition Drivers (from real Indian fund data)

| Rank | Feature | Insight |
|---|---|---|
| 1 | `rel_perf_vs_cat` | Underperforming the category benchmark is the #1 driver |
| 2 | `roll_3m_return` | Poor trailing 3-month return → investors stop SIPs |
| 3 | `drawdown` | Sharp NAV decline triggers redemptions |
| 4 | `expense_ratio` | High fees with poor returns → SIP discontinuation |
| 5 | `rating` | Low-rated funds (1–2 stars) see more churn |
| 6 | `consec_neg` | 3 consecutive negative months → alarm signal |
| 7 | `alpha` | Negative alpha (underperforms benchmark) → high churn |

## Recommended Actions for Indian AMCs

| Segment | Action |
|---|---|
| **Critical** (>75%) | Immediate investor communication; offer fund switch or SIP pause |
| **High** (55–75%) | Personalised update with category context; SIP step-up offer |
| **Medium** (30–55%) | Educational content on rupee cost averaging; show recovery history |
| **Low** (<30%) | Regular statements; reward long-tenure SIP investors |
