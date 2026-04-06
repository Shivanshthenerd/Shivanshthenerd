👋 Hi, I'm Shivansh and I'm currently exploring the world of computers.

---

# 📈 SIP Churn Rate Prediction — Deep Learning Project

A deep learning project that predicts whether a customer will **discontinue their Systematic Investment Plan (SIP)** before maturity, enabling fund houses and financial advisors to intervene proactively.

## Project Structure

```
├── data/
│   ├── generate_dataset.py      # Synthetic SIP customer dataset generator
│   └── sip_customers.csv        # Generated dataset (10,000 customers)
├── models/                      # Saved model artefacts (generated after training)
├── sip_churn_prediction.ipynb   # Main Jupyter notebook (full ML pipeline)
└── requirements.txt
```

## Dataset

A synthetic dataset of **10,000 SIP customers** (~31% churn rate) with 32 features covering:

| Category | Features |
|---|---|
| Demographics | Age, income, occupation, city tier, education, marital status |
| SIP Details | Amount, tenure, frequency, fund category, number of SIPs |
| Payment History | Missed/delayed payments, consecutive misses, failure rate |
| Market Context | Portfolio return, benchmark return, NAV volatility |
| Customer Engagement | App logins, support calls, portfolio reviews, last login |
| Financial Health | Credit score, existing loans, EMI burden, savings rate |

## Deep Learning Pipeline

1. **Exploratory Data Analysis** — distributions, churn rates by category, correlation matrix
2. **Feature Engineering** — derived ratios (SIP-to-income, missed payment ratio, engagement score)
3. **SMOTE** — handles class imbalance
4. **Baseline** — Logistic Regression
5. **Deep Neural Network (DNN)** — 4-layer fully connected network with BatchNorm & Dropout
6. **LSTM** — models 12-month sequential payment history
7. **Model Comparison** — ROC, Precision-Recall curves, confusion matrices
8. **Feature Importance** — permutation importance on the DNN
9. **Churn Scoring** — ensemble probability + risk segmentation (Low / Medium / High / Critical)

## Getting Started

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. (Optional) Regenerate the dataset
python data/generate_dataset.py

# 3. Open and run the notebook
jupyter notebook sip_churn_prediction.ipynb
```

## Key Churn Drivers

| Rank | Feature | Insight |
|---|---|---|
| 1 | `missed_payment_ratio` | Customers missing payments are most at risk |
| 2 | `consecutive_missed` | Streaks of missed payments are alarming |
| 3 | `last_login_days_ago` | Disengaged customers leave sooner |
| 4 | `payment_failure_rate` | Raw failure rate strongly predicts churn |
| 5 | `relative_return` | Underperforming the benchmark drives exits |

## Recommended Interventions

| Risk Segment | Action |
|---|---|
| **Critical** (>75%) | Immediate advisor call; offer SIP pause / amount reduction |
| **High** (55–75%) | Personalised outreach; highlight portfolio gains |
| **Medium** (30–55%) | Educational nudges; goal-based SIP reminders |
| **Low** (<30%) | Regular engagement emails; loyalty rewards |
