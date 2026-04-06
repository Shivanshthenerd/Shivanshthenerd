"""
Synthetic SIP (Systematic Investment Plan) Customer Dataset Generator
----------------------------------------------------------------------
Generates a realistic dataset that simulates SIP customer behaviour
and churn patterns for use in the SIP Churn Rate prediction project.
"""

import numpy as np
import pandas as pd

SEED = 42
N_CUSTOMERS = 10_000


def generate_dataset(n_customers: int = N_CUSTOMERS, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    # ── Demographics ─────────────────────────────────────────────────────────
    age = rng.integers(22, 65, n_customers)
    income_lpa = np.round(
        np.clip(rng.normal(8, 4, n_customers), 2.5, 60), 2
    )  # Lakhs per annum
    occupation = rng.choice(
        ["Salaried", "Self-Employed", "Business", "Retired", "Student"],
        n_customers,
        p=[0.50, 0.20, 0.18, 0.08, 0.04],
    )
    city_tier = rng.choice([1, 2, 3], n_customers, p=[0.30, 0.40, 0.30])
    education = rng.choice(
        ["Graduate", "Post-Graduate", "Professional", "Others"],
        n_customers,
        p=[0.45, 0.25, 0.20, 0.10],
    )
    marital_status = rng.choice(
        ["Married", "Single", "Divorced"], n_customers, p=[0.60, 0.35, 0.05]
    )
    dependents = rng.integers(0, 5, n_customers)

    # ── SIP Details ───────────────────────────────────────────────────────────
    sip_amount = rng.choice(
        [500, 1000, 2000, 3000, 5000, 7500, 10000, 15000, 20000, 25000, 50000],
        n_customers,
        p=[0.05, 0.15, 0.20, 0.15, 0.18, 0.10, 0.08, 0.04, 0.02, 0.02, 0.01],
    )
    sip_tenure_months = rng.integers(3, 121, n_customers)  # 3 months – 10 years
    sip_frequency = rng.choice(
        ["Monthly", "Quarterly", "Annual"],
        n_customers,
        p=[0.80, 0.15, 0.05],
    )
    fund_category = rng.choice(
        ["Equity", "Debt", "Hybrid", "ELSS", "Index"],
        n_customers,
        p=[0.45, 0.20, 0.18, 0.10, 0.07],
    )
    number_of_sips = rng.integers(1, 6, n_customers)

    # ── Payment History ───────────────────────────────────────────────────────
    missed_payments = rng.integers(0, min(sip_tenure_months.min(), 12), n_customers)
    delayed_payments = rng.integers(0, min(sip_tenure_months.min(), 15), n_customers)
    consecutive_missed = rng.integers(0, 4, n_customers)
    payment_failure_rate = np.round(
        np.clip(rng.beta(1.2, 8, n_customers), 0, 1), 3
    )

    # ── Market & Investment Context ───────────────────────────────────────────
    portfolio_return_pct = np.round(rng.normal(10, 8, n_customers), 2)  # %
    benchmark_return_pct = np.round(rng.normal(11, 5, n_customers), 2)
    relative_return = np.round(portfolio_return_pct - benchmark_return_pct, 2)
    nav_volatility = np.round(np.abs(rng.normal(15, 5, n_customers)), 2)
    market_downturn_flag = (rng.random(n_customers) < 0.20).astype(int)

    # ── Customer Engagement ───────────────────────────────────────────────────
    app_logins_last_3m = rng.integers(0, 31, n_customers)
    support_calls_last_6m = rng.integers(0, 10, n_customers)
    portfolio_reviews_last_yr = rng.integers(0, 13, n_customers)
    grievances_raised = rng.integers(0, 4, n_customers)
    email_open_rate = np.round(rng.beta(2, 3, n_customers), 3)
    last_login_days_ago = rng.integers(0, 181, n_customers)

    # ── Financial Health Indicators ───────────────────────────────────────────
    credit_score = rng.integers(550, 851, n_customers)
    existing_loans = rng.integers(0, 5, n_customers)
    loan_emi_pct_income = np.round(
        np.clip(rng.beta(2, 5, n_customers) * 60, 0, 60), 2
    )
    savings_rate_pct = np.round(
        np.clip(rng.beta(3, 4, n_customers) * 50, 0, 50), 2
    )

    # ── Churn Label (target) ──────────────────────────────────────────────────
    # Build a probability score from realistic churn drivers
    churn_score = (
        0.20 * (missed_payments / 12)
        + 0.15 * (consecutive_missed / 3)
        + 0.10 * payment_failure_rate
        + 0.10 * (last_login_days_ago / 180)
        + 0.08 * (support_calls_last_6m / 10)
        + 0.08 * (grievances_raised / 4)
        + 0.07 * (loan_emi_pct_income / 60)
        + 0.06 * (market_downturn_flag)
        + 0.06 * np.clip(-relative_return / 20, 0, 1)
        + 0.05 * (nav_volatility / 30)
        + 0.05 * ((65 - age) / 43)  # older investors churn less
        - 0.10 * (portfolio_reviews_last_yr / 12)
        - 0.08 * (app_logins_last_3m / 30)
        - 0.05 * (savings_rate_pct / 50)
    )
    churn_prob = 1 / (1 + np.exp(-6 * (churn_score - 0.35)))  # sigmoid squash
    churn_prob = np.clip(churn_prob, 0.03, 0.97)
    churn = (rng.random(n_customers) < churn_prob).astype(int)

    df = pd.DataFrame(
        {
            "customer_id": [f"CUST{i+1:05d}" for i in range(n_customers)],
            # Demographics
            "age": age,
            "income_lpa": income_lpa,
            "occupation": occupation,
            "city_tier": city_tier,
            "education": education,
            "marital_status": marital_status,
            "dependents": dependents,
            # SIP Details
            "sip_amount": sip_amount,
            "sip_tenure_months": sip_tenure_months,
            "sip_frequency": sip_frequency,
            "fund_category": fund_category,
            "number_of_sips": number_of_sips,
            # Payment History
            "missed_payments": missed_payments,
            "delayed_payments": delayed_payments,
            "consecutive_missed": consecutive_missed,
            "payment_failure_rate": payment_failure_rate,
            # Market Context
            "portfolio_return_pct": portfolio_return_pct,
            "benchmark_return_pct": benchmark_return_pct,
            "relative_return": relative_return,
            "nav_volatility": nav_volatility,
            "market_downturn_flag": market_downturn_flag,
            # Engagement
            "app_logins_last_3m": app_logins_last_3m,
            "support_calls_last_6m": support_calls_last_6m,
            "portfolio_reviews_last_yr": portfolio_reviews_last_yr,
            "grievances_raised": grievances_raised,
            "email_open_rate": email_open_rate,
            "last_login_days_ago": last_login_days_ago,
            # Financial Health
            "credit_score": credit_score,
            "existing_loans": existing_loans,
            "loan_emi_pct_income": loan_emi_pct_income,
            "savings_rate_pct": savings_rate_pct,
            # Target
            "churn": churn,
        }
    )
    return df


if __name__ == "__main__":
    df = generate_dataset()
    out = "data/sip_customers.csv"
    df.to_csv(out, index=False)
    print(f"Dataset saved to {out}")
    print(f"Shape : {df.shape}")
    print(f"Churn rate: {df['churn'].mean():.2%}")
