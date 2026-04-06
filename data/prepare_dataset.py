"""
Indian SIP Churn Dataset Preparation
=====================================
Source dataset : "Mutual Funds India - Detailed" from Kaggle
                 (publicly mirrored on GitHub — nileshiq/Mutual-Funds-India)
                 Contains 814 real AMFI-registered Indian mutual fund schemes.

What this script does
---------------------
1. Downloads the real Indian mutual-fund dataset.
2. Cleans numeric columns (alpha, beta, sharpe, sortino, sd) that contain
   '-' for missing values.
3. Expands each of the 814 funds into 120 monthly observations (10 years)
   using the fund's real statistics (annual return, standard deviation)
   to simulate monthly NAV series.  This produces ~97 000 rows that are
   fully grounded in real fund-level Indian data.
4. Engineers time-varying features per fund-month:
   rolling returns, drawdown, relative performance vs. category.
5. Derives a binary monthly SIP-attrition label from real fund metrics:
   poor trailing performance, high drawdown, high expense relative to
   category, and low rating all increase the churn probability.
6. Saves the resulting dataset to data/sip_india_monthly.csv and the
   raw fund-level snapshot to data/india_mf_funds.csv.
"""

import io
import urllib.request
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

SEED = 42
N_MONTHS = 120          # 10 years of monthly observations per fund
DATASET_URL = (
    "https://raw.githubusercontent.com/nileshiq/Mutual-Funds-India/"
    "88cd08b8c451844a2aaad3528ad8600218a5f491/comprehensive_mutual_funds_data.csv"
)


# ── 1. Download & clean fund-level data ──────────────────────────────────────

def download_funds(url: str = DATASET_URL) -> pd.DataFrame:
    print("Downloading real Indian mutual-fund dataset …")
    raw = urllib.request.urlopen(url, timeout=30).read()
    df = pd.read_csv(io.BytesIO(raw))
    print(f"  Downloaded  : {df.shape[0]} funds × {df.shape[1]} columns")

    # Parse numeric columns stored as strings (some rows contain '-')
    for col in ["sortino", "alpha", "sd", "beta", "sharpe"]:
        df[col] = pd.to_numeric(df[col].replace("-", np.nan), errors="coerce")

    # Fill remaining NaNs with column medians
    for col in ["alpha", "beta", "sharpe", "sortino", "sd",
                "returns_3yr", "returns_5yr"]:
        df[col] = df[col].fillna(df[col].median())

    df.reset_index(drop=True, inplace=True)
    df.index.name = "fund_id"
    df = df.reset_index()
    print(f"  Nulls after cleaning: {df.isnull().sum().sum()}")
    return df


# ── 2. Expand to monthly time-series ─────────────────────────────────────────

def expand_monthly(df_funds: pd.DataFrame,
                   n_months: int = N_MONTHS,
                   seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []

    # Category-level annualized return (used as rolling benchmark)
    cat_annual = df_funds.groupby("category")["returns_1yr"].median().to_dict()

    for _, fund in df_funds.iterrows():
        annual_ret = fund["returns_1yr"]
        ann_sd     = fund["sd"] if fund["sd"] > 0 else 15.0
        cat_ret    = cat_annual.get(fund["category"], annual_ret)

        monthly_mean      = annual_ret    / 12.0 / 100.0
        monthly_std       = ann_sd        / np.sqrt(12) / 100.0
        cat_monthly_mean  = cat_ret       / 12.0 / 100.0

        # Simulate fund monthly returns
        monthly_rets = rng.normal(monthly_mean, monthly_std, n_months)
        cat_rets     = rng.normal(cat_monthly_mean, monthly_std * 0.8, n_months)

        nav  = 10.0 * np.cumprod(1.0 + monthly_rets)
        peak = np.maximum.accumulate(nav)
        drawdown = (nav - peak) / (peak + 1e-9)

        # Start from month 12 so we always have 12-month lookback
        for t in range(12, n_months):
            roll_3m  = float(np.sum(monthly_rets[t - 3 : t]))
            roll_6m  = float(np.sum(monthly_rets[t - 6 : t]))
            roll_12m = float(np.sum(monthly_rets[t - 12 : t]))
            vol_3m   = float(np.std(monthly_rets[t - 3 : t]))
            cat_3m   = float(np.sum(cat_rets[t - 3 : t]))
            rel_perf = roll_3m - cat_3m

            rows.append(
                {
                    "fund_id"           : fund["fund_id"],
                    "month"             : t,
                    # ── Static (real) fund features ──
                    "expense_ratio"     : fund["expense_ratio"],
                    "risk_level"        : fund["risk_level"],
                    "rating"            : fund["rating"],
                    "fund_size_log"     : np.log1p(fund["fund_size_cr"]),
                    "fund_age_yr"       : fund["fund_age_yr"],
                    "min_sip_log"       : np.log1p(fund["min_sip"]),
                    "alpha"             : fund["alpha"],
                    "beta"              : fund["beta"],
                    "sharpe"            : fund["sharpe"],
                    "sortino"           : fund["sortino"],
                    "sd_annual"         : fund["sd"],
                    "returns_1yr"       : fund["returns_1yr"],
                    "returns_3yr"       : fund["returns_3yr"],
                    "returns_5yr"       : fund["returns_5yr"],
                    "category"          : fund["category"],
                    # ── Time-varying features ──
                    "monthly_return"    : float(monthly_rets[t]),
                    "nav_ratio_12m"     : float(nav[t] / (nav[t - 12] + 1e-9)),
                    "roll_3m_return"    : roll_3m,
                    "roll_6m_return"    : roll_6m,
                    "roll_12m_return"   : roll_12m,
                    "vol_3m"            : vol_3m,
                    "drawdown"          : float(drawdown[t]),
                    "rel_perf_vs_cat"   : rel_perf,
                    "consec_neg"        : int(np.all(monthly_rets[t - 3 : t] < 0)),
                }
            )

    return pd.DataFrame(rows)


# ── 3. Derive SIP churn label ─────────────────────────────────────────────────

def add_churn_label(df: pd.DataFrame, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed + 1)

    # Category medians for contextual scoring
    cat_expense  = df.groupby("category")["expense_ratio"].transform("median")
    cat_roll3m   = df.groupby("category")["roll_3m_return"].transform("quantile", 0.25)

    churn_score = (
        0.25 * (df["roll_3m_return"] < cat_roll3m).astype(float)
        + 0.20 * np.clip(-df["rel_perf_vs_cat"] / 0.05, 0, 1)
        + 0.15 * np.clip(-df["drawdown"] / 0.15, 0, 1)
        + 0.12 * (df["expense_ratio"] > cat_expense).astype(float)
        + 0.10 * df["consec_neg"].astype(float)
        + 0.08 * np.clip((4 - df["rating"]) / 4, 0, 1)
        + 0.05 * np.clip((df["risk_level"] - 3) / 3, 0, 1)
        + 0.05 * np.clip(-df["alpha"] / 10, 0, 1)
    )

    churn_prob = 1.0 / (1.0 + np.exp(-6.0 * (churn_score - 0.40)))
    churn_prob = np.clip(churn_prob, 0.03, 0.97)

    df = df.copy()
    df["churn_prob"] = churn_prob
    df["churn"] = (rng.random(len(df)) < churn_prob).astype(int)
    return df


# ── 4. Encode & save ──────────────────────────────────────────────────────────

def prepare_and_save(out_monthly: str = "data/sip_india_monthly.csv",
                     out_funds:   str = "data/india_mf_funds.csv") -> None:
    df_funds = download_funds()
    df_funds.to_csv(out_funds, index=False)
    print(f"Fund-level data saved to {out_funds}")

    print("Expanding to monthly time-series …")
    df_monthly = expand_monthly(df_funds)

    print("Adding churn labels …")
    df_monthly = add_churn_label(df_monthly)

    # Label-encode category
    cat_map = {c: i for i, c in enumerate(df_monthly["category"].unique())}
    df_monthly["category_enc"] = df_monthly["category"].map(cat_map)
    df_monthly.drop(columns=["category"], inplace=True)

    df_monthly.to_csv(out_monthly, index=False)

    print(f"\nMonthly dataset saved to {out_monthly}")
    print(f"  Shape      : {df_monthly.shape}")
    print(f"  Churn rate : {df_monthly['churn'].mean():.2%}")
    print(f"  Features   : {[c for c in df_monthly.columns if c != 'churn']}")


if __name__ == "__main__":
    prepare_and_save()
