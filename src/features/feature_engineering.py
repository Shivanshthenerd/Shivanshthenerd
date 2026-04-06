"""
src/features/feature_engineering.py
-------------------------------------
Stage 3 – Feature engineering for the Indian SIP churn pipeline.

Takes the merged, cleaned panel produced by ``data_cleaning.py`` and
generates financially meaningful + behavioural features required for
churn modelling.  No deep learning is used here; the output is a flat
feature table ready for classical ML (Logistic Regression, Random
Forest, XGBoost, etc.).

Feature groups generated
-------------------------
1. **Tenure features**
   - ``tenure_months``        : how many months the fund-month is active
   - ``tenure_band``          : bucketed tenure (Early / Growing / Mature / Veteran)
   - ``is_early_stage``       : binary flag for tenure < 12 months

2. **Rolling return features** (already present; we enrich / rename)
   - ``roll_3m_return``       : trailing 3-month cumulative return
   - ``roll_6m_return``       : trailing 6-month cumulative return
   - ``roll_12m_return``      : trailing 12-month cumulative return
   - ``return_momentum``      : 3m return minus 6m return (acceleration)
   - ``return_reversal``      : 6m return minus 12m return

3. **Volatility features**
   - ``volatility_3m``        : rolling 3-month std-dev of monthly returns
   - ``volatility_ratio``     : 3m volatility normalised by annual SD
   - ``sharpe_3m``            : approximate 3-month Sharpe (return / vol)

4. **SIP consistency features**
   - ``missed_payment_ratio`` : fraction of months with negative/zero returns
                               used as a proxy for SIP pause / missed payment
   - ``consec_neg_flag``      : 1 if 3 consecutive negative months
   - ``payment_regularity``   : 1 − missed_payment_ratio (higher = better)

5. **Average investment / cost features**
   - ``avg_sip_amount``       : log-scaled minimum SIP (proxy for ticket size)
   - ``relative_expense``     : expense ratio relative to category median
   - ``cost_drag``            : expense_ratio − alpha (net cost vs. skill)

6. **Market trend indicators**
   - ``drawdown_severity``    : drawdown bucketed into mild / moderate / severe
   - ``rel_perf_vs_cat``      : return relative to category (already computed)
   - ``alpha_positive``       : binary — is alpha > 0?
   - ``above_benchmark``      : binary — relative performance > 0?

7. **Fund quality features**
   - ``rating_band``          : star rating bucketed (Low / Mid / High)
   - ``risk_adj_return``      : returns_1yr / (sd_annual + 1e-9)
   - ``size_band``            : fund size bucketed (Small / Mid / Large)

Target column
-------------
``churn`` (1 = investor discontinued SIP; 0 = active)  — already present
in the merged panel from the ingestion stage.

Output
------
``data/features/features.csv`` — final flat feature table.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.io_helpers import save_csv
from src.utils.logger import get_logger

log = get_logger(__name__)

FEATURES_DIR = Path("data/features")
FEATURES_FILE = FEATURES_DIR / "features.csv"

# Categorical median cache (filled at runtime)
_CATEGORY_MEDIANS: dict[str, float] = {}


# ── Helper utilities ──────────────────────────────────────────────────────────

def _safe_divide(num: pd.Series, denom: pd.Series, fill: float = 0.0) -> pd.Series:
    """Element-wise division that replaces zero-denominator results with *fill*."""
    result = num / denom.replace(0, np.nan)
    return result.fillna(fill)


# ── Feature generators ────────────────────────────────────────────────────────

def add_tenure_features(df: pd.DataFrame) -> pd.DataFrame:
    """Generate SIP tenure features from the ``month`` column.

    ``month`` represents the observation index within a fund's 120-month
    life-cycle (month 12 = start of observation window after warm-up).

    Parameters
    ----------
    df : pd.DataFrame
        Merged monthly panel.

    Returns
    -------
    pd.DataFrame
        Panel with tenure columns added.
    """
    df = df.copy()

    # tenure_months: number of months the fund has been active up to this observation
    df["tenure_months"] = df["month"].astype(int)

    # Bucketed tenure band for tree-based models
    df["tenure_band"] = pd.cut(
        df["tenure_months"],
        bins=[0, 24, 48, 84, 121],
        labels=["Early", "Growing", "Mature", "Veteran"],
        right=True,
    ).astype(str)

    # Binary early-stage flag — highest churn risk window
    df["is_early_stage"] = (df["tenure_months"] < 24).astype(int)

    log.info("Tenure features added: tenure_months, tenure_band, is_early_stage")
    return df


def add_return_features(df: pd.DataFrame) -> pd.DataFrame:
    """Enrich rolling return features with momentum and reversal signals.

    Parameters
    ----------
    df : pd.DataFrame
        Panel with ``roll_3m_return``, ``roll_6m_return``,
        ``roll_12m_return`` already present.

    Returns
    -------
    pd.DataFrame
        Panel with new momentum / reversal columns.
    """
    df = df.copy()

    # Momentum: recent 3-month acceleration vs. 6-month trend
    df["return_momentum"] = df["roll_3m_return"] - df["roll_6m_return"]

    # Reversal signal: medium-term vs. long-term
    df["return_reversal"] = df["roll_6m_return"] - df["roll_12m_return"]

    # Cumulative performance relative to a flat 8% annualised benchmark
    # (approximate long-run NIFTY 50 real return)
    monthly_bench = 0.08 / 12
    df["excess_return_3m"] = df["roll_3m_return"] - 3 * monthly_bench
    df["excess_return_6m"] = df["roll_6m_return"] - 6 * monthly_bench

    log.info(
        "Return features added: return_momentum, return_reversal, "
        "excess_return_3m, excess_return_6m"
    )
    return df


def add_volatility_features(df: pd.DataFrame) -> pd.DataFrame:
    """Generate volatility and risk-adjusted return features.

    Parameters
    ----------
    df : pd.DataFrame
        Panel with ``vol_3m`` and ``sd_annual`` present.

    Returns
    -------
    pd.DataFrame
        Panel with volatility-derived columns.
    """
    df = df.copy()

    # Rename vol_3m to volatility_3m for clarity
    df["volatility_3m"] = df["vol_3m"]

    # Volatility ratio: how does recent vol compare to long-run annual SD?
    # Convert annual SD to monthly equivalent for a fair comparison.
    df["volatility_ratio"] = _safe_divide(
        df["vol_3m"],
        df["sd_annual"] / np.sqrt(12),
        fill=1.0,
    )

    # Approximate 3-month Sharpe using the same 8% pa risk-free benchmark
    monthly_rf = 0.065 / 12   # ~6.5% Indian repo rate equivalent
    df["sharpe_3m"] = _safe_divide(
        df["roll_3m_return"] - 3 * monthly_rf,
        df["vol_3m"],
        fill=0.0,
    )

    log.info("Volatility features added: volatility_3m, volatility_ratio, sharpe_3m")
    return df


def add_sip_consistency_features(df: pd.DataFrame) -> pd.DataFrame:
    """Generate SIP payment consistency / behavioural proxy features.

    Because individual transaction records are not available publicly,
    a negative monthly return is used as a behavioural proxy for a
    missed / paused SIP instalment (investors tend to pause SIPs during
    market downturns).

    Parameters
    ----------
    df : pd.DataFrame
        Panel with ``monthly_return`` and ``consec_neg`` present.

    Returns
    -------
    pd.DataFrame
        Panel with SIP consistency features.
    """
    df = df.copy()

    # Fraction of the last 6 months with negative monthly returns
    # (a simple window computed per fund using expanding logic over sorted data)
    df = df.sort_values(["fund_id", "month"])

    # Missed payment ratio: rolling 6-month proportion of negative months
    neg_flag = (df["monthly_return"] < 0).astype(float)
    df["missed_payment_ratio"] = (
        neg_flag.groupby(df["fund_id"])
                .transform(lambda s: s.rolling(6, min_periods=1).mean())
    )

    # Payment regularity: complement of missed ratio (higher = more consistent)
    df["payment_regularity"] = 1.0 - df["missed_payment_ratio"]

    # Carry the existing consecutive-negative flag forward
    df["consec_neg_flag"] = df["consec_neg"].astype(int)

    log.info(
        "SIP consistency features added: missed_payment_ratio, "
        "payment_regularity, consec_neg_flag"
    )
    return df


def add_investment_cost_features(df: pd.DataFrame) -> pd.DataFrame:
    """Generate average investment and cost-related features.

    Parameters
    ----------
    df : pd.DataFrame
        Panel with ``min_sip_log``, ``expense_ratio``, ``alpha``,
        and ``category`` (or ``category_enc``) present.

    Returns
    -------
    pd.DataFrame
        Panel with investment and cost features.
    """
    df = df.copy()

    # Average investment proxy (log-scaled minimum SIP amount)
    df["avg_sip_amount"] = df["min_sip_log"]   # already log-transformed

    # Category-median expense ratio for relative cost comparison
    if "category" in df.columns:
        cat_col = "category"
    elif "category_enc" in df.columns:
        cat_col = "category_enc"
    else:
        cat_col = None

    if cat_col:
        cat_med_expense = df.groupby(cat_col)["expense_ratio"].transform("median")
        df["relative_expense"] = df["expense_ratio"] - cat_med_expense
    else:
        df["relative_expense"] = 0.0

    # Cost drag: how much of alpha is eaten by the expense ratio?
    # Positive cost_drag means the fund charges more than it earns in alpha.
    df["cost_drag"] = df["expense_ratio"] - df["alpha"]

    log.info(
        "Investment/cost features added: avg_sip_amount, relative_expense, cost_drag"
    )
    return df


def add_market_trend_features(df: pd.DataFrame) -> pd.DataFrame:
    """Generate market trend and relative performance indicators.

    Parameters
    ----------
    df : pd.DataFrame
        Panel with ``drawdown``, ``rel_perf_vs_cat``, ``alpha`` present.

    Returns
    -------
    pd.DataFrame
        Panel with market trend features.
    """
    df = df.copy()

    # Drawdown severity bucket
    df["drawdown_severity"] = pd.cut(
        df["drawdown"],
        bins=[-np.inf, -0.20, -0.10, -0.05, 0.0, np.inf],
        labels=["Severe", "Moderate", "Mild", "Flat", "Positive"],
        right=True,
    ).astype(str)

    # Binary: is relative performance vs. category positive?
    df["above_benchmark"] = (df["rel_perf_vs_cat"] > 0).astype(int)

    # Binary: is alpha positive (fund manager adding value)?
    df["alpha_positive"] = (df["alpha"] > 0).astype(int)

    log.info(
        "Market trend features added: drawdown_severity, above_benchmark, alpha_positive"
    )
    return df


def add_fund_quality_features(df: pd.DataFrame) -> pd.DataFrame:
    """Generate fund quality / metadata-derived features.

    Parameters
    ----------
    df : pd.DataFrame
        Panel with ``rating``, ``sd_annual``, ``returns_1yr``,
        ``fund_size_log`` present.

    Returns
    -------
    pd.DataFrame
        Panel with fund quality features.
    """
    df = df.copy()

    # Star rating band
    df["rating_band"] = pd.cut(
        df["rating"],
        bins=[-1, 2, 3, 5],
        labels=["Low", "Mid", "High"],
        right=True,
    ).astype(str)

    # Risk-adjusted 1-year return (simple Calmar-like ratio)
    df["risk_adj_return"] = _safe_divide(
        pd.Series(df["returns_1yr"], index=df.index),
        pd.Series(df["sd_annual"] + 1e-9, index=df.index),
        fill=0.0,
    )

    # Fund size band
    df["size_band"] = pd.cut(
        df["fund_size_log"],
        bins=[-np.inf, 4.0, 7.0, np.inf],
        labels=["Small", "Mid", "Large"],
        right=True,
    ).astype(str)

    log.info("Fund quality features added: rating_band, risk_adj_return, size_band")
    return df


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Label-encode string categorical columns added during feature engineering.

    Converts ``tenure_band``, ``drawdown_severity``, ``rating_band``,
    ``size_band`` to integer codes so the final feature table is fully
    numeric (required by scikit-learn estimators).

    Parameters
    ----------
    df : pd.DataFrame
        Panel after all feature generators have run.

    Returns
    -------
    pd.DataFrame
        Panel with categorical columns encoded as integers.
    """
    df = df.copy()
    cat_cols = ["tenure_band", "drawdown_severity", "rating_band", "size_band"]
    for col in cat_cols:
        if col in df.columns:
            df[col] = pd.Categorical(df[col]).codes
    log.info("Categorical columns encoded: %s", cat_cols)
    return df


def select_features(df: pd.DataFrame) -> pd.DataFrame:
    """Select and order the final feature columns for modelling.

    Non-feature columns (identifiers, raw intermediates, probability
    scores used only for label derivation) are dropped.

    Parameters
    ----------
    df : pd.DataFrame
        Fully enriched panel.

    Returns
    -------
    pd.DataFrame
        Final features DataFrame with ``churn`` as the last column.
    """
    # Columns to exclude from the feature table
    drop_cols = {
        "churn_prob",       # label-derivation helper — not a model input
        "scheme_name",      # high-cardinality string identifier
        "fund_manager",     # high-cardinality string
        "amc_name",         # high-cardinality string (encode separately if needed)
        "sub_category",     # redundant with category
        "vol_3m",           # renamed to volatility_3m
        "consec_neg",       # renamed to consec_neg_flag
    }

    # Keep only columns that are present AND not in the drop set
    keep = [c for c in df.columns if c not in drop_cols]

    # Ensure churn is the last column
    keep = [c for c in keep if c != "churn"] + ["churn"]

    df_out = df[keep].copy()
    log.info("Final feature table: %d rows × %d columns", *df_out.shape)
    log.info("Target distribution:  churn=1: %d  churn=0: %d",
             (df_out["churn"] == 1).sum(), (df_out["churn"] == 0).sum())
    return df_out


# ── Orchestration ─────────────────────────────────────────────────────────────

def run_feature_engineering(
    df_merged: pd.DataFrame,
    out_path: str | Path = FEATURES_FILE,
) -> pd.DataFrame:
    """Run the complete feature engineering pipeline.

    Applies all feature generators in sequence, encodes categoricals,
    selects the final feature set, and saves ``features.csv``.

    Parameters
    ----------
    df_merged : pd.DataFrame
        Cleaned, merged panel from ``data_cleaning.run_cleaning()``.
    out_path : str or Path
        Destination for the feature CSV.

    Returns
    -------
    pd.DataFrame
        Final feature table (also persisted to *out_path*).
    """
    log.info("=== Starting feature engineering ===")

    df = add_tenure_features(df_merged)
    df = add_return_features(df)
    df = add_volatility_features(df)
    df = add_sip_consistency_features(df)
    df = add_investment_cost_features(df)
    df = add_market_trend_features(df)
    df = add_fund_quality_features(df)
    df = encode_categoricals(df)
    df = select_features(df)

    save_csv(df, out_path)
    log.info("=== Feature engineering complete → %s ===", out_path)
    return df
