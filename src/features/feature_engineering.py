"""
src/features/feature_engineering.py
-------------------------------------
Stage 3 – Feature engineering for the Indian SIP churn pipeline.

Takes the merged, cleaned panel produced by ``data_cleaning.py`` (AMFI
pipeline) or ``investor_cleaning.py`` (investor pipeline) and generates
financially meaningful + behavioural features required for churn modelling.
No deep learning is used here; the output is a flat feature table ready
for classical ML (Logistic Regression, Random Forest, XGBoost, etc.).

AMFI pipeline feature groups (fund × month panel)
---------------------------------------------------
1. Tenure features
2. Rolling return features (momentum, reversal, excess return)
3. Volatility features
4. SIP consistency features
5. Average investment / cost features
6. Market trend indicators
7. Fund quality features

Investor pipeline feature groups (investor × month panel)
----------------------------------------------------------
8.  CAGR features          — annualised compounded return on investor portfolio
9.  Max drawdown features  — rolling peak-to-trough decline
10. Risk-adjusted return   — Sharpe ratio computed from actual investor returns
11. NIFTY context features — market regime, NIFTY momentum vs. fund momentum
12. Investor behaviour     — missed payment streaks, payment regularity

Target column
-------------
``churn`` (1 = investor discontinued SIP; 0 = active)

Output
------
``data/features/features.csv`` — final flat feature table.

Running this script
-------------------
Can be invoked directly from any working directory::

    python src/features/feature_engineering.py
"""

import sys
from pathlib import Path

# Allow running this file directly: python src/features/feature_engineering.py
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import pandas as pd

from src.utils.io_helpers import save_csv
from src.utils.logger import get_logger

log = get_logger(__name__)

# Absolute paths — work from any working directory
FEATURES_DIR  = _PROJECT_ROOT / "data" / "features"
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


# ══════════════════════════════════════════════════════════════════════════════
# INVESTOR PIPELINE — additional feature generators
# ══════════════════════════════════════════════════════════════════════════════

def add_cagr_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute CAGR of the investor's SIP portfolio at each observation month.

    CAGR measures the annualised compounded growth rate of the portfolio::

        CAGR = (portfolio_value / total_invested) ^ (12 / months_active) - 1

    A CAGR of 0 is returned when ``total_invested`` ≤ 0 or
    ``months_active`` < 1 (avoids division-by-zero on the first month).

    Three additional derived signals are computed:
    - ``cagr_vs_nifty``   : CAGR minus estimated NIFTY CAGR (alpha proxy)
    - ``cagr_positive``   : binary — is the investor in profit?
    - ``cagr_band``       : bucketed (<0 %, 0–8 %, 8–15 %, >15 %)

    Parameters
    ----------
    df : pd.DataFrame
        Investor × month panel with ``portfolio_value``, ``total_invested``,
        ``total_months_active``, ``nifty_12m_return`` present.

    Returns
    -------
    pd.DataFrame
        Panel with CAGR feature columns added.
    """
    df = df.copy()

    # Raw CAGR
    ratio = _safe_divide(
        df["portfolio_value"],
        df["total_invested"].replace(0, np.nan),
        fill=1.0,
    )
    n = df["total_months_active"].clip(lower=1)
    df["cagr"] = ratio.pow(12.0 / n) - 1.0

    # Approximate NIFTY CAGR using rolling 12-month return as a proxy
    # (annualised: (1 + 12m_return)^1 - 1 is already annualised for 12 months)
    df["nifty_cagr_approx"] = df["nifty_12m_return"].fillna(0.0)

    # Investor CAGR relative to NIFTY
    df["cagr_vs_nifty"] = df["cagr"] - df["nifty_cagr_approx"]

    # Binary: investor in profit
    df["cagr_positive"] = (df["cagr"] > 0).astype(int)

    # Bucketed CAGR band (useful for tree splits)
    df["cagr_band"] = pd.cut(
        df["cagr"],
        bins=[-np.inf, 0.0, 0.08, 0.15, np.inf],
        labels=["Negative", "Low", "Moderate", "High"],
        right=True,
    ).astype(str)

    log.info("CAGR features added: cagr, cagr_vs_nifty, cagr_positive, cagr_band")
    return df


def add_max_drawdown_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute rolling max drawdown of the investor's portfolio.

    Max drawdown measures the largest peak-to-trough decline in portfolio
    value experienced by the investor up to each observation month::

        max_drawdown_t = min over s ≤ t of  (value_s - peak_s) / peak_s

    Two supporting features are also generated:
    - ``in_drawdown``      : 1 if current portfolio is below its historical peak
    - ``drawdown_recovery``: 1 if last month was the trough (recovering now)

    Parameters
    ----------
    df : pd.DataFrame
        Investor × month panel with ``portfolio_value``, ``investor_id``,
        ``date`` present.

    Returns
    -------
    pd.DataFrame
        Panel with drawdown feature columns added.
    """
    df = df.copy().sort_values(["investor_id", "date"])

    # Per-investor rolling peak portfolio value
    df["portfolio_peak"] = (
        df.groupby("investor_id")["portfolio_value"]
        .transform("cummax")
    )

    # Drawdown at each month: (value - peak) / peak  (always ≤ 0)
    df["portfolio_drawdown"] = _safe_divide(
        df["portfolio_value"] - df["portfolio_peak"],
        df["portfolio_peak"],
        fill=0.0,
    )

    # Max drawdown experienced so far (running minimum of drawdown column)
    df["max_drawdown"] = (
        df.groupby("investor_id")["portfolio_drawdown"]
        .transform("cummin")
    )

    # Binary: currently below peak
    df["in_drawdown"] = (df["portfolio_drawdown"] < 0).astype(int)

    # Binary: recovering (prev month was worse drawdown than this month)
    prev_dd = df.groupby("investor_id")["portfolio_drawdown"].shift(1)
    df["drawdown_recovery"] = (
        (df["portfolio_drawdown"] > prev_dd) & (df["in_drawdown"] == 1)
    ).astype(int)

    log.info(
        "Max drawdown features added: portfolio_drawdown, max_drawdown, "
        "in_drawdown, drawdown_recovery"
    )
    return df


def add_risk_adjusted_return_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute rolling risk-adjusted return metrics for each investor.

    Three metrics are computed over a trailing 12-month window:

    - **Sharpe ratio (12m)**::

          sharpe_12m = (mean_monthly_return - rf_monthly) / std_monthly_return × √12

      where ``rf_monthly ≈ 6.5 % / 12`` (approximate Indian repo rate).

    - **Sortino ratio (12m)** — like Sharpe but only penalises downside
      volatility (standard deviation of negative-return months).

    - **Calmar ratio** — CAGR / |max_drawdown| (return per unit of drawdown
      risk).

    Parameters
    ----------
    df : pd.DataFrame
        Investor × month panel with ``fund_monthly_return``,
        ``cagr``, ``max_drawdown`` present.

    Returns
    -------
    pd.DataFrame
        Panel with risk-adjusted return columns added.
    """
    df = df.copy().sort_values(["investor_id", "date"])
    rf_monthly = 0.065 / 12   # ~6.5% Indian risk-free rate / 12

    # Rolling 12-month Sharpe per investor
    def rolling_sharpe(s: pd.Series) -> pd.Series:
        excess = s - rf_monthly
        mean_e = excess.rolling(12, min_periods=3).mean()
        std_r  = s.rolling(12, min_periods=3).std()
        return (mean_e / std_r.replace(0, np.nan)).fillna(0.0) * np.sqrt(12)

    df["sharpe_12m"] = (
        df.groupby("investor_id")["fund_monthly_return"]
        .transform(rolling_sharpe)
    )

    # Rolling 12-month Sortino per investor (downside std only)
    def rolling_sortino(s: pd.Series) -> pd.Series:
        excess     = s - rf_monthly
        mean_e     = excess.rolling(12, min_periods=3).mean()
        # Downside deviation: std of returns below rf only
        def downside_std(window):
            neg = window[window < rf_monthly]
            return neg.std() if len(neg) >= 2 else np.nan
        down_std = s.rolling(12, min_periods=3).apply(downside_std, raw=False)
        return (mean_e / down_std.replace(0, np.nan)).fillna(0.0) * np.sqrt(12)

    df["sortino_12m"] = (
        df.groupby("investor_id")["fund_monthly_return"]
        .transform(rolling_sortino)
    )

    # Calmar ratio: CAGR / |max_drawdown|
    # Clamp max_drawdown away from 0 to avoid division by zero
    df["calmar_ratio"] = _safe_divide(
        df["cagr"],
        df["max_drawdown"].abs().replace(0, np.nan),
        fill=0.0,
    )

    log.info(
        "Risk-adjusted return features added: sharpe_12m, sortino_12m, calmar_ratio"
    )
    return df


def add_nifty_context_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add NIFTY-based market context features to the investor panel.

    Features generated:
    - ``market_regime``       : Bull / Flat / Bear based on NIFTY 3m return
    - ``fund_vs_nifty_3m``    : fund 3m return minus NIFTY 3m return
    - ``fund_vs_nifty_12m``   : fund 12m return minus NIFTY 12m return
    - ``nifty_drawdown_flag`` : 1 if NIFTY is in drawdown (< −5 %)

    Parameters
    ----------
    df : pd.DataFrame
        Investor × month panel with fund and NIFTY return columns.

    Returns
    -------
    pd.DataFrame
        Panel with NIFTY context features added.
    """
    df = df.copy()

    # Market regime based on NIFTY rolling 3m return
    df["market_regime"] = pd.cut(
        df["nifty_3m_return"].fillna(0.0),
        bins=[-np.inf, -0.05, 0.03, np.inf],
        labels=["Bear", "Flat", "Bull"],
        right=True,
    ).astype(str)

    # Fund alpha vs. NIFTY over 3m and 12m horizons
    # Approximate fund rolling returns from monthly return column
    roll_3m = (
        df.sort_values(["investor_id", "date"])
        .groupby("investor_id")["fund_monthly_return"]
        .transform(lambda s: s.rolling(3, min_periods=1).sum())
    )
    df["fund_vs_nifty_3m"] = roll_3m - df["nifty_3m_return"].fillna(0.0)

    roll_12m = (
        df.sort_values(["investor_id", "date"])
        .groupby("investor_id")["fund_monthly_return"]
        .transform(lambda s: s.rolling(12, min_periods=1).sum())
    )
    df["fund_vs_nifty_12m"] = roll_12m - df["nifty_12m_return"].fillna(0.0)

    # Flag deep NIFTY drawdown (often triggers panic selling / SIP pauses)
    df["nifty_drawdown_flag"] = (df["nifty_drawdown"] < -0.05).astype(int)

    log.info(
        "NIFTY context features added: market_regime, fund_vs_nifty_3m, "
        "fund_vs_nifty_12m, nifty_drawdown_flag"
    )
    return df


def add_investor_behaviour_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add investor behavioural features from payment history.

    Features generated:
    - ``missed_payment_ratio_12m``: fraction of last 12 months with missed payments
    - ``payment_streak``          : consecutive months without a missed payment
    - ``consec_missed``           : consecutive months with a missed payment
    - ``avg_monthly_investment``  : log(monthly_investment) — ticket-size proxy
    - ``tenure_band``             : bucketed tenure (Early/Growing/Mature/Veteran)
    - ``is_early_stage``          : binary flag for < 24 months tenure

    Parameters
    ----------
    df : pd.DataFrame
        Investor × month panel with ``missed_this_month``,
        ``monthly_investment``, ``total_months_active`` present.

    Returns
    -------
    pd.DataFrame
        Panel with investor behaviour features added.
    """
    df = df.copy().sort_values(["investor_id", "date"])

    # Rolling 12-month missed-payment ratio
    df["missed_payment_ratio_12m"] = (
        df.groupby("investor_id")["missed_this_month"]
        .transform(lambda s: s.rolling(12, min_periods=1).mean())
    )

    # Consecutive months without a missed payment (payment streak)
    def _consecutive_zeros(s: pd.Series) -> pd.Series:
        """Count consecutive 0s ending at each position."""
        result = np.zeros(len(s), dtype=float)
        streak = 0
        for i, v in enumerate(s):
            streak = 0 if v else streak + 1
            result[i] = streak
        return pd.Series(result, index=s.index)

    df["payment_streak"] = (
        df.groupby("investor_id")["missed_this_month"]
        .transform(_consecutive_zeros)
    )

    # Consecutive months WITH a missed payment (stress signal)
    def _consecutive_ones(s: pd.Series) -> pd.Series:
        result = np.zeros(len(s), dtype=float)
        streak = 0
        for i, v in enumerate(s):
            streak = streak + 1 if v else 0
            result[i] = streak
        return pd.Series(result, index=s.index)

    df["consec_missed"] = (
        df.groupby("investor_id")["missed_this_month"]
        .transform(_consecutive_ones)
    )

    # Log ticket size (log-transform reduces skew from wide SIP amount range)
    df["avg_monthly_investment"] = np.log1p(df["monthly_investment"])

    # Tenure features
    df["tenure_band"] = pd.cut(
        df["total_months_active"],
        bins=[0, 24, 48, 84, 121],
        labels=["Early", "Growing", "Mature", "Veteran"],
        right=True,
    ).astype(str)
    df["is_early_stage"] = (df["total_months_active"] < 24).astype(int)

    log.info(
        "Investor behaviour features added: missed_payment_ratio_12m, "
        "payment_streak, consec_missed, avg_monthly_investment, "
        "tenure_band, is_early_stage"
    )
    return df


def encode_investor_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Label-encode string categorical columns in the investor feature table.

    Converts ``tenure_band``, ``market_regime``, ``cagr_band``,
    ``fund_category``, ``age_group`` to integer codes.

    Parameters
    ----------
    df : pd.DataFrame
        Investor panel after all feature generators.

    Returns
    -------
    pd.DataFrame
        Panel with categorical columns encoded as integers.
    """
    df = df.copy()
    cat_cols = [
        "tenure_band", "market_regime", "cagr_band",
        "fund_category", "age_group",
    ]
    for col in cat_cols:
        if col in df.columns:
            df[col] = pd.Categorical(df[col]).codes
    log.info("Investor categorical columns encoded: %s",
             [c for c in cat_cols if c in df.columns])
    return df


def select_investor_features(df: pd.DataFrame) -> pd.DataFrame:
    """Select and order the final investor-level feature columns.

    Non-feature columns (high-cardinality strings, raw intermediates,
    intermediate portfolio tracking columns) are dropped.

    Parameters
    ----------
    df : pd.DataFrame
        Fully enriched investor × month panel.

    Returns
    -------
    pd.DataFrame
        Final feature table with ``churn`` as the last column.
    """
    drop_cols = {
        "investor_name",   # high-cardinality PII string
        "city",            # high-cardinality (encode separately if needed)
        "status",          # redundant with churn
        "sip_end_date",    # leakage risk (only known post-churn)
        "fund_name",       # high-cardinality string (category already encoded)
        "amc",             # high-cardinality string
        "portfolio_peak",  # intermediate SIP computation artefact
        "units_bought",    # per-month units — replaced by cumulative_units
    }

    keep = [c for c in df.columns if c not in drop_cols]
    # Ensure churn is last
    keep = [c for c in keep if c != "churn"] + ["churn"]

    df_out = df[keep].copy()
    log.info("Investor feature table: %d rows × %d columns", *df_out.shape)
    log.info(
        "Target distribution:  churn=1: %d  churn=0: %d",
        (df_out["churn"] == 1).sum(),
        (df_out["churn"] == 0).sum(),
    )
    return df_out


def run_feature_engineering_investor(
    df_panel: pd.DataFrame,
    out_path: str | Path = FEATURES_FILE,
) -> pd.DataFrame:
    """Run the investor-pipeline feature engineering sequence.

    Applies CAGR, max drawdown, risk-adjusted return, NIFTY context, and
    investor behaviour features; encodes categoricals; selects final columns;
    and saves ``features.csv``.

    Parameters
    ----------
    df_panel : pd.DataFrame
        Cleaned investor × month panel from
        ``investor_cleaning.run_investor_cleaning()``.
    out_path : str or Path
        Destination for the feature CSV.

    Returns
    -------
    pd.DataFrame
        Final feature table (also persisted to *out_path*).
    """
    log.info("=== Starting investor feature engineering ===")

    df = add_investor_behaviour_features(df_panel)   # adds tenure_band first
    df = add_cagr_features(df)
    df = add_max_drawdown_features(df)
    df = add_risk_adjusted_return_features(df)
    df = add_nifty_context_features(df)
    df = encode_investor_categoricals(df)
    df = select_investor_features(df)

    save_csv(df, out_path)
    log.info("=== Investor feature engineering complete → %s ===", out_path)
    return df


# ══════════════════════════════════════════════════════════════════════════════
# Standalone entry-points
# ══════════════════════════════════════════════════════════════════════════════

def main_amfi() -> None:
    """Run the AMFI feature-engineering stage end-to-end.

    Loads the cleaned merged panel from ``data/processed/merged_panel.csv``,
    engineers all AMFI features, and writes ``data/features/features_amfi.csv``.

    Run from any working directory::

        python src/features/feature_engineering.py --amfi
    """
    from src.utils.io_helpers import read_csv

    merged_path = _PROJECT_ROOT / "data" / "processed" / "merged_panel.csv"
    out_path    = _PROJECT_ROOT / "data" / "features"  / "features_amfi.csv"
    try:
        df_merged = read_csv(merged_path)
    except FileNotFoundError as exc:
        log.error("Merged AMFI panel not found: %s", exc)
        log.error(
            "Run `python src/ingestion/data_cleaning.py` first to generate "
            "the merged_panel.csv file."
        )
        sys.exit(1)

    run_feature_engineering(df_merged, out_path=out_path)


def main_investor() -> None:
    """Run the investor feature-engineering stage end-to-end.

    Loads the cleaned investor panel from
    ``data/processed/merged_investor_panel.csv``, engineers all investor
    features, and writes ``data/features/features.csv``.

    Run from any working directory::

        python src/features/feature_engineering.py
    """
    from src.utils.io_helpers import read_csv

    panel_path = _PROJECT_ROOT / "data" / "processed" / "merged_investor_panel.csv"
    try:
        df_panel = read_csv(panel_path, parse_dates=["date", "sip_start_date"])
    except FileNotFoundError as exc:
        log.error("Investor panel not found: %s", exc)
        log.error(
            "Run `python src/ingestion/investor_cleaning.py` first to generate "
            "the merged_investor_panel.csv file."
        )
        sys.exit(1)

    run_feature_engineering_investor(df_panel, out_path=FEATURES_FILE)


def main() -> None:
    """Default entry-point: run the investor feature-engineering pipeline.

    Pass ``--amfi`` as a command-line argument to run the AMFI pipeline
    instead::

        python src/features/feature_engineering.py --amfi
    """
    if "--amfi" in sys.argv:
        main_amfi()
    else:
        main_investor()


if __name__ == "__main__":
    main()
