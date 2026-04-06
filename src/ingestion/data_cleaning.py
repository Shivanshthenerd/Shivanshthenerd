"""
src/ingestion/data_cleaning.py
-------------------------------
Stage 2 – Data cleaning for the Indian SIP churn pipeline.

Takes the raw DataFrames produced by ``data_loader.py`` and applies:

1. Missing-value imputation (median for numerics, mode for categoricals).
2. Type coercion (ensure int / float columns have the right dtype).
3. Outlier capping for financial ratios (IQR-based, configurable).
4. Deduplication (exact duplicates on key columns).
5. Merging both datasets on ``fund_id`` to produce a single enriched
   monthly panel.
6. Persisting processed artefacts to ``data/processed/``.

Running this script
-------------------
Can be invoked directly from any working directory::

    python src/ingestion/data_cleaning.py
"""

import sys
from pathlib import Path

# Allow running this file directly: python src/ingestion/data_cleaning.py
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import pandas as pd

from src.utils.io_helpers import save_csv
from src.utils.logger import get_logger

log = get_logger(__name__)

# Absolute path — works from any working directory
PROCESSED_DIR = _PROJECT_ROOT / "data" / "processed"

# Columns that should be numeric in the fund-level dataset
FUND_NUMERIC_COLS = [
    "expense_ratio", "fund_size_cr", "fund_age_yr",
    "sortino", "alpha", "sd", "beta", "sharpe",
    "returns_1yr", "returns_3yr", "returns_5yr",
    "min_sip", "min_lumpsum",
]

# Columns that should be numeric in the monthly panel
MONTHLY_NUMERIC_COLS = [
    "monthly_return", "roll_3m_return", "roll_6m_return",
    "roll_12m_return", "nav_ratio_12m", "vol_3m",
    "drawdown", "rel_perf_vs_cat",
    "expense_ratio", "alpha", "beta", "sharpe", "sortino",
    "sd_annual", "returns_1yr", "returns_3yr", "returns_5yr",
]

# Ratio / return columns where we cap extreme outliers
OUTLIER_COLS = [
    "returns_1yr", "returns_3yr", "returns_5yr",
    "roll_3m_return", "roll_6m_return", "roll_12m_return",
    "monthly_return", "alpha", "rel_perf_vs_cat",
]


# ── Imputation ────────────────────────────────────────────────────────────────

def impute_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Fill missing values in numeric columns with the column median.

    Parameters
    ----------
    df:
        Input DataFrame.
    cols:
        List of numeric columns to impute.

    Returns
    -------
    pd.DataFrame
        DataFrame with imputed values.
    """
    df = df.copy()
    for col in cols:
        if col not in df.columns:
            continue
        # Coerce to numeric first so '-' strings become NaN
        df[col] = pd.to_numeric(df[col], errors="coerce")
        n_missing = df[col].isna().sum()
        if n_missing:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            log.info("Imputed %d missing values in '%s' with median=%.4f",
                     n_missing, col, median_val)
    return df


def impute_categorical(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Fill missing values in categorical columns with the column mode.

    Parameters
    ----------
    df:
        Input DataFrame.
    cols:
        List of categorical columns to impute.

    Returns
    -------
    pd.DataFrame
        DataFrame with imputed values.
    """
    df = df.copy()
    for col in cols:
        if col not in df.columns:
            continue
        n_missing = df[col].isna().sum()
        if n_missing:
            mode_val = df[col].mode().iloc[0]
            df[col] = df[col].fillna(mode_val)
            log.info("Imputed %d missing values in '%s' with mode='%s'",
                     n_missing, col, mode_val)
    return df


# ── Outlier capping ───────────────────────────────────────────────────────────

def cap_outliers(
    df: pd.DataFrame,
    cols: list[str],
    iqr_multiplier: float = 3.0,
) -> pd.DataFrame:
    """Cap extreme values using the IQR method (Tukey fences).

    Values below ``Q1 - k*IQR`` or above ``Q3 + k*IQR`` are clipped
    to the respective fence, where *k* = ``iqr_multiplier``.

    Parameters
    ----------
    df:
        Input DataFrame.
    cols:
        Columns to apply capping to.
    iqr_multiplier:
        Multiplier for the IQR fence (default 3.0 — only clips extreme
        outliers while keeping realistic financial extremes).

    Returns
    -------
    pd.DataFrame
        DataFrame with outliers capped.
    """
    df = df.copy()
    for col in cols:
        if col not in df.columns:
            continue
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - iqr_multiplier * iqr
        upper = q3 + iqr_multiplier * iqr
        n_clipped = ((df[col] < lower) | (df[col] > upper)).sum()
        if n_clipped:
            df[col] = df[col].clip(lower=lower, upper=upper)
            log.info("Capped %d outliers in '%s'  [%.4f, %.4f]",
                     n_clipped, col, lower, upper)
    return df


# ── Deduplication ─────────────────────────────────────────────────────────────

def deduplicate(df: pd.DataFrame, subset: list[str], label: str) -> pd.DataFrame:
    """Drop exact duplicate rows based on key columns.

    Parameters
    ----------
    df:
        Input DataFrame.
    subset:
        Columns that together form the natural key.
    label:
        Dataset name for log messages.

    Returns
    -------
    pd.DataFrame
        DataFrame with duplicates removed (first occurrence kept).
    """
    before = len(df)
    df = df.drop_duplicates(subset=subset, keep="first")
    dropped = before - len(df)
    if dropped:
        log.warning("[%s] Dropped %d duplicate rows on %s", label, dropped, subset)
    else:
        log.info("[%s] No duplicates found on %s", label, subset)
    return df


# ── Main cleaning functions ───────────────────────────────────────────────────

def clean_funds(df_funds: pd.DataFrame) -> pd.DataFrame:
    """Apply all cleaning steps to the fund-level snapshot.

    Parameters
    ----------
    df_funds:
        Raw fund-level DataFrame from ``data_loader.load_funds()``.

    Returns
    -------
    pd.DataFrame
        Cleaned fund-level DataFrame.
    """
    log.info("=== Cleaning fund-level data ===")
    df = impute_numeric(df_funds, FUND_NUMERIC_COLS)
    df = impute_categorical(df, ["category", "sub_category", "fund_manager", "amc_name"])
    df = cap_outliers(df, ["returns_1yr", "returns_3yr", "returns_5yr", "alpha"])
    df = deduplicate(df, subset=["fund_id"], label="india_mf_funds")

    # Ensure key integer columns stay int after imputation
    for col in ["fund_age_yr", "risk_level", "rating", "min_sip", "min_lumpsum"]:
        if col in df.columns:
            df[col] = df[col].astype(int)

    log.info("Fund-level cleaning done: %d rows", len(df))
    return df


def clean_monthly(df_monthly: pd.DataFrame) -> pd.DataFrame:
    """Apply all cleaning steps to the monthly panel.

    Parameters
    ----------
    df_monthly:
        Raw monthly DataFrame from ``data_loader.load_monthly()``.

    Returns
    -------
    pd.DataFrame
        Cleaned monthly panel DataFrame.
    """
    log.info("=== Cleaning monthly panel data ===")
    df = impute_numeric(df_monthly, MONTHLY_NUMERIC_COLS)
    df = cap_outliers(df, OUTLIER_COLS)
    df = deduplicate(df, subset=["fund_id", "month"], label="sip_india_monthly")

    # Ensure binary / integer columns are correct dtype
    for col in ["churn", "consec_neg"]:
        if col in df.columns:
            df[col] = df[col].astype(int)

    log.info("Monthly panel cleaning done: %d rows", len(df))
    return df


def merge_datasets(
    df_funds: pd.DataFrame,
    df_monthly: pd.DataFrame,
) -> pd.DataFrame:
    """Left-join the monthly panel with fund-level metadata on ``fund_id``.

    Static fund attributes (``scheme_name``, ``amc_name``, ``category``,
    ``sub_category``) are added to every fund-month row so downstream
    feature engineering can slice by category without additional joins.

    Parameters
    ----------
    df_funds:
        Cleaned fund-level DataFrame.
    df_monthly:
        Cleaned monthly panel DataFrame.

    Returns
    -------
    pd.DataFrame
        Merged panel with fund metadata columns added (suffixed ``_fund``
        if a column already exists in the monthly data).
    """
    log.info("Merging monthly panel with fund metadata on fund_id …")

    # Carry only metadata columns that do not already exist in monthly data
    meta_cols = ["fund_id", "scheme_name", "amc_name", "category", "sub_category"]
    meta_cols = [c for c in meta_cols if c in df_funds.columns]

    merged = df_monthly.merge(
        df_funds[meta_cols],
        on="fund_id",
        how="left",
        suffixes=("", "_fund"),
    )
    log.info("Merged dataset: %d rows × %d columns", *merged.shape)
    return merged


def run_cleaning(
    df_funds: pd.DataFrame,
    df_monthly: pd.DataFrame,
    out_dir: str | Path = PROCESSED_DIR,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run the full cleaning pipeline and persist outputs.

    Parameters
    ----------
    df_funds:
        Raw fund-level DataFrame.
    df_monthly:
        Raw monthly panel DataFrame.
    out_dir:
        Directory where processed CSVs are written.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        ``(clean_funds, clean_monthly, merged_panel)``
    """
    clean_f = clean_funds(df_funds)
    clean_m = clean_monthly(df_monthly)
    merged  = merge_datasets(clean_f, clean_m)

    save_csv(clean_f,  Path(out_dir) / "funds_clean.csv")
    save_csv(clean_m,  Path(out_dir) / "monthly_clean.csv")
    save_csv(merged,   Path(out_dir) / "merged_panel.csv")

    return clean_f, clean_m, merged


def main() -> None:
    """Entry-point: load raw AMFI data, clean it, and save processed CSVs.

    Run from any working directory::

        python src/ingestion/data_cleaning.py
    """
    from src.ingestion.data_loader import load_all

    try:
        df_funds, df_monthly = load_all()
    except FileNotFoundError as exc:
        log.error("Raw AMFI data not found: %s", exc)
        log.error(
            "Run `python data/prepare_dataset.py` first to download and "
            "generate the AMFI fund files."
        )
        sys.exit(1)

    run_cleaning(df_funds, df_monthly)
    log.info("Cleaning complete — outputs written to %s", PROCESSED_DIR)


if __name__ == "__main__":
    main()
