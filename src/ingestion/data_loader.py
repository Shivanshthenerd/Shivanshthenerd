"""
src/ingestion/data_loader.py
-----------------------------
Stage 1 – Raw data ingestion for the Indian SIP churn pipeline.

Two real-world datasets are supported out of the box:

1. **india_mf_funds.csv**  (fund-level snapshot)
   Source: Kaggle "Mutual Funds India – Detailed" (AMFI data)
   814 Indian MF schemes with expense ratio, returns, risk metrics.

2. **sip_india_monthly.csv**  (fund × month panel)
   Derived from the fund-level data: 814 funds × 120 months → 87 912
   fund-month rows carrying time-varying performance indicators.

Responsibilities of this module
--------------------------------
- Locate raw CSV files under ``data/raw/``.
- Standardise column names (lowercase, underscored).
- Validate that mandatory columns are present.
- Return clean DataFrames ready for the cleaning stage.
"""

from pathlib import Path

import pandas as pd

from src.utils.io_helpers import read_csv
from src.utils.logger import get_logger

log = get_logger(__name__)

# ── Default paths (relative to project root) ─────────────────────────────────
RAW_DIR = Path("data/raw")
FUNDS_FILE   = RAW_DIR / "india_mf_funds.csv"
MONTHLY_FILE = RAW_DIR / "sip_india_monthly.csv"

# Mandatory columns for each dataset
FUNDS_REQUIRED_COLS = {
    "fund_id", "scheme_name", "amc_name", "category", "expense_ratio",
    "risk_level", "rating", "fund_size_cr", "fund_age_yr",
    "returns_1yr", "returns_3yr", "returns_5yr",
}
MONTHLY_REQUIRED_COLS = {
    "fund_id", "month", "monthly_return", "roll_3m_return",
    "roll_6m_return", "roll_12m_return", "drawdown",
    "rel_perf_vs_cat", "churn",
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _standardise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Lowercase all column names and replace spaces / hyphens with underscores.

    Parameters
    ----------
    df:
        Input DataFrame.

    Returns
    -------
    pd.DataFrame
        DataFrame with cleaned column names.
    """
    df = df.copy()
    df.columns = (
        df.columns.str.strip()
                  .str.lower()
                  .str.replace(r"[\s\-]+", "_", regex=True)
                  .str.replace(r"[^\w]", "", regex=True)
    )
    return df


def _validate_columns(df: pd.DataFrame, required: set, label: str) -> None:
    """Assert that all required columns are present.

    Parameters
    ----------
    df:
        DataFrame to validate.
    required:
        Set of column names that must exist.
    label:
        Human-readable name for the dataset (used in error messages).

    Raises
    ------
    ValueError
        If one or more required columns are absent.
    """
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"[{label}] Missing required columns: {sorted(missing)}"
        )
    log.info("[%s] Schema validation passed (%d columns present)", label, len(df.columns))


# ── Public loaders ────────────────────────────────────────────────────────────

def load_funds(path: str | Path = FUNDS_FILE) -> pd.DataFrame:
    """Load and lightly standardise the fund-level snapshot.

    Parameters
    ----------
    path:
        Path to ``india_mf_funds.csv``.

    Returns
    -------
    pd.DataFrame
        Standardised fund-level DataFrame.
    """
    log.info("Loading fund-level data from %s", path)
    df = read_csv(path)
    df = _standardise_columns(df)
    _validate_columns(df, FUNDS_REQUIRED_COLS, "india_mf_funds")
    log.info("Fund-level data loaded: %d funds", len(df))
    return df


def load_monthly(path: str | Path = MONTHLY_FILE) -> pd.DataFrame:
    """Load and lightly standardise the fund × month panel data.

    Parameters
    ----------
    path:
        Path to ``sip_india_monthly.csv``.

    Returns
    -------
    pd.DataFrame
        Standardised monthly panel DataFrame.
    """
    log.info("Loading monthly panel data from %s", path)
    df = read_csv(path)
    df = _standardise_columns(df)
    _validate_columns(df, MONTHLY_REQUIRED_COLS, "sip_india_monthly")
    log.info("Monthly panel loaded: %d fund-month rows", len(df))
    return df


def load_all(
    funds_path: str | Path = FUNDS_FILE,
    monthly_path: str | Path = MONTHLY_FILE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Convenience wrapper that loads both datasets at once.

    Parameters
    ----------
    funds_path:
        Path to the fund-level CSV.
    monthly_path:
        Path to the monthly panel CSV.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        ``(df_funds, df_monthly)``
    """
    df_funds   = load_funds(funds_path)
    df_monthly = load_monthly(monthly_path)
    return df_funds, df_monthly
