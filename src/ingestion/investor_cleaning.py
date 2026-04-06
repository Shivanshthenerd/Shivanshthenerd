"""
src/ingestion/investor_cleaning.py
------------------------------------
Stage 2b – Investor-centric data cleaning and panel construction.

Takes the three raw datasets produced by ``simulate_investors.py`` and
``data_loader.py`` and builds a single investor × month panel ready for
feature engineering.

Steps
-----
1. **Timeline expansion** — expand each investor's SIP from their start
   date to their end date (or Dec 2024 if active), creating one row per
   investor per active month.
2. **Missed-payment distribution** — distribute each investor's total
   ``missed_payments`` across their active months, weighted toward months
   where the fund return was negative (mimicking realistic SIP pause
   behaviour).
3. **NAV join** — left-join with fund NAV on ``(fund_name, date)`` to
   obtain per-month NAV and fund return for each investor.
4. **NIFTY join** — left-join with NIFTY 50 on ``date`` for market
   context features.
5. **Unit accumulation (real SIP math)** — compute cumulative units held
   and current portfolio value month-by-month.
6. **Monthly aggregation validation** — the panel is already monthly;
   we validate alignment and forward-fill any sparse NAV gaps.
7. **Numeric imputation** — median fill for any remaining NaN cells.
8. **Save** ``data/processed/merged_investor_panel.csv``.

Key output columns
------------------
investor_id, date, fund_name, monthly_investment, sip_start_date,
missed_this_month, cumulative_missed, total_months_active,
nav, fund_monthly_return, cumulative_units, portfolio_value,
total_invested, nifty_close, nifty_monthly_return, nifty_drawdown,
churn (target: 1 = investor eventually stopped SIP)

Running this script
-------------------
Can be invoked directly from any working directory::

    python src/ingestion/investor_cleaning.py
"""

import sys
from pathlib import Path

# Allow running this file directly: python src/ingestion/investor_cleaning.py
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import pandas as pd

from src.utils.io_helpers import save_csv
from src.utils.logger import get_logger

log = get_logger(__name__)

# Absolute paths — work from any working directory
PROCESSED_DIR     = _PROJECT_ROOT / "data" / "processed"
PANEL_OUTPUT_FILE = PROCESSED_DIR / "merged_investor_panel.csv"

SIM_END = pd.Timestamp("2024-12-01")


# ── Timeline expansion ────────────────────────────────────────────────────────

def expand_investor_timelines(df_investors: pd.DataFrame) -> pd.DataFrame:
    """Expand investor records to an investor × month panel.

    For each investor, generates one row per calendar month between
    ``sip_start_date`` and ``sip_end_date`` (or ``SIM_END`` for active
    investors).  This gives the time-series structure needed for rolling
    window features and time-to-churn modelling.

    Parameters
    ----------
    df_investors : pd.DataFrame
        One row per investor from ``data_loader.load_investors()``.

    Returns
    -------
    pd.DataFrame
        Long-format panel with columns [investor_id, date, fund_name,
        monthly_investment, sip_start_date, sip_end_date, status, churn,
        investor_name, city, age_group].
    """
    log.info("Expanding investor timelines …")
    panels = []

    for _, row in df_investors.iterrows():
        start = row["sip_start_date"]
        # Active investors run to end of simulation window
        end   = row["sip_end_date"] if pd.notna(row["sip_end_date"]) else SIM_END

        # Generate monthly dates (first of each month)
        months = pd.date_range(start=start, end=end, freq="MS")
        if len(months) == 0:
            continue

        chunk = pd.DataFrame({
            "investor_id":        row["investor_id"],
            "date":               months,
            "fund_name":          row["fund_name"],
            "monthly_investment": row["monthly_investment"],
            "sip_start_date":     row["sip_start_date"],
            "sip_end_date":       row["sip_end_date"],
            "missed_payments":    row["missed_payments"],   # total over tenure
            "status":             row["status"],
            "churn":              row["churn"],
            "investor_name":      row.get("investor_name", ""),
            "city":               row.get("city", ""),
            "age_group":          row.get("age_group", ""),
        })
        panels.append(chunk)

    df_panel = pd.concat(panels, ignore_index=True)
    log.info("Timeline expansion: %d investor-month rows", len(df_panel))
    return df_panel


# ── Missed-payment distribution ───────────────────────────────────────────────

def distribute_missed_payments(
    df_panel: pd.DataFrame,
    df_nav:   pd.DataFrame,
) -> pd.DataFrame:
    """Assign monthly missed-payment flags across each investor's timeline.

    The *total* missed payments declared at the investor level are
    redistributed as a binary ``missed_this_month`` flag.  Months where
    the fund return was negative are 3× more likely to be assigned a
    missed payment, reflecting investor behaviour during downturns.

    Parameters
    ----------
    df_panel : pd.DataFrame
        Expanded investor × month panel.
    df_nav : pd.DataFrame
        Fund NAV monthly data (for negative-return weighting).

    Returns
    -------
    pd.DataFrame
        Panel with ``missed_this_month`` (0/1) and
        ``cumulative_missed`` columns added.
    """
    log.info("Distributing missed payments …")
    rng = np.random.default_rng(99)

    # Build a quick lookup: (fund_name, date) → monthly_return
    nav_ret = (
        df_nav[["fund_name", "date", "monthly_return"]]
        .set_index(["fund_name", "date"])["monthly_return"]
        .to_dict()
    )

    df_panel = df_panel.copy()
    df_panel = df_panel.sort_values(["investor_id", "date"])

    missed_flags = np.zeros(len(df_panel), dtype=np.int8)

    for inv_id, grp in df_panel.groupby("investor_id"):
        n_months    = len(grp)
        total_miss  = int(grp["missed_payments"].iloc[0])
        total_miss  = min(total_miss, n_months)   # can't miss more months than active

        if total_miss == 0:
            continue

        # Weight: negative-return months are 3× more likely to see a missed payment
        fund   = grp["fund_name"].iloc[0]
        weights = np.array([
            3.0 if nav_ret.get((fund, d), 0) < 0 else 1.0
            for d in grp["date"]
        ])
        weights = weights / weights.sum()

        chosen_idx = rng.choice(n_months, size=total_miss, replace=False, p=weights)
        global_idx = grp.index[chosen_idx]
        missed_flags[df_panel.index.get_indexer(global_idx)] = 1

    df_panel["missed_this_month"] = missed_flags

    # Cumulative missed payments per investor (running total)
    df_panel["cumulative_missed"] = (
        df_panel.groupby("investor_id")["missed_this_month"].cumsum()
    )

    log.info(
        "Missed payments distributed: %d months flagged across %d investors",
        int(missed_flags.sum()),
        df_panel["investor_id"].nunique(),
    )
    return df_panel


# ── NAV and NIFTY joins ───────────────────────────────────────────────────────

def join_nav_data(df_panel: pd.DataFrame, df_nav: pd.DataFrame) -> pd.DataFrame:
    """Left-join the investor panel with monthly fund NAV data.

    Joins on ``(fund_name, date)`` so each row gains the fund's NAV and
    monthly return for that calendar month.

    Parameters
    ----------
    df_panel : pd.DataFrame
        Expanded investor × month panel.
    df_nav : pd.DataFrame
        Fund NAV monthly data.

    Returns
    -------
    pd.DataFrame
        Panel enriched with ``nav``, ``fund_monthly_return``,
        ``fund_category``, ``amc``, ``expense_ratio``.
    """
    nav_cols = df_nav[["fund_name", "date", "nav", "monthly_return",
                        "category", "amc", "expense_ratio"]].copy()
    nav_cols = nav_cols.rename(columns={
        "monthly_return": "fund_monthly_return",
        "category":       "fund_category",
    })

    df_panel = df_panel.merge(nav_cols, on=["fund_name", "date"], how="left")

    # Forward-fill sparse NAV gaps within each fund (≤ 2 months)
    df_panel = df_panel.sort_values(["investor_id", "date"])
    for col in ["nav", "fund_monthly_return"]:
        df_panel[col] = (
            df_panel.groupby("investor_id")[col]
            .transform(lambda s: s.ffill(limit=2))
        )

    log.info("NAV join complete: %d rows, %.2f%% NAV coverage",
             len(df_panel), df_panel["nav"].notna().mean() * 100)
    return df_panel


def join_nifty_data(df_panel: pd.DataFrame, df_nifty: pd.DataFrame) -> pd.DataFrame:
    """Left-join the investor panel with NIFTY 50 monthly market data.

    Parameters
    ----------
    df_panel : pd.DataFrame
        Investor × month panel (post NAV join).
    df_nifty : pd.DataFrame
        NIFTY 50 monthly data.

    Returns
    -------
    pd.DataFrame
        Panel enriched with NIFTY columns.
    """
    nifty_cols = df_nifty[[
        "date", "nifty_close", "nifty_monthly_return",
        "nifty_3m_return", "nifty_6m_return", "nifty_12m_return",
        "nifty_drawdown",
    ]].copy()

    df_panel = df_panel.merge(nifty_cols, on="date", how="left")
    log.info("NIFTY join complete: %d rows", len(df_panel))
    return df_panel


# ── SIP unit accumulation (real SIP mathematics) ──────────────────────────────

def compute_sip_portfolio(df_panel: pd.DataFrame) -> pd.DataFrame:
    """Compute running portfolio value using real SIP unit-accumulation math.

    Each month where the investor did NOT miss a payment, they purchase
    ``monthly_investment / nav`` units.  Missed months contribute 0 units.
    Portfolio value at month *t* is ``cumulative_units_t × nav_t``.

    This is the standard way mutual fund platforms (Zerodha Coin, Groww,
    Paytm Money) calculate SIP returns in India.

    Parameters
    ----------
    df_panel : pd.DataFrame
        Investor × month panel with ``nav``, ``missed_this_month``,
        ``monthly_investment`` columns present.

    Returns
    -------
    pd.DataFrame
        Panel with ``units_bought``, ``cumulative_units``,
        ``portfolio_value``, ``total_invested`` columns added.
    """
    log.info("Computing SIP portfolio values …")
    df_panel = df_panel.copy().sort_values(["investor_id", "date"])

    # Units bought this month (0 if payment missed or NAV is NaN)
    valid_nav  = df_panel["nav"].fillna(1.0)    # fallback to avoid /0
    df_panel["units_bought"] = np.where(
        df_panel["missed_this_month"] == 1,
        0.0,
        df_panel["monthly_investment"] / valid_nav,
    )

    # Cumulative units and total invested per investor
    df_panel["cumulative_units"] = (
        df_panel.groupby("investor_id")["units_bought"].cumsum()
    )
    # total_invested: running sum of effective monthly contributions
    # (0 for missed months, monthly_investment otherwise)
    effective_investment = df_panel["monthly_investment"] * (1 - df_panel["missed_this_month"])
    df_panel["total_invested"] = effective_investment.groupby(df_panel["investor_id"]).cumsum()

    # Current portfolio value
    df_panel["portfolio_value"] = df_panel["cumulative_units"] * valid_nav

    log.info("Portfolio computation done")
    return df_panel


# ── Column additions: months_active ──────────────────────────────────────────

def add_months_active(df_panel: pd.DataFrame) -> pd.DataFrame:
    """Add ``total_months_active`` counter (1, 2, 3, … per investor).

    Parameters
    ----------
    df_panel : pd.DataFrame
        Sorted investor × month panel.

    Returns
    -------
    pd.DataFrame
        Panel with ``total_months_active`` column.
    """
    df_panel = df_panel.sort_values(["investor_id", "date"])
    df_panel["total_months_active"] = (
        df_panel.groupby("investor_id").cumcount() + 1
    )
    return df_panel


# ── Numeric imputation ────────────────────────────────────────────────────────

def impute_remaining_nulls(df_panel: pd.DataFrame) -> pd.DataFrame:
    """Median-fill any remaining NaN values in numeric columns.

    Parameters
    ----------
    df_panel : pd.DataFrame
        Investor panel after all joins.

    Returns
    -------
    pd.DataFrame
        Panel with no remaining NaN values in numeric columns.
    """
    num_cols = df_panel.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        n_null = df_panel[col].isna().sum()
        if n_null:
            med = df_panel[col].median()
            df_panel[col] = df_panel[col].fillna(med)
            log.info("Imputed %d nulls in '%s' with median=%.4f", n_null, col, med)
    return df_panel


# ── Orchestration ─────────────────────────────────────────────────────────────

def run_investor_cleaning(
    df_investors: pd.DataFrame,
    df_nav:       pd.DataFrame,
    df_nifty:     pd.DataFrame,
    out_path:     str | Path = PANEL_OUTPUT_FILE,
) -> pd.DataFrame:
    """Run the full investor cleaning pipeline.

    Parameters
    ----------
    df_investors : pd.DataFrame
        Raw investor data from ``data_loader.load_investors()``.
    df_nav : pd.DataFrame
        Fund NAV monthly data from ``data_loader.load_fund_nav()``.
    df_nifty : pd.DataFrame
        NIFTY 50 monthly data from ``data_loader.load_nifty()``.
    out_path : str or Path
        Destination CSV path (default: ``data/processed/merged_investor_panel.csv``).

    Returns
    -------
    pd.DataFrame
        Cleaned investor × month panel saved to *out_path*.
    """
    log.info("=== Starting investor cleaning pipeline ===")

    df = expand_investor_timelines(df_investors)
    df = distribute_missed_payments(df, df_nav)
    df = join_nav_data(df, df_nav)
    df = join_nifty_data(df, df_nifty)
    df = compute_sip_portfolio(df)
    df = add_months_active(df)
    df = impute_remaining_nulls(df)

    # Drop the static per-investor missed_payments total (redundant with
    # cumulative_missed which is per-month)
    df = df.drop(columns=["missed_payments"], errors="ignore")

    save_csv(df, Path(out_path))
    log.info(
        "=== Investor cleaning complete: %d rows × %d cols → %s ===",
        *df.shape, out_path,
    )
    return df


def main() -> None:
    """Entry-point: load raw investor data, build panel, and save.

    Run from any working directory::

        python src/ingestion/investor_cleaning.py
    """
    from src.ingestion.data_loader import load_investor_data_all
    from src.ingestion.simulate_investors import run_simulation

    # Ensure raw investor files exist
    run_simulation()

    try:
        df_nifty, df_nav, df_investors = load_investor_data_all()
    except FileNotFoundError as exc:
        log.error("Investor raw data not found: %s", exc)
        log.error(
            "Run `python src/ingestion/simulate_investors.py` to generate "
            "the investor dataset files."
        )
        sys.exit(1)

    run_investor_cleaning(df_investors, df_nav, df_nifty)
    log.info("Investor cleaning complete — output written to %s", PANEL_OUTPUT_FILE)


if __name__ == "__main__":
    main()
