"""
src/ingestion/simulate_investors.py
------------------------------------
Generates three realistic Indian financial datasets and writes them to
``data/raw/``.  All outputs are deterministic (fixed random seed).

Datasets produced
-----------------
1. ``nifty50_monthly.csv``
   Monthly NIFTY 50 price/return data, Jan 2015 – Dec 2024 (120 rows).
   Returns are calibrated to the historical NIFTY 50 trajectory
   (≈ 8 500 → 24 000, CAGR ≈ 11 %).  Key crash/rally events are
   hard-coded for realism (COVID Mar-2020 crash, recovery, 2022 rate-hike
   correction).

2. ``fund_nav_monthly.csv``
   Monthly NAV time-series for 20 popular Indian MF schemes (20 × 120 =
   2 400 rows).  Each fund is given realistic category, AMC, expense
   ratio, beta, and idiosyncratic alpha so that returns differ credibly.

3. ``sip_investors.csv``
   5 000 synthetic SIP investors with realistic Indian demographics.
   Fields: investor_id, investor_name, city, age_group, fund_name,
   monthly_investment, sip_start_date, sip_end_date, missed_payments,
   status (active/stopped).

Running this script
-------------------
Can be invoked directly from any working directory::

    python src/ingestion/simulate_investors.py

or imported and called programmatically::

    from src.ingestion.simulate_investors import run_simulation
    run_simulation()
"""

import sys
from pathlib import Path

# Allow running this file directly: python src/ingestion/simulate_investors.py
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import pandas as pd

from src.utils.io_helpers import save_csv
from src.utils.logger import get_logger

log = get_logger(__name__)

# Absolute path to data/raw — works from any working directory
RAW_DIR = _PROJECT_ROOT / "data" / "raw"

# ── Simulation parameters ──────────────────────────────────────────────────────
RANDOM_SEED   = 42
SIM_START     = pd.Timestamp("2015-01-01")
SIM_END       = pd.Timestamp("2024-12-01")
N_INVESTORS   = 5_000

# ── Indian MF fund catalogue ───────────────────────────────────────────────────
# Each entry: (fund_name, amc, category, beta, annual_alpha_pct, starting_nav)
# beta  = sensitivity to NIFTY 50 monthly return
# alpha = annualised excess return over market (gross, before expenses)
FUND_CATALOGUE = [
    ("SBI Bluechip Fund",                   "SBI Mutual Fund",       "Large Cap",   0.90,  1.5, 25.0),
    ("HDFC Mid-Cap Opportunities Fund",      "HDFC Mutual Fund",      "Mid Cap",     1.05,  3.0, 18.0),
    ("ICICI Pru Value Discovery Fund",       "ICICI Prudential MF",   "Value",       0.85,  2.5, 55.0),
    ("Axis Long Term Equity Fund",           "Axis Mutual Fund",      "ELSS",        0.92,  2.0, 30.0),
    ("Mirae Asset Large Cap Fund",           "Mirae Asset MF",        "Large Cap",   0.93,  1.8, 22.0),
    ("Nippon India Small Cap Fund",          "Nippon India MF",       "Small Cap",   1.18,  4.5, 12.0),
    ("Kotak Emerging Equity Fund",           "Kotak Mahindra MF",     "Mid Cap",     1.08,  3.2, 15.0),
    ("Parag Parikh Flexi Cap Fund",          "PPFAS Mutual Fund",     "Flexi Cap",   0.75,  3.8, 28.0),
    ("DSP Mid Cap Fund",                     "DSP Mutual Fund",       "Mid Cap",     1.02,  2.8, 20.0),
    ("Aditya Birla SL Frontline Equity",     "Aditya Birla SL MF",    "Large Cap",   0.95,  1.2, 45.0),
    ("Franklin India Prima Fund",            "Franklin Templeton MF", "Mid Cap",     1.00,  2.0, 65.0),
    ("UTI Flexi Cap Fund",                   "UTI Mutual Fund",       "Flexi Cap",   0.88,  1.5, 48.0),
    ("SBI Small Cap Fund",                   "SBI Mutual Fund",       "Small Cap",   1.20,  5.0, 10.0),
    ("HDFC Flexi Cap Fund",                  "HDFC Mutual Fund",      "Flexi Cap",   0.92,  1.8, 95.0),
    ("ICICI Pru Bluechip Fund",              "ICICI Prudential MF",   "Large Cap",   0.91,  1.4, 38.0),
    ("Axis Bluechip Fund",                   "Axis Mutual Fund",      "Large Cap",   0.88,  2.2, 28.0),
    ("Mirae Asset Emerging Bluechip Fund",   "Mirae Asset MF",        "Large & Mid", 1.00,  3.5, 36.0),
    ("Nippon India Growth Fund",             "Nippon India MF",       "Mid Cap",     1.05,  2.5, 70.0),
    ("Kotak Standard Multicap Fund",         "Kotak Mahindra MF",     "Flexi Cap",   0.90,  1.6, 32.0),
    ("Canara Robeco Emerging Equities",      "Canara Robeco MF",      "Mid & Small", 1.10,  3.0, 25.0),
]

# ── Indian demographics ────────────────────────────────────────────────────────
FIRST_NAMES = [
    "Rahul", "Priya", "Amit", "Sneha", "Vijay", "Ananya", "Karthik", "Pooja",
    "Sanjay", "Deepa", "Rohan", "Neha", "Arjun", "Kavya", "Suresh", "Lakshmi",
    "Manoj", "Divya", "Anil", "Sunita", "Vikram", "Meena", "Ravi", "Swati",
    "Dinesh", "Nandini", "Harish", "Rekha", "Rajesh", "Seema", "Gaurav", "Puja",
    "Nitin", "Shweta", "Manish", "Anjali", "Vivek", "Preeti", "Sunil", "Geeta",
    "Ashish", "Lata", "Pranav", "Rohini", "Yogesh", "Archana", "Kuldeep", "Mala",
]
LAST_NAMES = [
    "Sharma", "Patel", "Gupta", "Kumar", "Singh", "Joshi", "Verma", "Mehta",
    "Rao", "Nair", "Reddy", "Iyer", "Pillai", "Menon", "Chaudhary", "Sinha",
    "Malhotra", "Kapoor", "Chopra", "Saxena", "Agarwal", "Bansal", "Khanna",
    "Pandey", "Mishra", "Shukla", "Tiwari", "Chauhan", "Yadav", "Dubey",
]
CITIES = [
    "Mumbai", "Delhi", "Bangalore", "Chennai", "Hyderabad", "Pune",
    "Kolkata", "Ahmedabad", "Jaipur", "Lucknow", "Chandigarh", "Indore",
    "Bhopal", "Nagpur", "Visakhapatnam", "Kochi", "Surat", "Vadodara",
]
AGE_GROUPS = ["25-35", "36-45", "46-55", "55+"]
AGE_WEIGHTS = [0.35, 0.35, 0.20, 0.10]   # younger investors dominate SIP


# ══════════════════════════════════════════════════════════════════════════════
# NIFTY 50 monthly simulation
# ══════════════════════════════════════════════════════════════════════════════

def simulate_nifty50() -> pd.DataFrame:
    """Simulate NIFTY 50 monthly price and return series (Jan 2015–Dec 2024).

    Returns are normally distributed with a small positive drift calibrated
    to a ≈ 11 % annualised CAGR.  Five key market events are hard-coded to
    improve time-series realism:
    - COVID crash   : Feb–Mar 2020 (−7 %, −23 %)
    - COVID recovery: Apr–Jun 2020 (+14 %, +8 %, +8 %)
    - 2022 selloff  : Feb–Mar 2022 (−5 %, −6 %)
    - 2022 recovery : Jun 2022     (+6 %)
    - 2024 rally    : Oct–Nov 2024 (+5 %, +6 %)

    Returns
    -------
    pd.DataFrame
        120 rows × columns [date, nifty_close, nifty_monthly_return,
        nifty_3m_return, nifty_6m_return, nifty_12m_return, nifty_drawdown].
    """
    rng = np.random.default_rng(RANDOM_SEED)
    dates = pd.date_range(SIM_START, SIM_END, freq="MS")
    n = len(dates)   # 120 months

    # Base monthly returns: mean ≈ 0.9 %, σ ≈ 3.5 %
    monthly_ret = rng.normal(0.009, 0.035, n)

    # ── Hard-code key events by date index ──────────────────────────────────
    # Index 0 = Jan-2015, index 60 = Jan-2020
    event_idx = {
        61: -0.070,   # Feb 2020 – COVID fear
        62: -0.230,   # Mar 2020 – COVID crash
        63:  0.140,   # Apr 2020 – circuit breaker recovery
        64:  0.080,   # May 2020
        65:  0.075,   # Jun 2020
        85: -0.050,   # Feb 2022 – Russia-Ukraine / rate hike fears
        86: -0.060,   # Mar 2022
        89:  0.060,   # Jun 2022 – RBI pause relief
        117: 0.050,   # Oct 2024 – pre-election rally
        118: 0.060,   # Nov 2024
    }
    for idx, val in event_idx.items():
        monthly_ret[idx] = val

    # Build price series starting at ≈ 8 500 (NIFTY Jan-2015)
    nifty_close = np.empty(n)
    nifty_close[0] = 8_500.0
    for i in range(1, n):
        nifty_close[i] = nifty_close[i - 1] * (1.0 + monthly_ret[i])

    # Rolling multi-period returns (log-based for accuracy)
    log_ret = np.log1p(monthly_ret)
    roll_3  = np.full(n, np.nan)
    roll_6  = np.full(n, np.nan)
    roll_12 = np.full(n, np.nan)
    for i in range(n):
        if i >= 2:
            roll_3[i]  = np.expm1(log_ret[max(0, i - 2): i + 1].sum())
        if i >= 5:
            roll_6[i]  = np.expm1(log_ret[max(0, i - 5): i + 1].sum())
        if i >= 11:
            roll_12[i] = np.expm1(log_ret[max(0, i - 11): i + 1].sum())

    # Drawdown: decline from rolling peak
    rolling_peak = np.maximum.accumulate(nifty_close)
    drawdown = (nifty_close - rolling_peak) / rolling_peak

    df = pd.DataFrame({
        "date":                 dates,
        "nifty_close":          nifty_close.round(2),
        "nifty_monthly_return": monthly_ret.round(6),
        "nifty_3m_return":      roll_3.round(6),
        "nifty_6m_return":      roll_6.round(6),
        "nifty_12m_return":     roll_12.round(6),
        "nifty_drawdown":       drawdown.round(6),
    })
    log.info("NIFTY 50 simulated: %d months  close=[%.0f, %.0f]",
             n, nifty_close[0], nifty_close[-1])
    return df


# ══════════════════════════════════════════════════════════════════════════════
# Fund NAV simulation
# ══════════════════════════════════════════════════════════════════════════════

def simulate_fund_nav(df_nifty: pd.DataFrame) -> pd.DataFrame:
    """Simulate monthly NAV for 20 Indian MF schemes (Jan 2015–Dec 2024).

    Each fund's return is modelled as::

        fund_return_t = beta * nifty_return_t + monthly_alpha + epsilon_t

    where ``epsilon_t ~ N(0, idio_std)``.  The idiosyncratic standard
    deviation is derived from the fund category (small-cap funds have more
    idiosyncratic risk than large-cap).

    Parameters
    ----------
    df_nifty : pd.DataFrame
        Output of :func:`simulate_nifty50`.

    Returns
    -------
    pd.DataFrame
        2 400 rows × [date, fund_name, amc, category, nav, monthly_return,
        expense_ratio].
    """
    rng = np.random.default_rng(RANDOM_SEED + 1)
    nifty_ret = df_nifty["nifty_monthly_return"].values
    dates = df_nifty["date"].values
    n = len(dates)

    # Idiosyncratic std by category
    idio_std_map = {
        "Large Cap":    0.008,
        "Mid Cap":      0.015,
        "Small Cap":    0.022,
        "Flexi Cap":    0.010,
        "ELSS":         0.012,
        "Value":        0.012,
        "Large & Mid":  0.012,
        "Mid & Small":  0.018,
    }

    records = []
    for (fname, amc, cat, beta, ann_alpha, start_nav) in FUND_CATALOGUE:
        monthly_alpha = ann_alpha / 100.0 / 12.0   # annualised → monthly
        idio_std = idio_std_map.get(cat, 0.012)
        expense_ratio = round(rng.uniform(0.5, 2.0), 2)

        # Monthly return = systematic + alpha + noise - expense_ratio/12
        monthly_expense = expense_ratio / 100.0 / 12.0
        noise = rng.normal(0.0, idio_std, n)
        fund_ret = beta * nifty_ret + monthly_alpha + noise - monthly_expense

        # Build NAV series
        nav = np.empty(n)
        nav[0] = start_nav
        for i in range(1, n):
            nav[i] = nav[i - 1] * (1.0 + fund_ret[i])

        for i, dt in enumerate(dates):
            records.append({
                "date":           dt,
                "fund_name":      fname,
                "amc":            amc,
                "category":       cat,
                "expense_ratio":  expense_ratio,
                "nav":            round(float(nav[i]), 4),
                "monthly_return": round(float(fund_ret[i]), 6),
            })

    df = pd.DataFrame(records)
    log.info("Fund NAV simulated: %d funds × %d months = %d rows",
             len(FUND_CATALOGUE), n, len(df))
    return df


# ══════════════════════════════════════════════════════════════════════════════
# SIP investor simulation
# ══════════════════════════════════════════════════════════════════════════════

def simulate_investors(df_nifty: pd.DataFrame, df_nav: pd.DataFrame) -> pd.DataFrame:
    """Simulate 5 000 SIP investors with realistic Indian demographics.

    Churn (status = 'stopped') is driven by three factors:
    - Behavioural: too many missed payments (≥ 3 in a row → likely to stop)
    - Financial: fund experienced a large drawdown (> 20 %)
    - External: market crash during COVID window (2020) for risk-averse investors

    Parameters
    ----------
    df_nifty : pd.DataFrame
        NIFTY 50 monthly data (used to identify market crash windows).
    df_nav : pd.DataFrame
        Fund NAV monthly data (used to compute fund-level max drawdown).

    Returns
    -------
    pd.DataFrame
        5 000 rows × [investor_id, investor_name, city, age_group, fund_name,
        monthly_investment, sip_start_date, sip_end_date, missed_payments,
        status, churn].
    """
    rng = np.random.default_rng(RANDOM_SEED + 2)
    fund_names = [f[0] for f in FUND_CATALOGUE]

    # Pre-compute max-drawdown per fund (over full simulation window)
    fund_max_dd: dict[str, float] = {}
    for fname in fund_names:
        navs = df_nav.loc[df_nav["fund_name"] == fname, "nav"].values
        rolling_peak = np.maximum.accumulate(navs)
        dd = (navs - rolling_peak) / rolling_peak
        fund_max_dd[fname] = float(dd.min())   # most negative = worst drawdown

    # COVID crash window: Jan 2020 – Jun 2020
    covid_start = pd.Timestamp("2020-01-01")
    covid_end   = pd.Timestamp("2020-06-01")

    rows = []
    for i in range(N_INVESTORS):
        investor_id = f"INV{i + 1:05d}"

        # Name
        first = rng.choice(FIRST_NAMES)
        last  = rng.choice(LAST_NAMES)
        name  = f"{first} {last}"

        # Demographics
        city      = rng.choice(CITIES)
        age_group = rng.choice(AGE_GROUPS, p=AGE_WEIGHTS)

        # Fund assignment (weighted toward popular large/mid cap)
        fund_name = rng.choice(fund_names)

        # SIP amount: log-normal centred at ₹5 000, range ₹500 – ₹50 000
        monthly_investment = int(
            np.clip(rng.lognormal(np.log(5_000), 0.8), 500, 50_000)
        )
        # Round to nearest 500
        monthly_investment = max(500, round(monthly_investment / 500) * 500)

        # SIP start date: uniform between Jan 2015 and Dec 2022
        start_offset = int(rng.integers(0, 96))   # 0-95 months after Jan-2015
        sip_start = SIM_START + pd.DateOffset(months=start_offset)

        # Potential tenure up to Dec 2024
        max_tenure = int(
            (SIM_END.year - sip_start.year) * 12
            + (SIM_END.month - sip_start.month)
        )
        if max_tenure <= 0:
            max_tenure = 1

        # Missed payments: Poisson(lambda based on drawdown + income proxy)
        income_proxy = np.log(monthly_investment)            # higher income → fewer misses
        drawdown_factor = abs(fund_max_dd.get(fund_name, 0)) * 20   # 0–4 range
        lam = max(0.2, drawdown_factor - (income_proxy - 6) * 0.3)
        total_missed = int(rng.poisson(lam * max_tenure / 12))
        total_missed = min(total_missed, max_tenure)

        # Churn determination
        # Base churn probability from missed payments ratio and fund drawdown
        miss_ratio = total_missed / max(1, max_tenure)
        fund_dd    = abs(fund_max_dd.get(fund_name, 0))
        churn_prob = (
            0.20                          # baseline 20% churn
            + miss_ratio * 0.40           # more misses → higher churn
            + fund_dd * 0.30              # worse drawdown → higher churn
            + (0.10 if (sip_start <= covid_start <= sip_start + pd.DateOffset(months=max_tenure)) else 0.0)
        )
        # Younger investors slightly more likely to churn
        if age_group == "25-35":
            churn_prob += 0.05
        churn_prob = min(churn_prob, 0.95)

        churned = rng.random() < churn_prob

        # SIP end date (only for stopped investors)
        if churned:
            # Stop date: random in [start + 6 months, end]
            min_tenure = min(6, max_tenure)
            stop_offset = int(rng.integers(min_tenure, max(min_tenure + 1, max_tenure)))
            sip_end = sip_start + pd.DateOffset(months=stop_offset)
            sip_end = min(sip_end, SIM_END)
            status  = "stopped"
        else:
            sip_end = None
            status  = "active"

        rows.append({
            "investor_id":        investor_id,
            "investor_name":      name,
            "city":               city,
            "age_group":          age_group,
            "fund_name":          fund_name,
            "monthly_investment": monthly_investment,
            "sip_start_date":     sip_start.strftime("%Y-%m-%d"),
            "sip_end_date":       sip_end.strftime("%Y-%m-%d") if sip_end else "",
            "missed_payments":    total_missed,
            "status":             status,
            "churn":              int(churned),
        })

    df = pd.DataFrame(rows)
    log.info(
        "Investor dataset simulated: %d investors  active=%d  stopped=%d  churn_rate=%.1f%%",
        len(df),
        (df["status"] == "active").sum(),
        (df["status"] == "stopped").sum(),
        df["churn"].mean() * 100,
    )
    return df


# ══════════════════════════════════════════════════════════════════════════════
# Orchestration
# ══════════════════════════════════════════════════════════════════════════════

def run_simulation(out_dir: str | Path = RAW_DIR, force: bool = False) -> None:
    """Run the full simulation and save all three raw CSVs.

    Parameters
    ----------
    out_dir : str or Path
        Directory where raw CSVs are saved (default: ``<project_root>/data/raw``).
    force : bool
        If False (default), skip simulation when all three files already
        exist.  Set to True to re-generate from scratch.
    """
    out_dir = Path(out_dir)
    nifty_path     = out_dir / "nifty50_monthly.csv"
    nav_path       = out_dir / "fund_nav_monthly.csv"
    investors_path = out_dir / "sip_investors.csv"

    if not force and nifty_path.exists() and nav_path.exists() and investors_path.exists():
        log.info("Simulation files already exist — skipping (use force=True to regenerate)")
        return

    log.info("=== Running data simulation ===")
    df_nifty     = simulate_nifty50()
    df_nav       = simulate_fund_nav(df_nifty)
    df_investors = simulate_investors(df_nifty, df_nav)

    save_csv(df_nifty,     nifty_path)
    save_csv(df_nav,       nav_path)
    save_csv(df_investors, investors_path)
    log.info("=== Simulation complete — 3 raw CSVs written to %s ===", out_dir)


def main() -> None:
    """Entry-point when the script is run directly or as a module."""
    run_simulation()


if __name__ == "__main__":
    main()
