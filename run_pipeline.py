"""
run_pipeline.py
---------------
Top-level orchestration script for the Indian SIP churn feature pipeline.

Stages
------
1. Ingestion  — load raw CSVs via ``src.ingestion.data_loader``
2. Cleaning   — impute, cap, dedup, merge via ``src.ingestion.data_cleaning``
3. Features   — engineer all features via ``src.features.feature_engineering``

Usage
-----
    python run_pipeline.py

Outputs
-------
data/processed/funds_clean.csv
data/processed/monthly_clean.csv
data/processed/merged_panel.csv
data/features/features.csv          ← primary output for modelling
"""

import sys
from pathlib import Path

# Ensure the project root is on sys.path so src.* imports resolve correctly
# when the script is invoked from any working directory.
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ingestion.data_loader import load_all
from src.ingestion.data_cleaning import run_cleaning
from src.features.feature_engineering import run_feature_engineering
from src.utils.logger import get_logger

log = get_logger(__name__)


def main() -> None:
    """Execute the full three-stage SIP churn feature pipeline."""
    log.info("=" * 60)
    log.info("Indian SIP Churn — Feature Pipeline")
    log.info("=" * 60)

    # ── Stage 1: Ingestion ────────────────────────────────────────────────────
    log.info("STAGE 1 / 3  Ingestion")
    df_funds, df_monthly = load_all()

    # ── Stage 2: Cleaning ─────────────────────────────────────────────────────
    log.info("STAGE 2 / 3  Cleaning")
    _, _, df_merged = run_cleaning(df_funds, df_monthly)

    # ── Stage 3: Feature Engineering ─────────────────────────────────────────
    log.info("STAGE 3 / 3  Feature Engineering")
    df_features = run_feature_engineering(df_merged)

    log.info("=" * 60)
    log.info("Pipeline complete.")
    log.info("  Features file : data/features/features.csv")
    log.info("  Shape         : %s", df_features.shape)
    log.info("  Churn rate    : %.2f%%", df_features["churn"].mean() * 100)
    log.info("=" * 60)


if __name__ == "__main__":
    main()
