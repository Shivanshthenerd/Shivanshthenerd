"""
src/utils/io_helpers.py
-----------------------
Safe, logged helpers for reading and writing CSV files used
throughout the SIP churn pipeline.
"""

from pathlib import Path

import pandas as pd

from src.utils.logger import get_logger

log = get_logger(__name__)


def read_csv(path: str | Path, **kwargs) -> pd.DataFrame:
    """Read a CSV file into a DataFrame with logging.

    Parameters
    ----------
    path:
        Absolute or relative path to the CSV file.
    **kwargs:
        Extra keyword arguments forwarded to ``pd.read_csv``.

    Returns
    -------
    pd.DataFrame
        Loaded DataFrame.

    Raises
    ------
    FileNotFoundError
        If the file does not exist at *path*.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")

    df = pd.read_csv(path, **kwargs)
    log.info("Loaded %-50s  shape=%s", str(path), df.shape)
    return df


def save_csv(df: pd.DataFrame, path: str | Path, index: bool = False) -> None:
    """Write a DataFrame to CSV, creating parent directories if needed.

    Parameters
    ----------
    df:
        DataFrame to persist.
    path:
        Destination file path.
    index:
        Whether to write the row index (default: False).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=index)
    log.info("Saved  %-50s  shape=%s", str(path), df.shape)
