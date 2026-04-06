"""
src/utils/logger.py
-------------------
Shared logging configuration for the SIP churn pipeline.

All modules obtain a logger via ``get_logger(__name__)`` so that log
records carry the originating module path and a consistent timestamp
format.
"""

import logging
import sys


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a logger with a consistent format.

    Parameters
    ----------
    name:
        Typically ``__name__`` of the calling module.
    level:
        Logging level (default: INFO).

    Returns
    -------
    logging.Logger
        Configured logger instance.
    """
    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers when the module is re-imported.
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        logger.addHandler(handler)

    logger.setLevel(level)
    logger.propagate = False
    return logger
