import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def check_missing_values(data: dict):
    """Print missing value counts for each ticker's DataFrame."""
    if not data:
        raise ValueError("`data` dictionary is empty — nothing to check.")

    for ticker, df in data.items():
        if df is None or df.empty:
            logger.warning(f"{ticker}: DataFrame is empty.")
            continue
        missing = df.isnull().sum()
        total_missing = missing.sum()
        print(f"\n{ticker} — missing values:")
        print(missing[missing > 0] if total_missing > 0 else "No missing values")


def check_basic_stats(data: dict):
    """Print summary statistics for each ticker."""
    if not data:
        raise ValueError("`data` dictionary is empty — nothing to summarize.")

    for ticker, df in data.items():
        if df is None or df.empty:
            logger.warning(f"{ticker}: DataFrame is empty, skipping stats.")
            continue
        print(f"\n{ticker} — summary statistics:")
        print(df.describe())


def handle_missing_values(data: dict):
    """
    Forward-fill missing values, then backfill any remaining gaps at
    the start of the series. Also treats zero trading volume as a
    data anomaly and fills it the same way.

    Raises:
        ValueError: if `data` is empty or a DataFrame lacks a 'Volume' column.
    """
    if not data:
        raise ValueError("`data` dictionary is empty — nothing to clean.")

    cleaned = {}
    for ticker, df in data.items():
        if df is None or df.empty:
            logger.warning(f"{ticker}: DataFrame is empty, skipping cleaning.")
            continue
        try:
            df = df.copy()
            if "Volume" in df.columns:
                df.loc[df["Volume"] == 0, "Volume"] = np.nan
            else:
                logger.warning(f"{ticker}: no 'Volume' column found; skipping zero-volume check.")
            df = df.ffill().bfill()
            cleaned[ticker] = df
        except Exception as e:
            logger.error(f"{ticker}: failed to clean data — {e}")
            raise

    return cleaned