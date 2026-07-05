import yfinance as yf
import pandas as pd
import time
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

TICKERS = ["TSLA", "BND", "SPY"]
START_DATE = "2015-01-01"
END_DATE = "2026-06-30"


def fetch_data(tickers=TICKERS, start=START_DATE, end=END_DATE, max_retries=2):
    """
    Fetch historical daily data for all tickers in one batched request,
    then split into a dict of per-ticker DataFrames.

    Raises:
        ValueError: if tickers list is empty or dates are invalid.
        RuntimeError: if data cannot be fetched after retries.
    """
    if not tickers:
        raise ValueError("`tickers` must be a non-empty list of ticker symbols.")

    try:
        pd.Timestamp(start)
        pd.Timestamp(end)
    except (ValueError, TypeError) as e:
        raise ValueError(f"Invalid start/end date: {e}")

    attempt = 0
    raw = None
    while attempt <= max_retries:
        try:
            raw = yf.download(tickers, start=start, end=end, auto_adjust=True, group_by="ticker")
            if raw is None or raw.empty:
                raise ValueError("yfinance returned an empty DataFrame.")
            break
        except Exception as e:
            attempt += 1
            logger.warning(f"Fetch attempt {attempt} failed: {e}")
            if attempt > max_retries:
                raise RuntimeError(
                    f"Failed to fetch data for {tickers} after {max_retries + 1} attempts."
                ) from e
            time.sleep(3)

    data = {}
    for ticker in tickers:
        try:
            df = raw[ticker].copy()
            if df.empty:
                logger.warning(f"No data returned for {ticker}; skipping.")
                continue
            df["Ticker"] = ticker
            data[ticker] = df
        except KeyError:
            logger.warning(f"Ticker {ticker} missing from fetched data; skipping.")

    if not data:
        raise RuntimeError("No data was successfully fetched for any ticker.")

    return data


if __name__ == "__main__":
    data = fetch_data()
    for ticker, df in data.items():
        print(f"\n{ticker} - shape: {df.shape}")
        print(df.head())