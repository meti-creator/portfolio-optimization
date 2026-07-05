import pandas as pd
from data_loader import fetch_data

def check_missing_values(data: dict):
    """Print missing value counts for each ticker's DataFrame."""
    for ticker, df in data.items():
        missing = df.isnull().sum()
        total_missing = missing.sum()
        print(f"\n{ticker} — missing values:")
        print(missing[missing > 0] if total_missing > 0 else "No missing values")

def check_basic_stats(data: dict):
    """Print summary statistics for each ticker."""
    for ticker, df in data.items():
        print(f"\n{ticker} — summary statistics:")
        print(df.describe())

def handle_missing_values(data: dict):
    """
    Forward-fill missing values (common for financial time series —
    assumes price stayed flat until the next known value), then
    backfill any remaining gaps at the very start of the series.
    """
    cleaned = {}
    for ticker, df in data.items():
        df = df.ffill().bfill()
        cleaned[ticker] = df
    return cleaned

if __name__ == "__main__":
    data = fetch_data()
    check_missing_values(data)
    check_basic_stats(data)
    data = handle_missing_values(data)
    print("\n--- After cleaning ---")
    check_missing_values(data)