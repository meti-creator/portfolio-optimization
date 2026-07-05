import yfinance as yf
import pandas as pd
import time

TICKERS = ["TSLA", "BND", "SPY"]
START_DATE = "2015-01-01"
END_DATE = "2026-06-30"

def fetch_data(tickers=TICKERS, start=START_DATE, end=END_DATE):
    """
    Fetch historical daily data for all tickers in one batched request,
    then split into a dict of per-ticker DataFrames.
    """
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, group_by="ticker")

    data = {}
    for ticker in tickers:
        df = raw[ticker].copy()
        df["Ticker"] = ticker
        data[ticker] = df
    return data

if __name__ == "__main__":
    data = fetch_data()
    for ticker, df in data.items():
        print(f"\n{ticker} - shape: {df.shape}")
        print(df.head())
        print(df.dtypes)
        