import yfinance as yf

from data_loader import fetch_data

df = yf.download("SPY", start="2015-01-01", end="2026-06-30", auto_adjust=True)
print(df.shape)
print(df.head())

