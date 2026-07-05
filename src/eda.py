import logging
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def plot_closing_prices(data: dict, price_col="Close", save_path=None):
    """Plot closing price over time for each asset in `data`."""
    if not data:
        raise ValueError("`data` dictionary is empty.")

    fig, axes = plt.subplots(len(data), 1, figsize=(12, 4 * len(data)), sharex=True)
    if len(data) == 1:
        axes = [axes]

    for ax, (ticker, df) in zip(axes, data.items()):
        if price_col not in df.columns:
            logger.warning(f"{ticker}: '{price_col}' column not found; skipping.")
            continue
        ax.plot(df.index, df[price_col], label=ticker)
        ax.set_title(f"{ticker} — Closing Price Over Time")
        ax.set_ylabel("Price ($)")
        ax.legend()

    plt.xlabel("Date")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    return fig


def plot_daily_returns(data: dict, price_col="Close", save_path=None):
    """Plot daily percentage change for each asset in `data`."""
    if not data:
        raise ValueError("`data` dictionary is empty.")

    fig, axes = plt.subplots(len(data), 1, figsize=(12, 4 * len(data)), sharex=True)
    if len(data) == 1:
        axes = [axes]

    for ax, (ticker, df) in zip(axes, data.items()):
        if price_col not in df.columns:
            logger.warning(f"{ticker}: '{price_col}' column not found; skipping.")
            continue
        daily_return = df[price_col].pct_change() * 100
        ax.plot(df.index, daily_return, linewidth=0.7)
        ax.set_title(f"{ticker} — Daily % Change")
        ax.set_ylabel("Return (%)")

    plt.xlabel("Date")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    return fig


def plot_rolling_volatility(data: dict, price_col="Close", window=30, save_path=None):
    """Plot rolling standard deviation of daily returns for each asset."""
    if not data:
        raise ValueError("`data` dictionary is empty.")
    if window < 2:
        raise ValueError("`window` must be at least 2.")

    fig, axes = plt.subplots(len(data), 1, figsize=(12, 4 * len(data)), sharex=True)
    if len(data) == 1:
        axes = [axes]

    for ax, (ticker, df) in zip(axes, data.items()):
        if price_col not in df.columns:
            logger.warning(f"{ticker}: '{price_col}' column not found; skipping.")
            continue
        daily_return = df[price_col].pct_change() * 100
        rolling_std = daily_return.rolling(window=window).std()
        ax.plot(df.index, rolling_std, color="crimson")
        ax.set_title(f"{ticker} — Rolling {window}-Day Volatility")
        ax.set_ylabel("Volatility (%)")

    plt.xlabel("Date")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    return fig


def detect_outliers(data: dict, price_col="Close", z_threshold=3):
    """
    Detect outlier days using the Z-score method.
    Returns a dict keyed by ticker: DataFrame of outlier rows.
    """
    if not data:
        raise ValueError("`data` dictionary is empty.")

    from scipy import stats
    import pandas as pd

    outliers = {}
    for ticker, df in data.items():
        if price_col not in df.columns:
            logger.warning(f"{ticker}: '{price_col}' column not found; skipping.")
            continue
        daily_return = df[price_col].pct_change().dropna() * 100
        if daily_return.std() == 0:
            logger.warning(f"{ticker}: zero variance in returns; no outliers computed.")
            continue
        z_scores = stats.zscore(daily_return)
        z_series = pd.Series(z_scores, index=daily_return.index)
        outliers[ticker] = df.loc[z_series[z_series.abs() > z_threshold].index]

    return outliers