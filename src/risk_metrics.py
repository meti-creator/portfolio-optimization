import logging
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

TRADING_DAYS = 252


def calculate_var(daily_returns, confidence=0.95):
    """
    Historical Value at Risk at the given confidence level.
    `daily_returns` should be a decimal series (e.g. 0.02 for 2%), not %.
    """
    daily_returns = daily_returns.dropna()
    if daily_returns.empty:
        raise ValueError("daily_returns is empty — cannot compute VaR.")
    if not (0 < confidence < 1):
        raise ValueError("confidence must be between 0 and 1.")

    percentile = (1 - confidence) * 100
    return np.percentile(daily_returns, percentile)


def calculate_sharpe_ratio(daily_returns, risk_free_rate=0.02, trading_days=TRADING_DAYS):
    """
    Annualized Sharpe Ratio.
    `daily_returns` should be a decimal series (e.g. 0.02 for 2%), not %.
    """
    daily_returns = daily_returns.dropna()
    if daily_returns.empty:
        raise ValueError("daily_returns is empty — cannot compute Sharpe Ratio.")

    daily_std = daily_returns.std()
    if daily_std == 0 or np.isnan(daily_std):
        logger.warning("Standard deviation is zero or NaN — Sharpe Ratio undefined, returning NaN.")
        return float("nan")

    daily_risk_free = risk_free_rate / trading_days
    avg_daily_return = daily_returns.mean()

    return ((avg_daily_return - daily_risk_free) / daily_std) * np.sqrt(trading_days)


def calculate_risk_metrics(data: dict, price_col="Close", confidence=0.95, risk_free_rate=0.02):
    """
    Calculate VaR and Sharpe Ratio for every asset in `data`.
    Returns a dict keyed by ticker: {"var": ..., "sharpe": ...}
    """
    if not data:
        raise ValueError("`data` dictionary is empty.")

    metrics = {}
    for ticker, df in data.items():
        if price_col not in df.columns:
            logger.warning(f"{ticker}: '{price_col}' column not found; skipping.")
            continue
        try:
            daily_returns = df[price_col].pct_change().dropna()
            metrics[ticker] = {
                "var": calculate_var(daily_returns, confidence),
                "sharpe": calculate_sharpe_ratio(daily_returns, risk_free_rate),
            }
        except Exception as e:
            logger.error(f"{ticker}: failed to compute risk metrics — {e}")
            raise

    return metrics


if __name__ == "__main__":
    from data_loader import fetch_data
    from data_cleaning import handle_missing_values

    data = handle_missing_values(fetch_data())
    for ticker, m in calculate_risk_metrics(data).items():
        print(f"{ticker}: 95% VaR={m['var']*100:.2f}%, Sharpe={m['sharpe']:.2f}")