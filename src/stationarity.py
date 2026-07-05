import logging
from statsmodels.tsa.stattools import adfuller

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def run_adf_test(series, label="Series"):
    """
    Run the Augmented Dickey-Fuller test on a series and return results.

    Returns:
        dict with keys: label, adf_statistic, p_value, is_stationary
    """
    series = series.dropna()
    if len(series) < 10:
        raise ValueError(f"{label}: series too short ({len(series)} points) for a reliable ADF test.")

    try:
        result = adfuller(series)
    except Exception as e:
        logger.error(f"{label}: ADF test failed — {e}")
        raise

    adf_stat, p_value = result[0], result[1]
    is_stationary = p_value <= 0.05

    return {
        "label": label,
        "adf_statistic": adf_stat,
        "p_value": p_value,
        "is_stationary": is_stationary,
    }


def run_adf_on_assets(data: dict, price_col="Close"):
    """
    Run ADF tests on both closing price and daily return for every
    asset in `data`. Returns a list of result dicts.
    """
    if not data:
        raise ValueError("`data` dictionary is empty.")

    results = []
    for ticker, df in data.items():
        if price_col not in df.columns:
            logger.warning(f"{ticker}: '{price_col}' column not found; skipping.")
            continue

        results.append(run_adf_test(df[price_col], f"{ticker} — Closing Price"))

        returns = df[price_col].pct_change().dropna() * 100
        results.append(run_adf_test(returns, f"{ticker} — Daily Return"))

    return results


if __name__ == "__main__":
    from data_loader import fetch_data
    from data_cleaning import handle_missing_values

    data = handle_missing_values(fetch_data())
    for r in run_adf_on_assets(data):
        status = "Stationary" if r["is_stationary"] else "Non-stationary"
        print(f"{r['label']}: ADF={r['adf_statistic']:.4f}, p={r['p_value']:.4f} → {status}")