# Portfolio Optimization Project

A data-driven portfolio analysis and forecasting project for **GMF (Guide Me in Finance) Investments**, a personalized portfolio advisory firm. This project builds the analytical pipeline — from raw market data to a forecast-informed, optimized, and backtested portfolio recommendation — using three representative assets:

- **TSLA** — high-growth, high-risk individual stock
- **BND** — low-risk bond ETF (capital preservation)
- **SPY** — diversified, moderate-risk market benchmark

## Project Structure
## Setup

```powershell
# Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\Activate.ps1

# Install dependencies
python -m pip install -r requirements.txt
```

## Tasks

- [x] **Task 1 — Preprocess and Explore the Data**
  Data extraction (yfinance), cleaning, EDA, stationarity testing (ADF), volatility analysis, and risk metrics (VaR, Sharpe Ratio) for TSLA, BND, and SPY.
- [x] **Task 2 — Build Time Series Forecasting Models** *(initial progress)*
  ARIMA and LSTM models implemented, trained, and evaluated against a naive baseline.
- [ ] **Task 3 — Forecast Future Market Trends**
  Generate forward-looking TSLA forecasts with confidence intervals.
- [ ] **Task 4 — Optimize Portfolio Based on Forecast**
  Combine forecasts with historical risk/return profiles to recommend an optimal TSLA/BND/SPY allocation (Efficient Frontier).
- [ ] **Task 5 — Strategy Backtesting**
  Backtest the recommended portfolio against a benchmark allocation.

## Reports

See `reports/` for the interim report covering Task 1 and Task 2 progress.

## Data Source

Historical daily price data (2015-01-01 to 2026-06-30) via [yfinance](https://pypi.org/project/yfinance/).
