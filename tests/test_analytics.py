import pandas as pd
import numpy as np
import pytest
import sys, os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from risk_metrics import calculate_var, calculate_sharpe_ratio
from stationarity import run_adf_test


def make_returns(n=300, seed=1):
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(0, 0.02, n))


def test_var_returns_negative_number():
    returns = make_returns()
    var = calculate_var(returns)
    assert var < 0


def test_var_rejects_empty_series():
    with pytest.raises(ValueError):
        calculate_var(pd.Series(dtype=float))


def test_var_rejects_bad_confidence():
    with pytest.raises(ValueError):
        calculate_var(make_returns(), confidence=1.5)


def test_sharpe_ratio_is_finite():
    sharpe = calculate_sharpe_ratio(make_returns())
    assert np.isfinite(sharpe)


def test_sharpe_ratio_zero_std_returns_nan():
    flat_returns = pd.Series([0.0] * 50)
    assert np.isnan(calculate_sharpe_ratio(flat_returns))


def test_adf_detects_stationary_noise():
    result = run_adf_test(make_returns(500), "synthetic noise")
    assert result["is_stationary"] is True


def test_adf_rejects_short_series():
    with pytest.raises(ValueError):
        run_adf_test(pd.Series([1, 2, 3]), "too short")

        