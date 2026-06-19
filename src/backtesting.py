"""
backtesting.py
---------------
Backtests the VaR model the way a real risk desk would: count how often
actual losses exceeded the predicted VaR threshold ("breaches"), and test
whether the breach rate matches the expected rate using Kupiec's
Proportion-of-Failures (POF) test.
"""

import numpy as np
import pandas as pd
from scipy import stats


def count_var_breaches(actual_returns: pd.Series, var_series: pd.Series) -> pd.Series:
    """
    A breach occurs when the realized loss (negative return) exceeds
    the predicted VaR. Returns are in % (positive = gain, negative = loss).
    """
    aligned_actual, aligned_var = actual_returns.align(var_series, join="inner")
    breaches = (-aligned_actual) > aligned_var
    return breaches


def kupiec_pof_test(breaches: pd.Series, confidence: float = 0.95) -> dict:
    """
    Kupiec's Proportion-of-Failures test: checks whether the observed
    breach rate is statistically consistent with the expected
    (1 - confidence) breach rate, under a likelihood-ratio test.
    """
    n = len(breaches)
    x = breaches.sum()  # number of breaches
    p_expected = 1 - confidence
    p_observed = x / n if n > 0 else 0

    if x == 0 or x == n:
        lr_stat = np.nan
        p_value = np.nan
    else:
        lr_stat = -2 * (
            (n - x) * np.log(1 - p_expected) + x * np.log(p_expected)
            - (n - x) * np.log(1 - p_observed) - x * np.log(p_observed)
        )
        p_value = 1 - stats.chi2.cdf(lr_stat, df=1)

    return {
        "n_observations": n,
        "n_breaches": int(x),
        "expected_breach_rate_%": round(p_expected * 100, 2),
        "observed_breach_rate_%": round(p_observed * 100, 2),
        "lr_statistic": lr_stat,
        "p_value": p_value,
        "model_accepted_at_5%": (p_value > 0.05) if not np.isnan(p_value) else None,
    }


def backtest_summary(actual_returns: pd.Series, var_series: pd.Series,
                      confidence: float = 0.95) -> dict:
    breaches = count_var_breaches(actual_returns, var_series)
    return kupiec_pof_test(breaches, confidence)
