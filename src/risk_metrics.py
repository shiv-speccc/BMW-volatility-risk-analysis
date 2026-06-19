"""
risk_metrics.py
----------------
Translates conditional volatility forecasts into the risk metrics that
trading desks and risk teams actually report: Value-at-Risk (VaR) and
Expected Shortfall (Conditional VaR).
"""

import numpy as np
import pandas as pd
from scipy import stats


def parametric_var(mu: float, sigma: float, confidence: float = 0.95,
                    dist: str = "normal", nu: float = None) -> float:
    """
    One-day-ahead parametric VaR (as a positive % loss figure) given a
    mean and conditional volatility forecast.

    dist='t' uses a Student-t quantile (matches GARCH fit distribution,
    accounts for fat tails); dist='normal' uses the Gaussian quantile.
    """
    alpha = 1 - confidence
    if dist == "t" and nu is not None:
        # Standardize t quantile to unit variance
        q = stats.t.ppf(alpha, df=nu) / np.sqrt(nu / (nu - 2))
    else:
        q = stats.norm.ppf(alpha)
    var = -(mu + q * sigma)
    return var


def expected_shortfall(mu: float, sigma: float, confidence: float = 0.95,
                        dist: str = "normal", nu: float = None) -> float:
    """
    Expected Shortfall (a.k.a. Conditional VaR): the expected loss
    *given* that the loss exceeds the VaR threshold. Preferred by
    regulators (Basel III) because it captures tail severity, not just
    the threshold.
    """
    alpha = 1 - confidence
    if dist == "t" and nu is not None:
        q = stats.t.ppf(alpha, df=nu)
        es_factor = (stats.t.pdf(q, df=nu) / alpha) * ((nu + q ** 2) / (nu - 1))
        es_factor = es_factor / np.sqrt(nu / (nu - 2))
    else:
        q = stats.norm.ppf(alpha)
        es_factor = stats.norm.pdf(q) / alpha
    es = -(mu + (-es_factor) * sigma)
    return es


def rolling_var_series(cond_vol: pd.Series, mu: float = 0.0,
                        confidence: float = 0.95, dist: str = "normal",
                        nu: float = None) -> pd.Series:
    """Apply parametric_var across a full series of conditional volatility."""
    return cond_vol.apply(lambda s: parametric_var(mu, s, confidence, dist, nu))
