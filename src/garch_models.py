"""
garch_models.py
----------------
Fits and compares three volatility models commonly used in industry risk teams:

  1. GARCH(1,1)      -- symmetric, baseline volatility model
  2. GJR-GARCH(1,1)  -- captures the leverage effect (bad news raises vol more
                          than good news of the same size)
  3. EGARCH(1,1)     -- log-volatility specification, also captures asymmetry
                          and guarantees positive variance without parameter constraints

Model selection is done via AIC/BIC, then the best model is used for
volatility forecasting and downstream Value-at-Risk estimation.
"""

import pandas as pd
import numpy as np
from arch import arch_model


def fit_garch(returns: pd.Series, p: int = 1, q: int = 1, dist: str = "t"):
    """Standard symmetric GARCH(p,q) with Student-t errors (fat tails)."""
    model = arch_model(returns, vol="GARCH", p=p, q=q, dist=dist)
    return model.fit(disp="off")


def fit_gjr_garch(returns: pd.Series, p: int = 1, o: int = 1, q: int = 1, dist: str = "t"):
    """GJR-GARCH adds an asymmetry term 'o' for the leverage effect."""
    model = arch_model(returns, vol="GARCH", p=p, o=o, q=q, dist=dist)
    return model.fit(disp="off")


def fit_egarch(returns: pd.Series, p: int = 1, o: int = 1, q: int = 1, dist: str = "t"):
    """EGARCH models log-variance; naturally asymmetric and unconstrained."""
    model = arch_model(returns, vol="EGARCH", p=p, o=o, q=q, dist=dist)
    return model.fit(disp="off")


def compare_models(returns: pd.Series) -> pd.DataFrame:
    """Fit all three models and compare on AIC / BIC / log-likelihood."""
    results = {
        "GARCH(1,1)": fit_garch(returns),
        "GJR-GARCH(1,1,1)": fit_gjr_garch(returns),
        "EGARCH(1,1,1)": fit_egarch(returns),
    }

    comparison = pd.DataFrame({
        name: {
            "AIC": res.aic,
            "BIC": res.bic,
            "Log-Likelihood": res.loglikelihood,
        }
        for name, res in results.items()
    }).T.sort_values("AIC")

    return comparison, results


def forecast_volatility(fitted_model, horizon: int = 30) -> pd.DataFrame:
    """
    Forecast conditional volatility 'horizon' days ahead.
    Returns annualized volatility forecast for readability.

    EGARCH has no closed-form multi-step forecast, so we use Monte Carlo
    simulation (method='simulation') for horizon > 1; this works for all
    three model types.
    """
    method = "analytic" if horizon == 1 else "simulation"
    forecast = fitted_model.forecast(horizon=horizon, reindex=False, method=method)
    variance_forecast = forecast.variance.iloc[-1]
    daily_vol_pct = np.sqrt(variance_forecast)
    annualized_vol_pct = daily_vol_pct * np.sqrt(252)

    return pd.DataFrame({
        "day_ahead": range(1, horizon + 1),
        "daily_volatility_%": daily_vol_pct.values,
        "annualized_volatility_%": annualized_vol_pct.values,
    })
