"""
returns.py
----------
Computes log returns and realized volatility measures used as inputs
to the GARCH-family models and as ground truth for backtesting.
"""

import numpy as np
import pandas as pd


def compute_log_returns(df: pd.DataFrame, price_col: str = "Close") -> pd.Series:
    """
    Compute daily log returns (in percent) from a price series.
    Percent scale is used because arch_model converges more reliably
    on returns scaled to roughly unit variance.
    """
    log_returns = np.log(df[price_col] / df[price_col].shift(1)) * 100
    return log_returns.dropna()


def compute_realized_volatility(returns: pd.Series, window: int = 21) -> pd.Series:
    """
    Rolling realized volatility (annualized) using a trailing window
    of daily log returns. Default window of 21 trading days ~ 1 month.
    """
    rolling_std = returns.rolling(window=window).std()
    annualized = rolling_std * np.sqrt(252)
    return annualized.dropna()


def train_test_split_returns(returns: pd.Series, test_size: float = 0.2):
    """
    Chronological train/test split (no shuffling -- this is time series data).
    """
    split_idx = int(len(returns) * (1 - test_size))
    train = returns.iloc[:split_idx]
    test = returns.iloc[split_idx:]
    return train, test
