"""
eda.py
------
Exploratory data analysis: price trends, return distributions,
stylized facts of financial returns (volatility clustering, fat tails),
and stationarity testing -- all standard checks before fitting GARCH models.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf
import os

sns.set_theme(style="whitegrid")


def plot_price_history(df: pd.DataFrame, out_dir: str = "outputs/figures"):
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df.index, df["Close"], color="#1f4e79", linewidth=1)
    ax.set_title("BMW Closing Price (1996 - 2026)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (EUR)")
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(f"{out_dir}/price_history.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_returns_distribution(returns: pd.Series, out_dir: str = "outputs/figures"):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    axes[0].plot(returns.index, returns, color="#c0392b", linewidth=0.5)
    axes[0].set_title("Daily Log Returns (%)")
    axes[0].set_xlabel("Date")

    sns.histplot(returns, bins=100, kde=True, ax=axes[1], color="#2c3e50")
    axes[1].set_title("Return Distribution (fat tails vs Normal)")
    axes[1].set_xlabel("Daily log return (%)")

    os.makedirs(out_dir, exist_ok=True)
    fig.tight_layout()
    fig.savefig(f"{out_dir}/returns_distribution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_volatility_clustering(returns: pd.Series, out_dir: str = "outputs/figures"):
    """Squared returns are a common proxy for volatility -- clustering here
    is the visual justification for using a GARCH model."""
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(returns.index, returns ** 2, color="#8e44ad", linewidth=0.5)
    ax.set_title("Squared Returns -- Visual Evidence of Volatility Clustering")
    ax.set_ylabel("Squared daily return (%^2)")
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(f"{out_dir}/volatility_clustering.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_acf_squared_returns(returns: pd.Series, out_dir: str = "outputs/figures", lags: int = 40):
    fig, ax = plt.subplots(figsize=(10, 4))
    plot_acf(returns ** 2, lags=lags, ax=ax)
    ax.set_title("ACF of Squared Returns (autocorrelation = GARCH effects present)")
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(f"{out_dir}/acf_squared_returns.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def stationarity_report(returns: pd.Series) -> dict:
    """Augmented Dickey-Fuller test -- confirms returns are stationary
    (a precondition for fitting GARCH models), unlike raw price levels."""
    result = adfuller(returns.dropna())
    return {
        "adf_statistic": result[0],
        "p_value": result[1],
        "is_stationary": result[1] < 0.05,
        "critical_values": result[4],
    }


def summary_statistics(returns: pd.Series) -> pd.Series:
    stats = returns.describe()
    stats["skewness"] = returns.skew()
    stats["kurtosis"] = returns.kurtosis()  # excess kurtosis; >0 means fat tails
    stats["annualized_volatility_%"] = returns.std() * np.sqrt(252)
    return stats


def run_full_eda(df: pd.DataFrame, returns: pd.Series, out_dir: str = "outputs/figures"):
    plot_price_history(df, out_dir)
    plot_returns_distribution(returns, out_dir)
    plot_volatility_clustering(returns, out_dir)
    plot_acf_squared_returns(returns, out_dir)
    stats = summary_statistics(returns)
    adf = stationarity_report(returns)
    return stats, adf
