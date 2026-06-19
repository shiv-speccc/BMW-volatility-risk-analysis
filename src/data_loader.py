"""
data_loader.py
----------------
Handles loading and basic cleaning of the BMW OHLCV dataset.
"""

import pandas as pd
import os


def load_bmw_data(filepath: str = "data/bmw.csv") -> pd.DataFrame:
    """
    Load the BMW stock dataset from CSV, parse dates, sort chronologically,
    and run basic sanity checks.

    Parameters
    ----------
    filepath : str
        Path to the bmw.csv file.

    Returns
    -------
    pd.DataFrame
        Cleaned dataframe indexed by Date with columns:
        Open, High, Low, Close, Volume
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Could not find dataset at {filepath}")

    df = pd.read_csv(filepath, parse_dates=["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    df = df.set_index("Date")

    # Basic cleaning
    df = df[~df.index.duplicated(keep="first")]
    df = df.dropna(subset=["Open", "High", "Low", "Close", "Volume"])

    # Sanity checks
    assert (df["High"] >= df["Low"]).all(), "Found rows where High < Low"
    assert (df[["Open", "High", "Low", "Close"]] > 0).all().all(), "Found non-positive prices"

    return df


if __name__ == "__main__":
    data = load_bmw_data()
    print(data.shape)
    print(data.head())
    print(data.tail())
