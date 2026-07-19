"""Download SPY/VIX history and compute the variance risk premium (VRP).

VRP = VIX - realized_vol_21d: the gap between the market's implied
volatility (VIX, forward-looking) and what volatility actually
realised over the trailing month. A persistently positive VRP is the
premium that volatility-selling strategies try to harvest.
"""

import os

import numpy as np
import pandas as pd
import yfinance as yf

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
OUTPUT_FILE = os.path.join(DATA_DIR, "vix_spy_data.csv")

START_DATE = "2004-01-01"
END_DATE = "2026-01-01"
REALIZED_VOL_WINDOW = 21
TRADING_DAYS_PER_YEAR = 252


def _as_series(closes, name):
    # yf.download(...)["Close"] is a Series for a single ticker on most
    # yfinance versions, but a one-column DataFrame on others -- handle both.
    if isinstance(closes, pd.DataFrame):
        closes = closes.iloc[:, 0]
    closes = closes.copy()
    closes.name = name
    return closes


def download_spy_vix(start=START_DATE, end=END_DATE):
    """Download adjusted SPY closes and VIX index levels from yfinance."""
    spy = yf.download("SPY", start=start, end=end, auto_adjust=True, progress=False)["Close"]
    vix = yf.download("^VIX", start=start, end=end, auto_adjust=True, progress=False)["Close"]
    return _as_series(spy, "SPY"), _as_series(vix, "VIX")


def compute_realized_vol(spy_prices, window=REALIZED_VOL_WINDOW):
    """Annualised realised volatility (%) from trailing daily log returns."""
    log_returns = np.log(spy_prices / spy_prices.shift(1))
    return log_returns.rolling(window).std() * np.sqrt(TRADING_DAYS_PER_YEAR) * 100


def build_dataset(start=START_DATE, end=END_DATE, window=REALIZED_VOL_WINDOW):
    spy, vix = download_spy_vix(start, end)
    df = pd.concat([spy, vix], axis=1)
    df.columns = ["SPY", "VIX"]
    df = df.dropna(how="any")
    df["realized_vol_21d"] = compute_realized_vol(df["SPY"], window)
    df["VRP"] = df["VIX"] - df["realized_vol_21d"]
    return df.dropna()


def save_dataset(df, path=OUTPUT_FILE):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path)


def load_or_build(manual_csv_path=None, force_refresh=False):
    """Load the VIX/SPY/VRP dataset, downloading or rebuilding as needed.

    manual_csv_path: a CSV with at least Date, SPY, VIX columns (e.g.
    downloaded elsewhere and uploaded here); realized_vol_21d and VRP
    are (re)computed from SPY/VIX if not already present.
    """
    if manual_csv_path:
        df = pd.read_csv(manual_csv_path, index_col=0, parse_dates=True)
        if "realized_vol_21d" not in df.columns or "VRP" not in df.columns:
            df["realized_vol_21d"] = compute_realized_vol(df["SPY"])
            df["VRP"] = df["VIX"] - df["realized_vol_21d"]
            df = df.dropna()
        save_dataset(df)
        return df

    if os.path.exists(OUTPUT_FILE) and not force_refresh:
        return pd.read_csv(OUTPUT_FILE, index_col=0, parse_dates=True)

    df = build_dataset()
    save_dataset(df)
    return df


def print_summary(df):
    print(f"Rows: {len(df)}  ({df.index.min().date()} -> {df.index.max().date()})")
    print(f"Average VIX:              {df['VIX'].mean():.2f}")
    print(f"Average realized vol 21d: {df['realized_vol_21d'].mean():.2f}")
    print(f"Average VRP:              {df['VRP'].mean():.2f}")
    print(f"% of days VRP positive:   {(df['VRP'] > 0).mean():.1%}")


if __name__ == "__main__":
    import sys

    manual_path = sys.argv[1] if len(sys.argv) > 1 else None
    dataset = load_or_build(manual_csv_path=manual_path)
    print_summary(dataset)
