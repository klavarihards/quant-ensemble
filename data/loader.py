"""Price data loading with local CSV caching."""

import os
import time
from datetime import datetime

import pandas as pd
import yfinance as yf

import config


def _cache_is_fresh(path, max_age_days):
    if not os.path.exists(path):
        return False
    age_seconds = time.time() - os.path.getmtime(path)
    return age_seconds < max_age_days * 86400


def _download_prices(tickers, start, end):
    raw = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=True,   # adjusted closes
        progress=False,
        group_by="column",
    )

    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"]
    else:
        # Single ticker requests come back without a MultiIndex.
        prices = raw[["Close"]]
        prices.columns = tickers

    prices = prices[sorted(prices.columns)]
    prices.index.name = "Date"
    return prices


def _clean(prices):
    prices = prices.ffill()
    prices = prices.dropna(how="any")
    return prices


def load_prices(cfg=config, force_refresh=False):
    """Return a clean DataFrame of adjusted daily closes, date-indexed.

    Uses data/prices.csv as a cache and only re-downloads when the cache
    is missing, stale (older than cfg.CACHE_MAX_AGE_DAYS), or force_refresh
    is set.
    """
    cache_path = cfg.PRICE_CACHE_FILE
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    if not force_refresh and _cache_is_fresh(cache_path, cfg.CACHE_MAX_AGE_DAYS):
        prices = pd.read_csv(cache_path, index_col="Date", parse_dates=True)
        return _clean(prices)

    prices = _download_prices(cfg.TICKERS, cfg.START_DATE, cfg.END_DATE)
    prices = _clean(prices)
    prices.to_csv(cache_path)
    return prices


if __name__ == "__main__":
    df = load_prices()
    print(df.shape)
    print(df.tail())
