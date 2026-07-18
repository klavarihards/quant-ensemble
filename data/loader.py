"""Price data loading with local CSV caching.

Supports three ways of getting price data, in order of preference:
  1. An explicit manual CSV path (e.g. a file you downloaded on a machine
     with real internet access and uploaded here).
  2. A fresh local cache at cfg.PRICE_CACHE_FILE.
  3. A live yfinance download, which is also written to the cache.

If a live download isn't possible (e.g. this sandbox's network policy
blocks Yahoo Finance) and no cache exists yet, load_prices raises a
clear error pointing at download_data.py instead of failing silently.
"""

import os
import time

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


def _read_csv(path):
    return pd.read_csv(path, index_col="Date", parse_dates=True)


def _clean(prices):
    prices = prices.ffill()
    prices = prices.dropna(how="any")
    return prices


def _validate_tickers(prices, tickers):
    missing = sorted(set(tickers) - set(prices.columns))
    if missing:
        raise ValueError(
            f"Price data is missing required tickers: {missing}. "
            f"Available columns: {sorted(prices.columns)}"
        )


def refresh_cache(cfg=config):
    """Download fresh prices from yfinance and write them to the cache.

    Used directly by download_data.py (run on a machine with real
    internet access) and internally by load_prices() when the cache is
    stale or missing.
    """
    cache_path = cfg.PRICE_CACHE_FILE
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    prices = _download_prices(cfg.TICKERS, cfg.START_DATE, cfg.END_DATE)
    prices = _clean(prices)
    if prices.empty:
        raise ValueError("Downloaded price data is empty.")

    _validate_tickers(prices, cfg.TICKERS)
    prices.to_csv(cache_path)
    return prices


def load_prices(cfg=config, force_refresh=False, manual_csv_path=None):
    """Return a clean DataFrame of adjusted daily closes, date-indexed.

    - If manual_csv_path is given, it is loaded directly and promoted to
      the cache (useful for a CSV downloaded elsewhere and uploaded here).
    - Otherwise, data/prices.csv is used as a cache and only refreshed
      via yfinance when missing, stale (older than
      cfg.CACHE_MAX_AGE_DAYS), or force_refresh is set.
    - If a live download fails (e.g. network access is blocked) but a
      cache already exists, that cache is used as a fallback instead of
      raising.
    """
    cache_path = cfg.PRICE_CACHE_FILE
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    if manual_csv_path:
        prices = _clean(_read_csv(manual_csv_path))
        _validate_tickers(prices, cfg.TICKERS)
        prices.to_csv(cache_path)
        return prices

    if not force_refresh and _cache_is_fresh(cache_path, cfg.CACHE_MAX_AGE_DAYS):
        prices = _clean(_read_csv(cache_path))
        _validate_tickers(prices, cfg.TICKERS)
        return prices

    try:
        return refresh_cache(cfg)
    except Exception as exc:
        if os.path.exists(cache_path):
            print(
                f"Warning: live download failed ({exc}); "
                f"falling back to existing cache at {cache_path}"
            )
            prices = _clean(_read_csv(cache_path))
            _validate_tickers(prices, cfg.TICKERS)
            return prices

        raise RuntimeError(
            "Could not download price data (no network access to Yahoo "
            "Finance?) and no cached CSV was found at "
            f"{cache_path}. Run download_data.py on a machine with "
            "internet access, then upload the resulting prices.csv to "
            f"{cache_path} (or pass manual_csv_path=... to load_prices)."
        ) from exc


if __name__ == "__main__":
    import sys

    manual_path = sys.argv[1] if len(sys.argv) > 1 else None
    df = load_prices(manual_csv_path=manual_path)
    print(df.shape)
    print(df.tail())
