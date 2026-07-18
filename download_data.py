"""Run this on a machine with real internet access to Yahoo Finance.

It downloads adjusted daily closes for every ticker in config.PAIRS over
the configured backtest window and writes them to data/prices.csv. Copy
that file into this project's data/ folder (e.g. upload it here) so the
sandbox can run the full backtest without needing network access itself.

Usage:
    pip install -r requirements.txt
    python3 download_data.py
"""

import config
from data.loader import refresh_cache


def main():
    print(f"Downloading {config.TICKERS} from {config.START_DATE} to {config.END_DATE} ...")
    prices = refresh_cache(config)
    print(f"Saved {prices.shape[0]} rows x {prices.shape[1]} tickers to {config.PRICE_CACHE_FILE}")
    print(f"Date range: {prices.index.min().date()} -> {prices.index.max().date()}")
    print(prices.tail())


if __name__ == "__main__":
    main()
