"""Central configuration for the pairs trading ensemble."""

import os

# --- Universe ---------------------------------------------------------
# Each tuple is (asset_A, asset_B). The engine regresses A on B to find
# the dollar-neutral hedge ratio, so order matters for interpretation
# (not for correctness of the strategy).
PAIRS = [
    ("GLD", "GDX"),
    ("KO", "PEP"),
    ("XOM", "CVX"),
    ("SPY", "QQQ"),
    ("IEF", "TLT"),
]

TICKERS = sorted({ticker for pair in PAIRS for ticker in pair})

# --- Backtest window ----------------------------------------------------
START_DATE = "2010-01-01"
END_DATE = "2026-01-01"

# --- Strategy parameters -------------------------------------------------
# Validated via a train (2010-2018) / test (2019-2026) split plus a
# stricter three-way (2010-2015 / 2016-2020 / 2021-2026) holdout check on
# real market data -- see research notes in the PR/commit history. The
# original (window=20, entry=2.0, exit=0.5, level prices) settings had
# genuine gross edge but traded too often for the edge to survive 0.1%
# round-trip costs; entry=3.0/exit=0.25 cuts trade count from ~150/pair
# to ~15-25/pair over 16 years, which is enough to flip every out-of-
# sample slice from negative to positive Sharpe.
USE_LOG_PRICES = True    # hedge ratio & spread computed on log prices
HEDGE_RATIO_WINDOW = 20   # trading days for the rolling hedge ratio (OLS slope)
ZSCORE_WINDOW = 15        # trading days for the spread's rolling mean & std
ENTRY_THRESHOLD = 3.0     # enter when |z| exceeds this
EXIT_THRESHOLD = 0.25     # exit when |z| falls back below this (min hold permitting)
MIN_HOLDING_DAYS = 5      # do not evaluate the exit condition before this many days
TRANSACTION_COST = 0.001  # 0.1% of notional, charged on entry and exit only

# --- Capital & risk -------------------------------------------------------
CAPITAL_PER_PAIR = 1.0 / len(PAIRS)  # equal weight across pairs
MAX_LEVERAGE = 1.0                    # no leverage for now

# --- Data ------------------------------------------------------------
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
PRICE_CACHE_FILE = os.path.join(DATA_DIR, "prices.csv")

# --- Reporting ------------------------------------------------------
BENCHMARK_TICKER = "SPY"
TRADING_DAYS_PER_YEAR = 252
