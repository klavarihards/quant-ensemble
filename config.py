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
ZSCORE_WINDOW = 20        # trading days used for rolling hedge ratio, mean & std
ENTRY_THRESHOLD = 2.0     # enter when |z| exceeds this
EXIT_THRESHOLD = 0.5      # exit when |z| falls back below this
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
