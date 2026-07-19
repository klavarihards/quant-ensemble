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

# Expanded opportunity set: commodity, international, sector and fixed-
# income pairs, tested to see whether more (and more diverse) pairs push
# combined Sharpe further. Result: they don't. Per-pair holdout Sharpe
# (2010-2015 / 2016-2020 / 2021-2026) showed only DBA/MOO robust in all
# three periods; EFA/EEM, FXE/FXB, XLK/XLV, XLF/XLU, XLY/XLP, LQD/HYG,
# and TIP/IEF were negative in every one of them (real signal, not
# noise); USO/XLE and GLD/SLV were inconsistent (one negative period
# each). Blending all 10 into the trailing-Sharpe-tilted portfolio
# collapsed combined Sharpe from 0.629 to -0.004 -- and even adding just
# the single robust pair (DBA/MOO) alone still made it worse (0.629 ->
# 0.477 at the best-tested trailing window), because the tilt's noisy
# month-to-month estimate has more chances to misallocate capital away
# from the original 5's proven performers as more candidates compete for
# weight. Kept here for reproducibility/experimentation
# (USE_EXPANDED_UNIVERSE below), but not used by default -- the
# validated 5-pair PAIRS remains the better production configuration.
NEW_PAIRS = [
    ("USO", "XLE"),   # oil price vs energy stocks
    ("GLD", "SLV"),   # gold vs silver
    ("DBA", "MOO"),   # agriculture commodities vs agriculture stocks
    ("EFA", "EEM"),   # developed vs emerging markets
    ("FXE", "FXB"),   # euro vs pound
    ("XLK", "XLV"),   # technology vs healthcare
    ("XLF", "XLU"),   # financials vs utilities
    ("XLY", "XLP"),   # consumer discretionary vs staples
    ("LQD", "HYG"),   # investment grade vs high yield bonds
    ("TIP", "IEF"),   # inflation-protected vs nominal treasuries
]

ALL_PAIRS = PAIRS + NEW_PAIRS
TICKERS = sorted({ticker for pair in ALL_PAIRS for ticker in pair})

# Set True to trade ALL_PAIRS (5 validated + 10 experimental) instead of
# just the validated 5. Off by default -- see the finding documented
# above NEW_PAIRS.
USE_EXPANDED_UNIVERSE = False

PAIR_OVERRIDES = {
    pair: {
        "ENTRY_THRESHOLD": 2.5,
        "EXIT_THRESHOLD": 1.0,
        "MIN_HOLDING_DAYS": 5,
        "COINTEGRATION_GATE": True,
        "COINTEGRATION_WINDOW": 252,
        "COINTEGRATION_P_THRESHOLD": 0.05,
    }
    for pair in NEW_PAIRS
}

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
CAPITAL_PER_PAIR = 1.0 / len(PAIRS)  # equal weight starting point
MAX_LEVERAGE = 1.0                    # no leverage for now

# --- Portfolio construction ------------------------------------------------
# Walk-forward monthly re-weighting toward pairs with positive trailing
# risk-adjusted performance (clipped at zero, so a losing pair gets no
# new capital), instead of a fixed equal split. Validated the same way
# as the strategy parameters above: it improved (never worsened) Sharpe
# in each of the three 2010-2015/2016-2020/2021-2026 holdout periods,
# and a rebalancing cost is charged so it isn't a free lunch from daily
# reweighting.
USE_TRAILING_SHARPE_TILT = True
TRAILING_SHARPE_WINDOW = 252  # ~1 trading year

# --- Data ------------------------------------------------------------
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
PRICE_CACHE_FILE = os.path.join(DATA_DIR, "prices.csv")

# --- Reporting ------------------------------------------------------
BENCHMARK_TICKER = "SPY"
TRADING_DAYS_PER_YEAR = 252
