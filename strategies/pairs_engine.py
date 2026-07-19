"""Single-pair statistical arbitrage engine."""

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller


class PairsEngine:
    """Dollar-neutral mean-reversion engine for a single pair.

    Given two price series (asset A and asset B), this:
      1. Estimates a rolling OLS hedge ratio (beta of A on B), optionally
         on log prices, over cfg.HEDGE_RATIO_WINDOW.
      2. Builds the spread = A - hedge_ratio * B (in the same, optionally
         logged, price space) and its rolling Z-score over
         cfg.ZSCORE_WINDOW -- a separate, shorter window than the hedge
         ratio's so the entry/exit signal stays responsive while the
         hedge estimate stays stable.
      3. Runs a stateful signal loop over the Z-score to decide when to
         enter/exit, freezing the hedge ratio at the moment of entry so
         it is never re-estimated mid-trade, and refusing to hold a
         position for less than cfg.MIN_HOLDING_DAYS (avoids paying
         entry+exit costs on trades that reverse within a few noisy days).
      4. Shifts signals by one day (trade on next bar, no look-ahead)
         and converts the held position into P&L, net of transaction
         costs charged only on entry and exit.

    With cfg.USE_LOG_PRICES, steps 1-2 operate on log(A)/log(B): the
    resulting spread and hedge ratio are in log-return space, so P&L is
    already a normalised (unit-capital) return and no separate dollar
    capital base is needed. With it off, P&L is a literal dollar P&L on
    A - hedge_ratio * B, normalised by the dollar capital committed
    (A + |hedge_ratio| * B).

    With cfg.COINTEGRATION_GATE, a walk-forward cointegration filter is
    added: once a month, an ADF test runs on the trailing
    cfg.COINTEGRATION_WINDOW days of the spread (strictly prior data
    only, no look-ahead), and new entries are only allowed while the
    most recent test's p-value is below cfg.COINTEGRATION_P_THRESHOLD.
    Existing trades still exit normally regardless of gate state -- the
    gate only blocks opening new positions on a pair that currently
    looks non-stationary.
    """

    def __init__(self, price_a, price_b, config, name=None):
        self.price_a = price_a.astype(float)
        self.price_b = price_b.astype(float)
        self.config = config
        self.name = name or f"{price_a.name}/{price_b.name}"
        self.result = None

    # -- hedge ratio & signal construction -------------------------------

    def _rolling_hedge_ratio(self, a, b, window):
        # Rolling OLS slope of A on B (with intercept) is exactly
        # Cov(A, B) / Var(B) over the same window -- this avoids running
        # an explicit regression at every bar while remaining identical
        # to OLS.
        cov = a.rolling(window).cov(b)
        var = b.rolling(window).var()
        return cov / var

    def _zscore(self, spread, window):
        mean = spread.rolling(window).mean()
        std = spread.rolling(window).std()
        return (spread - mean) / std

    def _cointegration_gate(self, spread, window, p_threshold):
        """Walk-forward monthly ADF gate: True while the pair currently
        looks cointegrated, based only on data strictly before each test
        date. Re-tested once a month; held constant in between."""
        idx = spread.index
        month_starts = set(spread.index.to_series().groupby(idx.to_period("M")).head(1))

        gate = np.zeros(len(idx), dtype=bool)
        last_ok = False
        for i, dt in enumerate(idx):
            if dt in month_starts:
                hist = spread.iloc[:i].tail(window).dropna()
                if len(hist) >= max(60, window // 2):
                    try:
                        p_value = adfuller(hist.to_numpy(), autolag="AIC")[1]
                        last_ok = p_value < p_threshold
                    except Exception:
                        last_ok = False
            gate[i] = last_ok

        return pd.Series(gate, index=idx, name="coint_gate")

    def _generate_positions(self, zscore, hedge_ratio, cfg, gate_ok=None):
        """Stateful loop: no look-ahead, hedge ratio fixed at entry."""
        n = len(zscore)
        z_vals = zscore.to_numpy()
        h_vals = hedge_ratio.to_numpy()
        gate_vals = gate_ok.to_numpy() if gate_ok is not None else np.ones(n, dtype=bool)
        min_hold = getattr(cfg, "MIN_HOLDING_DAYS", 0)

        state = 0
        current_hedge = np.nan
        trade_counter = 0
        days_in_trade = 0

        positions = np.zeros(n)
        hedge_used = np.full(n, np.nan)
        trade_ids = np.full(n, np.nan)

        for i in range(n):
            z = z_vals[i]
            h = h_vals[i]
            if not np.isnan(z):
                if state == 0:
                    # A non-positive hedge ratio means the rolling OLS
                    # estimate has broken down (the two assets no longer
                    # look positively related over the lookback window),
                    # so there is no valid dollar-neutral trade to open
                    # today even if the z-score looks extreme.
                    if not np.isnan(h) and h > 0 and gate_vals[i]:
                        if z > cfg.ENTRY_THRESHOLD:
                            state = -1  # spread too high: short A, long B
                            current_hedge = h
                            trade_counter += 1
                            days_in_trade = 0
                        elif z < -cfg.ENTRY_THRESHOLD:
                            state = 1  # spread too low: long A, short B
                            current_hedge = h
                            trade_counter += 1
                            days_in_trade = 0
                else:
                    days_in_trade += 1
                    if days_in_trade >= min_hold and abs(z) < cfg.EXIT_THRESHOLD:
                        state = 0
                        current_hedge = np.nan
                        days_in_trade = 0

            positions[i] = state
            hedge_used[i] = current_hedge if state != 0 else np.nan
            trade_ids[i] = trade_counter if state != 0 else np.nan

        idx = zscore.index
        return (
            pd.Series(positions, index=idx, name="position_raw"),
            pd.Series(hedge_used, index=idx, name="hedge_used"),
            pd.Series(trade_ids, index=idx, name="trade_id"),
        )

    # -- main entry point --------------------------------------------------

    def run(self):
        cfg = self.config
        hedge_window = getattr(cfg, "HEDGE_RATIO_WINDOW", cfg.ZSCORE_WINDOW)
        z_window = cfg.ZSCORE_WINDOW
        use_log = getattr(cfg, "USE_LOG_PRICES", False)

        a, b = self.price_a.align(self.price_b, join="inner")
        working_a, working_b = (np.log(a), np.log(b)) if use_log else (a, b)

        hedge_ratio = self._rolling_hedge_ratio(working_a, working_b, hedge_window)
        spread = working_a - hedge_ratio * working_b
        zscore = self._zscore(spread, z_window)

        if getattr(cfg, "COINTEGRATION_GATE", False):
            coint_window = getattr(cfg, "COINTEGRATION_WINDOW", 252)
            coint_p_threshold = getattr(cfg, "COINTEGRATION_P_THRESHOLD", 0.05)
            gate_ok = self._cointegration_gate(spread, coint_window, coint_p_threshold)
        else:
            gate_ok = pd.Series(True, index=spread.index)

        position_raw, hedge_used, trade_id = self._generate_positions(
            zscore, hedge_ratio, cfg, gate_ok=gate_ok
        )

        # No look-ahead: today's held position/hedge/trade-id reflect
        # yesterday's decision.
        position = position_raw.shift(1).fillna(0)
        hedge_shifted = hedge_used.shift(1).ffill()
        trade_id_shifted = trade_id.shift(1)

        delta_a = working_a.diff()
        delta_b = working_b.diff()

        if use_log:
            # Log-price deltas approximate simple returns, so the spread
            # is already a normalised, unit-capital quantity.
            capital_base = pd.Series(1.0, index=a.index)
        else:
            capital_base = a.shift(1) + hedge_shifted.abs() * b.shift(1)

        dollar_pnl = position * (delta_a - hedge_shifted * delta_b)

        # Transaction costs only when the held position actually changes
        # (entry or exit) -- our state machine never flips sign in one
        # step, it always passes through 0, so |diff| is 0 or 1.
        turnover = position.diff().abs().fillna(position.abs())
        cost = cfg.TRANSACTION_COST * capital_base * turnover

        returns = (dollar_pnl - cost) / capital_base
        returns = returns.where(capital_base != 0, 0.0)
        returns = returns.fillna(0.0)

        self.result = pd.DataFrame(
            {
                "price_a": a,
                "price_b": b,
                "hedge_ratio": hedge_ratio,
                "spread": spread,
                "zscore": zscore,
                "position_raw": position_raw,
                "position": position,
                "trade_id": trade_id_shifted,
                "coint_gate": gate_ok,
                "turnover": turnover,
                "dollar_pnl": dollar_pnl,
                "cost": cost,
                "returns": returns,
            }
        )
        return self.result

    # -- trade-level summary -----------------------------------------------

    def trade_summary(self):
        """Aggregate returns per trade to compute counts and win rate."""
        if self.result is None:
            self.run()

        trades = self.result.dropna(subset=["trade_id"])
        if trades.empty:
            return {"num_trades": 0, "win_rate": np.nan, "avg_trade_return": np.nan}

        pnl_by_trade = trades.groupby("trade_id")["returns"].sum()
        num_trades = len(pnl_by_trade)
        win_rate = (pnl_by_trade > 0).sum() / num_trades if num_trades else np.nan

        return {
            "num_trades": num_trades,
            "win_rate": win_rate,
            "avg_trade_return": pnl_by_trade.mean(),
        }
