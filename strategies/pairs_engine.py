"""Single-pair statistical arbitrage engine."""

import numpy as np
import pandas as pd


class PairsEngine:
    """Dollar-neutral mean-reversion engine for a single pair.

    Given two price series (asset A and asset B), this:
      1. Estimates a rolling OLS hedge ratio (beta of A on B).
      2. Builds the spread = price_A - hedge_ratio * price_B and its
         rolling Z-score.
      3. Runs a stateful signal loop over the Z-score to decide when to
         enter/exit, freezing the hedge ratio at the moment of entry so
         it is never re-estimated mid-trade.
      4. Shifts signals by one day (trade on next bar, no look-ahead)
         and converts the held position into dollar P&L, normalised by
         the capital committed to the trade, net of transaction costs
         charged only on entry and exit.
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

    def _generate_positions(self, zscore, hedge_ratio, cfg):
        """Stateful loop: no look-ahead, hedge ratio fixed at entry."""
        n = len(zscore)
        z_vals = zscore.to_numpy()
        h_vals = hedge_ratio.to_numpy()

        state = 0
        current_hedge = np.nan
        trade_counter = 0

        positions = np.zeros(n)
        hedge_used = np.full(n, np.nan)
        trade_ids = np.full(n, np.nan)

        for i in range(n):
            z = z_vals[i]
            if not np.isnan(z):
                if state == 0:
                    if z > cfg.ENTRY_THRESHOLD:
                        state = -1  # spread too high: short A, long B
                        current_hedge = h_vals[i]
                        trade_counter += 1
                    elif z < -cfg.ENTRY_THRESHOLD:
                        state = 1  # spread too low: long A, short B
                        current_hedge = h_vals[i]
                        trade_counter += 1
                elif abs(z) < cfg.EXIT_THRESHOLD:
                    state = 0
                    current_hedge = np.nan

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
        window = cfg.ZSCORE_WINDOW

        a, b = self.price_a.align(self.price_b, join="inner")

        hedge_ratio = self._rolling_hedge_ratio(a, b, window)
        spread = a - hedge_ratio * b
        zscore = self._zscore(spread, window)

        position_raw, hedge_used, trade_id = self._generate_positions(
            zscore, hedge_ratio, cfg
        )

        # No look-ahead: today's held position/hedge/trade-id reflect
        # yesterday's decision.
        position = position_raw.shift(1).fillna(0)
        hedge_shifted = hedge_used.shift(1).ffill()
        trade_id_shifted = trade_id.shift(1)

        delta_a = a.diff()
        delta_b = b.diff()

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
