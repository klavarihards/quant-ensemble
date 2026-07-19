"""Combine per-pair strategies into a single portfolio."""

import numpy as np
import pandas as pd


class Portfolio:
    """Combine independent pair strategies into one portfolio return.

    Two weighting schemes:
      - Equal weight (config.CAPITAL_PER_PAIR each), the default.
      - A walk-forward trailing-Sharpe tilt (config.USE_TRAILING_SHARPE_TILT):
        once a month, re-weight pairs in proportion to their trailing
        config.TRAILING_SHARPE_WINDOW-day Sharpe (clipped at zero, so a
        pair with negative trailing performance gets no new capital
        until it recovers), falling back to equal weight if every pair's
        trailing Sharpe is non-positive. Only uses data strictly before
        the rebalance date, so there is no look-ahead, and a rebalancing
        cost (config.TRANSACTION_COST times the weight turnover) is
        charged on every rebalance to account for resizing positions.

    This was chosen over daily re-weighting and over per-pair-optimised
    entry/exit parameters after both were tested: daily re-weighting
    ignores realistic rebalancing costs, and per-pair optimisation
    overfit badly out-of-sample (some pairs' "best" in-sample parameters
    produced strongly negative holdout Sharpe). The monthly trailing-
    Sharpe tilt held up (improved, never worsened) across three
    independent 2010-2015/2016-2020/2021-2026 holdout periods.
    """

    def __init__(self, returns_df, config):
        self.returns_df = returns_df.fillna(0.0)
        self.config = config
        self.weights = pd.Series(
            config.CAPITAL_PER_PAIR, index=returns_df.columns, dtype=float
        )
        self.weights_history = None

    def combined_returns(self):
        if getattr(self.config, "USE_TRAILING_SHARPE_TILT", False):
            return self._trailing_sharpe_tilt_returns()
        return (self.returns_df * self.weights).sum(axis=1)

    def _trailing_sharpe_tilt_returns(self):
        cfg = self.config
        df = self.returns_df
        n_pairs = df.shape[1]
        window = cfg.TRAILING_SHARPE_WINDOW
        cost = cfg.TRANSACTION_COST
        equal_weight = pd.Series(1.0 / n_pairs, index=df.columns)

        month_starts = set(df.index.to_series().groupby(df.index.to_period("M")).head(1))

        weights = pd.DataFrame(index=df.index, columns=df.columns, dtype=float)
        current_w = equal_weight.copy()
        rebal_cost = pd.Series(0.0, index=df.index)

        for dt in df.index:
            if dt in month_starts:
                # Strictly-prior history only: no look-ahead into today's return.
                hist = df.loc[:dt].iloc[:-1].tail(window)
                if len(hist) >= max(40, window // 4):
                    trailing_sharpe = (hist.mean() / hist.std()).clip(lower=0)
                    total = trailing_sharpe.sum()
                    new_w = trailing_sharpe / total if total > 0 else equal_weight
                    rebal_cost.loc[dt] = cost * (new_w - current_w).abs().sum()
                    current_w = new_w
            weights.loc[dt] = current_w

        self.weights_history = weights
        return (df * weights).sum(axis=1) - rebal_cost

    def correlation_matrix(self):
        return self.returns_df.corr()
