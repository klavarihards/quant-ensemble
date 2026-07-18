"""Combine per-pair strategies into a single equal-weight portfolio."""

import numpy as np
import pandas as pd


class Portfolio:
    """Equal-capital combination of independent pair strategies.

    Each pair's return series is already normalised by its own capital
    base, so combining them into a portfolio return is a weighted sum
    using each pair's share of total capital (config.CAPITAL_PER_PAIR).
    """

    def __init__(self, returns_df, config):
        self.returns_df = returns_df.fillna(0.0)
        self.config = config
        self.weights = pd.Series(
            config.CAPITAL_PER_PAIR, index=returns_df.columns, dtype=float
        )

    def combined_returns(self):
        return (self.returns_df * self.weights).sum(axis=1)

    def correlation_matrix(self):
        return self.returns_df.corr()
