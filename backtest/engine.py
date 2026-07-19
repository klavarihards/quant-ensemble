"""Full backtest pipeline: data -> signals -> portfolio -> metrics."""

import pandas as pd

from backtest.metrics import compute_metrics
from data.loader import load_prices
from strategies.portfolio import Portfolio
from strategies.signals import run_all_pairs


class Backtester:
    """Runs the end-to-end pairs-ensemble backtest for a given config."""

    def __init__(self, config):
        self.config = config
        self.prices = None
        self.pair_returns = None
        self.portfolio = None
        self.combined_returns = None
        self.results = None

    def run(self):
        cfg = self.config

        self.prices = load_prices(cfg)
        pairs = getattr(cfg, "ALL_PAIRS", cfg.PAIRS)
        self.pair_returns = run_all_pairs(self.prices, cfg, pairs=pairs)

        self.portfolio = Portfolio(self.pair_returns, cfg)
        self.combined_returns = self.portfolio.combined_returns()

        rows = [
            compute_metrics(self.pair_returns[col], col)
            for col in self.pair_returns.columns
        ]
        rows.append(compute_metrics(self.combined_returns, "Combined Portfolio"))

        benchmark_price = self.prices[cfg.BENCHMARK_TICKER]
        benchmark_returns = benchmark_price.pct_change().fillna(0.0)
        rows.append(
            compute_metrics(benchmark_returns, f"{cfg.BENCHMARK_TICKER} Buy & Hold")
        )

        results = pd.DataFrame(rows).set_index("Name")
        results = results.sort_values("Sharpe Ratio", ascending=False)
        self.results = results

        return results

    def correlation_matrix(self):
        return self.portfolio.correlation_matrix()
