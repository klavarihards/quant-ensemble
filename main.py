"""Entry point: run the full pairs trading ensemble backtest."""

import pandas as pd

import config
from backtest.engine import Backtester


def main():
    pd.set_option("display.width", 160)
    pd.set_option("display.float_format", lambda x: f"{x:,.4f}")

    backtester = Backtester(config)
    results = backtester.run()

    print("\nBacktest results (sorted by Sharpe Ratio)")
    print("=" * 100)
    print(results.to_string())
    print("=" * 100)

    print("\nPair return correlation matrix")
    print("-" * 60)
    print(backtester.correlation_matrix().round(2).to_string())

    return results


if __name__ == "__main__":
    main()
