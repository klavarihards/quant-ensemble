"""Run the pairs engine across the full universe of pairs."""

import types

import pandas as pd

from backtest.metrics import compute_metrics
from strategies.pairs_engine import PairsEngine


def _effective_config(base_config, overrides):
    """Build a per-pair config: base config's UPPERCASE attributes with
    specific ones overridden, without mutating the shared base config."""
    if not overrides:
        return base_config
    merged = types.SimpleNamespace(
        **{key: getattr(base_config, key) for key in dir(base_config) if key.isupper()}
    )
    for key, value in overrides.items():
        setattr(merged, key, value)
    return merged


def run_all_pairs(prices, config, pairs=None, pair_overrides=None):
    """Run PairsEngine on every configured pair.

    pairs defaults to config.PAIRS; pair_overrides defaults to
    config.PAIR_OVERRIDES (a {(A, B): {...}} dict of per-pair config
    attribute overrides, e.g. a different entry/exit threshold or a
    cointegration gate for newly added pairs that haven't been tuned
    the same way as the core set).

    Returns a DataFrame of daily returns, one column per pair
    (named "A/B"), and prints a per-pair summary of trade count,
    win rate and Sharpe ratio.
    """
    pairs = pairs if pairs is not None else config.PAIRS
    pair_overrides = (
        pair_overrides if pair_overrides is not None else getattr(config, "PAIR_OVERRIDES", {})
    )

    returns = {}
    summaries = []

    for asset_a, asset_b in pairs:
        name = f"{asset_a}/{asset_b}"
        pair_config = _effective_config(config, pair_overrides.get((asset_a, asset_b)))
        engine = PairsEngine(prices[asset_a], prices[asset_b], pair_config, name=name)
        engine.run()

        returns[name] = engine.result["returns"]
        trade_stats = engine.trade_summary()
        metrics = compute_metrics(engine.result["returns"], name)

        summaries.append(
            {
                "Pair": name,
                "Trades": trade_stats["num_trades"],
                "Win Rate": trade_stats["win_rate"],
                "Sharpe": metrics["Sharpe Ratio"],
            }
        )

    returns_df = pd.DataFrame(returns)

    print("\nPer-pair signal summary")
    print("-" * 46)
    print(f"{'Pair':<10}{'Trades':>10}{'Win Rate':>12}{'Sharpe':>12}")
    for row in summaries:
        win_rate = row["Win Rate"]
        win_rate_str = f"{win_rate:.1%}" if pd.notna(win_rate) else "n/a"
        sharpe_str = f"{row['Sharpe']:.2f}" if pd.notna(row["Sharpe"]) else "n/a"
        print(f"{row['Pair']:<10}{row['Trades']:>10}{win_rate_str:>12}{sharpe_str:>12}")
    print("-" * 46)

    return returns_df
