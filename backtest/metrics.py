"""Performance metrics for a daily return series."""

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252


def _max_drawdown_and_duration(cum_returns):
    running_max = cum_returns.cummax()
    drawdown = cum_returns / running_max - 1.0
    max_dd = drawdown.min()

    in_drawdown = drawdown < 0
    if not in_drawdown.any():
        return max_dd, 0

    group_id = (~in_drawdown).cumsum()
    durations = in_drawdown.groupby(group_id).sum()
    max_duration = int(durations.max())

    return max_dd, max_duration


def compute_metrics(returns, name="Strategy", periods_per_year=TRADING_DAYS_PER_YEAR):
    """Compute a standard set of risk/return metrics for a return series."""
    returns = returns.fillna(0.0)
    n = len(returns)

    cum_returns = (1.0 + returns).cumprod()
    total_return = cum_returns.iloc[-1] - 1.0 if n else np.nan

    years = n / periods_per_year if n else np.nan
    if n and cum_returns.iloc[-1] > 0 and years > 0:
        ann_return = cum_returns.iloc[-1] ** (1.0 / years) - 1.0
    else:
        ann_return = np.nan

    ann_vol = returns.std(ddof=0) * np.sqrt(periods_per_year)

    sharpe = (
        returns.mean() / returns.std(ddof=0) * np.sqrt(periods_per_year)
        if returns.std(ddof=0) > 0
        else np.nan
    )

    downside = returns[returns < 0]
    downside_std = downside.std(ddof=0)
    sortino = (
        returns.mean() / downside_std * np.sqrt(periods_per_year)
        if downside_std and downside_std > 0
        else np.nan
    )

    max_dd, max_dd_duration = _max_drawdown_and_duration(cum_returns)
    calmar = ann_return / abs(max_dd) if max_dd not in (0, np.nan) and not np.isnan(ann_return) else np.nan

    nonzero = returns[returns != 0]
    win_rate = (nonzero > 0).sum() / len(nonzero) if len(nonzero) else np.nan

    gains = returns[returns > 0].sum()
    losses = -returns[returns < 0].sum()
    profit_factor = gains / losses if losses > 0 else np.nan

    return {
        "Name": name,
        "Total Return": total_return,
        "Annualised Return": ann_return,
        "Annualised Vol": ann_vol,
        "Sharpe Ratio": sharpe,
        "Sortino Ratio": sortino,
        "Calmar Ratio": calmar,
        "Max Drawdown": max_dd,
        "Max DD Duration (days)": max_dd_duration,
        "Win Rate": win_rate,
        "Profit Factor": profit_factor,
    }
