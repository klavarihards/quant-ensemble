"""Monthly short straddle with daily delta hedging.

Cycle: on the first trading day of each month, sell one ATM straddle
(1 call + 1 put, 30 calendar days to expiry, strike = SPY rounded to
the nearest $1), sized so a 3-standard-deviation adverse move would
cost no more than 5% of current capital. Delta-hedge daily with SPY
shares using that day's actual VIX as implied vol. Close 5 days before
expiry, or immediately if the mark-to-market loss on the straddle alone
exceeds 200% of premium collected. Skip a month entirely if VIX is at
or below 15 at the scheduled entry (premium considered too cheap).

P&L is decomposed and tracked separately every day:
  - option_pnl:  change in the straddle's own mark-to-market value
                 (what we'd pay to buy it back) -- this is where
                 premium decay/gamma/vega losses show up.
  - hedge_pnl:   P&L from the SPY shares held to stay delta-neutral,
                 accrued as yesterday's share count times today's
                 price move.
Their daily sum telescopes exactly to
  total_pnl = (premium collected - final buyback cost) + cumulative hedge P&L
by construction (see run_backtest), so there's no reconciliation gap
between the day-by-day series and the per-trade summary.

Run as a module (not a bare script) so the sibling backtest/ package
resolves: `python3 -m vol_strategy.short_straddle` from the repo root.
"""

import math

import numpy as np
import pandas as pd

from backtest.metrics import compute_metrics
from vol_strategy.data_loader import load_or_build
from vol_strategy.options_pricer import DEFAULT_RISK_FREE_RATE, compute_greeks, price_option

CAPITAL_BASE = 100_000.0
OPTION_MULTIPLIER = 100          # 1 contract = 100 shares, standard US equity option
DTE_DAYS = 30                    # calendar days to expiry at entry
CLOSE_DTE_DAYS = 5               # close this many calendar days before expiry
STOP_LOSS_MULTIPLE = 2.0         # close if MTM loss > this multiple of premium collected
VIX_ENTRY_MIN = 15.0             # skip the month if VIX <= this at entry
MAX_LOSS_PCT_OF_CAPITAL = 0.05   # size so the 3-sigma scenario loss <= this fraction of capital
SIGMA_MULTIPLE = 3.0
RISK_FREE_RATE = DEFAULT_RISK_FREE_RATE
MIN_T_YEARS = 1.0 / 365.0        # floor to keep Black-Scholes well-defined


def _estimate_num_contracts(S, K, sigma, T, premium_per_contract, capital,
                             multiplier=OPTION_MULTIPLIER, sigma_multiple=SIGMA_MULTIPLE,
                             max_loss_pct=MAX_LOSS_PCT_OF_CAPITAL):
    """Size the straddle from a 3-sigma adverse-move scenario.

    Short options have unlimited theoretical loss, so "maximum
    theoretical loss" is approximated here as the expiry intrinsic
    value if SPY moves +/- sigma_multiple standard deviations (a GBM
    approximation: S * sigma * sqrt(T)) by expiry, net of premium
    collected -- the worse of the up-move and down-move scenarios.
    """
    move = S * sigma * math.sqrt(T) * sigma_multiple
    loss_up = max((S + move) - K, 0.0) * multiplier
    loss_down = max(K - (S - move), 0.0) * multiplier
    worst_case_intrinsic = max(loss_up, loss_down)
    loss_per_contract = worst_case_intrinsic - premium_per_contract * multiplier
    loss_per_contract = max(loss_per_contract, 0.01 * S * multiplier)  # numerical floor
    budget = capital * max_loss_pct
    return max(int(budget // loss_per_contract), 0)


def run_backtest(df=None, capital_base=CAPITAL_BASE, r=RISK_FREE_RATE):
    """Run the full day-by-day simulation.

    Returns (trades_df, daily_df): trades_df has one row per calendar
    month (traded or skipped); daily_df has one row per trading day
    covering the whole sample, with pnl/return/equity columns suitable
    for backtest.metrics.compute_metrics.
    """
    if df is None:
        df = load_or_build()
    df = df.sort_index()

    month_start_dates = set(df.index.to_series().groupby(df.index.to_period("M")).head(1))

    equity = capital_base
    state = None  # dict describing the currently open trade, or None if flat
    trades = []
    daily_rows = []

    for t in df.index:
        S_t = float(df.loc[t, "SPY"])
        vix_t = float(df.loc[t, "VIX"])
        sigma_t = vix_t / 100.0
        daily_pnl = 0.0

        if state is None:
            if t in month_start_dates:
                if vix_t <= VIX_ENTRY_MIN:
                    trades.append(dict(entry_date=t, skipped=True, reason="VIX filter",
                                        vix_entry=vix_t))
                else:
                    K = round(S_t)
                    expiry_date = t + pd.Timedelta(days=DTE_DAYS)
                    planned_close_date = expiry_date - pd.Timedelta(days=CLOSE_DTE_DAYS)
                    T0 = (expiry_date - t).days / 365.0

                    call0 = price_option(S_t, K, T0, r, sigma_t, "call")
                    put0 = price_option(S_t, K, T0, r, sigma_t, "put")
                    premium_per_contract = call0 + put0

                    num_contracts = _estimate_num_contracts(S_t, K, sigma_t, T0,
                                                             premium_per_contract, equity)
                    if num_contracts <= 0:
                        trades.append(dict(entry_date=t, skipped=True,
                                            reason="position size rounded to 0",
                                            vix_entry=vix_t))
                    else:
                        premium_collected_total = premium_per_contract * OPTION_MULTIPLIER * num_contracts
                        call_delta = compute_greeks(S_t, K, T0, r, sigma_t, "call")["delta"]
                        put_delta = compute_greeks(S_t, K, T0, r, sigma_t, "put")["delta"]
                        target_shares = (call_delta + put_delta) * OPTION_MULTIPLIER * num_contracts

                        state = dict(
                            entry_date=t, K=K, expiry_date=expiry_date,
                            planned_close_date=planned_close_date,
                            num_contracts=num_contracts,
                            premium_collected_total=premium_collected_total,
                            premium_per_contract=premium_per_contract,
                            shares_held=target_shares, S_prev=S_t,
                            V_prev=premium_collected_total,  # V_entry == premium just collected
                            cumulative_hedge_pnl=0.0,
                            capital_before=equity, vix_entry=vix_t,
                        )
                        # Entry day: hedge just established, V_prev == V_now by
                        # construction, so daily_pnl is correctly 0 here.
            # else: flat and not a month start -> idle day, daily_pnl stays 0
        else:
            T_t = max((state["expiry_date"] - t).days / 365.0, MIN_T_YEARS)

            call_delta = compute_greeks(S_t, state["K"], T_t, r, sigma_t, "call")["delta"]
            put_delta = compute_greeks(S_t, state["K"], T_t, r, sigma_t, "put")["delta"]
            target_shares = (call_delta + put_delta) * OPTION_MULTIPLIER * state["num_contracts"]

            hedge_pnl_today = state["shares_held"] * (S_t - state["S_prev"])
            state["cumulative_hedge_pnl"] += hedge_pnl_today
            state["shares_held"] = target_shares
            state["S_prev"] = S_t

            call_now = price_option(S_t, state["K"], T_t, r, sigma_t, "call")
            put_now = price_option(S_t, state["K"], T_t, r, sigma_t, "put")
            V_t = (call_now + put_now) * OPTION_MULTIPLIER * state["num_contracts"]
            option_pnl_today = state["V_prev"] - V_t
            state["V_prev"] = V_t

            daily_pnl = option_pnl_today + hedge_pnl_today

            mtm_loss = V_t - state["premium_collected_total"]
            stop_triggered = mtm_loss > STOP_LOSS_MULTIPLE * state["premium_collected_total"]
            scheduled_close = t >= state["planned_close_date"]

            if stop_triggered or scheduled_close:
                option_pnl_total = state["premium_collected_total"] - V_t
                total_pnl = option_pnl_total + state["cumulative_hedge_pnl"]
                trades.append(dict(
                    entry_date=state["entry_date"], close_date=t, skipped=False,
                    reason="stop_loss" if stop_triggered else "scheduled",
                    K=state["K"], num_contracts=state["num_contracts"],
                    vix_entry=state["vix_entry"], vix_close=vix_t,
                    premium_collected=state["premium_collected_total"],
                    buyback_cost=V_t, option_pnl=option_pnl_total,
                    hedge_pnl=state["cumulative_hedge_pnl"], total_pnl=total_pnl,
                    capital_before=state["capital_before"],
                    return_pct=total_pnl / state["capital_before"],
                ))
                state = None

        equity += daily_pnl
        daily_rows.append(dict(date=t, pnl=daily_pnl, equity=equity))

    # Force-close a trade still open at the end of the data (last available MTM).
    if state is not None:
        V_t = state["V_prev"]
        option_pnl_total = state["premium_collected_total"] - V_t
        total_pnl = option_pnl_total + state["cumulative_hedge_pnl"]
        trades.append(dict(
            entry_date=state["entry_date"], close_date=df.index[-1], skipped=False,
            reason="data ended", K=state["K"], num_contracts=state["num_contracts"],
            vix_entry=state["vix_entry"], vix_close=float(df.iloc[-1]["VIX"]),
            premium_collected=state["premium_collected_total"], buyback_cost=V_t,
            option_pnl=option_pnl_total, hedge_pnl=state["cumulative_hedge_pnl"],
            total_pnl=total_pnl, capital_before=state["capital_before"],
            return_pct=total_pnl / state["capital_before"],
        ))

    trades_df = pd.DataFrame(trades)
    daily_df = pd.DataFrame(daily_rows).set_index("date")
    daily_df["capital_prev"] = daily_df["equity"].shift(1).fillna(capital_base)
    daily_df["return"] = daily_df["pnl"] / daily_df["capital_prev"]

    return trades_df, daily_df


def summarize(trades_df, daily_df, spy_prices, capital_base=CAPITAL_BASE):
    metrics = compute_metrics(daily_df["return"], "Short Straddle + Delta Hedge")

    traded = trades_df[~trades_df["skipped"]].copy()
    n_months_total = len(trades_df)
    n_skipped = int(trades_df["skipped"].sum())
    n_traded = len(traded)
    win_rate = (traded["total_pnl"] > 0).mean() if n_traded else float("nan")
    avg_monthly_pnl = traded["total_pnl"].mean() if n_traded else float("nan")
    worst5 = traded.sort_values("total_pnl").head(5)

    spy_ret = spy_prices.pct_change().fillna(0.0)
    spy_metrics = compute_metrics(spy_ret, "SPY Buy & Hold")

    print("=== Short Straddle + Delta Hedge: 2004-2026 ===")
    print(f"Months in sample: {n_months_total}  (traded: {n_traded}, skipped by VIX filter/sizing: {n_skipped})")
    print(f"Stop-loss exits: {(traded['reason'] == 'stop_loss').sum()}  |  Scheduled exits: {(traded['reason'] == 'scheduled').sum()}")
    print()
    print(f"Total Return:        {metrics['Total Return']:+.2%}")
    print(f"Annualised Return:   {metrics['Annualised Return']:+.2%}")
    print(f"Annualised Vol:      {metrics['Annualised Vol']:.2%}")
    print(f"Sharpe Ratio:        {metrics['Sharpe Ratio']:.3f}")
    print(f"Max Drawdown:        {metrics['Max Drawdown']:+.2%}")
    print(f"Max DD Duration:     {metrics['Max DD Duration (days)']} days")
    print()
    print(f"Win rate (% profitable months, traded only): {win_rate:.1%}")
    print(f"Average monthly P&L: ${avg_monthly_pnl:,.0f}")
    print()
    print("Worst 5 months:")
    cols = ["entry_date", "close_date", "reason", "vix_entry", "vix_close",
            "premium_collected", "option_pnl", "hedge_pnl", "total_pnl", "return_pct"]
    with pd.option_context("display.width", 200, "display.float_format", lambda x: f"{x:,.2f}"):
        print(worst5[cols].to_string(index=False))
    print()
    print("=== vs. SPY Buy & Hold (same period) ===")
    print(f"{'Metric':<22}{'Straddle':>14}{'SPY B&H':>14}")
    for key in ["Total Return", "Annualised Return", "Annualised Vol", "Sharpe Ratio", "Max Drawdown"]:
        fmt = "{:.2%}" if key != "Sharpe Ratio" else "{:.3f}"
        print(f"{key:<22}{fmt.format(metrics[key]):>14}{fmt.format(spy_metrics[key]):>14}")

    return metrics, spy_metrics


if __name__ == "__main__":
    prices = load_or_build()
    trades, daily = run_backtest(prices)
    summarize(trades, daily, prices["SPY"])
