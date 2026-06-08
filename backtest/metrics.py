"""
metrics.py — Portfolio PnL, Sharpe, drawdown, and Buy & Hold benchmark calculations.
"""

from __future__ import annotations
import numpy as np
import pandas as pd

def compute_pnl(weights: pd.DataFrame,
                returns: pd.DataFrame,
                spread_costs: pd.Series,
                is_causal_strat: bool = False) -> dict:
    """
    Compute portfolio PnL and key performance metrics.

    - is_causal_strat=True  → Strategy 1: weights already capacity-constrained,
                               use as-is (no rescaling).
    - is_causal_strat=False → external strategies: equal-weight 1/N rescaling
                               across the N columns returned by the strategy.
    """
    tickers = weights.columns.tolist()
    n_stocks = max(len(tickers), 1)

    if is_causal_strat:
        alloc_weights = weights
    else:
        alloc_weights = weights * (1.0 / n_stocks)

    common_idx = alloc_weights.index.intersection(returns.index)
    common_tickers = [t for t in tickers if t in returns.columns]
    alloc_w = alloc_weights.loc[common_idx, common_tickers]
    ret = returns.loc[common_idx, common_tickers]

    # Gross returns: W(t-1) * r(t)
    gross_returns = (alloc_w.shift(1) * ret).sum(axis=1).fillna(0)

    # Spread costs
    if is_causal_strat:
        sc = spread_costs.reindex(common_idx).fillna(0)
    else:
        sc = spread_costs.reindex(common_idx).fillna(0) / n_stocks
    net_returns = gross_returns - sc

    cum_gross = gross_returns.cumsum()
    cum_net   = net_returns.cumsum()

    # Annualise
    n_days = (common_idx[-1] - common_idx[0]).days
    ann_factor = 365.25 / max(n_days, 1)

    periods_per_day = len(common_idx) / max(n_days, 1)
    periods_per_year = periods_per_day * 252

    net_std = net_returns.std()
    sharpe = (net_returns.mean() / net_std * np.sqrt(periods_per_year)) if net_std > 0 else 0

    # Max Drawdown
    cum_max = cum_net.cummax()
    drawdown = cum_net - cum_max
    max_dd = drawdown.min()

    # Trade count
    n_trades = (alloc_w.diff().fillna(0).abs().sum(axis=1) > 1e-6).sum()

    metrics = {
        "gross_cumulative_return": float(cum_gross.iloc[-1]),
        "net_cumulative_return": float(cum_net.iloc[-1]),
        "annualised_gross": float(cum_gross.iloc[-1] * ann_factor),
        "annualised_net": float(cum_net.iloc[-1] * ann_factor),
        "sharpe_ratio": float(sharpe),
        "max_drawdown": float(max_dd),
        "n_trades": int(n_trades),
        "n_periods": int(len(common_idx)),
        "n_days": int(n_days),
        "total_spread_cost": float(sc.sum()),
        "avg_gross_exposure": float(alloc_w.abs().sum(axis=1).mean()),
        "avg_net_exposure": float(alloc_w.sum(axis=1).mean()),
    }

    return {
        "metrics": metrics,
        "cum_gross": cum_gross,
        "cum_net": cum_net,
        "drawdown": drawdown,
        "weights": alloc_w,
        "gross_returns": gross_returns,
        "net_returns": net_returns,
    }

def buy_and_hold_benchmark(df_returns: pd.DataFrame) -> dict:
    """
    Equal-weight Buy & Hold on all SP100 tickers.
    Portfolio weight = 1/N on every ticker, held the entire period.
    No transaction costs (one-time buy at t=0 is negligible).
    """
    n = df_returns.shape[1]
    if n == 0:
        return {}

    # Equal-weight portfolio
    port_returns = df_returns.mean(axis=1).fillna(0)

    cum_gross = port_returns.cumsum()
    cum_net   = cum_gross  # no TC for buy-and-hold

    n_days = (df_returns.index[-1] - df_returns.index[0]).days
    ann_factor = 365.25 / max(n_days, 1)
    periods_per_day = len(df_returns) / max(n_days, 1)
    periods_per_year = periods_per_day * 252

    std = port_returns.std()
    sharpe = (port_returns.mean() / std * np.sqrt(periods_per_year)) if std > 0 else 0

    cum_max = cum_net.cummax()
    drawdown = cum_net - cum_max
    max_dd = drawdown.min()

    metrics = {
        "gross_cumulative_return": float(cum_gross.iloc[-1]),
        "net_cumulative_return":   float(cum_net.iloc[-1]),
        "annualised_gross":        float(cum_gross.iloc[-1] * ann_factor),
        "annualised_net":          float(cum_net.iloc[-1] * ann_factor),
        "sharpe_ratio":            float(sharpe),
        "max_drawdown":            float(max_dd),
        "n_trades":                1,
        "n_periods":               int(len(df_returns)),
        "n_days":                  int(n_days),
        "total_spread_cost":       0.0,
        "avg_gross_exposure":      1.0,
        "avg_net_exposure":        1.0,
    }

    return {
        "metrics":       metrics,
        "cum_gross":     cum_gross,
        "cum_net":       cum_net,
        "drawdown":      drawdown,
        "weights":       pd.DataFrame({"SP100_EW": pd.Series(1.0 / n, index=df_returns.index)}),
        "gross_returns": port_returns,
        "net_returns":   port_returns,
    }
