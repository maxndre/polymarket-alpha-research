"""
engine.py — Backtest engine for Polymarket → S&P 100 strategies.

Architecture
------------
Three decoupled concerns:
  1. Signal generation  — each strategy provides target weights W(t) via generate_signals()
  2. Execution delay    — all signals shifted by ε minutes (default: 1 min)
  3. PnL calculation    — realistic costs via bid/ask spread from Q4 quote data

Strategy interface (plug-in contract):
    generate_signals(df_poly, df_bars, **kwargs) -> pd.DataFrame
    Returns: DatetimeIndex (UTC), columns = tickers, values ∈ [-1, 1]

Usage
-----
    python backtest/engine.py                      # Strategy 1 on test set
    python backtest/engine.py --strategy all       # Compare all strategies
    python backtest/engine.py --strategy path/to/strategy.py
    python backtest/engine.py --epsilon 2          # 2-min execution delay
    python backtest/engine.py --period train       # run on train set
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import re
import sqlite3
import sys
import time
from pathlib import Path

import matplotlib
if __name__ == "__main__":
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

from .metrics import compute_pnl, buy_and_hold_benchmark

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("backtest.engine")

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT         = Path(__file__).resolve().parent.parent
FEATURES_DIR = ROOT / "data" / "features"
STRATEGIES_DIR = ROOT / "strategies"
OUTPUT_DIR   = ROOT / "backtest_results"

DB_BARS   = ROOT / "data" / "raw" / "sp100_2025_1min_bars.sqlite3"
DB_QUOTES = ROOT / "data" / "raw" / "sp100_2025_q4_1min_quotes_partial.sqlite3"

TRAIN_END = "2025-11-01 00:00:00+00:00"


# ═════════════════════════════════════════════════════════════════════════════
#  DATA LOADING
# ═════════════════════════════════════════════════════════════════════════════

def load_sp100_bars(tickers: list[str] | None = None,
                    start: str | None = None,
                    end: str | None = None,
                    resample: str = "1min") -> pd.DataFrame:
    """Load S&P 100 1-min OHLCV bars and return close prices (UTC index, tickers as columns)."""
    logger.info("Loading S&P 100 bars (resample=%s) ...", resample)
    conn = sqlite3.connect(str(DB_BARS))

    if tickers:
        ph    = ",".join(["?"] * len(tickers))
        query = f"SELECT time, ticker, close FROM bars WHERE ticker IN ({ph})"
        df    = pd.read_sql(query, conn, params=tickers)
    else:
        df = pd.read_sql("SELECT time, ticker, close FROM bars", conn)
    conn.close()

    df["time"] = pd.to_datetime(df["time"], utc=True)
    df_close   = df.pivot_table(index="time", columns="ticker", values="close", aggfunc="last")

    if resample != "1min":
        df_close = df_close.resample(resample).last()
    if start:
        df_close = df_close[df_close.index >= start]
    if end:
        df_close = df_close[df_close.index < end]

    logger.info("  Loaded: %d rows × %d tickers", len(df_close), df_close.shape[1])
    return df_close


def load_sp100_returns(**kwargs) -> pd.DataFrame:
    """Load bars and compute 1-period pct_change returns."""
    return load_sp100_bars(**kwargs).pct_change(1, fill_method=None)


def load_quotes() -> pd.DataFrame:
    """Load Q4 2025 bid/ask quotes indexed by (time, ticker)."""
    if not DB_QUOTES.exists():
        logger.warning("Quote database not found — will use flat 1.5 bps cost.")
        return pd.DataFrame()

    logger.info("Loading Q4 2025 bid/ask quotes ...")
    conn = sqlite3.connect(str(DB_QUOTES))
    df   = pd.read_sql("SELECT time, ticker, bid_price, ask_price FROM quote_1m", conn)
    conn.close()

    df["time"] = pd.to_datetime(df["time"], utc=True)
    df = df.set_index(["time", "ticker"]).sort_index()
    logger.info("  Loaded: %d rows", len(df))
    return df


def load_polymarket_features(split: str = "test", resample: str = "15min") -> pd.DataFrame:
    """Load pre-computed Polymarket group features (wide format)."""
    suffix = {"full": "", "train": "_train", "test": "_test"}[split]
    path   = FEATURES_DIR / f"features_1m{suffix}_wide.parquet"

    if not path.exists():
        raise FileNotFoundError(f"Missing features file: {path}\nRun pipeline/run_pipeline.py first.")

    logger.info("Loading Polymarket features: %s", path.name)
    df = pd.read_parquet(path)
    df["minute_ts"] = pd.to_datetime(df["minute_ts"], utc=True)
    df = df.set_index("minute_ts")

    if resample != "1min":
        df = df.resample(resample).last()

    logger.info("  Loaded: %d rows × %d cols", len(df), df.shape[1])
    return df


# ═════════════════════════════════════════════════════════════════════════════
#  STRATEGY 1 — Causal Granger signal generation
# ═════════════════════════════════════════════════════════════════════════════

def _load_calibration() -> dict:
    """Load pre-calibrated S1 parameters from strategies/s1_causal_granger/."""
    calib_file = STRATEGIES_DIR / "s1_causal_granger" / "calibration_params.json"
    if not calib_file.exists():
        raise FileNotFoundError(f"Calibration file not found: {calib_file}")

    with open(calib_file) as f:
        raw = json.load(f)

    stats_x = {tuple(k.split("|||")): v for k, v in raw["stats_x"].items()}
    regime_thresholds = {tuple(k.split("|||")): v for k, v in raw["regime_thresholds"].items()}
    signs = {tuple(k.split("|||")): v for k, v in raw["signs"].items()}

    return {
        "optimal_z":               raw["optimal_z"],
        "scaling_factor":          raw.get("scaling_factor", 0.5),
        "asymmetric_short_penalty": raw.get("asymmetric_short_penalty", 0.5),
        "p_value_threshold":       raw.get("p_value_threshold", 0.02),
        "resample_freq":           raw.get("resample_freq", "15min"),
        "stats_x":                 stats_x,
        "regime_thresholds":       regime_thresholds,
        "signs":                   signs,
        "holding_periods":         raw.get("holding_periods", {}),
    }


def _feature_expr(df: pd.DataFrame, category: str, expr: str) -> pd.Series:
    """Evaluate a feature expression for a given Polymarket category."""
    words     = set(re.findall(r'[a-zA-Z_]+', expr))
    local_env = {w: df[f"{category}_{w}"].values for w in words if f"{category}_{w}" in df.columns}
    if not local_env:
        return pd.Series(np.nan, index=df.index)
    try:
        res = eval(expr, {"__builtins__": None}, local_env)
        return pd.Series(res, index=df.index).replace([np.inf, -np.inf], np.nan)
    except Exception:
        return pd.Series(np.nan, index=df.index)


def strategy1_generate_signals(df_poly: pd.DataFrame,
                                df_stocks_returns: pd.DataFrame,
                                period: str = "test") -> pd.DataFrame:
    """
    Strategy 1: Causal-graph-driven Z-score regime signals.

    All parameters are loaded from calibration_params.json — never recalculated.
    Z-scores are normalized using train-set statistics only (no look-ahead).

    Returns
    -------
    pd.DataFrame — DatetimeIndex (UTC), columns = tickers,
                   values = target portfolio weights ∈ [-1, 1]
    """
    calib         = _load_calibration()
    optimal_z     = calib["optimal_z"]
    scaling       = calib["scaling_factor"]
    short_penalty = calib["asymmetric_short_penalty"]
    p_threshold   = calib["p_value_threshold"]
    stats_x       = calib["stats_x"]
    regime_thr    = calib["regime_thresholds"]
    signs         = calib["signs"]

    z_long  = optimal_z
    z_short = optimal_z + short_penalty

    # Load causal edges
    edges_file = FEATURES_DIR / "causal_edges.parquet"
    if not edges_file.exists():
        raise FileNotFoundError(f"Causal edges not found: {edges_file}\nRun pipeline first.")

    edges_df = pd.read_parquet(edges_file)
    if "p_value_fdr" in edges_df.columns:
        edges_df = edges_df[edges_df["p_value_fdr"] <= p_threshold].copy()
    if "weight_new" in edges_df.columns:
        edges_df["weight"] = edges_df["weight_new"]

    if edges_df.empty:
        logger.warning("No causal edges pass p < %.4f.", p_threshold)
        return pd.DataFrame()

    tickers    = edges_df["target"].unique().tolist()
    common_idx = df_poly.index.intersection(df_stocks_returns.index)
    df_poly_s  = df_poly.loc[common_idx]

    logger.info("S1 — %d causal edges, %d tickers, Z=%.2f", len(edges_df), len(tickers), optimal_z)

    # Build delta-z scores
    delta_z: dict = {}
    for cat in edges_df["source"].unique():
        for feat in edges_df[edges_df["source"] == cat]["feature"].unique():
            key = (cat, feat)
            if key not in stats_x:
                continue
            x_raw  = _feature_expr(df_poly_s, cat, feat)
            x_diff = x_raw.diff().fillna(0)
            roll_std = x_diff.rolling(8).std().fillna(0)
            regime_mask = roll_std >= regime_thr[key]
            z = (x_diff - stats_x[key]["mu"]) / stats_x[key]["std"]
            z.loc[~regime_mask] = 0.0
            delta_z[key] = z

    # Aggregate per ticker with stop-loss state machine
    portfolio_weights = pd.DataFrame(0.0, index=common_idx, columns=tickers)

    for tick, group in edges_df.groupby("target"):
        sig = pd.Series(0.0, index=common_idx)
        for _, row in group.iterrows():
            key  = (row["source"], row["feature"])
            lag  = max(1, int(row["lag_minutes"] // 15))
            sign = signs.get((row["source"], tick, row["feature"]), 1)
            if key in delta_z:
                sig += sign * row["weight"] * delta_z[key].shift(lag - 1).fillna(0.0)

        # State machine: hold position + 2.5% stop-loss
        pos      = np.zeros(len(sig))
        state    = 0      # 0: flat, 1: long, -1: short
        entry_idx = -1
        entry_val = 0.0
        ret_series = df_stocks_returns[tick].fillna(0.0) if tick in df_stocks_returns.columns else pd.Series(0.0, index=common_idx)
        cum_prod   = (1.0 + ret_series).cumprod().values
        sig_arr    = sig.values

        for t in range(len(sig)):
            s = sig_arr[t]
            if state == 0:
                if s > z_long:
                    state, entry_idx = 1, t
                    entry_val = np.clip(s * scaling, 0.0, 1.0)
                    pos[t] = entry_val
                elif s < -z_short:
                    state, entry_idx = -1, t
                    entry_val = np.clip(s * scaling, -1.0, 0.0)
                    pos[t] = entry_val
            elif state == 1:
                if s < -z_short:
                    state, entry_idx = -1, t
                    entry_val = np.clip(s * scaling, -1.0, 0.0)
                    pos[t] = entry_val
                elif cum_prod[entry_idx] != 0 and (cum_prod[t] / cum_prod[entry_idx] - 1) <= -0.025:
                    state, entry_idx = 0, -1
                else:
                    pos[t] = entry_val
            elif state == -1:
                if s > z_long:
                    state, entry_idx = 1, t
                    entry_val = np.clip(s * scaling, 0.0, 1.0)
                    pos[t] = entry_val
                elif cum_prod[entry_idx] != 0 and (cum_prod[t] / cum_prod[entry_idx] - 1) >= 0.025:
                    state, entry_idx = 0, -1
                else:
                    pos[t] = entry_val

        portfolio_weights[tick] = pos

    # Capacity constraint: max gross leverage = 1.0
    sum_abs = portfolio_weights.abs().sum(axis=1)
    scale   = np.minimum(1.0, 1.0 / sum_abs.replace(0, np.nan)).fillna(1.0)
    portfolio_weights = portfolio_weights.multiply(scale, axis=0)

    logger.info("S1 — signals generated: %d timestamps", len(portfolio_weights))
    return portfolio_weights


# ═════════════════════════════════════════════════════════════════════════════
#  EXECUTION SIMULATOR
# ═════════════════════════════════════════════════════════════════════════════

def apply_epsilon_delay(weights: pd.DataFrame, epsilon_minutes: int = 1) -> pd.DataFrame:
    """
    Shift signal weights forward by ε minutes.
    Enforces: 'a signal usable only at ε=0 is not tradable in production.'
    """
    if epsilon_minutes <= 0:
        return weights

    diffs       = weights.index.to_series().diff().dropna()
    median_diff = diffs.median()
    if pd.isna(median_diff) or median_diff <= pd.Timedelta(0):
        median_diff = pd.Timedelta(minutes=15)

    shift_periods = max(1, int(pd.Timedelta(minutes=epsilon_minutes) / median_diff))
    logger.info("ε-delay: %d min → shift %d periods", epsilon_minutes, shift_periods)
    return weights.shift(shift_periods).fillna(0)


def compute_spread_costs(weights: pd.DataFrame, df_quotes: pd.DataFrame) -> pd.Series:
    """
    Compute execution costs using bid/ask quotes.
    Buy  (ΔW > 0) → fill at ask → cost = half_spread × |ΔW|
    Sell (ΔW < 0) → fill at bid → same formula
    Falls back to flat 1.5 bps when quotes are unavailable.
    """
    dW = weights.diff().fillna(0)

    if df_quotes.empty:
        return dW.abs().sum(axis=1) * 0.00015

    mid = (df_quotes["ask_price"] + df_quotes["bid_price"]) / 2
    df_quotes = df_quotes.copy()
    df_quotes["half_spread"] = (df_quotes["ask_price"] - df_quotes["bid_price"]) / (2 * mid)

    hs_wide = (
        df_quotes[["half_spread"]].reset_index()
        .pivot_table(index="time", columns="ticker", values="half_spread", aggfunc="last")
    )
    hs = hs_wide.reindex(index=weights.index, columns=weights.columns)
    median_spread = hs.stack().median()
    hs = hs.fillna(median_spread if not pd.isna(median_spread) else 0.00015)

    logger.info("  Spread cost: median half-spread = %.2f bps", hs.stack().median() * 10_000)
    return (dW.abs() * hs).sum(axis=1)


# ═════════════════════════════════════════════════════════════════════════════
#  PLOTTING
# ═════════════════════════════════════════════════════════════════════════════

_COLORS = {
    "Strategy_1_Causal":           "#3B82F6",
    "strategy_h1_earnings":        "#10B981",
    "strategy_disasters_short":    "#EF4444",
    "strategy_disasters_cash":     "#F59E0B",
    "strategy_disasters_mixed":    "#8B5CF6",
}
_NAMES = {
    "Strategy_1_Causal":        "S1 — Causal Granger",
    "strategy_h1_earnings":     "S2 — Earnings Spike",
    "strategy_disasters_short": "S3 — Disasters Long/Short",
    "strategy_disasters_cash":  "S3 — Disasters Long/Cash",
    "strategy_disasters_mixed": "S3 — Disasters Mixed",
}


def plot_results(result: dict, strategy_name: str, output_dir: Path,
                 bh_result: dict | None = None) -> None:
    """Generate cumulative PnL, drawdown, and portfolio weights plots."""
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False})

    cum_net   = result["cum_net"]
    cum_gross = result["cum_gross"]
    drawdown  = result["drawdown"]
    weights   = result["weights"]
    metrics   = result["metrics"]
    color     = _COLORS.get(strategy_name, "#10B981")
    disp      = _NAMES.get(strategy_name, strategy_name)

    # ── Cumulative PnL ────────────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 9), sharex=True,
                                   gridspec_kw={"height_ratios": [3, 1]})

    ax1.plot(cum_net.index,   cum_net.values   * 100, label=f"Net ({cum_net.iloc[-1]*100:+.2f}%)",   color=color, lw=2.5)
    ax1.plot(cum_gross.index, cum_gross.values * 100, label=f"Gross ({cum_gross.iloc[-1]*100:+.2f}%)", color=color, lw=1.5, ls="--", alpha=0.7)
    ax1.fill_between(cum_net.index, cum_net.values * 100, cum_gross.values * 100,
                     color="#EF4444", alpha=0.15,
                     label=f"Spread cost ({metrics['total_spread_cost']*100:.2f}%)")

    if bh_result:
        bh = bh_result["cum_net"]
        ax1.plot(bh.index, bh.values * 100, label=f"S&P 100 B&H ({bh.iloc[-1]*100:+.2f}%)",
                 color="#64748B", lw=2.0, ls=":")

    ax1.axhline(0, color="grey", lw=0.8, ls="--")
    ax1.set_title(
        f"{disp} (Nov–Dec 2025, ε=1 min)\n"
        f"Sharpe={metrics['sharpe_ratio']:+.3f}  |  Net={metrics['net_cumulative_return']*100:+.2f}%  "
        f"|  MaxDD={metrics['max_drawdown']*100:.2f}%  |  {metrics['n_trades']} trades",
        fontsize=13, fontweight="bold",
    )
    ax1.set_ylabel("Cumulative Return (%)")
    ax1.legend(loc="upper left")
    ax1.grid(True, ls=":", alpha=0.5)

    # Rolling 7-day Sharpe (bottom panel)
    window = 182
    ppy    = 6552
    nr     = result["net_returns"]
    rs     = (nr.rolling(window).mean() / nr.rolling(window).std() * np.sqrt(ppy))
    ax2.plot(rs.index, rs.values, label=f"7-day rolling Sharpe", color=color, lw=2.0)
    ax2.axhline(0, color="grey", lw=0.8, ls="--")
    ax2.set_ylabel("Rolling Sharpe (7d)")
    ax2.legend(loc="upper left")
    ax2.grid(True, ls=":", alpha=0.5)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))

    plt.tight_layout()
    fig.savefig(output_dir / f"{strategy_name}_pnl.png", dpi=300)
    plt.close(fig)

    # ── Drawdown ──────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.fill_between(drawdown.index, drawdown.values * 100, 0, color="#E53935", alpha=0.4)
    ax.plot(drawdown.index, drawdown.values * 100, color="#E53935", lw=0.8)
    ax.set_title(f"{disp} — Drawdown", fontsize=13, fontweight="bold")
    ax.set_ylabel("Drawdown (%)")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(output_dir / f"{strategy_name}_drawdown.png", dpi=300)
    plt.close(fig)

    # ── Portfolio weights (daily resampled) ───────────────────────────────────
    plot_w = weights.resample("1D").mean()
    plot_w = plot_w.loc[:, (plot_w != 0).any(axis=0)]
    if not plot_w.empty:
        fig, ax = plt.subplots(figsize=(15, 6))
        colors  = plt.get_cmap("tab20")(np.linspace(0, 1, len(plot_w.columns)))
        ax.stackplot(plot_w.index, plot_w.clip(lower=0).T, labels=plot_w.columns, colors=colors)
        ax.stackplot(plot_w.index, plot_w.clip(upper=0).T, colors=colors)
        ax.axhline(0, color="black", lw=1)
        ax.set_title(f"{disp} — Portfolio Weights", fontsize=13, fontweight="bold")
        ax.set_ylabel("Allocated Weight")
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(dict(zip(labels, handles)).values(), dict(zip(labels, handles)).keys(),
                  loc="upper left", bbox_to_anchor=(1, 1), ncol=2, fontsize=8)
        plt.tight_layout()
        fig.savefig(output_dir / f"{strategy_name}_weights.png", dpi=300)
        plt.close(fig)

    logger.info("Plots saved → %s", output_dir)


# ═════════════════════════════════════════════════════════════════════════════
#  DYNAMIC STRATEGY LOADING
# ═════════════════════════════════════════════════════════════════════════════

def load_external_strategy(strategy_path: str | Path):
    """
    Dynamically import a strategy file.
    The file must define: generate_signals(df_poly, df_bars, **kwargs) -> pd.DataFrame
    """
    path = Path(strategy_path)
    if not path.exists():
        raise FileNotFoundError(f"Strategy file not found: {path}")

    spec   = importlib.util.spec_from_file_location(f"strategy_{path.stem}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, "generate_signals"):
        raise AttributeError(
            f"{path.name} must define generate_signals(df_poly, df_bars, **kwargs) -> pd.DataFrame"
        )
    return module


# ═════════════════════════════════════════════════════════════════════════════
#  MAIN ORCHESTRATOR
# ═════════════════════════════════════════════════════════════════════════════

def run_backtest(strategy: str = "1", epsilon: int = 1, period: str = "test") -> dict:
    """
    Run the full backtest pipeline for a given strategy.

    Parameters
    ----------
    strategy : "1" (S1 Causal Granger), path to .py file, or "all"
    epsilon  : execution delay in minutes
    period   : "test" (Nov–Dec), "train" (Jan–Oct), or "full"
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    logger.info("=" * 60)
    logger.info("  BACKTEST ENGINE | strategy=%s | ε=%dm | period=%s", strategy, epsilon, period)
    logger.info("=" * 60)

    start = TRAIN_END if period == "test" else None
    end   = TRAIN_END if period == "train" else None
    resample = "15min"

    df_returns = load_sp100_returns(start=start, end=end, resample=resample)
    df_poly    = load_polymarket_features(split=period if period != "full" else "full", resample=resample)
    df_quotes  = load_quotes()

    # Enumerate strategies
    strategies_to_run = []
    if strategy == "all":
        strategies_to_run.append(("Strategy_1_Causal", "builtin_1"))
        for f in sorted(STRATEGIES_DIR.rglob("strategy.py")):
            strategies_to_run.append((f.parent.name, str(f)))
    elif strategy == "1":
        strategies_to_run.append(("Strategy_1_Causal", "builtin_1"))
    else:
        p = Path(strategy)
        if p.exists():
            strategies_to_run.append((p.stem, str(p)))
        else:
            logger.error("Strategy not found: %s", strategy)
            return {}

    all_results: dict = {}

    # Buy & Hold benchmark
    bh_result = buy_and_hold_benchmark(df_returns)
    if bh_result:
        m = bh_result["metrics"]
        logger.info("[B&H] Sharpe=%.3f | Net=%.4f", m["sharpe_ratio"], m["net_cumulative_return"])
        plot_results(bh_result, "BuyAndHold_SP100", OUTPUT_DIR)
        with open(OUTPUT_DIR / "BuyAndHold_SP100_metrics.json", "w") as f:
            json.dump(m, f, indent=2)
        all_results["BuyAndHold_SP100"] = bh_result

    for strat_name, strat_src in strategies_to_run:
        logger.info("--- Running: %s ---", strat_name)

        if strat_src == "builtin_1":
            raw_weights  = strategy1_generate_signals(df_poly, df_returns, period=period)
            df_ret_strat = df_returns
        else:
            module       = load_external_strategy(strat_src)
            df_bars_ext  = load_sp100_bars(start=start, end=end, resample=resample)
            df_ret_strat = df_bars_ext.pct_change(1, fill_method=None)
            raw_weights  = module.generate_signals(df_poly, df_bars_ext)

        if raw_weights is None or raw_weights.empty:
            logger.warning("No signals for %s — skipping.", strat_name)
            continue

        delayed = apply_epsilon_delay(raw_weights, epsilon_minutes=epsilon)
        costs   = compute_spread_costs(delayed, df_quotes)
        is_s1   = "Causal" in strat_name or "Strategy_1" in strat_name
        result  = compute_pnl(delayed, df_ret_strat, costs, is_capacity_constrained=is_s1)
        m       = result["metrics"]

        logger.info("  Net:    %.4f (ann: %.4f)", m["net_cumulative_return"],   m["annualised_net"])
        logger.info("  Sharpe: %.3f | MaxDD: %.4f | Trades: %d",
                    m["sharpe_ratio"], m["max_drawdown"], m["n_trades"])

        plot_results(result, strat_name, OUTPUT_DIR, bh_result=bh_result)
        with open(OUTPUT_DIR / f"{strat_name}_metrics.json", "w") as f:
            json.dump(m, f, indent=2)

        # Persist full time series for notebook analysis
        result["net_returns"].to_csv(OUTPUT_DIR / f"{strat_name}_net_returns.csv")
        result["cum_net"].to_csv(OUTPUT_DIR / f"{strat_name}_cum_net.csv")
        if not result["weights"].empty:
            result["weights"].to_csv(OUTPUT_DIR / f"{strat_name}_weights.csv")

        all_results[strat_name] = result

    # Comparison table
    if len(all_results) > 1:
        bh_ann = all_results.get("BuyAndHold_SP100", {}).get("metrics", {}).get("annualised_net")
        print(f"\n{'='*70}")
        print(f"{'Strategy':<34} {'Net Ann.':>10} {'Sharpe':>10} {'MaxDD':>10} {'Trades':>8}")
        print("-" * 70)
        for name, res in all_results.items():
            m    = res["metrics"]
            star = " ★" if bh_ann and name != "BuyAndHold_SP100" and m["annualised_net"] > bh_ann else ""
            print(f"{name+star:<36} {m['annualised_net']*100:>8.2f}%  {m['sharpe_ratio']:>8.3f}  "
                  f"{m['max_drawdown']*100:>8.2f}%  {m['n_trades']:>6d}")
        print("=" * 70)

    logger.info("Backtest done in %.1fs. Results: %s", time.time() - t0, OUTPUT_DIR)
    return all_results


# ═════════════════════════════════════════════════════════════════════════════
#  CLI
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Polymarket → S&P 100 Backtest Engine")
    parser.add_argument("--strategy", "-s", default="1",
                        help="Strategy: '1' (S1), path to .py file, or 'all'")
    parser.add_argument("--epsilon",  "-e", type=int, default=1,
                        help="Execution delay in minutes (default: 1)")
    parser.add_argument("--period",   "-p", default="test",
                        choices=["test", "train", "full"],
                        help="Data period: test (Nov+), train (<Nov), full")
    args = parser.parse_args()
    run_backtest(strategy=args.strategy, epsilon=args.epsilon, period=args.period)
