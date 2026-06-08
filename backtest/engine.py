"""
engine.py — Master Backtest Engine for Day 3
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import os
import re
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
if __name__ == "__main__":
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Import Metrics ───────────────────────────────────────────────────────────
try:
    from backtest.metrics import compute_pnl, buy_and_hold_benchmark
except ImportError:
    from metrics import compute_pnl, buy_and_hold_benchmark

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("backtest_engine")

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent      # poly-hugo/backtest
REPO_ROOT = ROOT.parent                    # poly-hugo
WORKSPACE_ROOT = REPO_ROOT.parent          # EI/

def find_path(rel_path: str) -> Path:
    """Find path relative to repo root, workspace root, or via env vars."""
    p_repo = REPO_ROOT / rel_path
    if p_repo.exists():
        return p_repo
    p_work = WORKSPACE_ROOT / rel_path
    if p_work.exists():
        return p_work
    env_val = os.environ.get(rel_path.replace("/", "_").replace(".", "_").upper())
    if env_val:
        return Path(env_val)
    return p_repo

DIR_FEATURES   = find_path("Features")
DIR_STRATEGIES = REPO_ROOT / "strategies"
DIR_OUTPUT     = REPO_ROOT / "backtest_results"

DB_BARS   = find_path("Donnees_Brutes/sp100_2025_1min_bars.sqlite3")
DB_QUOTES = find_path("Donnees_Brutes/sp100_2025_q4_1min_quotes_partial.sqlite3")

EDGES_FILE = DIR_STRATEGIES / "s1_causal_granger" / "causal_edges.parquet"
if not EDGES_FILE.exists():
    # Fallback to general workspace strategies dir
    EDGES_FILE = find_path("STRATEGIES/causal_edges.parquet")

CALIB_FILE = DIR_STRATEGIES / "s1_causal_granger" / "calibration_params.json"
if not CALIB_FILE.exists():
    CALIB_FILE = find_path("STRATEGIES/calibration_strat1.json")

TRAIN_END   = "2025-11-01 00:00:00+00:00"
QUOTE_START = "2025-10-01"  # Q4 quotes start here

# ═════════════════════════════════════════════════════════════════════════════
#  DATA LOADING
# ═════════════════════════════════════════════════════════════════════════════

def load_sp100_bars(tickers: list[str] | None = None,
                    start: str | None = None,
                    end: str | None = None,
                    resample: str = "1min") -> pd.DataFrame:
    """
    Load S&P 100 1-min OHLCV bars and return close prices.
    Index: DatetimeIndex (UTC), columns: tickers.
    """
    if not DB_BARS.exists():
        logger.error("Database not found: %s", DB_BARS)
        return pd.DataFrame()

    logger.info("Loading S&P 100 bars (resample=%s) …", resample)
    conn = sqlite3.connect(str(DB_BARS))

    if tickers:
        ph = ",".join(["?"] * len(tickers))
        query = f"SELECT time, ticker, close FROM bars WHERE ticker IN ({ph})"
        df = pd.read_sql(query, conn, params=tickers)
    else:
        df = pd.read_sql("SELECT time, ticker, close FROM bars", conn)

    conn.close()
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df_close = df.pivot_table(index="time", columns="ticker", values="close", aggfunc="last")

    if resample != "1min":
        df_close = df_close.resample(resample).last()
    if start:
        df_close = df_close[df_close.index >= start]
    if end:
        df_close = df_close[df_close.index < end]

    logger.info("  Bars loaded: %d rows × %d tickers", len(df_close), df_close.shape[1])
    return df_close


def load_sp100_returns(tickers: list[str] | None = None,
                       start: str | None = None,
                       end: str | None = None,
                       resample: str = "1min") -> pd.DataFrame:
    """Load bars and compute 1-period pct_change returns."""
    prices = load_sp100_bars(tickers, start, end, resample)
    if prices.empty:
        return pd.DataFrame()
    return prices.pct_change(1, fill_method=None)


def load_quotes() -> pd.DataFrame:
    """
    Load Q4 2025 bid/ask quote data.
    Returns DataFrame indexed by (time, ticker) with bid/ask prices.
    """
    if not DB_QUOTES.exists():
        logger.warning("Quote database not found — using flat cost instead.")
        return pd.DataFrame()

    logger.info("Loading Q4 2025 quotes …")
    conn = sqlite3.connect(str(DB_QUOTES))
    df = pd.read_sql("SELECT time, ticker, bid_price, ask_price FROM quote_1m", conn)
    conn.close()

    df["time"] = pd.to_datetime(df["time"], utc=True)
    df = df.set_index(["time", "ticker"]).sort_index()
    logger.info("  Quotes loaded: %d rows", len(df))
    return df


def load_strategy_bars(strat_name: str,
                       start: str | None = None,
                       end: str | None = None,
                       resample: str = "15min") -> pd.DataFrame:
    """
    Load the close prices for a given strategy.
    """
    if "disasters" in strat_name.lower():
        logger.info("Loading insurance bars for strategy %s …", strat_name)
        db_path = find_path("Stratégie Catastrophes naturelles/sp100_insurance_1min_bars.sqlite3")
        if not db_path.exists():
            logger.error("Insurance database not found: %s", db_path)
            return pd.DataFrame()
        
        conn = sqlite3.connect(str(db_path))
        df = pd.read_sql("SELECT time, ticker, close FROM bar_1m", conn)
        conn.close()
        df["time"] = pd.to_datetime(df["time"].str.replace("Z", "+00:00", regex=False), utc=True)
        df_close = df.pivot_table(index="time", columns="ticker", values="close", aggfunc="last")
        
        if resample != "1min":
            df_close = df_close.resample(resample).last()
        if start:
            df_close = df_close[df_close.index >= start]
        if end:
            df_close = df_close[df_close.index < end]
        return df_close
    else:
        return load_sp100_bars(start=start, end=end, resample=resample)


def load_strategy_returns(strat_name: str,
                           start: str | None = None,
                           end: str | None = None,
                           resample: str = "15min") -> pd.DataFrame:
    """Load close prices and compute returns for a given strategy."""
    prices = load_strategy_bars(strat_name, start, end, resample)
    if prices.empty:
        return pd.DataFrame()
    return prices.pct_change(1, fill_method=None)


def load_polymarket_features(split: str = "test", resample: str = "15min") -> pd.DataFrame:
    """Load pre-computed Polymarket group features (wide format)."""
    suffix = {"full": "", "train": "_train", "test": "_test"}[split]
    path = DIR_FEATURES / f"polymarket_group_features_1m{suffix}_wide.parquet"

    if not path.exists():
        # Try finding in general workspace Features folder
        path = find_path(f"Features/polymarket_group_features_1m{suffix}_wide.parquet")

    if not path.exists():
        raise FileNotFoundError(f"Missing Polymarket features at: {path}")

    logger.info("Loading Polymarket features: %s", path.name)
    df = pd.read_parquet(path)
    df["minute_ts"] = pd.to_datetime(df["minute_ts"], utc=True)
    df = df.set_index("minute_ts")

    if resample != "1min":
        df = df.resample(resample).last()

    logger.info("  Features loaded: %d rows × %d cols", len(df), df.shape[1])
    return df

# ═════════════════════════════════════════════════════════════════════════════
#  STRATEGY 1 — Load calibrated parameters and generate signals
# ═════════════════════════════════════════════════════════════════════════════

def _load_calibration() -> dict:
    """
    Load pre-calibrated parameters from calibration_params.json or calibration_strat1.json.
    """
    if not CALIB_FILE.exists():
        raise FileNotFoundError(
            f"Calibration file not found: {CALIB_FILE}\n"
            "Ensure calibration files are placed correctly."
        )

    with open(CALIB_FILE) as f:
        raw = json.load(f)

    # Deserialise keys back to tuples
    stats_x = {}
    for k, v in raw["stats_x"].items():
        cat, feat = k.split("|||")
        stats_x[(cat, feat)] = v

    regime_thresholds = {}
    for k, v in raw["regime_thresholds"].items():
        cat, feat = k.split("|||")
        stats_x_key = (cat, feat)
        regime_thresholds[stats_x_key] = v

    signs = {}
    for k, v in raw["signs"].items():
        parts = k.split("|||")
        signs[(parts[0], parts[1], parts[2])] = v

    holding_periods = raw.get("holding_periods", {})

    return {
        "optimal_z": raw["optimal_z"],
        "scaling_factor": raw.get("scaling_factor", 0.5),
        "asymmetric_short_penalty": raw.get("asymmetric_short_penalty", 0.5),
        "p_value_threshold": raw.get("p_value_threshold", 0.02),
        "resample_freq": raw.get("resample_freq", "15min"),
        "stats_x": stats_x,
        "regime_thresholds": regime_thresholds,
        "signs": signs,
        "holding_periods": holding_periods,
    }


def _compute_feature_expr(df: pd.DataFrame, category: str, expr: str) -> pd.Series:
    """Evaluate a feature expression for a given Polymarket category."""
    words = set(re.findall(r'[a-zA-Z_]+', expr))
    local_env = {}
    for w in words:
        col = f"{category}_{w}"
        if col in df.columns:
            local_env[w] = df[col].values
    if not local_env:
        return pd.Series(np.nan, index=df.index)
    try:
        res = eval(expr, {"__builtins__": None}, local_env)
        return pd.Series(res, index=df.index).replace([np.inf, -np.inf], np.nan)
    except Exception:
        return pd.Series(np.nan, index=df.index)


def strategy1_generate_signals(df_poly: pd.DataFrame,
                               df_stocks_returns: pd.DataFrame,
                               period: str = "test",
                               **kwargs) -> pd.DataFrame:
    """
    Strategy 1: Causal-graph-driven Z-score regime signals.
    """
    calib = _load_calibration()
    optimal_z        = calib["optimal_z"]
    scaling_factor   = calib["scaling_factor"]
    short_penalty    = calib["asymmetric_short_penalty"]
    p_threshold      = calib["p_value_threshold"]
    stats_x          = calib["stats_x"]
    regime_thresholds = calib["regime_thresholds"]
    signs            = calib["signs"]
    
    z_long  = optimal_z
    z_short = optimal_z + short_penalty

    logger.info("Strategy 1 — Loaded calibration: Z=%.2f, %d normalisation stats, %d signs",
                optimal_z, len(stats_x), len(signs))

    if not EDGES_FILE.exists():
        raise FileNotFoundError(f"Causal edges not found: {EDGES_FILE}")

    edges_df = pd.read_parquet(EDGES_FILE)
    if "p_value_fdr" in edges_df.columns:
        edges_df = edges_df[edges_df["p_value_fdr"] <= p_threshold].copy()
    if "weight_new" in edges_df.columns:
        edges_df["weight"] = edges_df["weight_new"]

    if edges_df.empty:
        logger.warning("No causal edges pass p < %.4f filter.", p_threshold)
        return pd.DataFrame()

    tickers = edges_df["target"].unique().tolist()
    logger.info("Strategy 1 — %d causal edges, %d tickers", len(edges_df), len(tickers))

    common_idx = df_poly.index.intersection(df_stocks_returns.index)
    df_poly_sig = df_poly.loc[common_idx]

    delta_z = {}
    for cat in edges_df["source"].unique():
        cat_edges = edges_df[edges_df["source"] == cat]
        for feat in cat_edges["feature"].unique():
            key = (cat, feat)
            if key not in stats_x:
                logger.warning("  Missing calibration for (%s, %s) — skipping", cat, feat)
                continue

            x_raw = _compute_feature_expr(df_poly_sig, cat, feat)
            x_diff = x_raw.diff().fillna(0)
            rolling_std = x_diff.rolling(8).std().fillna(0)
            regime_mask = rolling_std >= regime_thresholds.get(key, 0.0)

            mu    = stats_x[key]["mu"]
            sigma = stats_x[key]["std"]

            z_score = (x_diff - mu) / sigma
            z_score.loc[~regime_mask] = 0.0
            delta_z[key] = z_score

    portfolio_weights = pd.DataFrame(0.0, index=common_idx, columns=tickers)

    for tick, group in edges_df.groupby("target"):
        sig = pd.Series(0.0, index=common_idx)
        for _, row in group.iterrows():
            cat = row["source"]
            feat = row["feature"]
            w = row["weight"]
            l = max(1, int(row["lag_minutes"] // 15))
            key = (cat, feat)
            if key not in delta_z:
                continue
            sign_beta = signs.get((cat, tick, feat), 1)
            sig += sign_beta * w * delta_z[key].shift(l - 1).fillna(0.0)

        pos = np.zeros(len(sig))
        state = 0  # 0: Flat, 1: Long, -1: Short
        entry_idx = -1
        entry_val = 0.0
        
        ret_series = df_stocks_returns[tick].fillna(0.0) if tick in df_stocks_returns.columns else pd.Series(0.0, index=common_idx)
        cum_prod = (1.0 + ret_series).cumprod().values
        sig_arr = sig.values
        
        for t in range(len(sig)):
            current_sig = sig_arr[t]
            
            if state == 0:
                if current_sig > z_long:
                    state = 1
                    entry_idx = t
                    entry_val = np.clip(current_sig * scaling_factor, 0.0, 1.0)
                    pos[t] = entry_val
                elif current_sig < -z_short:
                    state = -1
                    entry_idx = t
                    entry_val = np.clip(current_sig * scaling_factor, -1.0, 0.0)
                    pos[t] = entry_val
                else:
                    pos[t] = 0.0
            elif state == 1:
                if current_sig < -z_short:
                    state = -1
                    entry_idx = t
                    entry_val = np.clip(current_sig * scaling_factor, -1.0, 0.0)
                    pos[t] = entry_val
                else:
                    ratio = cum_prod[t] / cum_prod[entry_idx] if cum_prod[entry_idx] != 0 else 1.0
                    ret_since_entry = ratio - 1.0
                    if ret_since_entry <= -0.025:
                        state = 0
                        entry_idx = -1
                        pos[t] = 0.0
                    else:
                        pos[t] = entry_val
            elif state == -1:
                if current_sig > z_long:
                    state = 1
                    entry_idx = t
                    entry_val = np.clip(current_sig * scaling_factor, 0.0, 1.0)
                    pos[t] = entry_val
                else:
                    ratio = cum_prod[t] / cum_prod[entry_idx] if cum_prod[entry_idx] != 0 else 1.0
                    ret_since_entry = ratio - 1.0
                    if ret_since_entry >= 0.025:
                        state = 0
                        entry_idx = -1
                        pos[t] = 0.0
                    else:
                        pos[t] = entry_val
                        
        portfolio_weights[tick] = pos

    sum_abs = portfolio_weights.abs().sum(axis=1)
    scale = np.minimum(1.0, 1.0 / sum_abs.replace(0, np.nan)).fillna(1.0)
    portfolio_weights = portfolio_weights.multiply(scale, axis=0)

    logger.info("Strategy 1 — Signals generated: %d timestamps, %d tickers",
                len(portfolio_weights), len(tickers))
    return portfolio_weights

# ═════════════════════════════════════════════════════════════════════════════
#  EXECUTION SIMULATOR
# ═════════════════════════════════════════════════════════════════════════════

def apply_epsilon_delay(weights: pd.DataFrame, epsilon_minutes: int = 1) -> pd.DataFrame:
    """Shift weights forward by ε minutes."""
    if epsilon_minutes <= 0:
        return weights

    diffs = weights.index.to_series().diff().dropna()
    median_diff = diffs.median()
    if pd.isna(median_diff) or median_diff <= pd.Timedelta(0):
        median_diff = pd.Timedelta(minutes=15)

    shift_periods = max(1, int(pd.Timedelta(minutes=epsilon_minutes) / median_diff))
    logger.info("Applying ε-delay: %d min → shift by %d periods (period = %s)",
                epsilon_minutes, shift_periods, median_diff)
    return weights.shift(shift_periods).fillna(0)


def compute_spread_costs(weights: pd.DataFrame, df_quotes: pd.DataFrame) -> pd.Series:
    """Compute spread execution costs using quotes database or flat 1.5 bps fallback."""
    dW = weights.diff().fillna(0)

    if df_quotes.empty:
        logger.warning("No quote data — using flat 1.5 bps spread cost")
        return dW.abs().sum(axis=1) * 0.00015

    df_hs = df_quotes.copy()
    mid = (df_hs["ask_price"] + df_hs["bid_price"]) / 2
    df_hs["half_spread"] = (df_hs["ask_price"] - df_hs["bid_price"]) / (2 * mid)

    hs_wide = df_hs[["half_spread"]].reset_index().pivot_table(
        index="time", columns="ticker", values="half_spread", aggfunc="last"
    )

    hs_aligned = hs_wide.reindex(index=weights.index, columns=weights.columns)
    median_spread = hs_aligned.stack().median()
    if pd.isna(median_spread):
        median_spread = 0.00015
    hs_aligned = hs_aligned.fillna(median_spread)

    coverage = hs_aligned.notna().mean().mean() * 100
    spread_cost = (dW.abs() * hs_aligned).sum(axis=1)

    logger.info("  Spread costs: median half-spread = %.2f bps, quote coverage = %.1f%%",
                median_spread * 10000, coverage)
    return spread_cost

# ═════════════════════════════════════════════════════════════════════════════
#  PLOTTING
# ═════════════════════════════════════════════════════════════════════════════

def plot_results(result: dict, strategy_name: str, output_dir: Path, bh_result: dict = None, bh_traded_result: dict = None):
    """Generate quality backtest performance plots."""
    import matplotlib.dates as mdates
    output_dir.mkdir(parents=True, exist_ok=True)

    cum_gross = result["cum_gross"]
    cum_net   = result["cum_net"]
    drawdown  = result["drawdown"]
    weights   = result["weights"]
    metrics   = result["metrics"]

    DISPLAY_NAMES = {
        "Strategy_1_Causal": "S1 — Causal",
        "strategy_h1_earnings": "S2 — Earnings Spike (H1)",
        "strategy_disasters_short": "S3a — Disaster Long/Short",
        "strategy_disasters_cash": "S3b — Disaster Long/Cash",
        "strategy_disasters_mixed": "S3c — Disaster Mixed"
    }

    COLORS_STRAT = {
        "Strategy_1_Causal": "#3B82F6",
        "strategy_h1_earnings": "#10B981",
        "strategy_disasters_short": "#EF4444",
        "strategy_disasters_cash": "#F59E0B",
        "strategy_disasters_mixed": "#8B5CF6",
    }

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 150,
    })

    if strategy_name == "BuyAndHold_SP100":
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.plot(cum_net.index, cum_net.values * 100, label="Buy & Hold SP100",
                color="#64748B", linewidth=1.5)
        ax.axhline(0, color="grey", linewidth=0.5, linestyle="--")
        ax.set_title("Buy & Hold SP100 — Cumulative PnL", fontsize=13, fontweight="bold")
        ax.set_ylabel("Cumulative Return (%)")
        ax.set_xlabel("Date")
        ax.legend(loc="upper left")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        fig.savefig(output_dir / f"{strategy_name}_cumulative_pnl.png", dpi=300)
        plt.close(fig)
    else:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 9), sharex=True,
                                       gridspec_kw={"height_ratios": [3, 1]})

        disp_name = DISPLAY_NAMES.get(strategy_name, strategy_name)
        net_color = COLORS_STRAT.get(strategy_name, "#10B981")

        ax1.plot(cum_net.index, cum_net.values * 100, label=f"Net return ({cum_net.iloc[-1]*100:+.2f}%)",
                 color=net_color, linewidth=2.5)
        ax1.plot(cum_gross.index, cum_gross.values * 100, label=f"Gross return ({cum_gross.iloc[-1]*100:+.2f}%)",
                 color=net_color, linewidth=1.5, linestyle="--", alpha=0.7)

        total_spread = metrics["total_spread_cost"]
        ax1.fill_between(cum_net.index, cum_net.values * 100, cum_gross.values * 100,
                         color="#EF4444", alpha=0.15, label=f"Spread cost ({total_spread*100:.2f}%)")

        if bh_result:
            bh_cum_net = bh_result["cum_net"]
            ax1.plot(bh_cum_net.index, bh_cum_net.values * 100,
                     label=f"S&P 100 B&H ({bh_cum_net.iloc[-1]*100:+.2f}%)",
                     color="#64748B", linewidth=2.0, linestyle=":")

        if bh_traded_result:
            bh_t_cum_net = bh_traded_result["cum_net"]
            ax1.plot(bh_t_cum_net.index, bh_t_cum_net.values * 100,
                     label=f"Traded Universe B&H ({bh_t_cum_net.iloc[-1]*100:+.2f}%)",
                     color="#E11D48", linewidth=2.0, linestyle="-.")

        ax1.axhline(0, color="grey", linewidth=0.8, linestyle="--")

        sharpe_val = metrics["sharpe_ratio"]
        net_val = metrics["net_cumulative_return"] * 100
        gross_val = metrics["gross_cumulative_return"] * 100
        max_dd = metrics["max_drawdown"] * 100
        n_trades = metrics["n_trades"]
        
        ax1.set_title(f"{disp_name} (Nov-Dec 2025, ε=1 min)\n"
                      f"Sharpe={sharpe_val:+.3f} | Net={net_val:+.2f}% | Gross={gross_val:+.2f}% | MaxDD={max_dd:.2f}% | {n_trades} trades",
                      fontsize=13, fontweight="bold")
        ax1.set_ylabel("Cumulative Return (%)")
        ax1.legend(loc="upper left")
        ax1.grid(True, linestyle=":", alpha=0.5)

        # Bottom Panel (ax2)
        window = 182
        ppy = 6552
        nr = result["net_returns"]
        rmean = nr.rolling(window).mean()
        rstd  = nr.rolling(window).std()
        rsharpe = (rmean / rstd * np.sqrt(ppy))
        rsharpe = rsharpe.where(rstd > 1e-8, np.nan)

        short_name = disp_name.split(" ")[0]
        ax2.plot(rsharpe.index, rsharpe.values, label=f"7-day rolling Sharpe — {short_name}",
                 color=net_color, linewidth=2.5)

        if bh_result:
            bh_nr = bh_result["net_returns"]
            bh_rmean = bh_nr.rolling(window).mean()
            bh_rstd  = bh_nr.rolling(window).std()
            bh_rsharpe = (bh_rmean / bh_rstd * np.sqrt(ppy))
            bh_rsharpe = bh_rsharpe.where(bh_rstd > 1e-8, np.nan)

            ax2.plot(bh_rsharpe.index, bh_rsharpe.values, label="S&P 100 B&H",
                     color="#64748B", linewidth=1.5, linestyle=":")

        if bh_traded_result:
            bh_t_nr = bh_traded_result["net_returns"]
            bh_t_rmean = bh_t_nr.rolling(window).mean()
            bh_t_rstd  = bh_t_nr.rolling(window).std()
            bh_t_rsharpe = (bh_t_rmean / bh_t_rstd * np.sqrt(ppy))
            bh_t_rsharpe = bh_t_rsharpe.where(bh_t_rstd > 1e-8, np.nan)

            ax2.plot(bh_t_rsharpe.index, bh_t_rsharpe.values, label="Traded Universe B&H",
                     color="#E11D48", linewidth=1.5, linestyle="-.")

        ax2.axhline(0, color="grey", linewidth=0.8, linestyle="--")
        ax2.set_ylabel("Rolling Sharpe (7d)")
        ax2.legend(loc="upper left")
        ax2.grid(True, linestyle=":", alpha=0.5)
        ax2.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))

        plt.tight_layout()
        fig.savefig(output_dir / f"{strategy_name}_cumulative_pnl.png", dpi=300)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.fill_between(drawdown.index, drawdown.values * 100, 0, color="#E53935", alpha=0.4)
    ax.plot(drawdown.index, drawdown.values * 100, color="#E53935", linewidth=0.8)
    ax.set_title(f"{strategy_name} — Drawdown", fontsize=13, fontweight="bold")
    ax.set_ylabel("Drawdown (%)")
    ax.set_xlabel("Date")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(output_dir / f"{strategy_name}_drawdown.png", dpi=300)
    plt.close(fig)

    plot_w = weights.resample("1D").mean()
    plot_w = plot_w.loc[:, (plot_w != 0).any(axis=0)]

    if not plot_w.empty:
        fig, ax = plt.subplots(figsize=(15, 7))
        pos_w = plot_w.clip(lower=0)
        neg_w = plot_w.clip(upper=0)
        cmap = plt.get_cmap("tab20")
        colors = cmap(np.linspace(0, 1, len(plot_w.columns)))

        ax.stackplot(plot_w.index, pos_w.T, labels=plot_w.columns,
                     colors=colors, baseline="zero")
        ax.stackplot(plot_w.index, neg_w.T, colors=colors, baseline="zero")
        ax.axhline(0, color="black", linewidth=1)
        ax.set_title(f"{strategy_name} — Portfolio Weights Evolution",
                     fontsize=13, fontweight="bold")
        ax.set_ylabel("Allocated Weight")
        ax.set_xlabel("Date")
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc="upper left",
                  bbox_to_anchor=(1, 1), ncol=2, fontsize=8)
        plt.tight_layout()
        fig.savefig(output_dir / f"{strategy_name}_weights.png", dpi=300)
        plt.close(fig)

    logger.info("Plots saved to %s", output_dir)

# ═════════════════════════════════════════════════════════════════════════════
#  DYNAMIC STRATEGY LOADING
# ═════════════════════════════════════════════════════════════════════════════

def load_external_strategy(strategy_path: str | Path):
    """Dynamically import teammate strategy module."""
    path = Path(strategy_path)
    if not path.exists():
        raise FileNotFoundError(f"Strategy file not found: {path}")

    spec = importlib.util.spec_from_file_location(f"strategy_{path.stem}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, "generate_signals"):
        raise AttributeError(
            f"Strategy file {path.name} must define: "
            "generate_signals(df_poly, df_bars, **kwargs) -> pd.DataFrame"
        )
    return module

# ═════════════════════════════════════════════════════════════════════════════
#  MAIN ORCHESTRATOR
# ═════════════════════════════════════════════════════════════════════════════

def run_backtest(strategy: str = "1",
                 epsilon: int = 1,
                 period: str = "test") -> dict:
    """Run full backtest for selected strategy."""
    DIR_OUTPUT.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    logger.info("╔══════════════════════════════════════════════════════════╗")
    logger.info("║  BACKTEST ENGINE — Day 3                                ║")
    logger.info("║  Strategy: %-10s | ε: %d min | Period: %-6s       ║", strategy, epsilon, period)
    logger.info("╚══════════════════════════════════════════════════════════╝")

    if period == "test":
        start, end = TRAIN_END, None
    elif period == "train":
        start, end = None, TRAIN_END
    else:
        start, end = None, None

    logger.info("Loading market data …")
    resample = "15min"
    df_returns = load_sp100_returns(start=start, end=end, resample=resample)
    if df_returns.empty:
        logger.error("No return data loaded. Is sp100_2025_1min_bars.sqlite3 present?")
        return {}

    df_poly = load_polymarket_features(split=period if period != "full" else "full", resample=resample)
    df_quotes = load_quotes()

    strategies_to_run = []

    if strategy == "all":
        strategies_to_run.append(("Strategy_1_Causal", "builtin_1"))
        for f in sorted(DIR_STRATEGIES.rglob("strategy.py")):
            if f.parent.name != "s1_causal_granger":
                strategies_to_run.append((f.parent.name, str(f)))
    elif strategy == "1":
        strategies_to_run.append(("Strategy_1_Causal", "builtin_1"))
    else:
        p = Path(strategy)
        if p.exists():
            strategies_to_run.append((p.parent.name if p.name == "strategy.py" else p.stem, str(p)))
        else:
            # Fallback path resolve
            resolved_p = find_path(strategy)
            if resolved_p.exists():
                strategies_to_run.append((resolved_p.parent.name if resolved_p.name == "strategy.py" else resolved_p.stem, str(resolved_p)))
            else:
                logger.error("Strategy not found: %s", strategy)
                return {}

    all_results = {}

    logger.info("━━━ Computing: Buy & Hold SP100 (benchmark) ━━━")
    bh_result = buy_and_hold_benchmark(df_returns)
    if bh_result:
        m_bh = bh_result["metrics"]
        logger.info("  [BnH] Net Cumulative: %.4f (Ann: %.4f)", m_bh["net_cumulative_return"], m_bh["annualised_net"])
        logger.info("  [BnH] Sharpe: %.3f  |  MaxDD: %.4f", m_bh["sharpe_ratio"], m_bh["max_drawdown"])
        plot_results(bh_result, "BuyAndHold_SP100", DIR_OUTPUT)
        with open(DIR_OUTPUT / "BuyAndHold_SP100_metrics.json", "w") as f:
            json.dump(m_bh, f, indent=4)
        all_results["BuyAndHold_SP100"] = bh_result

    for strat_name, strat_source in strategies_to_run:
        logger.info("━━━ Running: %s ━━━", strat_name)

        if strat_source == "builtin_1":
            raw_weights = strategy1_generate_signals(df_poly, df_returns, period=period)
            df_returns_strat = df_returns
        else:
            module = load_external_strategy(strat_source)
            df_bars_ext = load_strategy_bars(strat_name, start=start, end=end, resample=resample)
            raw_weights = module.generate_signals(df_poly, df_bars_ext)
            df_returns_strat = load_strategy_returns(strat_name, start=start, end=end, resample=resample)

        if raw_weights.empty:
            logger.warning("Strategy %s produced no signals — skipping.", strat_name)
            continue

        delayed_weights = apply_epsilon_delay(raw_weights, epsilon_minutes=epsilon)
        spread_costs = compute_spread_costs(delayed_weights, df_quotes)

        is_causal = "Causal" in strat_name or "Strategy_1" in strat_name
        result = compute_pnl(delayed_weights, df_returns_strat, spread_costs, is_causal_strat=is_causal)
        m = result["metrics"]

        logger.info("  Gross Cumulative: %.4f (Ann: %.4f)", m["gross_cumulative_return"], m["annualised_gross"])
        logger.info("  Net Cumulative:   %.4f (Ann: %.4f)", m["net_cumulative_return"], m["annualised_net"])
        logger.info("  Sharpe Ratio:     %.3f", m["sharpe_ratio"])
        logger.info("  Max Drawdown:     %.4f", m["max_drawdown"])
        logger.info("  Trades:           %d", m["n_trades"])
        logger.info("  Total Spread Cost:%.6f", m["total_spread_cost"])

        w = result["weights"]
        traded_tickers = w.columns[(w.abs().sum(axis=0) > 1e-6)].tolist()
        if traded_tickers:
            df_returns_traded = df_returns_strat[traded_tickers]
            bh_traded_result = buy_and_hold_benchmark(df_returns_traded)
        else:
            bh_traded_result = None

        plot_results(result, strat_name, DIR_OUTPUT, bh_result=bh_result, bh_traded_result=bh_traded_result)
        with open(DIR_OUTPUT / f"{strat_name}_metrics.json", "w") as f:
            json.dump(m, f, indent=4)

        try:
            nr = result.get("net_returns")
            cn = result.get("cum_net")
            wts = result.get("weights")
            if nr is not None:
                nr.to_csv(DIR_OUTPUT / f"{strat_name}_net_returns.csv")
            if cn is not None:
                cn.to_csv(DIR_OUTPUT / f"{strat_name}_cum_net.csv")
            if wts is not None and not wts.empty:
                wts.to_csv(DIR_OUTPUT / f"{strat_name}_weights.csv")
        except Exception:
            logger.exception("Failed to persist full result timeseries for %s", strat_name)

        all_results[strat_name] = result

    if len(all_results) > 1:
        _print_comparison(all_results)

    elapsed = time.time() - t_start
    logger.info("Backtest completed in %.1fs. Results in: %s", elapsed, DIR_OUTPUT)
    return all_results


def _print_comparison(all_results: dict):
    """Print side-by-side strategy metrics comparison."""
    print("\n" + "═" * 80)
    print("STRATEGY COMPARISON  (★ = beats Buy & Hold)")
    print("═" * 80)
    bh_net = all_results.get("BuyAndHold_SP100", {}).get("metrics", {}).get("annualised_net", None)
    header = f"{'Strategy':<32s} {'Net Ann.':<12s} {'Sharpe':<10s} {'MaxDD':<10s} {'Trades':<8s}"
    print(header)
    print("─" * 80)
    for name, res in all_results.items():
        m = res["metrics"]
        beat = ""
        if bh_net is not None and name != "BuyAndHold_SP100":
            beat = " ★" if m["annualised_net"] > bh_net else ""
        label = f"{name}{beat}"
        print(f"{label:<34s} {m['annualised_net']*100:>8.2f}%   {m['sharpe_ratio']:>8.3f}  "
              f"{m['max_drawdown']*100:>8.2f}%  {m['n_trades']:>6d}")
    print("═" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Day 3 Backtest Engine — Polymarket → S&P 100 Trading Strategies",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python engine.py                               # Strategy 1 on test set
  python engine.py --strategy all                # Compare all strategies
  python engine.py --epsilon 5                   # 5-min execution delay
  python engine.py --period train                # Run on train set
        """,
    )
    parser.add_argument("--strategy", "-s", default="1",
                        help="Strategy to run: '1' (Hugo), path to .py, or 'all'")
    parser.add_argument("--epsilon", "-e", type=int, default=1,
                        help="Execution delay in minutes (default: 1)")
    parser.add_argument("--period", "-p", default="test",
                        choices=["test", "train", "full"],
                        help="Data period: test (Nov+), train (<Nov), full")

    args = parser.parse_args()
    run_backtest(strategy=args.strategy, epsilon=args.epsilon, period=args.period)
