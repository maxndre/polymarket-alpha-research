"""
Strategy 1 — Causal Granger: Multi-resolution Granger Causal Portfolio Optimization

This strategy uses pre-computed causal edges from VAR/Granger-causality modeling
to map Polymarket categories to S&P 100 stocks. It opens long/short positions
when Polymarket activity signals cross a z-score threshold in high-variance regimes.

Author: Hugo Dunias
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_HERE = Path(__file__).resolve().parent
CALIB_FILE = _HERE / "calibration_params.json"
EDGES_FILE = _HERE / "causal_edges.parquet"

def _load_calibration() -> dict:
    """Load pre-calibrated parameters from calibration_params.json."""
    if not CALIB_FILE.exists():
        raise FileNotFoundError(
            f"Calibration file not found at: {CALIB_FILE}\n"
            "Please ensure calibration_params.json is in the strategy directory."
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
        regime_thresholds[(cat, feat)] = v

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

def generate_signals(
    df_poly: pd.DataFrame,
    df_bars: pd.DataFrame,
    **kwargs
) -> pd.DataFrame:
    """
    Strategy 1 signal generation.

    Returns
    -------
    pd.DataFrame — DatetimeIndex (UTC), columns = tickers, values ∈ [-1, 1]
    """
    df_stocks_returns = df_bars.pct_change(1, fill_method=None)

    # ── Load pre-calibrated parameters ───────────────────────────────────────
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

    logger.info("S1 — Loaded calibration: Z=%.2f, %d normalisation stats, %d signs",
                optimal_z, len(stats_x), len(signs))

    # ── Load causal edges ────────────────────────────────────────────────────
    if not EDGES_FILE.exists():
        raise FileNotFoundError(f"Causal edges file not found at: {EDGES_FILE}")

    edges_df = pd.read_parquet(EDGES_FILE)
    if "p_value_fdr" in edges_df.columns:
        edges_df = edges_df[edges_df["p_value_fdr"] <= p_threshold].copy()
    if "weight_new" in edges_df.columns:
        edges_df["weight"] = edges_df["weight_new"]

    if edges_df.empty:
        logger.warning("S1 — No causal edges pass p < %.4f filter.", p_threshold)
        return pd.DataFrame(0.0, index=df_poly.index, columns=df_bars.columns)

    tickers = edges_df["target"].unique().tolist()
    logger.info("S1 — %d causal edges, %d tickers", len(edges_df), len(tickers))

    # ── Align indices ────────────────────────────────────────────────────────
    common_idx = df_poly.index.intersection(df_stocks_returns.index)
    df_poly_sig = df_poly.loc[common_idx]

    # ── Build delta-z scores for each (category, feature) ────────────────────
    delta_z = {}
    for cat in edges_df["source"].unique():
        cat_edges = edges_df[edges_df["source"] == cat]
        for feat in cat_edges["feature"].unique():
            key = (cat, feat)
            if key not in stats_x:
                logger.warning("  S1 — Missing calibration for (%s, %s) — skipping", cat, feat)
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

    # ── Aggregate signals per ticker and convert to portfolio weights ────────
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

        # State machine for holding and 2.5% stop-loss
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

    # Dynamic capacity-constrained scaling (maximum gross leverage of 1.0)
    sum_abs = portfolio_weights.abs().sum(axis=1)
    scale = np.minimum(1.0, 1.0 / sum_abs.replace(0, np.nan)).fillna(1.0)
    portfolio_weights = portfolio_weights.multiply(scale, axis=0)

    # Reindex to full df_poly index to match execution engine format
    full_weights = pd.DataFrame(0.0, index=df_poly.index, columns=df_bars.columns)
    full_weights.update(portfolio_weights)

    logger.info("S1 — Signals generated: %d timestamps, %d tickers",
                len(portfolio_weights), len(tickers))
    return full_weights
