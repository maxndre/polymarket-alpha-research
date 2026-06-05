"""
step7_causal_discovery.py — Step 7: Non-linear causal discovery.

Finds which Polymarket category features Granger-cause S&P 100 stock returns
using a tree-based (HistGradientBoosting) permutation importance framework:

  For each S&P 100 ticker:
    1. Build a lagged feature matrix from Polymarket category signals (15-min resolution).
    2. Train a gradient-boosted tree model in purged time-series cross-validation.
    3. Keep features whose out-of-sample permutation importance is strictly positive
       (i.e., they improve prediction, not just in-sample fit).

Outputs causal_edges.parquet: (source_category, target_ticker, feature, lag_minutes, importance).
These edges drive the S1 signal generation in strategies/s1_causal_granger/strategy.py.
"""

from __future__ import annotations
import logging
import re
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import TimeSeriesSplit

from .config import FEATURES_DIR, RAW_DIR

logger = logging.getLogger(__name__)

TRAIN_END      = "2025-11-01 00:00:00+00:00"
RESAMPLE_FREQ  = "15min"
LAGS_15M       = [1, 2, 4, 8, 16, 32]  # 15m, 30m, 1h, 2h, 4h, 8h
N_SPLITS       = 3
GAP            = 4  # 1-hour gap between train and validation folds (in 15-min bars)
DB_BARS        = RAW_DIR / "sp100_2025_1min_bars.sqlite3"


def _compute_feature_expr(df: pd.DataFrame, category: str, expr: str) -> pd.Series:
    """Evaluate a feature expression for a given Polymarket category column."""
    words = set(re.findall(r'[a-zA-Z_]+', expr))
    local_env = {
        w: df[f"{category}_{w}"].values
        for w in words
        if f"{category}_{w}" in df.columns
    }
    if not local_env:
        return pd.Series(np.nan, index=df.index)
    try:
        res = eval(expr, {"__builtins__": None}, local_env)
        return pd.Series(res, index=df.index).replace([np.inf, -np.inf], np.nan)
    except Exception:
        return pd.Series(np.nan, index=df.index)


def _load_sp100_returns() -> pd.DataFrame:
    """Load S&P 100 bars, resample to 15-min, return pct_change returns (train only)."""
    conn = sqlite3.connect(str(DB_BARS))
    df   = pd.read_sql("SELECT time, ticker, close FROM bars", conn)
    conn.close()

    df["time"] = pd.to_datetime(df["time"], utc=True)
    df_close   = df.pivot(index="time", columns="ticker", values="close")
    df_close   = df_close[df_close.index < TRAIN_END].resample(RESAMPLE_FREQ).last()
    return df_close.pct_change(1, fill_method=None)


def extract_causal_edges() -> pd.DataFrame:
    """
    Step 7 — Tree-based causal discovery.
    Outputs causal_edges.parquet with columns:
        source, target, feature, lag_minutes, importance, weight
    """
    logger.info("=== STEP 7: CAUSAL DISCOVERY ===")

    logger.info("Loading and resampling Polymarket features to %s ...", RESAMPLE_FREQ)
    df_poly = pd.read_parquet(FEATURES_DIR / "features_1m_train_wide.parquet")
    df_poly["minute_ts"] = pd.to_datetime(df_poly["minute_ts"], utc=True)
    df_poly = df_poly.set_index("minute_ts").resample(RESAMPLE_FREQ).last()

    suffix_features = ["price_weighted_mean", "shannon_entropy", "volume_shares_total"]
    categories = list({
        c.split("_price_weighted_mean")[0]
        for c in df_poly.columns if "_price_weighted_mean" in c
    })

    logger.info("Loading and resampling S&P 100 returns to %s ...", RESAMPLE_FREQ)
    df_stocks = _load_sp100_returns()
    tickers   = df_stocks.columns.tolist()

    # Build differenced Polymarket feature matrix
    X_dict = {}
    for cat in categories:
        for feat in suffix_features:
            x_raw = _compute_feature_expr(df_poly, cat, feat)
            X_dict[f"{cat}::{feat}"] = x_raw.diff()
    df_X = pd.DataFrame(X_dict)

    # Build lagged feature matrix
    logger.info("Building lagged feature matrix (%d categories × %d features × %d lags) ...",
                len(categories), len(suffix_features), len(LAGS_15M))
    lagged, names = [], []
    for col in df_X.columns:
        for lag in LAGS_15M:
            lagged.append(df_X[col].shift(lag))
            names.append(f"{col}::lag_{lag}")
    df_X_lags = pd.concat(lagged, axis=1)
    df_X_lags.columns = names

    tscv        = TimeSeriesSplit(n_splits=N_SPLITS, gap=GAP)
    valid_edges = []

    logger.info("Running purged TS cross-validation with HistGradientBoosting (%d tickers) ...",
                len(tickers))

    for i, tick in enumerate(tickers):
        if i % 10 == 0:
            logger.info("  Processing %d/%d: %s", i + 1, len(tickers), tick)

        y      = df_stocks[tick]
        y_lag1 = y.shift(1).rename("y_lag1")

        df_model = pd.concat([y, y_lag1, df_X_lags], axis=1).dropna()
        if len(df_model) < 200:
            continue

        y_target = df_model[tick]
        X_input  = df_model.drop(columns=[tick])

        model = HistGradientBoostingRegressor(
            max_iter=100, learning_rate=0.05, max_depth=5, random_state=42
        )
        importances_per_fold = []

        for train_idx, test_idx in tscv.split(X_input):
            X_tr, y_tr = X_input.iloc[train_idx], y_target.iloc[train_idx]
            X_te, y_te = X_input.iloc[test_idx],  y_target.iloc[test_idx]
            model.fit(X_tr, y_tr)
            r = permutation_importance(
                model, X_te, y_te,
                n_repeats=5, random_state=42,
                scoring="neg_mean_squared_error",
            )
            importances_per_fold.append(r.importances_mean)

        avg_imp = np.mean(importances_per_fold, axis=0)

        for idx, col in enumerate(X_input.columns):
            if col == "y_lag1":
                continue
            if avg_imp[idx] > 0.0:
                cat, feat, lag_str = col.split("::")
                valid_edges.append({
                    "source":       cat,
                    "target":       tick,
                    "feature":      feat,
                    "lag_minutes":  int(lag_str.split("_")[1]) * 15,
                    "importance":   float(avg_imp[idx]),
                })

    df_edges = pd.DataFrame(valid_edges)
    logger.info("Discovered %d causal edges.", len(df_edges))

    if not df_edges.empty:
        df_edges["weight"] = df_edges.groupby("target")["importance"].transform(
            lambda x: x / x.sum()
        )
        out = FEATURES_DIR / "causal_edges.parquet"
        df_edges.to_parquet(out, index=False)
        logger.info("Saved causal edges → %s", out)
    else:
        logger.warning("No causal edges found.")

    return df_edges
