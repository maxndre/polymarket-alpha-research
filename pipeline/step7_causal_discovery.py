import pandas as pd
import numpy as np
import sqlite3
import re
from pathlib import Path
import logging
import json
import itertools
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import TimeSeriesSplit

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
from pipeline.config import DB_STOCKS_PATH, DIR_STRATEGIES, DIR_FEATURES

DB_PATH = DB_STOCKS_PATH

# Train Period
TRAIN_END = "2025-11-01 00:00:00+00:00"
RESAMPLE_FREQ = "15min"
LAGS_15M = [1, 2, 4, 8, 16, 32]  # Corresponding to 15m, 30m, 1h, 2h, 4h, 8h
N_SPLITS = 3
GAP = 4 # 1 hour gap between train and validation in 15m bars

# ── Feature Evaluation ────────────────────────────────────────────────────────
def compute_feature_expr(df: pd.DataFrame, category: str, expr: str) -> pd.Series:
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
    except Exception as exc:
        return pd.Series(np.nan, index=df.index)

def load_resampled_stocks() -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    query = "SELECT time, ticker, close FROM bars"
    df = pd.read_sql(query, conn)
    conn.close()
    
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df = df.pivot(index="time", columns="ticker", values="close")
    # Resample to 15 minutes, filter to train period
    df = df[df.index < TRAIN_END]
    df = df.resample(RESAMPLE_FREQ).last()
    return df.pct_change(1, fill_method=None)

def extract_non_linear_causality():
    logger.info("=== STEP 7: Non-Linear Causal Discovery (Tree-based) ===")
    
    # 1. Dynamically extract categories and features
    logger.info(f"Loading and resampling Polymarket Train data to {RESAMPLE_FREQ}...")
    df_poly = pd.read_parquet(DIR_FEATURES / "polymarket_group_features_1m_train_wide.parquet")
    df_poly["minute_ts"] = pd.to_datetime(df_poly["minute_ts"], utc=True)
    df_poly = df_poly.set_index("minute_ts")
    df_poly = df_poly.resample(RESAMPLE_FREQ).last()
    
    # Auto-detect categories and base features from columns
    # We look for common suffixes
    suffix_features = ["price_weighted_mean", "shannon_entropy", "volume_shares_total"]
    categories = list(set([c.split("_price_weighted_mean")[0] for c in df_poly.columns if "_price_weighted_mean" in c]))
    features_list = {f: f for f in suffix_features}
    
    logger.info(f"Loading and resampling S&P100 Train data to {RESAMPLE_FREQ}...")
    df_stocks = load_resampled_stocks()
    tickers = df_stocks.columns.tolist()

    # Create Polymarket features diffs
    X_diff_dict = {}
    for cat in categories:
        for feat_name, expr in features_list.items():
            x_raw = compute_feature_expr(df_poly, cat, expr)
            X_diff_dict[f"{cat}::{feat_name}"] = x_raw.diff()

    df_X = pd.DataFrame(X_diff_dict)

    # Build Master Lagged Matrix
    logger.info("Building Master Lagged Feature Matrix...")
    lagged_features = []
    feature_names = []
    
    for col in df_X.columns:
        for lag in LAGS_15M:
            lagged_features.append(df_X[col].shift(lag))
            feature_names.append(f"{col}::lag_{lag}")
            
    df_X_lags = pd.concat(lagged_features, axis=1)
    df_X_lags.columns = feature_names
    
    tscv = TimeSeriesSplit(n_splits=N_SPLITS, gap=GAP)
    
    valid_edges = []
    
    # Evaluate Non-Linear Causality per stock
    logger.info("Running Purged Time-Series Cross-Validation with HistGradientBoosting...")
    
    for i, tick in enumerate(tickers):
        if i % 10 == 0:
            logger.info(f"Processing stock {i+1}/{len(tickers)}: {tick}")
            
        y = df_stocks[tick]
        y_lag1 = y.shift(1).rename("y_lag1")
        
        # Combine X and y
        df_model = pd.concat([y, y_lag1, df_X_lags], axis=1).dropna()
        if len(df_model) < 200:
            continue
            
        y_target = df_model[tick]
        X_train = df_model.drop(columns=[tick])
        
        # Train one global non-linear model for the stock
        model = HistGradientBoostingRegressor(max_iter=100, learning_rate=0.05, max_depth=5, random_state=42)
        
        # Compute out-of-sample permutation importance across splits
        importances_list = []
        for train_idx, test_idx in tscv.split(X_train):
            X_tr, y_tr = X_train.iloc[train_idx], y_target.iloc[train_idx]
            X_te, y_te = X_train.iloc[test_idx], y_target.iloc[test_idx]
            
            model.fit(X_tr, y_tr)
            
            # Permutation importance
            r = permutation_importance(model, X_te, y_te, n_repeats=5, random_state=42, scoring='neg_mean_squared_error')
            importances_list.append(r.importances_mean)
            
        # Average importance
        avg_importances = np.mean(importances_list, axis=0)
        
        # Only keep features where Out-of-Sample importance is strictly positive (meaning they help predict)
        # y_lag1 is index 0
        for idx, col in enumerate(X_train.columns):
            if col == "y_lag1": continue
            imp = avg_importances[idx]
            if imp > 0.0:  # Threshold for non-linear causality (any positive out-of-sample value)
                cat, feat, lag_str = col.split("::")
                lag_val = int(lag_str.split("_")[1])
                
                valid_edges.append({
                    "source": cat,
                    "target": tick,
                    "feature": feat,
                    "lag_minutes": lag_val * 15, # Convert back to minutes
                    "importance": imp
                })

    df_edges = pd.DataFrame(valid_edges)
    logger.info(f"Discovered {len(df_edges)} non-linear causal edges.")
    
    # Save edges
    if not df_edges.empty:
        # Scale weights based on importance
        df_edges["weight"] = df_edges.groupby("target")["importance"].transform(lambda x: x / x.sum())
        out_path = DIR_STRATEGIES / "causal_edges.parquet"
        df_edges.to_parquet(out_path)
        logger.info(f"Saved non-linear causal edges to {out_path}")
    else:
        logger.warning("No non-linear causal edges found.")

if __name__ == "__main__":
    extract_non_linear_causality()
