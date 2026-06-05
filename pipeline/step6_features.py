"""
step6_features.py — Step 6: Advanced 1-minute feature generation.

For each (minute_ts, llm_category) group, computes:
  - Price statistics: mean, std, VWAP, quantiles, skewness, kurtosis
  - Entropy: Shannon entropy of binary market uncertainty
  - Concentration: Herfindahl-Hirschman Index (HHI) of within-group volume
  - Rolling windows: 5m, 1h returns, volume sums, trades counts, entropy change

Output: dense 1-minute wide table (one row per minute, one column per feature × category).
Saved as both long-format and wide-format Parquet, with train/test splits.
"""

from __future__ import annotations
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .config import PROCESSED_DIR, FEATURES_DIR

logger = logging.getLogger(__name__)

TRAIN_END = "2025-11-01 00:00:00+00:00"


def run_features_generation() -> None:
    """Step 6 — Generate advanced category-level features from 1-min trade data."""
    logger.info("=== STEP 6: FEATURE GENERATION ===")
    t_start = time.time()

    FEATURES_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load data ──────────────────────────────────────────────────────────────
    logger.info("Loading parquet datasets ...")
    df_market     = pd.read_parquet(PROCESSED_DIR / "market.parquet",           columns=["market_id", "condition_id"])
    df_categories = pd.read_parquet(PROCESSED_DIR / "market_categories.parquet", columns=["market_id", "llm_category"])
    df_polarities = pd.read_parquet(PROCESSED_DIR / "market_polarities.parquet", columns=["market_id", "polarity"])
    df_trades     = pd.read_parquet(PROCESSED_DIR / "trade_price_1m_usable.parquet")
    logger.info("Loaded %d trade rows", len(df_trades))

    # ── Map condition_id to category and polarity ──────────────────────────────
    for df in [df_market, df_categories, df_polarities]:
        df["market_id"] = df["market_id"].astype(int)

    mapping = (
        df_market.merge(df_categories, on="market_id")
        .merge(df_polarities, on="market_id")
        [["condition_id", "llm_category", "polarity"]]
        .drop_duplicates()
    )
    df_trades = df_trades.merge(mapping, on="condition_id", how="left")
    df_trades["llm_category"] = df_trades["llm_category"].fillna("Other")
    df_trades["polarity"]     = df_trades["polarity"].fillna(0).astype(int)

    # ── Pre-compute signed price and notional ──────────────────────────────────
    logger.info("Pre-computing vectorized features ...")

    df_trades["signed_close"] = np.where(
        df_trades["polarity"] != 0,
        df_trades["close_price"] * df_trades["polarity"],
        np.nan,
    )
    df_trades["price_x_notional_signed"] = df_trades["signed_close"]    * df_trades["notional_usdc_1m"]
    df_trades["notional_signed"]         = np.where(df_trades["polarity"] != 0, df_trades["notional_usdc_1m"], np.nan)
    df_trades["price_x_notional"]        = df_trades["close_price"]     * df_trades["notional_usdc_1m"]
    df_trades["notional_squared"]        = df_trades["notional_usdc_1m"] ** 2
    df_trades["high_minus_low"]          = df_trades["high_price"] - df_trades["low_price"]

    # Shannon entropy: -[p*log(p) + (1-p)*log(1-p)]
    eps = 1e-9
    p = df_trades["close_price"].clip(0.0, 1.0)
    df_trades["shannon_entropy"] = -(p * np.log(p + eps) + (1.0 - p) * np.log(1.0 - p + eps))

    # ── Groupby aggregations ───────────────────────────────────────────────────
    logger.info("Aggregating by (minute_ts, llm_category) ...")
    t_agg = time.time()

    grp = df_trades.groupby(["minute_ts", "llm_category"])

    df_agg = grp.agg({
        "signed_close":             ["mean", "std", "max", "min", "median"],
        "price_x_notional_signed":  "sum",
        "notional_signed":          "sum",
        "notional_usdc_1m":         "sum",
        "notional_squared":         "sum",
        "volume_shares_1m":         "sum",
        "trades_count_1m":          "sum",
        "condition_id":             "nunique",
        "high_minus_low":           "mean",
        "shannon_entropy":          "mean",
    })
    df_agg.columns = [f"{c[0]}_{c[1]}" for c in df_agg.columns]
    df_agg["price_skew"] = grp["signed_close"].skew()
    df_agg["price_kurt"] = grp["signed_close"].apply(lambda x: x.kurt())
    df_agg["price_q25"]  = grp["signed_close"].quantile(0.25)
    df_agg["price_q75"]  = grp["signed_close"].quantile(0.75)
    df_agg = df_agg.reset_index()

    # ── Derived features ───────────────────────────────────────────────────────
    # VWAP (volume-weighted average price) for signed markets
    df_agg["vwap"] = (df_agg["price_x_notional_signed_sum"] / df_agg["notional_signed_sum"])
    df_agg["vwap"] = df_agg["vwap"].fillna(df_agg["signed_close_mean"])

    # HHI of within-group volume concentration
    df_agg["hhi_volume"] = (df_agg["notional_squared_sum"] / df_agg["notional_usdc_1m_sum"] ** 2).fillna(1.0).clip(0.0, 1.0)

    df_agg = df_agg.rename(columns={
        "signed_close_mean":     "price_mean",
        "signed_close_std":      "price_std",
        "signed_close_max":      "price_max",
        "signed_close_min":      "price_min",
        "signed_close_median":   "price_median",
        "volume_shares_1m_sum":  "volume_shares_total",
        "notional_usdc_1m_sum":  "notional_usdc_total",
        "trades_count_1m_sum":   "trades_count_total",
        "condition_id_nunique":  "active_markets_count",
        "high_minus_low_mean":   "spread_mean",
        "shannon_entropy_mean":  "shannon_entropy",
    }).drop(columns=["price_x_notional_signed_sum", "notional_signed_sum", "notional_squared_sum"])

    logger.info("Aggregations done in %.1fs — %d rows", time.time() - t_agg, len(df_agg))

    # ── Reindex onto dense 1-minute grid ──────────────────────────────────────
    logger.info("Building dense 1-minute timeline ...")
    categories    = df_agg["llm_category"].unique()
    full_timeline = pd.date_range("2025-01-01 00:00+00:00", "2025-12-31 23:59+00:00", freq="min", name="minute_ts")
    mux           = pd.MultiIndex.from_product([full_timeline, categories], names=["minute_ts", "llm_category"])
    df_dense      = df_agg.set_index(["minute_ts", "llm_category"]).reindex(mux).reset_index()

    df_dense["active_markets_count"] = df_dense["active_markets_count"].fillna(0).astype(int)
    df_dense["volume_shares_total"]  = df_dense["volume_shares_total"].fillna(0.0)
    df_dense["notional_usdc_total"]  = df_dense["notional_usdc_total"].fillna(0.0)
    df_dense["trades_count_total"]   = df_dense["trades_count_total"].fillna(0).astype(int)

    # ── Rolling features per category ─────────────────────────────────────────
    logger.info("Computing rolling features ...")
    t_roll = time.time()
    cat_dfs = []
    for cat in categories:
        df_cat = df_dense[df_dense["llm_category"] == cat].sort_values("minute_ts").copy()
        df_cat["return_1m"]       = df_cat["vwap"].pct_change(1)
        df_cat["return_5m"]       = df_cat["vwap"].pct_change(5)
        df_cat["vol_roll_5m"]     = df_cat["notional_usdc_total"].rolling(5,  min_periods=1).sum()
        df_cat["vol_roll_1h"]     = df_cat["notional_usdc_total"].rolling(60, min_periods=1).sum()
        df_cat["trades_roll_5m"]  = df_cat["trades_count_total"].rolling(5,  min_periods=1).sum()
        df_cat["trades_roll_1h"]  = df_cat["trades_count_total"].rolling(60, min_periods=1).sum()
        df_cat["entropy_delta_5m"]= df_cat["shannon_entropy"].diff(5)
        cat_dfs.append(df_cat)

    df_long = pd.concat(cat_dfs, ignore_index=True)
    logger.info("Rolling features done in %.1fs", time.time() - t_roll)

    # Wide format: each row = one minute, columns = feature × category
    df_wide = df_long.pivot(index="minute_ts", columns="llm_category")
    df_wide.columns = [f"{col[1]}_{col[0]}" for col in df_wide.columns]
    df_wide = df_wide.reset_index()

    # ── Save train / test splits ───────────────────────────────────────────────
    train_mask_long = df_long["minute_ts"] < TRAIN_END
    train_mask_wide = df_wide["minute_ts"] < TRAIN_END

    files: dict[str, pd.DataFrame] = {
        "features_1m_long.parquet":       df_long,
        "features_1m_train_long.parquet": df_long[train_mask_long],
        "features_1m_test_long.parquet":  df_long[~train_mask_long],
        "features_1m_wide.parquet":       df_wide,
        "features_1m_train_wide.parquet": df_wide[train_mask_wide],
        "features_1m_test_wide.parquet":  df_wide[~train_mask_wide],
    }

    logger.info("Saving Parquet files ...")
    for name, df in files.items():
        out = FEATURES_DIR / name
        df.to_parquet(out, index=False)
        logger.info("  Saved: %s (%d rows, %.1f MB)", name, len(df), out.stat().st_size / 1e6)

    logger.info("=== Step 6 complete in %.1fs ===", time.time() - t_start)
