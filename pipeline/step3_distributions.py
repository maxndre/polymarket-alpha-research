"""
step3_distributions.py – ÉTAPE 3 : Propriétés statistiques et distributionnelles.

Computes full distributional profiles for each key market variable.
Forbidden: iterrows / itertuples. All aggregations use groupby + vectorized ops.
"""

from __future__ import annotations
import logging

import numpy as np
import pandas as pd

from .utils import distribution_summary

logger = logging.getLogger(__name__)


def compute_market_distributions(
    df_usable_markets: pd.DataFrame,
    df_usable_bars:    pd.DataFrame,
) -> dict:
    """
    ÉTAPE 3 – Analyse statistique et distributionnelle de l'univers exploitable.

    For each variable we compute:
      median, mean, std, q05, q25, q75, q95, skewness, min, max, n

    Variables analysed
    ------------------
    1. Trade count per market
    2. Volume USDC (cumulative notional)
    3. Volume in shares (notional)
    4. Bid-ask spread proxy per market (mean of bar-level high-low spread)
    5. Price-update frequency (mean Δt between consecutive active bars, minutes)
    6. Market duration in hours
    7. Resolution outcome distribution (YES vs NO ratio)

    Returns
    -------
    dict with one key per variable + "resolution_outcomes".
    """
    logger.info("=== ÉTAPE 3 : PROPRIÉTÉS STATISTIQUES ET DISTRIBUTIONNELLES ===")

    # ── Per-market aggregates (vectorized groupby) ────────────────────────────
    logger.info("Aggregating per-market statistics …")

    # Active bars only (bars with actual trades)
    active_bars = df_usable_bars[df_usable_bars["trades_count_1m"] > 0].copy()

    # 1-2. Trade count & volume
    agg_base = (
        df_usable_bars
        .groupby("condition_id", observed=True)
        .agg(
            total_trades       = ("trades_count_1m",  "sum"),
            total_vol_usdc     = ("notional_usdc_1m", "sum"),
            total_vol_shares   = ("volume_shares_1m", "sum"),
        )
        .reset_index()
    )

    # 3. Bid-ask spread proxy: mean(high - low) per bar, then mean per market
    #    This is a within-bar spread; for markets with a real order book this
    #    underestimates the true spread, but is consistent across all markets.
    df_usable_bars_copy = df_usable_bars.copy()
    df_usable_bars_copy["bar_spread"] = (
        df_usable_bars_copy["high_price"] - df_usable_bars_copy["low_price"]
    )
    spread_agg = (
        df_usable_bars_copy
        .groupby("condition_id", observed=True)["bar_spread"]
        .agg(spread_mean="mean", spread_std="std")
        .reset_index()
    )

    # 4. Price-update frequency (Δt between consecutive active bars)
    logger.info("Computing Δt (price-update frequency) …")
    active_sorted = active_bars.sort_values(["condition_id", "minute_ts"])
    active_sorted["prev_ts"] = (
        active_sorted.groupby("condition_id", observed=True)["minute_ts"].shift(1)
    )
    active_sorted["delta_t_min"] = (
        (active_sorted["minute_ts"] - active_sorted["prev_ts"])
        .dt.total_seconds() / 60.0
    )
    freq_agg = (
        active_sorted.dropna(subset=["delta_t_min"])
        .groupby("condition_id", observed=True)["delta_t_min"]
        .agg(mean_dt_min="mean", median_dt_min="median")
        .reset_index()
    )

    # 5. Market duration from market table
    duration_h = (
        (df_usable_markets["end_date"] - df_usable_markets["start_date"])
        .dt.total_seconds() / 3600.0
    )

    # Merge everything into one per-market table
    pm = (
        agg_base
        .merge(spread_agg, on="condition_id", how="left")
        .merge(freq_agg,   on="condition_id", how="left")
        .merge(
            df_usable_markets[["condition_id", "resolved_yes", "resolved_no"]],
            on="condition_id", how="left"
        )
    )

    # ── Distributional summaries ──────────────────────────────────────────────
    logger.info("Building distributional summaries …")
    results = {}

    results["trade_count"]      = distribution_summary(pm["total_trades"],       "trade_count")
    results["volume_usdc"]      = distribution_summary(pm["total_vol_usdc"],      "volume_usdc")
    results["volume_shares"]    = distribution_summary(pm["total_vol_shares"],    "volume_shares")
    results["spread_mean"]      = distribution_summary(pm["spread_mean"],         "spread_mean")
    results["spread_std"]       = distribution_summary(pm["spread_std"],          "spread_std")
    results["mean_dt_min"]      = distribution_summary(pm["mean_dt_min"],         "mean_dt_min")
    results["market_duration_h"]= distribution_summary(duration_h,               "duration_h")

    # 6. Resolution outcomes (YES vs NO share)
    n_yes = int(pm["resolved_yes"].sum())
    n_no  = int(pm["resolved_no"].sum())
    n_tot = n_yes + n_no
    results["resolution_outcomes"] = {
        "n_yes":   n_yes,
        "n_no":    n_no,
        "n_total": n_tot,
        "pct_yes": round(n_yes / n_tot * 100, 2) if n_tot > 0 else np.nan,
        "pct_no":  round(n_no  / n_tot * 100, 2) if n_tot > 0 else np.nan,
    }

    # 7. Cross-sectional table for downstream use
    results["_per_market_df"] = pm  # stored for step 4; prefix _ = internal

    logger.info(
        "Distributions computed for %d usable markets.",
        pm["condition_id"].nunique()
    )
    for key, val in results.items():
        if key.startswith("_") or not isinstance(val, dict) or "median" not in val:
            continue
        logger.info(
            "  %-22s  median=%.4f  mean=%.4f  skew=%.2f",
            key, val["median"], val["mean"], val["skewness"],
        )

    return results
