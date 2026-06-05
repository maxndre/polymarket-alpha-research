"""
step3_distributions.py — Step 3: Statistical distribution analysis.

Computes full distributional profiles for each key market variable.
All aggregations use groupby + vectorized ops; no iterrows/itertuples.
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
    Step 3 — Statistical and distributional analysis of the usable market universe.

    Variables analysed
    ------------------
    1. Trade count per market
    2. Volume USDC (cumulative notional)
    3. Volume in shares
    4. Bid-ask spread proxy (mean bar-level high-low spread per market)
    5. Price-update frequency (mean Δt between consecutive active bars, minutes)
    6. Market duration in hours
    7. Resolution outcome distribution (YES vs NO ratio)

    Returns
    -------
    dict with one key per variable + "_per_market_df" (DataFrame for step 4).
    """
    logger.info("=== STEP 3: STATISTICAL DISTRIBUTIONS ===")

    active_bars = df_usable_bars[df_usable_bars["trades_count_1m"] > 0].copy()

    # 1-3. Trade count, volume, shares
    agg_base = (
        df_usable_bars
        .groupby("condition_id", observed=True)
        .agg(
            total_trades     = ("trades_count_1m",  "sum"),
            total_vol_usdc   = ("notional_usdc_1m", "sum"),
            total_vol_shares = ("volume_shares_1m", "sum"),
        )
        .reset_index()
    )

    # 4. Bid-ask spread proxy: mean(high - low) per bar, then mean per market
    bars_copy = df_usable_bars.copy()
    bars_copy["bar_spread"] = bars_copy["high_price"] - bars_copy["low_price"]
    spread_agg = (
        bars_copy
        .groupby("condition_id", observed=True)["bar_spread"]
        .agg(spread_mean="mean", spread_std="std")
        .reset_index()
    )

    # 5. Price-update frequency (Δt between consecutive active bars)
    logger.info("Computing Δt (price-update frequency) ...")
    active_sorted = active_bars.sort_values(["condition_id", "minute_ts"])
    active_sorted["prev_ts"]    = active_sorted.groupby("condition_id", observed=True)["minute_ts"].shift(1)
    active_sorted["delta_t_min"] = (
        (active_sorted["minute_ts"] - active_sorted["prev_ts"]).dt.total_seconds() / 60.0
    )
    freq_agg = (
        active_sorted.dropna(subset=["delta_t_min"])
        .groupby("condition_id", observed=True)["delta_t_min"]
        .agg(mean_dt_min="mean", median_dt_min="median")
        .reset_index()
    )

    # 6. Market duration
    duration_h = (
        (df_usable_markets["end_date"] - df_usable_markets["start_date"])
        .dt.total_seconds() / 3600.0
    )

    pm = (
        agg_base
        .merge(spread_agg, on="condition_id", how="left")
        .merge(freq_agg,   on="condition_id", how="left")
        .merge(
            df_usable_markets[["condition_id", "resolved_yes", "resolved_no"]],
            on="condition_id", how="left",
        )
    )

    results = {}
    results["trade_count"]       = distribution_summary(pm["total_trades"],      "trade_count")
    results["volume_usdc"]       = distribution_summary(pm["total_vol_usdc"],     "volume_usdc")
    results["volume_shares"]     = distribution_summary(pm["total_vol_shares"],   "volume_shares")
    results["spread_mean"]       = distribution_summary(pm["spread_mean"],        "spread_mean")
    results["spread_std"]        = distribution_summary(pm["spread_std"],         "spread_std")
    results["mean_dt_min"]       = distribution_summary(pm["mean_dt_min"],        "mean_dt_min")
    results["market_duration_h"] = distribution_summary(duration_h,              "duration_h")

    # 7. Resolution outcome ratio
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

    results["_per_market_df"] = pm  # for step 4; prefix _ marks internal use

    logger.info("Distributions computed for %d usable markets.", pm["condition_id"].nunique())
    for key, val in results.items():
        if key.startswith("_") or not isinstance(val, dict) or "median" not in val:
            continue
        logger.info("  %-22s  median=%.4f  mean=%.4f  skew=%.2f",
                    key, val["median"], val["mean"], val["skewness"])

    return results
