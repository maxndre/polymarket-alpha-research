"""
step4_concentration.py – ÉTAPE 4 : Analyse de Concentration.

Quantifies how activity (volume, trades) is distributed across:
  A) Individual markets (Gini, top-1%/5%/10% share)
  B) Categories / tag buckets
  C) Time (last-48h volume share = end-of-life surge proxy)

All vectorized; no iterrows/itertuples.
"""

from __future__ import annotations
import logging

import numpy as np
import pandas as pd

from .utils import gini_coefficient, top_n_share, distribution_summary

logger = logging.getLogger(__name__)


def analyze_market_concentration(
    df_usable_markets: pd.DataFrame,
    df_usable_bars:    pd.DataFrame,
    df_market_tag:     pd.DataFrame,
    per_market_df:     pd.DataFrame,          # from step3 _per_market_df
) -> dict:
    """
    ÉTAPE 4 – Analyse de concentration.

    Parameters
    ----------
    df_usable_markets : filtered market DataFrame (output of step2).
    df_usable_bars    : filtered 1-min bars (output of step2).
    df_market_tag     : raw market_tag join table.
    per_market_df     : per-market aggregates from step3.

    Returns
    -------
    Nested dict with market, categorical, and temporal concentration metrics.
    """
    logger.info("=== ÉTAPE 4 : ANALYSE DE CONCENTRATION ===")
    results: dict = {}

    usable_cids = set(df_usable_markets["condition_id"])

    # ── 4.A  Market-level concentration ──────────────────────────────────────
    logger.info("4.A – Concentration par marché …")

    vol = per_market_df["total_vol_usdc"].fillna(0.0)
    trd = per_market_df["total_trades"].fillna(0.0)

    gini_vol = gini_coefficient(vol.values)
    gini_trd = gini_coefficient(trd.values)

    top1_vol  = top_n_share(vol, 0.01)
    top5_vol  = top_n_share(vol, 0.05)
    top10_vol = top_n_share(vol, 0.10)
    top1_trd  = top_n_share(trd, 0.01)
    top5_trd  = top_n_share(trd, 0.05)
    top10_trd = top_n_share(trd, 0.10)

    logger.info(
        "  Volume: Gini=%.4f  Top-1%%=%.1f%%  Top-5%%=%.1f%%  Top-10%%=%.1f%%",
        gini_vol, top1_vol * 100, top5_vol * 100, top10_vol * 100
    )

    results["market_concentration"] = {
        "gini_volume":        round(gini_vol,   4),
        "gini_trades":        round(gini_trd,   4),
        "top01_pct_volume":   round(top1_vol  * 100, 2),
        "top05_pct_volume":   round(top5_vol  * 100, 2),
        "top10_pct_volume":   round(top10_vol * 100, 2),
        "top01_pct_trades":   round(top1_trd  * 100, 2),
        "top05_pct_trades":   round(top5_trd  * 100, 2),
        "top10_pct_trades":   round(top10_trd * 100, 2),
        "n_markets_analysed": int(len(per_market_df)),
    }

    # ── 4.B  Categorical concentration ───────────────────────────────────────
    logger.info("4.B – Concentration catégorielle (tag buckets) …")

    # Use market_tag to assign a primary bucket to each usable market
    # We take the first tag_slug match from selected_filter_tag buckets
    mtag_usable = df_market_tag[df_market_tag["condition_id"].isin(usable_cids)].copy()

    # Merge volume info
    mtag_vol = mtag_usable.merge(
        per_market_df[["condition_id", "total_vol_usdc", "total_trades"]],
        on="condition_id", how="left"
    )

    # Aggregate by tag_label
    cat_agg = (
        mtag_vol.groupby("tag_label", observed=True)
        .agg(
            n_markets           = ("condition_id", "nunique"),
            total_volume_usdc   = ("total_vol_usdc", "sum"),
            total_trades        = ("total_trades",   "sum"),
        )
        .reset_index()
        .sort_values("total_volume_usdc", ascending=False)
    )

    grand_vol = cat_agg["total_volume_usdc"].sum()
    grand_trd = cat_agg["total_trades"].sum()
    cat_agg["pct_volume"] = (cat_agg["total_volume_usdc"] / grand_vol * 100).round(2)
    cat_agg["pct_trades"] = (cat_agg["total_trades"]       / grand_trd * 100).round(2)

    results["categorical_concentration"] = cat_agg.head(40).to_dict(orient="records")
    logger.info("  Top-5 tags by volume: %s",
                cat_agg.head(5)["tag_label"].tolist())

    # ── 4.C  Temporal concentration (end-of-life surge) ──────────────────────
    logger.info("4.C – Concentration temporelle (volume last-48h / total) …")

    # For each usable market, compute:
    #   total volume  AND  volume in last 48 hours before end_date
    end_dates = df_usable_markets.set_index("condition_id")["end_date"]

    # Merge end_date into bars (vectorized via map)
    bars_with_end = df_usable_bars.copy()
    bars_with_end["end_date"] = bars_with_end["condition_id"].map(end_dates)

    bars_with_end["hours_to_end"] = (
        (bars_with_end["end_date"] - bars_with_end["minute_ts"])
        .dt.total_seconds() / 3600.0
    )

    # Total volume per market
    total_vol = (
        df_usable_bars
        .groupby("condition_id", observed=True)["notional_usdc_1m"]
        .sum()
        .rename("total_vol")
    )

    # Last-48h volume per market (vectorized boolean mask)
    last48h_vol = (
        bars_with_end[bars_with_end["hours_to_end"].between(0, 48)]
        .groupby("condition_id", observed=True)["notional_usdc_1m"]
        .sum()
        .rename("last48h_vol")
    )

    temporal_df = (
        total_vol.to_frame()
        .join(last48h_vol, how="left")
        .fillna({"last48h_vol": 0.0})
        .reset_index()
    )
    temporal_df["last48h_share"] = (
        temporal_df["last48h_vol"] / temporal_df["total_vol"].replace(0, np.nan)
    )

    results["temporal_concentration"] = {
        "last48h_share_distribution": distribution_summary(
            temporal_df["last48h_share"], "last48h_share"
        ),
        "n_markets_with_last48h_surge_gt50pct": int(
            (temporal_df["last48h_share"] > 0.50).sum()
        ),
        "n_markets_with_last48h_surge_gt80pct": int(
            (temporal_df["last48h_share"] > 0.80).sum()
        ),
    }

    logger.info(
        "  last-48h share: median=%.3f  mean=%.3f  (markets with >50%% in last 48h: %d)",
        results["temporal_concentration"]["last48h_share_distribution"]["median"],
        results["temporal_concentration"]["last48h_share_distribution"]["mean"],
        results["temporal_concentration"]["n_markets_with_last48h_surge_gt50pct"],
    )

    # ── 4.D  Lorenz curve data (for plotting in LaTeX / matplotlib) ───────────
    logger.info("4.D – Computing Lorenz curve data …")
    vol_sorted = np.sort(vol.values)
    vol_cumsum = np.cumsum(vol_sorted)
    lorenz_y = vol_cumsum / vol_cumsum[-1] if vol_cumsum[-1] > 0 else vol_cumsum
    lorenz_x = np.linspace(0, 1, len(lorenz_y))
    # Downsample to 200 points for JSON
    idx = np.round(np.linspace(0, len(lorenz_x) - 1, 200)).astype(int)
    results["lorenz_curve"] = {
        "x": lorenz_x[idx].tolist(),
        "y": lorenz_y[idx].tolist(),
    }

    logger.info("Concentration analysis complete.")
    return results
