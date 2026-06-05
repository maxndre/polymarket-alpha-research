"""
step4_concentration.py — Step 4: Market concentration analysis.

Quantifies how activity (volume, trades) is distributed across:
  A) Individual markets (Gini coefficient, top-1/5/10% share)
  B) Categories / tag buckets
  C) Time (last-48h volume share = end-of-life surge proxy)
  D) Lorenz curve data (for visualization)

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
    per_market_df:     pd.DataFrame,   # from step3 _per_market_df
) -> dict:
    """
    Step 4 — Market concentration analysis.

    Returns a nested dict with market-level, categorical, and temporal metrics.
    """
    logger.info("=== STEP 4: CONCENTRATION ANALYSIS ===")
    results: dict = {}
    usable_cids = set(df_usable_markets["condition_id"])

    # ── 4.A  Market-level concentration ──────────────────────────────────────
    logger.info("4.A — Market-level concentration ...")

    vol = per_market_df["total_vol_usdc"].fillna(0.0)
    trd = per_market_df["total_trades"].fillna(0.0)

    results["market_concentration"] = {
        "gini_volume":        round(gini_coefficient(vol.values),   4),
        "gini_trades":        round(gini_coefficient(trd.values),   4),
        "top01_pct_volume":   round(top_n_share(vol, 0.01) * 100,  2),
        "top05_pct_volume":   round(top_n_share(vol, 0.05) * 100,  2),
        "top10_pct_volume":   round(top_n_share(vol, 0.10) * 100,  2),
        "top01_pct_trades":   round(top_n_share(trd, 0.01) * 100,  2),
        "top05_pct_trades":   round(top_n_share(trd, 0.05) * 100,  2),
        "top10_pct_trades":   round(top_n_share(trd, 0.10) * 100,  2),
        "n_markets_analysed": int(len(per_market_df)),
    }
    logger.info(
        "  Volume: Gini=%.4f  Top-1%%=%.1f%%  Top-5%%=%.1f%%",
        results["market_concentration"]["gini_volume"],
        results["market_concentration"]["top01_pct_volume"],
        results["market_concentration"]["top05_pct_volume"],
    )

    # ── 4.B  Categorical concentration ───────────────────────────────────────
    logger.info("4.B — Categorical concentration (tag buckets) ...")

    mtag_usable = df_market_tag[df_market_tag["condition_id"].isin(usable_cids)].copy()
    mtag_vol = mtag_usable.merge(
        per_market_df[["condition_id", "total_vol_usdc", "total_trades"]],
        on="condition_id", how="left",
    )
    cat_agg = (
        mtag_vol.groupby("tag_label", observed=True)
        .agg(
            n_markets         = ("condition_id",    "nunique"),
            total_volume_usdc = ("total_vol_usdc",  "sum"),
            total_trades      = ("total_trades",    "sum"),
        )
        .reset_index()
        .sort_values("total_volume_usdc", ascending=False)
    )
    grand_vol = cat_agg["total_volume_usdc"].sum()
    grand_trd = cat_agg["total_trades"].sum()
    cat_agg["pct_volume"] = (cat_agg["total_volume_usdc"] / grand_vol * 100).round(2)
    cat_agg["pct_trades"] = (cat_agg["total_trades"]       / grand_trd * 100).round(2)

    results["categorical_concentration"] = cat_agg.head(40).to_dict(orient="records")
    logger.info("  Top-5 tags by volume: %s", cat_agg.head(5)["tag_label"].tolist())

    # ── 4.C  Temporal concentration (end-of-life surge) ──────────────────────
    logger.info("4.C — Temporal concentration (last-48h volume share) ...")

    end_dates       = df_usable_markets.set_index("condition_id")["end_date"]
    bars_with_end   = df_usable_bars.copy()
    bars_with_end["end_date"]     = bars_with_end["condition_id"].map(end_dates)
    bars_with_end["hours_to_end"] = (
        (bars_with_end["end_date"] - bars_with_end["minute_ts"]).dt.total_seconds() / 3600.0
    )

    total_vol  = df_usable_bars.groupby("condition_id", observed=True)["notional_usdc_1m"].sum().rename("total_vol")
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
        "last48h_share_distribution":          distribution_summary(temporal_df["last48h_share"], "last48h_share"),
        "n_markets_last48h_share_gt50pct":     int((temporal_df["last48h_share"] > 0.50).sum()),
        "n_markets_last48h_share_gt80pct":     int((temporal_df["last48h_share"] > 0.80).sum()),
    }

    # ── 4.D  Lorenz curve (200-point downsample for JSON export) ─────────────
    vol_sorted = np.sort(vol.values)
    vol_cumsum = np.cumsum(vol_sorted)
    lorenz_y   = vol_cumsum / vol_cumsum[-1] if vol_cumsum[-1] > 0 else vol_cumsum
    lorenz_x   = np.linspace(0, 1, len(lorenz_y))
    idx        = np.round(np.linspace(0, len(lorenz_x) - 1, 200)).astype(int)
    results["lorenz_curve"] = {"x": lorenz_x[idx].tolist(), "y": lorenz_y[idx].tolist()}

    logger.info("Concentration analysis complete.")
    return results
