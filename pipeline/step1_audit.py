"""
step1_audit.py — Step 1: Database audit and quality mapping.

Produces a comprehensive quality report as a nested dictionary.
Fully vectorized — no iterrows/itertuples on large DataFrames.
"""

from __future__ import annotations
import json
import logging
from collections import Counter

import numpy as np
import pandas as pd

from .utils import distribution_summary

logger = logging.getLogger(__name__)


def _explode_tags(df_markets: pd.DataFrame) -> pd.DataFrame:
    """
    Parse the tags_json TEXT column and return a long-form DataFrame:
        condition_id | tag_id | tag_slug | tag_label
    """
    import json as _json

    def _parse_row(s: str):
        if not isinstance(s, str) or not s.strip():
            return []
        try:
            return _json.loads(s)
        except Exception:
            return []

    tags_series = df_markets["tags_json"].apply(_parse_row)
    tag_records = []
    for cid, tag_list in zip(df_markets["condition_id"], tags_series):
        for t in tag_list:
            tag_records.append({
                "condition_id": cid,
                "tag_id":       t.get("tag_id"),
                "tag_slug":     t.get("slug"),
                "tag_label":    t.get("label"),
            })
    return pd.DataFrame(tag_records)


def map_and_audit_database(
    df_markets: pd.DataFrame,
    df_bars: pd.DataFrame,
    df_token: pd.DataFrame,
    df_market_tag: pd.DataFrame,
) -> tuple[dict, pd.DataFrame]:
    """
    Step 1 — Database audit and quality mapping.

    Parameters
    ----------
    df_markets    : market table (~58k rows).
    df_bars       : trade_price_1m table (~10M rows).
    df_token      : token_outcome table (~116k rows).
    df_market_tag : market_tag join table (~392k rows).

    Returns
    -------
    (report dict, per_market DataFrame)
    """
    logger.info("=== STEP 1: DATABASE AUDIT ===")
    report: dict = {}

    # ── 1.1 Basic counts ──────────────────────────────────────────────────────
    n_markets   = df_markets["condition_id"].nunique()
    n_events    = df_markets["event_id"].nunique()
    n_token_ids = df_token["token_id"].nunique()
    n_bars      = len(df_bars)

    logger.info("Markets (condition_id): %d", n_markets)
    logger.info("Events  (event_id):     %d", n_events)
    logger.info("Token IDs:              %d", n_token_ids)
    logger.info("1-min OHLCV bars:       %d", n_bars)

    df_tags_long = _explode_tags(df_markets)
    tag_counts = (
        df_tags_long.groupby("tag_label")["condition_id"]
        .nunique()
        .sort_values(ascending=False)
    )
    bucket_counts = (
        df_market_tag.groupby("tag_slug")["condition_id"]
        .nunique()
        .sort_values(ascending=False)
        .head(20)
        .to_dict()
    )
    outcome_dist = df_token["outcome_label"].value_counts().to_dict()

    report["overview"] = {
        "n_markets_unique":             n_markets,
        "n_events_unique":              n_events,
        "n_token_ids_unique":           n_token_ids,
        "n_bar_rows_total":             n_bars,
        "n_distinct_tags":              int(tag_counts.shape[0]),
        "top_30_tags_by_market_count":  {str(k): int(v) for k, v in tag_counts.head(30).items()},
        "top_20_slugs_in_market_tag":   {str(k): int(v) for k, v in bucket_counts.items()},
        "outcome_label_distribution":   {str(k): int(v) for k, v in outcome_dist.items()},
    }

    # ── 1.2 Temporal analysis ─────────────────────────────────────────────────
    logger.info("Computing market lifetimes ...")

    start = df_markets["start_date"]
    end   = df_markets["end_date"]
    duration_h = (end - start).dt.total_seconds() / 3600.0

    report["temporal"] = {
        "bar_date_range":     [str(df_bars["minute_ts"].min()), str(df_bars["minute_ts"].max())],
        "market_start_range": [str(start.min()), str(start.max())],
        "market_end_range":   [str(end.min()),   str(end.max())],
        "duration_hours":     distribution_summary(duration_h, "duration_hours"),
    }

    # ── 1.3 Missing-value audit ───────────────────────────────────────────────
    logger.info("Auditing missing values ...")

    def _missing_rates(df: pd.DataFrame) -> dict:
        total = len(df)
        if total == 0:
            return {}
        return (df.isnull().sum() / total * 100).round(2).to_dict()

    report["missing_rates"] = {
        "market":       _missing_rates(df_markets),
        "trade_price":  _missing_rates(df_bars),
        "token":        _missing_rates(df_token),
    }

    # ── 1.4 Microstructure anomaly audit ─────────────────────────────────────
    logger.info("Detecting microstructure anomalies ...")

    price_cols = ["open_price", "high_price", "low_price", "close_price"]

    def _pct_out_of_bounds(series: pd.Series, lo=0.0, hi=1.0) -> float:
        valid = series.dropna()
        if len(valid) == 0:
            return np.nan
        return float(((valid < lo) | (valid > hi)).sum() / len(valid) * 100)

    price_anomalies: dict = {}
    for col in price_cols:
        if col not in df_bars.columns:
            continue
        s = df_bars[col]
        price_anomalies[col] = {
            "pct_nan":       float(s.isna().mean() * 100),
            "pct_out_of_01": _pct_out_of_bounds(s),
            "pct_negative":  float((s.dropna() < 0).mean() * 100),
        }

    negative_spread = zero_spread = -1
    if {"high_price", "low_price"}.issubset(df_bars.columns):
        negative_spread = int((df_bars["high_price"] < df_bars["low_price"]).sum())
        zero_spread     = int((df_bars["high_price"] == df_bars["low_price"]).sum())

    active_bars  = df_bars[df_bars["trades_count_1m"] > 0]
    ts_inverted  = int((df_markets["end_date"] < df_markets["start_date"]).sum())

    report["microstructure_anomalies"] = {
        "price_anomalies_by_column":     price_anomalies,
        "n_bars_high_lt_low":            negative_spread,
        "n_bars_high_eq_low":            zero_spread,
        "n_active_bars_missing_price":   int(active_bars[price_cols].isna().any(axis=1).sum()),
        "n_markets_timestamp_inverted":  ts_inverted,
    }

    # ── 1.5 Activity distribution per market ──────────────────────────────────
    logger.info("Computing per-market activity summaries ...")

    per_market = (
        df_bars.groupby("condition_id", observed=True)
        .agg(
            total_trades       = ("trades_count_1m",  "sum"),
            total_volume_usdc  = ("notional_usdc_1m", "sum"),
            n_bars_total       = ("minute_ts",         "count"),
            n_active_bars      = ("trades_count_1m",  lambda x: (x > 0).sum()),
            first_bar          = ("minute_ts",         "min"),
            last_bar           = ("minute_ts",         "max"),
        )
        .reset_index()
    )

    report["activity_summary"] = {
        "total_trades":      distribution_summary(per_market["total_trades"],      "total_trades"),
        "total_volume_usdc": distribution_summary(per_market["total_volume_usdc"], "volume_usdc"),
        "n_active_bars":     distribution_summary(per_market["n_active_bars"],     "active_bars"),
        "n_markets_zero_trades": int((per_market["total_trades"]     == 0).sum()),
        "n_markets_zero_volume": int((per_market["total_volume_usdc"] == 0).sum()),
    }

    logger.info("Audit complete.")
    return report, per_market
