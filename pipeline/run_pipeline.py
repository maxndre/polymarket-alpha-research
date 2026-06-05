"""
run_pipeline.py — Main pipeline orchestrator.

Usage:
    python pipeline/run_pipeline.py           # run all steps
    python pipeline/run_pipeline.py --rebuild # force re-read from SQLite

Outputs (in data/processed/):
    audit_report.json
    funnel_attrition.json
    distribution_report.json
    concentration_report.json
    per_market_usable.parquet
    trade_price_1m_usable.parquet
    market_polarities.parquet

Outputs (in data/features/):
    features_1m_[train|test]_[long|wide].parquet
    causal_edges.parquet
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE.parent) not in sys.path:
    sys.path.insert(0, str(_HERE.parent))

from pipeline.utils                import setup_logging
from pipeline.config               import PROCESSED_DIR, FunnelConfig, LOG_LEVEL, LOG_FORMAT, LOG_DATE_FORMAT
from pipeline.db_loader            import load_all_tables
from pipeline.step1_audit          import map_and_audit_database
from pipeline.step2_funnel         import apply_usability_funnel
from pipeline.step3_distributions  import compute_market_distributions
from pipeline.step4_concentration       import analyze_market_concentration
from pipeline.step4b_market_categories  import run_market_categorization
from pipeline.step5_market_polarity     import run_polarity_labeling
from pipeline.step6_features       import run_features_generation
from pipeline.step7_causal_discovery import extract_causal_edges

logger = logging.getLogger(__name__)


class _NumpyEncoder(json.JSONEncoder):
    """Serialize numpy / pandas types to JSON-native Python types."""
    def default(self, obj):
        import numpy as np
        import pandas as pd
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return None if (np.isnan(obj) or np.isinf(obj)) else float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, pd.Timestamp):
            return str(obj)
        return super().default(obj)


def _save_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, cls=_NumpyEncoder, indent=2, ensure_ascii=False)
    logger.info("Saved → %s  (%.1f KB)", path.name, path.stat().st_size / 1024)


def main(rebuild: bool = False) -> dict:
    setup_logging(LOG_LEVEL, LOG_FORMAT, LOG_DATE_FORMAT)
    t_start = time.time()

    logger.info("=" * 60)
    logger.info("  POLYMARKET ANALYSIS PIPELINE — START")
    logger.info("=" * 60)

    # Step 0 — Load raw tables
    logger.info("Loading raw tables ...")
    tables       = load_all_tables(force_rebuild=rebuild)
    df_markets   = tables["market"]
    df_bars      = tables["bars"]
    df_token     = tables["token_outcome"]
    df_market_tag = tables["market_tag"]

    # Step 1 — Audit
    audit_report, per_market_raw = map_and_audit_database(df_markets, df_bars, df_token, df_market_tag)
    _save_json(audit_report, PROCESSED_DIR / "audit_report.json")

    # Step 2 — Funnel
    cfg = FunnelConfig()
    df_usable_mkts, df_usable_bars, attrition = apply_usability_funnel(
        df_markets, df_bars, df_token,
        tables["filter_tag"], df_market_tag,
        per_market_raw, cfg,
    )
    _save_json({"attrition": attrition}, PROCESSED_DIR / "funnel_attrition.json")
    logger.info("Usable markets: %d | Usable bars: %d", len(df_usable_mkts), len(df_usable_bars))

    # Step 3 — Distributions
    dist_report   = compute_market_distributions(df_usable_mkts, df_usable_bars)
    per_market_df = dist_report.pop("_per_market_df")
    _save_json(dist_report, PROCESSED_DIR / "distribution_report.json")

    per_market_df.to_parquet(str(PROCESSED_DIR / "per_market_usable.parquet"), index=False)
    df_usable_bars.to_parquet(str(PROCESSED_DIR / "trade_price_1m_usable.parquet"), index=False)

    # Step 4 — Concentration
    conc_report = analyze_market_concentration(df_usable_mkts, df_usable_bars, df_market_tag, per_market_df)
    _save_json(conc_report, PROCESSED_DIR / "concentration_report.json")

    # Step 4b — Market category classification (required by steps 5 and 6)
    run_market_categorization(df_usable_mkts)

    # Step 5 — Market polarity
    run_polarity_labeling()

    # Step 6 — Feature generation
    run_features_generation()

    # Step 7 — Causal discovery
    extract_causal_edges()

    elapsed = time.time() - t_start
    logger.info("=" * 60)
    logger.info("  PIPELINE COMPLETE in %.1fs", elapsed)
    logger.info("=" * 60)

    # Summary
    n_raw    = audit_report["overview"]["n_markets_unique"]
    n_usable = len(df_usable_mkts)
    print(f"\n{'='*60}")
    print("PIPELINE SUMMARY")
    print(f"{'='*60}")
    print(f"  Raw markets:           {n_raw:>8,}")
    print(f"  Usable markets:        {n_usable:>8,}  ({n_usable/n_raw*100:.1f}%)")
    print(f"  Usable 1-min bars:     {len(df_usable_bars):>8,}")
    print()
    print("  FUNNEL ATTRITION:")
    for step in attrition:
        print(f"    [{step['step']}] {step['label']:<40s}  {step['n_after']:>6,}  (−{step['pct_removed']:.1f}%)")
    print()
    print(f"  Volume Gini:           {conc_report['market_concentration']['gini_volume']:.4f}")
    print(f"  Top-1% volume share:   {conc_report['market_concentration']['top01_pct_volume']:.1f}%")
    print(f"{'='*60}")

    return {
        "audit":            audit_report,
        "attrition":        attrition,
        "distributions":    dist_report,
        "concentration":    conc_report,
        "df_usable_markets": df_usable_mkts,
        "df_usable_bars":   df_usable_bars,
        "per_market":       per_market_df,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Polymarket Analysis Pipeline")
    parser.add_argument("--rebuild", action="store_true", help="Force rebuild of Parquet cache from SQLite")
    args = parser.parse_args()
    main(rebuild=args.rebuild)
