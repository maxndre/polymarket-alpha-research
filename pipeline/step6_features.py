"""
step6_features.py – Generation of exhaustive 1-minute features per Polymarket market group (LLM category).
Calculates advanced financial and statistical metrics (simple & weighted prices, quantiles, skewness,
kurtosis, Shannon entropy of uncertainty, Herfindahl-Hirschman index, and rolling aggregations) on a dense 1-minute grid.
"""
from pathlib import Path
import logging
import time
import numpy as np
import pandas as pd

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Paths & Config ────────────────────────────────────────────────────────────
from pipeline.config import PARQUET_DIR, DIR_STRATEGIES, DIR_FEATURES

DIR_NETTOYEES = PARQUET_DIR

def run_features_generation():
    logger.info("=== STEP 6 : Generation of Advanced Category Features ===")
    t_start = time.time()
    
    # Create Features directory if it doesn't exist
    DIR_FEATURES.mkdir(exist_ok=True, parents=True)
    
    # ── 1. Load Data ──────────────────────────────────────────────────────────
    logger.info("Loading required parquet datasets …")
    df_market = pd.read_parquet(DIR_NETTOYEES / "market.parquet", columns=["market_id", "condition_id"])
    df_categories = pd.read_parquet(DIR_NETTOYEES / "market_categories.parquet", columns=["market_id", "llm_category"])
    df_polarities = pd.read_parquet(DIR_NETTOYEES / "market_polarities.parquet", columns=["market_id", "polarity"])
    df_trades = pd.read_parquet(DIR_STRATEGIES / "trade_price_1m_usable.parquet")
    
    logger.info(f"Loaded trades: {len(df_trades):,} rows")
    
    # ── 2. Map condition_id to llm_category and polarity ─────────────────────
    logger.info("Mapping condition_id to LLM categories and polarities …")
    
    # Cast market_id to int for robust merging
    df_market["market_id"] = df_market["market_id"].astype(int)
    df_categories["market_id"] = df_categories["market_id"].astype(int)
    df_polarities["market_id"] = df_polarities["market_id"].astype(int)
    
    mapping = df_market.merge(df_categories, on="market_id")
    mapping = mapping.merge(df_polarities, on="market_id")
    mapping = mapping[["condition_id", "llm_category", "polarity"]].drop_duplicates()
    
    # Left join to associate trades with LLM categories and polarities
    df_trades = df_trades.merge(mapping, on="condition_id", how="left")
    df_trades["llm_category"] = df_trades["llm_category"].fillna("Other")
    df_trades["polarity"] = df_trades["polarity"].fillna(0).astype(int)
    
    # ── 3. Vectorized Pre-calculations ────────────────────────────────────────
    logger.info("Performing fast vectorial pre-calculations …")
    
    # Signed close price: only compute for non-zero polarity (neutral markets ignored in price metrics)
    df_trades["signed_close_price"] = np.where(df_trades["polarity"] != 0, df_trades["close_price"] * df_trades["polarity"], np.nan)
    
    # Signed price notional for VWAP
    df_trades["price_notional_signed"] = df_trades["signed_close_price"] * df_trades["notional_usdc_1m"]
    
    # Volume associated with signed markets only
    df_trades["notional_usdc_1m_signed"] = np.where(df_trades["polarity"] != 0, df_trades["notional_usdc_1m"], np.nan)
    
    # Raw sums for HHI and non-signed features
    df_trades["price_notional"] = df_trades["close_price"] * df_trades["notional_usdc_1m"]
    df_trades["notional_squared"] = df_trades["notional_usdc_1m"] ** 2
    df_trades["high_minus_low"] = df_trades["high_price"] - df_trades["low_price"]
    
    # Shannon Entropy for each individual binary market: - [p * log(p) + (1-p) * log(1-p)]
    eps = 1e-9
    p = df_trades["close_price"].clip(0.0, 1.0)
    df_trades["shannon_entropy_m"] = -(p * np.log(p + eps) + (1.0 - p) * np.log(1.0 - p + eps))
    
    # ── 4. Groupby & Fast Aggregations ────────────────────────────────────────
    logger.info("Aggregating data by (minute_ts, llm_category) …")
    t_agg = time.time()
    
    # Groupby object
    df_grp = df_trades.groupby(["minute_ts", "llm_category"])
    
    # Standard fast aggregates
    df_basic = df_grp.agg({
        "signed_close_price": ["mean", "std", "max", "min", "median"],
        "price_notional_signed": "sum",
        "notional_usdc_1m_signed": "sum",
        "notional_usdc_1m": "sum",
        "notional_squared": "sum",
        "volume_shares_1m": "sum",
        "trades_count_1m": "sum",
        "condition_id": "nunique",
        "high_minus_low": "mean",
        "shannon_entropy_m": "mean",
    })
    
    # Flatten hierarchical index columns
    df_basic.columns = [f"{col[0]}_{col[1]}" for col in df_basic.columns]
    
    # Highly optimized vectorized moments & quantiles (avoiding slow lambdas)
    logger.info("Computing moments and quantiles via optimized pandas vector operations …")
    df_basic["price_skew"] = df_grp["signed_close_price"].skew()
    
    # Fast C-level kurtosis calculation via apply (kurt() automatically ignores NaNs and handles < 4 elements)
    df_basic["price_kurt"] = df_grp["signed_close_price"].apply(lambda x: x.kurt())
    
    df_basic["price_q25"] = df_grp["signed_close_price"].quantile(0.25)
    df_basic["price_q75"] = df_grp["signed_close_price"].quantile(0.75)
    
    # Reset index to access minute_ts and llm_category
    df_basic = df_basic.reset_index()
    
    # ── 5. Derived Features ───────────────────────────────────────────────────
    logger.info("Computing derived features (HHI & Volume-Weighted Price) …")
    
    # Volume-weighted average price (VWAP proxy) for signed markets
    df_basic["price_weighted_mean"] = df_basic["price_notional_signed_sum"] / df_basic["notional_usdc_1m_signed_sum"]
    # Fallback to simple mean if volume is 0
    df_basic["price_weighted_mean"] = df_basic["price_weighted_mean"].fillna(df_basic["signed_close_price_mean"])
    
    # Herfindahl-Hirschman Index (HHI) for volume concentration within the group
    df_basic["hhi_volume"] = df_basic["notional_squared_sum"] / (df_basic["notional_usdc_1m_sum"] ** 2)
    # HHI is 1.0 if only 1 market or volume sum is 0
    df_basic["hhi_volume"] = df_basic["hhi_volume"].fillna(1.0).clip(0.0, 1.0)
    
    # Clean up intermediate sums
    df_basic = df_basic.rename(columns={
        "signed_close_price_mean": "price_mean",
        "signed_close_price_std": "price_std",
        "signed_close_price_max": "price_max",
        "signed_close_price_min": "price_min",
        "signed_close_price_median": "price_median",
        "volume_shares_1m_sum": "volume_shares_total",
        "notional_usdc_1m_sum": "notional_usdc_total",
        "trades_count_1m_sum": "trades_count_total",
        "condition_id_nunique": "active_markets_count",
        "high_minus_low_mean": "spread_mean",
        "shannon_entropy_m_mean": "shannon_entropy",
    })
    
    # Drop raw sums
    df_basic = df_basic.drop(columns=["price_notional_signed_sum", "notional_usdc_1m_signed_sum", "notional_squared_sum"])
    
    logger.info(f"Aggregations complete in {time.time() - t_agg:.1f}s. Row count: {len(df_basic):,}")
    
    # ── 6. Reindexing onto a DENSE 1-minute Grid ──────────────────────────────
    logger.info("Generating dense 1-minute timeline for all categories in 2025 …")
    categories = df_basic["llm_category"].unique()
    
    # Full 1-min datetime range for 2025 in UTC
    full_timeline = pd.date_range(
        start="2025-01-01 00:00:00+00:00",
        end="2025-12-31 23:59:00+00:00",
        freq="min",
        name="minute_ts"
    )
    
    # MultiIndex representing all combinations of (minute_ts, llm_category)
    mux = pd.MultiIndex.from_product([full_timeline, categories], names=["minute_ts", "llm_category"])
    
    # Set index and reindex
    df_dense = df_basic.set_index(["minute_ts", "llm_category"]).reindex(mux).reset_index()
    logger.info(f"Dense reindexed dataframe rows: {len(df_dense):,}")
    
    # Fill counts and volumes with 0/0.0, keeping prices and entropy as NaN
    df_dense["active_markets_count"] = df_dense["active_markets_count"].fillna(0).astype(int)
    df_dense["volume_shares_total"] = df_dense["volume_shares_total"].fillna(0.0)
    df_dense["notional_usdc_total"] = df_dense["notional_usdc_total"].fillna(0.0)
    df_dense["trades_count_total"] = df_dense["trades_count_total"].fillna(0).astype(int)
    
    # ── 7. Calculate Time-Series & Rolling Features ───────────────────────────
    logger.info("Calculating category-specific time-series & rolling window features …")
    t_roll = time.time()
    
    category_dfs = []
    for cat in categories:
        df_cat = df_dense[df_dense["llm_category"] == cat].sort_values("minute_ts").copy()
        
        # Returns on weighted mean price
        df_cat["return_1m"] = df_cat["price_weighted_mean"].pct_change(1)
        df_cat["return_5m"] = df_cat["price_weighted_mean"].pct_change(5)
        
        # Volume rolling sums
        df_cat["vol_roll_5m"] = df_cat["notional_usdc_total"].rolling(5, min_periods=1).sum()
        df_cat["vol_roll_1h"] = df_cat["notional_usdc_total"].rolling(60, min_periods=1).sum()
        
        # Trades rolling sums
        df_cat["trades_roll_5m"] = df_cat["trades_count_total"].rolling(5, min_periods=1).sum()
        df_cat["trades_roll_1h"] = df_cat["trades_count_total"].rolling(60, min_periods=1).sum()
        
        # Shannon Entropy change
        df_cat["entropy_change_5m"] = df_cat["shannon_entropy"].diff(5)
        
        category_dfs.append(df_cat)
        
    df_final_long = pd.concat(category_dfs, ignore_index=True)
    logger.info(f"Rolling calculations complete in {time.time() - t_roll:.1f}s")
    
    # ── 8. Create Wide Format DataFrame ───────────────────────────────────────
    logger.info("Pivoting to WIDE format (each minute is a single row) …")
    t_pivot = time.time()
    
    # Pivot
    df_pivot = df_final_long.pivot(index="minute_ts", columns="llm_category")
    # Flatten MultiIndex columns: e.g. "Crypto Prices_price_mean"
    df_pivot.columns = [f"{col[1]}_{col[0]}" for col in df_pivot.columns]
    df_final_wide = df_pivot.reset_index()
    
    logger.info(f"Pivot complete in {time.time() - t_pivot:.1f}s. Wide columns: {len(df_final_wide.columns)}")
    
    # ── 9. Save Datasets & Train/Test Splits ──────────────────────────────────
    train_mask_long = df_final_long["minute_ts"] < "2025-11-01 00:00:00+00:00"
    train_mask_wide = df_final_wide["minute_ts"] < "2025-11-01 00:00:00+00:00"
    
    # Files to save
    files_to_save = {
        # Long format
        "polymarket_group_features_1m_long.parquet": df_final_long,
        "polymarket_group_features_1m_train_long.parquet": df_final_long[train_mask_long],
        "polymarket_group_features_1m_test_long.parquet": df_final_long[~train_mask_long],
        # Wide format
        "polymarket_group_features_1m_wide.parquet": df_final_wide,
        "polymarket_group_features_1m_train_wide.parquet": df_final_wide[train_mask_wide],
        "polymarket_group_features_1m_test_wide.parquet": df_final_wide[~train_mask_wide],
    }
    
    logger.info("Saving generated Parquet datasets to Features/ directory …")
    for name, df in files_to_save.items():
        out_path = DIR_FEATURES / name
        df.to_parquet(out_path, index=False)
        logger.info(f"  Saved: {name} ({len(df):,} rows, {out_path.stat().st_size / 1e6:.1f} MB)")
        
    elapsed = time.time() - t_start
    logger.info(f"=== Advanced Category Features completed successfully in {elapsed:.1f}s ===")


if __name__ == "__main__":
    run_features_generation()
