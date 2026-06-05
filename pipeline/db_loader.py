"""
db_loader.py — SQLite → pandas → Parquet with intelligent caching.

Design choices:
  - Convert once to Parquet (columnar, compressed); subsequent runs read ~10x faster.
  - All TEXT price columns cast to float64 at load time.
  - Timestamps parsed to timezone-aware datetime64[ns, UTC].
  - No row-level loops; uses read_sql with chunksize for the large bars table.
"""

from __future__ import annotations
import logging
import sqlite3
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .config import (
    DB_PATH, PROCESSED_DIR,
    PARQUET_MARKET, PARQUET_TOKEN, PARQUET_BARS,
    PARQUET_MTAG, PARQUET_FTAG,
)

logger = logging.getLogger(__name__)

_PRICE_COLS  = ["open_price", "high_price", "low_price", "close_price",
                "volume_shares_1m", "notional_usdc_1m"]
_TS_COLS_MKT = ["start_date", "end_date"]
_TS_COL_BAR  = "minute_ts"


def _parse_ts(series: pd.Series) -> pd.Series:
    """Parse ISO-8601 string timestamps → datetime64[ns, UTC] (vectorized)."""
    return pd.to_datetime(series, utc=True, errors="coerce")


def _cast_prices(df: pd.DataFrame) -> pd.DataFrame:
    """Cast TEXT price / volume columns to float64 in-place."""
    for col in _PRICE_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _load_market_from_db(conn: sqlite3.Connection) -> pd.DataFrame:
    logger.info("Loading 'market' table from SQLite ...")
    df = pd.read_sql("SELECT * FROM market", conn)
    for col in _TS_COLS_MKT:
        if col in df.columns:
            df[col] = _parse_ts(df[col])
    logger.info("  market: %d rows", len(df))
    return df


def _load_token_from_db(conn: sqlite3.Connection) -> pd.DataFrame:
    logger.info("Loading 'token_outcome' table from SQLite ...")
    df = pd.read_sql("SELECT * FROM token_outcome", conn)
    logger.info("  token_outcome: %d rows", len(df))
    return df


def _load_bars_from_db(conn: sqlite3.Connection, chunksize: int = 500_000) -> pd.DataFrame:
    """Stream the ~10M-row bars table in chunks to avoid OOM, then concat."""
    logger.info("Loading 'trade_price_1m' in chunks of %d ...", chunksize)
    t0 = time.time()
    chunks = []
    for i, chunk in enumerate(
        pd.read_sql("SELECT * FROM trade_price_1m", conn, chunksize=chunksize)
    ):
        chunk = _cast_prices(chunk)
        chunk[_TS_COL_BAR] = _parse_ts(chunk[_TS_COL_BAR])
        chunks.append(chunk)
        if (i + 1) % 5 == 0:
            logger.info("  ... %d chunks loaded (%ds elapsed)", i + 1, int(time.time() - t0))
    df = pd.concat(chunks, ignore_index=True)
    logger.info("  trade_price_1m: %d rows in %.1fs", len(df), time.time() - t0)
    return df


def _load_market_tag_from_db(conn: sqlite3.Connection) -> pd.DataFrame:
    logger.info("Loading 'market_tag' table from SQLite ...")
    df = pd.read_sql("SELECT * FROM market_tag", conn)
    logger.info("  market_tag: %d rows", len(df))
    return df


def _load_filter_tag_from_db(conn: sqlite3.Connection) -> pd.DataFrame:
    logger.info("Loading 'selected_filter_tag' table from SQLite ...")
    df = pd.read_sql("SELECT * FROM selected_filter_tag", conn)
    logger.info("  selected_filter_tag: %d rows", len(df))
    return df


def _save_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(table, str(path), compression="snappy")
    logger.info("  Saved → %s  (%.1f MB)", path.name, path.stat().st_size / 1e6)


def _load_parquet(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(str(path))
    for col in _TS_COLS_MKT + [_TS_COL_BAR]:
        if col in df.columns and not pd.api.types.is_datetime64_any_dtype(df[col]):
            df[col] = _parse_ts(df[col])
    return df


def load_all_tables(force_rebuild: bool = False) -> dict[str, pd.DataFrame]:
    """
    Load all tables from SQLite and cache as Parquet.

    Parameters
    ----------
    force_rebuild : bool
        If True, ignore existing Parquet cache and re-read from SQLite.

    Returns
    -------
    dict with keys: 'market', 'token_outcome', 'bars', 'market_tag', 'filter_tag'
    """
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    parquet_map = {
        "market":        PARQUET_MARKET,
        "token_outcome": PARQUET_TOKEN,
        "bars":          PARQUET_BARS,
        "market_tag":    PARQUET_MTAG,
        "filter_tag":    PARQUET_FTAG,
    }

    all_cached = all(p.exists() for p in parquet_map.values())

    if all_cached and not force_rebuild:
        logger.info("All Parquet caches found — loading directly (fast path).")
        return {name: _load_parquet(path) for name, path in parquet_map.items()}

    logger.info("Building Parquet cache from SQLite: %s", DB_PATH)
    conn = sqlite3.connect(str(DB_PATH))

    loaders = {
        "market":        _load_market_from_db,
        "token_outcome": _load_token_from_db,
        "bars":          _load_bars_from_db,
        "market_tag":    _load_market_tag_from_db,
        "filter_tag":    _load_filter_tag_from_db,
    }

    tables: dict[str, pd.DataFrame] = {}
    for name, loader_fn in loaders.items():
        path = parquet_map[name]
        if path.exists() and not force_rebuild:
            logger.info("Cache hit for '%s'.", name)
            tables[name] = _load_parquet(path)
        else:
            df = loader_fn(conn)
            _save_parquet(df, path)
            tables[name] = df

    conn.close()
    logger.info("All tables loaded:")
    for name, df in tables.items():
        logger.info("  %-20s %8d rows  %d cols", name, len(df), df.shape[1])

    return tables
