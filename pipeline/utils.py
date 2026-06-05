"""
utils.py — Pure statistical utilities shared across pipeline steps.
All functions are vectorized; no Python-level row iteration.
"""

from __future__ import annotations
import logging
import numpy as np
import pandas as pd
from scipy import stats as sp_stats

logger = logging.getLogger(__name__)


def gini_coefficient(values: np.ndarray) -> float:
    """
    Gini coefficient for a 1-D array of non-negative values.
    O(n log n) via sorted cumsum. Returns float in [0, 1].
    0 = perfect equality, 1 = maximal concentration.
    """
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v) & (v >= 0)]
    if len(v) == 0:
        return np.nan
    if v.sum() == 0:
        return 0.0
    v_sorted = np.sort(v)
    n = len(v_sorted)
    idx = np.arange(1, n + 1)
    return float(
        (2 * (idx * v_sorted).sum() - (n + 1) * v_sorted.sum())
        / (n * v_sorted.sum())
    )


def distribution_summary(series: pd.Series, name: str = "") -> dict:
    """
    Robust distributional statistics for a numeric Series.
    Returns a flat dict suitable for JSON serialization.
    """
    s = series.dropna().astype(np.float64)
    if s.empty:
        logger.warning("distribution_summary: empty series for '%s'", name)
        return {k: np.nan for k in ["n", "mean", "median", "std",
                                     "q05", "q25", "q75", "q95",
                                     "skewness", "min", "max"]}
    return {
        "n":        int(len(s)),
        "mean":     float(s.mean()),
        "median":   float(s.median()),
        "std":      float(s.std()),
        "q05":      float(s.quantile(0.05)),
        "q25":      float(s.quantile(0.25)),
        "q75":      float(s.quantile(0.75)),
        "q95":      float(s.quantile(0.95)),
        "skewness": float(sp_stats.skew(s.values, bias=False)),
        "min":      float(s.min()),
        "max":      float(s.max()),
    }


def top_n_share(series: pd.Series, pct: float) -> float:
    """Fraction of total held by the top `pct` share of items (e.g., pct=0.01 → top-1%)."""
    s = series.dropna().sort_values(ascending=False)
    if s.empty or s.sum() == 0:
        return np.nan
    cutoff = max(1, int(np.ceil(len(s) * pct)))
    return float(s.iloc[:cutoff].sum() / s.sum())


def setup_logging(level: str = "INFO",
                  fmt: str = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                  datefmt: str = "%Y-%m-%d %H:%M:%S") -> None:
    """Configure root logger once for the whole pipeline."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format=fmt,
        datefmt=datefmt,
        force=True,
    )
