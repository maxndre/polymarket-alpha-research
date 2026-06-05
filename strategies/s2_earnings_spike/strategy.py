"""
Strategy 2 — Earnings Spike: Post-announcement mean-reversion via Polymarket IC

The strategy identifies S&P 100 earnings announcements via Polymarket earnings
markets, waits for the post-announcement price spike, and takes a mean-reversion
position when the Polymarket IC (Information Coefficient) is high enough.

Author: Maxandre Goillot
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_HERE       = Path(__file__).resolve().parent
_PARAMS_PATH = _HERE / "calibration_params.json"

EARNINGS_KEYWORDS = [
    "earnings", "eps", "revenue", "quarterly results",
    "beats estimates", "misses estimates", "guidance",
]

WINDOW_PRE_BARS  = 4   # 4 × 15min = 1h before announcement
WINDOW_POST_BARS = 8   # 8 × 15min = 2h post-announcement holding window


def _load_params() -> dict:
    if _PARAMS_PATH.exists():
        with open(_PARAMS_PATH) as f:
            return json.load(f)
    return {
        "ic_threshold": 0.6,
        "z_entry": 1.0,
        "max_weight": 0.2,
        "holding_bars": 8,
    }


def _identify_earnings_events(df_poly: pd.DataFrame) -> list[tuple]:
    """
    Find Polymarket categories that match earnings keywords and return
    a list of (timestamp, category) pairs where market activity spikes.
    """
    earnings_cats = [
        c for c in df_poly.columns
        if any(kw in c.lower() for kw in EARNINGS_KEYWORDS)
    ]
    events = []
    for cat in earnings_cats:
        if f"{cat}_volume_shares_total" not in df_poly.columns:
            continue
        vol = df_poly[f"{cat}_volume_shares_total"].fillna(0.0)
        mu, sig = vol.mean(), vol.std()
        if sig == 0:
            continue
        spikes = vol[vol > mu + 3.0 * sig].index.tolist()
        for t in spikes:
            ticker = _guess_ticker_from_category(cat)
            if ticker:
                events.append((t, cat, ticker))
    return events


def _guess_ticker_from_category(cat: str) -> str | None:
    """Heuristic: match Polymarket category name to S&P 100 ticker."""
    ticker_map = {
        "apple": "AAPL", "microsoft": "MSFT", "amazon": "AMZN",
        "nvidia": "NVDA", "meta": "META", "alphabet": "GOOGL",
        "tesla": "TSLA", "jpmorgan": "JPM", "berkshire": "BRK.B",
        "unitedhealth": "UNH", "johnson": "JNJ", "walmart": "WMT",
        "exxon": "XOM", "chevron": "CVX", "visa": "V",
    }
    cat_lower = cat.lower()
    for key, ticker in ticker_map.items():
        if key in cat_lower:
            return ticker
    return None


def _compute_ic(df_poly: pd.DataFrame, cat: str, df_returns: pd.DataFrame,
                ticker: str, event_time: pd.Timestamp, lag_bars: int) -> float:
    """
    Compute the Information Coefficient between pre-announcement Polymarket
    VWAP delta and post-announcement stock return.
    IC > 0 means the crowd correctly anticipated the direction.
    """
    col = f"{cat}_vwap" if f"{cat}_vwap" in df_poly.columns else f"{cat}_price_weighted_mean"
    if col not in df_poly.columns or ticker not in df_returns.columns:
        return 0.0

    try:
        idx = df_poly.index.get_loc(event_time)
    except KeyError:
        return 0.0

    pre_start  = max(0, idx - WINDOW_PRE_BARS)
    post_end   = min(len(df_poly) - 1, idx + WINDOW_POST_BARS)

    poly_signal = df_poly[col].iloc[pre_start:idx].diff().mean()
    stock_ret   = df_returns[ticker].iloc[idx:post_end].sum()

    if np.isnan(poly_signal) or np.isnan(stock_ret) or poly_signal == 0:
        return 0.0
    return np.sign(poly_signal) * np.sign(stock_ret)


def generate_signals(
    df_poly:  pd.DataFrame,
    df_bars:  pd.DataFrame,
    **kwargs,
) -> pd.DataFrame:
    """
    Strategy 2 entry point.

    Scans for high-volume earnings events on Polymarket, checks IC, and
    opens a post-announcement mean-reversion position.

    Returns
    -------
    pd.DataFrame — DatetimeIndex (UTC), columns = tickers, values ∈ [-1, 1]
    """
    params  = _load_params()
    ic_thr  = params["ic_threshold"]
    max_w   = params["max_weight"]
    hold    = params["holding_bars"]

    df_returns = df_bars.pct_change(1, fill_method=None)
    tickers    = df_bars.columns.tolist()

    weights = pd.DataFrame(0.0, index=df_poly.index, columns=tickers)
    events  = _identify_earnings_events(df_poly)

    if not events:
        logger.warning("S2 — No earnings events detected in Polymarket features.")
        return weights

    logger.info("S2 — %d earnings events detected.", len(events))

    for event_time, cat, ticker in events:
        if ticker not in weights.columns:
            continue

        ic = _compute_ic(df_poly, cat, df_returns, ticker, event_time, lag_bars=1)

        if abs(ic) < ic_thr:
            continue

        try:
            entry_iloc = df_poly.index.get_loc(event_time)
        except KeyError:
            continue

        exit_iloc = min(entry_iloc + hold, len(df_poly) - 1)
        entry_idx = df_poly.index[entry_iloc]
        exit_idx  = df_poly.index[exit_iloc]

        # Mean-reversion: if crowd was bullish (ic>0) and stock spiked up → short
        direction = -np.sign(ic)
        weights.loc[entry_idx:exit_idx, ticker] = direction * max_w

        logger.info("  Event: %-10s | cat=%-30s | IC=%+.2f | dir=%+d | hold=%d bars",
                    ticker, cat[:30], ic, direction, hold)

    return weights
