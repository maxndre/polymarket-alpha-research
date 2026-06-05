"""
placebo_analysis.py — Placebo tests for Strategy 2 (Earnings Spike)

Tests whether the predictive signal is specific to the post-announcement window,
or whether it is a spurious correlation present throughout the year.

Method
------
For each real earnings event (t*):
  1. Compute IC between Polymarket VWAP and stock return in the TRUE post-announcement window.
  2. Compute IC in N random control windows shifted ±k days from t*.
  3. Compare: if the signal is genuine, IC(post-announcement) >> IC(placebo).

The test is a time-series equivalent of a permutation test.

Usage
-----
    python strategies/s2_earnings_spike/placebo_analysis.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent.parent))

from backtest.engine import load_polymarket_features, load_sp100_returns  # noqa: E402
from strategies.s2_earnings_spike.strategy import (  # noqa: E402
    _identify_earnings_events, _compute_ic, WINDOW_POST_BARS,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
logger = logging.getLogger("s2.placebo")

N_PLACEBO_SHIFTS = 30   # placebo windows per event
OUTPUT_DIR       = _HERE / "placebo_results"


def run_placebo_analysis(period: str = "full") -> dict:
    """
    Run placebo tests and produce comparison charts.

    Returns
    -------
    dict with keys: real_ics, placebo_ics, t_stat, p_value, n_events
    """
    OUTPUT_DIR.mkdir(exist_ok=True)
    logger.info("Loading data ...")

    df_poly    = load_polymarket_features(split=period, resample="15min")
    df_returns = load_sp100_returns(resample="15min")

    events = _identify_earnings_events(df_poly)
    logger.info("Found %d earnings events.", len(events))

    real_ics    = []
    placebo_ics = []

    for event_time, cat, ticker in events:
        ic_real = _compute_ic(df_poly, cat, df_returns, ticker, event_time, lag_bars=1)
        real_ics.append(ic_real)

        # Generate N placebo windows
        event_iloc = df_poly.index.get_loc(event_time) if event_time in df_poly.index else None
        if event_iloc is None:
            continue

        for shift in range(-N_PLACEBO_SHIFTS // 2, N_PLACEBO_SHIFTS // 2 + 1):
            if shift == 0:
                continue
            shift_iloc = event_iloc + shift * WINDOW_POST_BARS
            if shift_iloc < 0 or shift_iloc >= len(df_poly):
                continue
            placebo_t = df_poly.index[shift_iloc]
            ic_p = _compute_ic(df_poly, cat, df_returns, ticker, placebo_t, lag_bars=1)
            placebo_ics.append(ic_p)

    if not real_ics:
        logger.warning("No events with valid ICs.")
        return {}

    real_arr    = np.array(real_ics)
    placebo_arr = np.array(placebo_ics)

    t_stat, p_val = stats.ttest_ind(real_arr, placebo_arr, alternative="greater")

    logger.info("Real IC:    mean=%.3f  std=%.3f  n=%d", real_arr.mean(), real_arr.std(), len(real_arr))
    logger.info("Placebo IC: mean=%.3f  std=%.3f  n=%d", placebo_arr.mean(), placebo_arr.std(), len(placebo_arr))
    logger.info("t-stat: %.3f  p-value: %.4f", t_stat, p_val)

    _plot_ic_distribution(real_arr, placebo_arr, t_stat, p_val)
    _plot_ic_timeline(events, df_poly, df_returns)

    result = {
        "real_ics":    real_arr.tolist(),
        "placebo_ics": placebo_arr.tolist(),
        "real_mean":   float(real_arr.mean()),
        "placebo_mean": float(placebo_arr.mean()),
        "t_stat":      float(t_stat),
        "p_value":     float(p_val),
        "n_events":    len(real_arr),
        "n_placebo":   len(placebo_arr),
    }

    import json
    with open(OUTPUT_DIR / "placebo_summary.json", "w") as f:
        json.dump(result, f, indent=2)

    return result


def _plot_ic_distribution(real_ics: np.ndarray, placebo_ics: np.ndarray,
                           t_stat: float, p_val: float) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))

    bins = np.linspace(-1.0, 1.0, 25)
    ax.hist(placebo_ics, bins=bins, alpha=0.5, label=f"Placebo (n={len(placebo_ics)})", color="#94A3B8")
    ax.hist(real_ics,    bins=bins, alpha=0.8, label=f"Real post-announcement (n={len(real_ics)})",  color="#3B82F6")

    ax.axvline(real_ics.mean(),    color="#1D4ED8", lw=2.0, ls="--", label=f"Real mean = {real_ics.mean():+.3f}")
    ax.axvline(placebo_ics.mean(), color="#6B7280", lw=2.0, ls="--", label=f"Placebo mean = {placebo_ics.mean():+.3f}")
    ax.axvline(0, color="black", lw=0.8)

    ax.set_title(
        f"S2 Placebo Test — IC Distribution\n"
        f"t={t_stat:.3f}, p={p_val:.4f}",
        fontsize=13, fontweight="bold",
    )
    ax.set_xlabel("Information Coefficient (IC)")
    ax.set_ylabel("Count")
    ax.legend()
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "ic_distribution.png", dpi=300)
    plt.close(fig)
    logger.info("Saved: %s", OUTPUT_DIR / "ic_distribution.png")


def _plot_ic_timeline(events: list, df_poly: pd.DataFrame,
                      df_returns: pd.DataFrame) -> None:
    """IC per event, sorted by time."""
    rows = []
    for event_time, cat, ticker in events:
        ic = _compute_ic(df_poly, cat, df_returns, ticker, event_time, lag_bars=1)
        rows.append({"time": event_time, "ticker": ticker, "ic": ic})

    if not rows:
        return

    df_ev = pd.DataFrame(rows).sort_values("time").reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(12, 5))
    colors = ["#22C55E" if ic > 0 else "#EF4444" for ic in df_ev["ic"]]
    ax.bar(df_ev.index, df_ev["ic"], color=colors, width=0.6)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(df_ev.index)
    ax.set_xticklabels(
        [f"{row['ticker']}\n{row['time'].strftime('%b %d')}" for _, row in df_ev.iterrows()],
        fontsize=8,
    )
    ax.set_title("S2 — IC per Earnings Event", fontsize=13, fontweight="bold")
    ax.set_ylabel("Information Coefficient")
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "ic_per_event.png", dpi=300)
    plt.close(fig)
    logger.info("Saved: %s", OUTPUT_DIR / "ic_per_event.png")


if __name__ == "__main__":
    result = run_placebo_analysis(period="full")
    if result:
        print(f"\nPlacebo Test Summary")
        print(f"  Real IC:    {result['real_mean']:+.3f}  (n={result['n_events']})")
        print(f"  Placebo IC: {result['placebo_mean']:+.3f}  (n={result['n_placebo']})")
        print(f"  t-stat:     {result['t_stat']:+.3f}")
        print(f"  p-value:    {result['p_value']:.4f}")
        verdict = "SIGNAL IS SPECIFIC to announcement windows" if result["p_value"] < 0.05 else "INCONCLUSIVE (p ≥ 0.05)"
        print(f"  Verdict:    {verdict}")
