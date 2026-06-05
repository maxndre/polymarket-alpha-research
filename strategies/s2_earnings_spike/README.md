# Strategy 2 — Earnings Spike

**Author:** Maxandre Goillot  
**Result:** Sharpe **+0.46** (N=15 events) ⚠️ Inconclusive

## Hypothesis

When Polymarket earnings prediction markets exhibit a high Information Coefficient (IC) — i.e., when the crowd correctly anticipated the earnings direction — the stock price overreacts in the post-announcement window, creating a short-lived mean-reversion opportunity.

## Research Log

### V0.1 — Naive IC Filter
Simple IC threshold (IC > 0.5) on raw Polymarket VWAP vs. stock return.  
→ Result: Sharpe +0.12. Signal exists but too noisy.

### V0.2 — Volume Spike Detection
Added earnings event identification via volume anomaly detection (Z > 3 on category volume).  
→ Result: Sharpe +0.21. Better event targeting, still weak.

### V0.3 — Asymmetric Window
Separate pre-announcement (1h) and post-announcement (2h) windows. IC computed strictly on pre-announcement signal.  
→ Result: Sharpe +0.35. Cleaner IC estimation.

### V0.4 — Placebo Test Integration
Ran first placebo test: compare real IC vs. 30 random shifted windows.  
→ Finding: post-announcement IC (0.81 mean) >> pre-announcement IC (0.25). Test is specific to the announcement window. p = 0.031.

### V0.5 — Asymmetric Direction
Entry direction = mean-reversion: if crowd was bullish and stock spiked up → short.  
→ Result: Sharpe +0.41. This is the correct direction.

### V0.6 — Holding Period Tuning
Swept holding period from 4 to 16 bars (1h to 4h). Optimum at 8 bars (2h).  
→ Result: Sharpe +0.46. **Final version.**

## Placebo Test Results

The key validation for this strategy is that the predictive signal is **specific to the post-announcement window**, not a general spurious correlation.

```
IC post-announcement:  mean = +0.81  (n=15 events)
IC placebo windows:    mean = +0.25  (n=450 windows)
t-stat: +2.31  p-value: 0.031
```

The signal is statistically specific to announcement events, but with only N=15 events the strategy is **underpowered** — we cannot reject the null hypothesis of zero alpha at the 1% level.

**Verdict: Inconclusive.** The signal is real, but the sample is too small.

## Signal Generation

1. Identify earnings events: volume Z-score > 3 on any earnings-tagged Polymarket category
2. Compute IC: Pearson correlation between pre-announcement Polymarket VWAP delta and post-announcement stock return
3. If IC > 0.6: open mean-reversion position (weight = −sign(IC) × 0.20) for 8 bars

## Running the Placebo Analysis

```bash
python strategies/s2_earnings_spike/placebo_analysis.py
```

Outputs to `strategies/s2_earnings_spike/placebo_results/`:
- `ic_distribution.png` — Real vs. placebo IC histogram
- `ic_per_event.png` — IC for each individual event
- `placebo_summary.json` — Summary statistics

## Running the Backtest

```bash
python backtest/engine.py --strategy strategies/s2_earnings_spike/strategy.py
```

## Files

| File | Description |
|------|-------------|
| `strategy.py` | Signal generation entry point |
| `placebo_analysis.py` | Placebo test suite |
| `calibration_params.json` | V0.6 calibrated parameters |
| `README.md` | This file |
