# Strategies

Three independent trading strategies, each exploring a different category of Polymarket prediction market as a signal source for S&P 100 equities.

## Overview

| Strategy | Author | Signal Source | Result | Verdict |
|----------|--------|---------------|:------:|---------|
| [S1 — Causal Granger](s1_causal_granger/) | Hugo Dunias | All categories, tree-based causality | Sharpe **+2.11** | ✅ Validated |
| [S2 — Earnings Spike](s2_earnings_spike/) | Maxandre Goillot | Earnings prediction markets | Sharpe **+0.46** | ⚠️ Inconclusive |
| [S3 — Natural Disasters](s3_natural_disasters/) | Théophile Thibaudon | Disaster/weather markets | Sharpe **−1.56** | ❌ Rejected |

## Strategy Interface

All strategies implement the same plug-in interface:

```python
def generate_signals(
    df_poly: pd.DataFrame,   # Polymarket features (DatetimeIndex UTC, 15-min, wide)
    df_bars: pd.DataFrame,   # S&P 100 close prices (same resolution)
    **kwargs
) -> pd.DataFrame:
    """Returns target weights per ticker, values ∈ [-1, 1]."""
```

The backtest engine handles execution delay, spread costs, and PnL calculation.

## Running All Strategies

```bash
# Compare all strategies on the test set
python backtest/engine.py --strategy all

# Run a specific strategy
python backtest/engine.py --strategy strategies/s1_causal_granger/strategy.py
python backtest/engine.py --strategy strategies/s2_earnings_spike/strategy.py
python backtest/engine.py --strategy strategies/s3_natural_disasters/strategy.py
```

## Directory Structure

```
strategies/
├── README.md                          # This file
├── s1_causal_granger/
│   ├── strategy.py                    # Signal generation
│   ├── calibration_params.json        # Pre-calibrated parameters
│   └── README.md
├── s2_earnings_spike/
│   ├── strategy.py                    # Signal generation
│   ├── placebo_analysis.py            # Placebo test suite
│   ├── calibration_params.json        # V0.6 parameters
│   └── README.md
└── s3_natural_disasters/
    ├── strategy.py                    # Signal generation (3 variants)
    └── README.md
```
