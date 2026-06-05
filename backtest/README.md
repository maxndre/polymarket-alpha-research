# Backtest Module

Implementation of a realistic backtest engine for Polymarket → S&P 100 strategies.

## Architecture

```
backtest/
├── engine.py      # Orchestrator: data loading, signal execution, plotting
├── metrics.py     # Portfolio PnL, Sharpe, drawdown, Buy & Hold benchmark
└── README.md      # This file
```

## Design Principles

### 1. Signal vs. Execution Separation

A signal computed at time `t` can only be **acted upon at `t + ε`**, where `ε` is the execution delay (default: 1 minute). This prevents look-ahead bias from a signal-generation latency that would not exist in production.

```python
delayed_weights = apply_epsilon_delay(raw_weights, epsilon_minutes=1)
```

### 2. Realistic Transaction Costs

Execution costs are computed from **Q4 2025 bid/ask quotes** (1-min resolution):

```
cost(t) = Σ_i |ΔW_i(t)| × half_spread_i(t)
```

where `half_spread = (ask - bid) / (2 × mid)`. Falls back to 1.5 bps flat if quotes are unavailable.

### 3. No Data Leakage

- Strategy calibration uses **train set only** (Jan–Oct 2025)
- Z-score normalization statistics come from training period only
- Test set: Nov–Dec 2025 (held out throughout development)

---

## Usage

```bash
# Run S1 (default) on the test set
python backtest/engine.py

# Run S1 with a 2-minute execution delay
python backtest/engine.py --epsilon 2

# Run an external strategy
python backtest/engine.py --strategy strategies/s2_earnings_spike/strategy.py

# Compare all strategies
python backtest/engine.py --strategy all

# Run on the training period
python backtest/engine.py --period train
```

Output is written to `backtest_results/`:
- `{strategy}_pnl.png` — cumulative PnL with benchmark overlay
- `{strategy}_drawdown.png` — drawdown chart
- `{strategy}_weights.png` — portfolio weights over time
- `{strategy}_metrics.json` — all scalar metrics
- `{strategy}_net_returns.csv` — time series for downstream analysis

---

## Strategy Interface

Any `.py` file passed via `--strategy` must expose:

```python
def generate_signals(
    df_poly:  pd.DataFrame,  # Polymarket features (DatetimeIndex UTC, wide format)
    df_bars:  pd.DataFrame,  # S&P 100 close prices (DatetimeIndex UTC)
    **kwargs
) -> pd.DataFrame:
    """
    Returns: DatetimeIndex (UTC), columns = tickers, values ∈ [-1, 1]
    """
```

The engine handles delay, cost, and PnL computation — the strategy only needs to produce target weights.

---

## Metrics

| Metric                     | Description                                      |
|----------------------------|--------------------------------------------------|
| `sharpe_ratio`             | Annualized Sharpe (net of costs)                 |
| `net_cumulative_return`    | Total return over the test period                |
| `annualised_net`           | Net return annualized (prorated from period)     |
| `max_drawdown`             | Peak-to-trough decline on net cumulative PnL     |
| `n_trades`                 | Number of rebalancing events (periods with ΔW≠0) |
| `total_spread_cost`        | Sum of all execution costs paid                  |
| `avg_gross_exposure`       | Average Σ|W_i| across all periods               |

---

## Results Summary (Test Set: Nov–Dec 2025)

| Strategy                      | Net Return | Sharpe | Max DD | Verdict          |
|-------------------------------|:----------:|:------:|:------:|------------------|
| S&P 100 Buy & Hold            |  +2.8%     |  +1.04 | −2.1%  | Benchmark        |
| **S1 — Causal Granger**       | **+5.9%**  |**+2.11**| −1.3% | ✅ Validated      |
| S2 — Earnings Spike           |  +1.1%     |  +0.46 | −0.9%  | ⚠️ Inconclusive (N=15) |
| S3 — Natural Disasters        |  −4.4%     |  −1.56 | −5.2%  | ❌ Rejected       |

> All results include 1.5 bps average execution costs and 1-min ε-delay.
