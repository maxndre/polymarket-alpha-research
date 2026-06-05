# Polymarket Alpha Research

> **Can decentralized prediction markets contain alpha signals that precede traditional equity markets?**

A quantitative research project from CentraleSupélec investigating whether real-money prediction markets on [Polymarket](https://polymarket.com) carry information that leads price discovery in S&P 100 equities.

**3 independent strategies** | **14,465 prediction markets** | **$1.21B USDC volume** | **S&P 100 1-min bars, full year 2025**

---

## Results

| Strategy | Hypothesis | Universe | Sharpe (test) | N trades | Verdict |
|----------|-----------|----------|:-------------:|:--------:|:-------:|
| [S1 — Causal Granger](strategies/s1_causal_granger/) | Polymarket financials markets Granger-cause S&P 100 financials | JPM, GS, BAC, MS, WFC... | **+2.11** | 60d | ✅ Signal found |
| [S2 — Earnings Spike](strategies/s2_earnings_spike/) | Anomalous volume on earnings prediction markets leaks directional info | S&P 100 | +0.46 | 15 | ⚠️ Inconclusive |
| [S3 — Natural Disasters](strategies/s3_natural_disasters/) | Natural disaster markets lead P&C insurance stocks | ACGL, AIG, ALL, CB, TRV... | −1.56 | 42 | ❌ No edge |

*Train: Jan–Oct 2025 · Test: Nov–Dec 2025 · All Sharpe ratios annualized, net of bid-ask spreads*

---

## Notebooks

The research story for each strategy is told as a self-contained Jupyter notebook — hypothesis, data, signal generation, backtest, and conclusions in one place.

| Notebook | Description | Author |
|----------|-------------|--------|
| [`01_dataset_overview`](notebooks/01_dataset_overview.ipynb) | Polymarket & S&P 100 data exploration, 8-step filtering pipeline walkthrough | Hugo |
| [`02_s1_causal_granger`](notebooks/02_s1_causal_granger.ipynb) | Granger causality discovery, causal graph, signal generation, backtest | Hugo |
| [`03_s2_earnings_spike`](notebooks/03_s2_earnings_spike.ipynb) | 6-iteration research process, direction inversion finding, final results | Maxandre |
| [`04_s3_natural_disasters`](notebooks/04_s3_natural_disasters.ipynb) | Cross-market correlation, optimization surface, why no edge was found | Théophile |

---

## Architecture

```
Raw Polymarket DB                 S&P 100 1-min OHLCV
  (~58k markets)                    (101 tickers)
       │                                  │
       ▼                                  │
  pipeline/                              │
  8-step filtering funnel                │
  (58k → 14,465 markets)                │
       │                                  │
       ├── Granger causality  ────────────┤
       │   (causal graph)                 │
       │                                  │
       ├── Volume z-score spike  ─────────┤
       │   (earnings markets)             │
       │                                  │
       └── Disaster signals  ─────────────┤
           (P&C insurance)                │
                                          │
                              backtest/engine.py
                              (bid-ask fills, mark-to-market,
                               realistic microstructure)
                                          │
                                    Metrics & P&L
```

---

## Repository Structure

```
polymarket-alpha-research/
│
├── data/
│   ├── README.md              ← Dataset documentation & how to reproduce
│   └── samples/               ← Small extracts for notebook demos
│
├── pipeline/                  ← 8-step Polymarket data processing
│   ├── step1_audit.py         ← Schema validation & missing data
│   ├── step2_funnel.py        ← 7-stage market filter (58k → 14.5k)
│   ├── step3_distributions.py
│   ├── step4_concentration.py
│   ├── step5_market_polarity.py
│   ├── step6_features.py      ← Z-scores, normalized metrics
│   ├── step7_causal_discovery.py ← Granger causality → GML graph
│   └── run_pipeline.py        ← CLI orchestrator
│
├── backtest/
│   ├── engine.py              ← Event-driven backtest (bid-ask fills)
│   └── metrics.py             ← Sharpe, max drawdown, calmar, etc.
│
├── strategies/
│   ├── s1_causal_granger/     ← Hugo: Granger causality signal
│   ├── s2_earnings_spike/     ← Maxandre: earnings volume anomaly
│   └── s3_natural_disasters/  ← Théophile: disaster → insurance signal
│
├── notebooks/                 ← Research narratives (render on GitHub)
│
└── presentation/
    └── final_slides.pdf
```

---

## Data

Raw data is not included in this repository due to size and licensing constraints. See [`data/README.md`](data/README.md) for full documentation, schema, and instructions to reproduce the datasets.

| Dataset | Source | Coverage | Size |
|---------|--------|----------|------|
| Polymarket markets | Polymarket API | Jan–Dec 2025, 14,465 markets after filtering | ~150 MB |
| S&P 100 1-min OHLCV | Market data provider | Jan–Dec 2025, 101 tickers | ~310 MB |
| S&P 100 Q4 bid-ask quotes | Market data provider | Oct–Dec 2025 | ~50 MB |

Small samples for running the notebooks are available in [`data/samples/`](data/samples/).

---

## Quickstart

```bash
git clone https://github.com/YOUR_ORG/polymarket-alpha-research
cd polymarket-alpha-research
pip install -r requirements.txt

# Run the data pipeline (requires raw data — see data/README.md)
python pipeline/run_pipeline.py

# Run a strategy backtest
python strategies/s1_causal_granger/strategy.py --period test

# Or open any notebook directly
jupyter notebook notebooks/02_s1_causal_granger.ipynb
```

---

## Team

| Name | Role | GitHub |
|------|------|--------|
| **Hugo Dunias** | Data pipeline, backtest engine, S1 Causal Granger | [@hugo-dunias](https://github.com/) |
| **Maxandre Goillot** | S2 Earnings Spike (6 iterations) | [@maxandregoillot](https://github.com/maxandregoillot) |
| **Théophile Thibaudon** | S3 Natural Disasters | [@theophile-thibaudon](https://github.com/) |

*CentraleSupélec ST4-EI · June 2026*

---

## Key Findings

**S1 — Causal Granger (✅):** Polymarket financial sector markets exhibit statistically significant Granger causality with S&P 100 financials. A signal built on this lead-lag relationship achieves Sharpe +2.11 on the held-out test set (Nov–Dec 2025).

**S2 — Earnings Spike (⚠️):** Large volume spikes on Polymarket earnings markets reliably detect the *timing* of information release, but not the *direction* of price movement — the signal inverts perfectly between train and test sets. Main insight: Polymarket acts as an event timer, not a directional predictor. N=15 test trades is too small to conclude. See the [research log](strategies/s2_earnings_spike/RESEARCH_LOG.md) for the full 6-iteration investigation.

**S3 — Natural Disasters (❌):** No consistent lead-lag relationship found between Polymarket natural disaster markets and P&C insurance stock returns across any tested parameterization (Sharpe range: −9.7 to −1.56).

---

## License

MIT License — see [LICENSE](LICENSE) for details.
