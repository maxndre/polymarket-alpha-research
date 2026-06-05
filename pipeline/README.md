# Data Pipeline

Eight-step pipeline transforming raw Polymarket SQLite databases into features ready for strategy backtesting.

## Steps

```
step1_audit              Map and audit the raw database
step2_funnel             Apply 7-filter usability funnel (58k → 14.5k markets)
step3_distributions      Compute per-market distributional statistics
step4_concentration      Gini, HHI, top-N share analysis
step4b_market_categories LLM-based thematic classification into 19 categories
step5_market_polarity    Assign +1/−1/0 polarity using heuristics + LLM
step6_features           Generate dense 1-min category-level features
step7_causal_discovery   Tree-based Granger causality via HistGradientBoosting
```

## Quick Start

```bash
# Run all 7 steps in sequence
python pipeline/run_pipeline.py

# Force rebuild of Parquet cache from raw SQLite
python pipeline/run_pipeline.py --rebuild
```

---

## Step-by-Step Reference

### Step 1 — Database Audit (`step1_audit.py`)
Reads all tables and produces a structured report on:
- Missing values per column
- Microstructure anomalies (negative spreads, zero volumes, stale prices)
- Per-market activity (trade count, total volume, date range, resolution ratio)

Output: `data/processed/audit_report.json`

### Step 2 — Usability Funnel (`step2_funnel.py`)
Seven sequential filters:

| # | Filter | Threshold |
|---|--------|-----------|
| 1 | Minimum total notional volume | > $1,000 |
| 2 | Minimum trade count | ≥ 20 trades |
| 3 | Minimum active lifetime | ≥ 7 days |
| 4 | Exclude unresolved markets | resolution ratio > 0 |
| 5 | Exclude markets with >40% missing bars | completeness > 0.6 |
| 6 | Exclude markets tagged as excluded (filter_tag table) | — |
| 7 | LLM semantic filter: keep if relevant to macro/equity events | async batch via DeepSeek |

The LLM filter uses `AsyncOpenAI` with a semaphore of 15 concurrent requests and caches results to `data/processed/semantic_clusters.parquet` for reproducibility.

**Attrition:** 58,000 raw markets → 14,500 usable markets (−75%)

### Step 3 — Distributions (`step3_distributions.py`)
For each usable market, computes:
- Trade count, total volume, average spread, Δt distribution, duration
- Resolution ratio (resolved/total)

Output: `data/processed/distribution_report.json` + `per_market_usable.parquet`

### Step 4 — Concentration (`step4_concentration.py`)
Market-level concentration analysis:
- **Gini coefficient** on trade count and volume
- **Top-N shares**: top 0.1%, 1%, 10% of markets by volume
- **Temporal concentration**: last-48h volume surge detection (Lorenz curve data)
- **Categorical breakdown** by LLM category

### Step 4b — Market Category Classification (`step4b_market_categories.py`)
Classifies each usable market question into one of 19 thematic categories using an LLM:

`Crypto Prices`, `Equities & Earnings`, `Fed Policy`, `Trade & Tariffs`, `Inflation`,
`Macro / Economy`, `War & Conflict`, `Geopolitics`, `Crypto Regulation`, `Law & Justice`,
`Energy & Commodities`, `AI & Technology`, `Science & Tech`, `Health & Pharma`,
`Immigration`, `Acquisitions`, `Sports`, `Entertainment & Showbiz`, `Other`

Results are processed in batches of 50 with checkpoint/resume support. Unknown LLM
responses are mapped to `"Other"` to prevent silent data corruption.

Output: `data/processed/market_categories.parquet`  — columns: `market_id`, `llm_category`

### Step 5 — Market Polarity (`step5_market_polarity.py`)
Assigns directional polarity to each market outcome:
- `+1` — outcome price correlates positively with good equity news (e.g., "Will AAPL beat earnings?")
- `−1` — outcome price correlates negatively (e.g., "Will GDP miss?")
- `0` — ambiguous / no clear direction

Heuristics handle known categories; LLM handles edge cases. Saves progress via JSONL checkpoint for resume support.

Requires: `LLM_API_KEY` environment variable.

### Step 6 — Feature Generation (`step6_features.py`)
Aggregates tick data into dense 1-min × category features:

| Feature | Description |
|---------|-------------|
| `price_weighted_mean` | Signed VWAP across all active markets in category |
| `shannon_entropy` | Mean binary market uncertainty: −[p·log(p) + (1−p)·log(1−p)] |
| `volume_shares_total` | Total share volume traded |
| `hhi_volume` | Herfindahl-Hirschman Index of within-category volume |
| `return_5m`, `return_1h` | Rolling VWAP returns |
| `entropy_delta_5m` | 5-min change in Shannon entropy |

Output: wide Parquet (one row per minute, one column per feature × category) with train/test splits.

### Step 7 — Causal Discovery (`step7_causal_discovery.py`)
Tree-based Granger causality test:

1. Build lagged feature matrix from Polymarket category features (lags: 15m, 30m, 1h, 2h, 4h, 8h)
2. For each S&P 100 ticker, fit a `HistGradientBoostingRegressor` using purged time-series cross-validation (3 splits, 1-hour gap)
3. Retain features with strictly positive out-of-sample permutation importance

Output: `data/features/causal_edges.parquet`
Schema: `(source_category, target_ticker, feature, lag_minutes, importance, weight)`

---

## Configuration

All thresholds and paths are in `pipeline/config.py`. No hardcoded values in step files.

```python
# config.py (excerpt)
class FunnelConfig:
    min_volume: float = 1_000.0
    min_trades: int   = 20
    min_days:   int   = 7
    completeness_threshold: float = 0.6

LLM_API_KEY  = os.environ.get("LLM_API_KEY", "")
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "https://api.deepseek.com")
LLM_MODEL    = os.environ.get("LLM_MODEL",    "deepseek-chat")
```

Set the API key via environment variable before running Step 2 or Step 5:

```bash
export LLM_API_KEY="your-api-key-here"
python pipeline/run_pipeline.py
```

---

## Outputs

```
data/
├── processed/
│   ├── audit_report.json
│   ├── funnel_attrition.json
│   ├── distribution_report.json
│   ├── concentration_report.json
│   ├── per_market_usable.parquet
│   ├── trade_price_1m_usable.parquet
│   ├── market_categories.parquet
│   ├── market_polarities.parquet
│   └── semantic_clusters.parquet
└── features/
    ├── features_1m_long.parquet
    ├── features_1m_train_long.parquet
    ├── features_1m_test_long.parquet
    ├── features_1m_wide.parquet
    ├── features_1m_train_wide.parquet
    ├── features_1m_test_wide.parquet
    └── causal_edges.parquet
```
