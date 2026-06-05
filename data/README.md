# Data

Raw datasets are **not included** in this repository due to licensing and size constraints.
This file documents the full schema and the steps to reproduce them.

---

## Datasets

### 1. Polymarket — Prediction Markets (train + test)

| File | Period | Markets | Bars | Size |
|------|--------|---------|------|------|
| `raw/polymarket_train_2025.sqlite3` | Jan–Oct 2025 | 58,182 raw → 14,465 after filtering | ~5M 1-min bars | ~650 MB |
| `processed/trade_price_1m_usable_train.parquet` | Jan–Oct 2025 | 14,465 | 3,397,419 | ~105 MB |
| `processed/trade_price_1m_usable_test.parquet` | Nov–Dec 2025 | 14,465 | 1,524,257 | ~42 MB |

**SQLite schema:**

```sql
market           (condition_id, event_id, question, start_date, end_date, tags_json, ...)
token_outcome    (token_id, condition_id, outcome_label, outcome_index)
trade_price_1m   (condition_id, token_id, minute_ts,
                  open_price, high_price, low_price, close_price,
                  volume_shares_1m, notional_usdc_1m, trades_count_1m)
market_tag       (condition_id, tag_slug, tag_label)
selected_filter_tag (tag_slug, bucket_label)
```

All `*_price` columns are stored as `TEXT` in SQLite and cast to `float64` on load.
`minute_ts` is an ISO-8601 UTC string; the loader converts it to `datetime64[ns, UTC]`.

**Key statistics (post-filtering):**
- 14,465 usable markets (24.8% of raw 58,182)
- $1.21B USDC total volume (train)
- Gini coefficient of volume: 0.689 — top 5% of markets hold ~60% of volume
- Outcome distribution: Yes 27.6%, No 38.6%, Up 17.0%, Down 16.8%
- Top categories by volume: Crypto Prices, Bitcoin, Ethereum, Equities & Earnings, Macro

**Filtering pipeline attrition (step2_funnel.py):**

| Filter | Remaining | Removed |
|--------|-----------|---------|
| Start (raw) | 58,182 | — |
| 1. Standard binary (Yes/No or Up/Down) | ~42,000 | −28% |
| 2. Inferred resolution (last price → 0 or 1) | ~30,000 | −29% |
| 3. Temporal validity (1h – 2yr duration) | ~28,000 | −7% |
| 4. Volume ≥ $1,000 USDC | ~20,000 | −29% |
| 5. Activity ≥ 300 trades | ~16,000 | −20% |
| 6. Price continuity (gap ≤ 24h) | ~14,800 | −7% |
| 7. Semantic filter (TradFi-relevant via LLM) | **14,465** | −2% |

---

### 2. S&P 100 — 1-min OHLCV Bars

| File | Period | Tickers | Bars | Size |
|------|--------|---------|------|------|
| `raw/sp100_2025_1min_bars.sqlite3` | Jan–Dec 2025 | 101 | ~9.6M | ~420 MB |
| `processed/sp100_2025_1min_bars_train.parquet` | Jan–Oct 2025 | 101 | 8,052,572 | ~256 MB |
| `processed/sp100_2025_1min_bars_test.parquet` | Nov–Dec 2025 | 101 | 1,570,372 | ~52 MB |

```sql
-- bars table
bars (time TEXT, ticker TEXT, open REAL, high REAL, low REAL, close REAL, volume REAL)
```

101 tickers including: NVDA, GOOGL, AAPL, MSFT, AMZN, TSLA, META, JPM, GS, BAC, MS...

---

### 3. S&P 100 — Q4 Bid/Ask Quotes (for backtest spread costs)

| File | Period | Size |
|------|--------|------|
| `raw/sp100_2025_q4_1min_quotes_partial.sqlite3` | Oct–Dec 2025 | ~50 MB |

```sql
quote_1m (time TEXT, ticker TEXT, bid_price REAL, ask_price REAL)
```

Median spread: **2.82 bps** per 1-min candle on liquid names.
Used by `backtest/engine.py` for realistic execution cost simulation.

---

## Train / Test Split

```
Jan 2025 ──────────────── Oct 31, 2025 │ Nov 1, 2025 ── Dec 31, 2025
          TRAIN (10 months)             │       TEST (2 months)
          Signal calibration            │    Out-of-sample backtest
```

---

## How to Reproduce

1. **Polymarket data** — Download via the [Polymarket API](https://docs.polymarket.com)
   or request the academic dataset.
2. **S&P 100 bars** — Any financial data provider (Polygon.io, Alpaca, etc.) for
   1-min OHLCV on the S&P 100 constituents for calendar year 2025.
3. Place raw files in `data/raw/` then run:

```bash
python pipeline/run_pipeline.py
```

This regenerates all `data/processed/` Parquet files from scratch.

---

## Samples

`data/samples/` contains small CSV extracts (~500 rows each) for running the
notebooks without the full datasets:

- `polymarket_sample.csv` — 500 rows from `trade_price_1m_usable_train`
- `sp100_sample.csv` — 500 rows of 1-min OHLCV bars
