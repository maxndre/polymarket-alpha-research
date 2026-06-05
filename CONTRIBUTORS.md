# Contributors

This project was built by three students from the **École CentraleSupélec** as part of the ST4 EI Alpha research seminar (2024–2025 cohort).

---

## Hugo Dunias — Strategy 1: Causal Granger
`strategies/s1_causal_granger/` · `pipeline/step7_causal_discovery.py`

Hugo designed and implemented the full causal discovery pipeline and the primary trading strategy.

**Contributions:**
- Data pipeline: Steps 1–7 (audit, funnel, distributions, concentration, polarity, features, causal discovery)
- Non-linear Granger causality framework using `HistGradientBoosting` + permutation importance
- Signal generation with Z-score regime detection and state-machine position management
- Backtest engine architecture (execution delay, spread costs, purged cross-validation)

**Key results:** Annualized Sharpe **+2.11** on Nov–Dec 2025 test set (vs. S&P 100 B&H +1.04)

---

## Maxandre Goillot — Strategy 2: Earnings Spike
`strategies/s2_earnings_spike/`

Maxandre designed a mean-reversion strategy around Polymarket earnings prediction markets and S&P 100 post-announcement price dynamics.

**Contributions:**
- LLM-based earnings market identification and classification
- Post-announcement window analysis and IC (Information Coefficient) computation
- Placebo tests confirming that predictive power is concentrated in the post-announcement window
- Iterative calibration across 6 strategy versions (V0.1 → V0.6)

**Key results:** Sharpe **+0.46** (N=15 events, ⚠️ inconclusive — sample too small)

---

## Théophile Thibaudon — Strategy 3: Natural Disasters
`strategies/s3_natural_disasters/`

Théophile explored whether Polymarket disaster/weather prediction markets carry advance information about the insurance and energy sectors of the S&P 100.

**Contributions:**
- Disaster/weather market identification and severity scoring
- Sector exposure mapping: insurance (CB, CI, MET), energy (CVX, XOM)
- Three signal variants: long/short, long/cash, mixed regime
- Robustness checks across event sizes and lead times

**Key results:** Sharpe **−1.56** (❌ rejected — no predictive edge found)

---

## Academic Context

> **ST4 EI Alpha** — CentraleSupélec Quantitative Finance Research Seminar, Academic Year 2024–2025
>
> Research question: *Can decentralized prediction market prices on Polymarket carry advance information about S&P 100 equity returns?*
>
> Supervised by the EI (Engineering and Industry) department.
> All data is historical. No real capital was deployed.
