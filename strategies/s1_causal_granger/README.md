# Strategy 1 — Causal Granger (S1)

This strategy utilizes Granger causality relations inferred from Polymarket predictive activity to trade components of the S&P 100 index.

## Architecture & Logic

1. **Causal Mapping (Offline):**
   Multi-resolution Vector Autoregression (VAR) modeling is used to test whether Polymarket prediction categories Granger-cause S&P 100 stock returns on the training period (Jan–Oct 2025). Edges are selected after False Discovery Rate (FDR) Bonferroni corrections and directionality checks (ensuring significance only in the direction Poly → Stock).
   
2. **Signal Generation (Online):**
   For each active edge (Category → Stock):
   - Computes rolling changes in category feature scores (e.g. price, volume, open interest).
   - Verifies if the feature's variance exceeds a local regime threshold (ensuring we trade only in high-activity regimes).
   - Normalizes the signal to a Z-score using training-set mean and standard deviation.
   - Triggers long (+1) or short (-1) positions when the aggregated Z-score exceeds `z_long` or falls below `z_short`.

3. **Stops & Leverage Constraints:**
   - Incorporates a **2.5% stop-loss** from the entry stock price.
   - Enforces **dynamic leverage scaling** so that the maximum gross portfolio exposure never exceeds 1.0 (fully collateralized).
