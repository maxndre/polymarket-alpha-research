"""
step2_funnel.py — Step 2: Usability funnel (7 sequential filters).

Reduces 58k raw Polymarket markets to ~14.5k tradable ones.
All filtering is vectorized (no iterrows/itertuples).

Filter order
------------
1. Standard binary outcome type (Yes/No or Up/Down only)
2. Inferred resolution (last close price → 0 or 1)
3. Temporal validity (duration in [1h, 2yr])
4. Volume / liquidity (≥ $1,000 USDC)
5. Minimum activity (≥ 300 trades)
6. Price continuity (max gap between active bars ≤ 24h)
7. Semantic relevance (LLM-based TradFi filter via async batches)

Note on resolution:
  The `resolved` column is masked in the dataset.
  We infer resolution from the last close price of the YES/Up token.
"""

from __future__ import annotations
import asyncio
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd
from openai import AsyncOpenAI
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

from .config import FunnelConfig, PROCESSED_DIR

logger = logging.getLogger(__name__)


# ── Text preprocessing (for LLM semantic filter) ─────────────────────────────

def _clean_question(text: str) -> str:
    """Strip dates, prices, and percentages — improves LLM clustering quality."""
    text = re.sub(r'(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2}(?:st|nd|rd|th)?', '', text, flags=re.IGNORECASE)
    text = re.sub(r'(?:January|February|March|April|May|June|July|August|September|October|November|December)', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\d{1,2}(?::\d{2})?\s*(?:AM|PM)\s*(?:ET|EST|EDT|UTC)?', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\$[\d,]+(?:\.\d+)?', '', text)
    text = re.sub(r'\d+(?:\.\d+)?%', '', text)
    text = re.sub(r'\b202[4-9]\b', '', text)
    return re.sub(r'\s+', ' ', text).strip()


# ── Attrition logger ──────────────────────────────────────────────────────────

def _log_step(step: int, label: str, n_before: int, n_after: int) -> dict:
    removed     = n_before - n_after
    pct_removed = removed / n_before * 100 if n_before > 0 else 0.0
    pct_left    = n_after  / n_before * 100 if n_before > 0 else 0.0
    logger.info(
        "  [Filter %d — %s]  %d → %d  (−%d, −%.1f%%  |  %.1f%% remaining)",
        step, label, n_before, n_after, removed, pct_removed, pct_left,
    )
    return {
        "step": step, "label": label,
        "n_before": n_before, "n_after": n_after,
        "n_removed": removed,
        "pct_removed":   round(pct_removed, 2),
        "pct_remaining": round(pct_left,    2),
    }


# ── Filter helpers ────────────────────────────────────────────────────────────

def _infer_resolution(df_bars: pd.DataFrame,
                      df_token: pd.DataFrame,
                      cfg: FunnelConfig) -> pd.DataFrame:
    """
    Returns DataFrame: condition_id | last_close_yes | resolved_yes | resolved_no | is_resolved
    Uses outcome_index == 0 (first token = YES/Up side).
    """
    yes_tokens = df_token[df_token["outcome_index"] == 0][["token_id", "condition_id"]].copy()
    bars_yes   = df_bars.merge(yes_tokens, on=["token_id", "condition_id"], how="inner")

    last_close = (
        bars_yes.sort_values("minute_ts")
        .groupby("condition_id", observed=True)["close_price"]
        .last()
        .reset_index()
        .rename(columns={"close_price": "last_close_yes"})
    )
    last_close["resolved_yes"] = last_close["last_close_yes"] >= cfg.RESOLUTION_YES_THRESHOLD
    last_close["resolved_no"]  = last_close["last_close_yes"] <= cfg.RESOLUTION_NO_THRESHOLD
    last_close["is_resolved"]  = last_close["resolved_yes"] | last_close["resolved_no"]
    return last_close


def _standard_binary_cids(df_token: pd.DataFrame, cfg: FunnelConfig) -> set:
    """Keep only markets whose outcome labels exactly match Yes/No or Up/Down."""
    token_labels = (
        df_token.copy()
        .assign(label_low=df_token["outcome_label"].str.lower().str.strip())
        .groupby("condition_id", observed=True)["label_low"]
        .apply(frozenset)
        .reset_index(name="label_set")
    )
    accepted = {frozenset(pair) for pair in cfg.ACCEPTED_OUTCOME_PAIRS}
    mask     = token_labels["label_set"].isin(accepted)
    return set(token_labels.loc[mask, "condition_id"])


# ── Semantic LLM filter ───────────────────────────────────────────────────────

async def _process_batch(client: AsyncOpenAI, batch: list, cfg: FunnelConfig) -> str:
    prompt = (
        "Below are prediction market questions. For each, decide if it is relevant "
        "to financial trading (finance, macro, crypto) with 'KEEP', otherwise 'REJECT' "
        "(noise, pop culture, entertainment).\n"
        "Also give exactly 5 keywords separated by commas.\n"
        "Strict format per line: ID: KEEP | kw1, kw2, kw3, kw4, kw5\n\n"
    )
    for i, q in batch:
        prompt += f"{i}: {q}\n"
    try:
        resp = await client.chat.completions.create(
            model=cfg.LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        return resp.choices[0].message.content
    except Exception as e:
        logger.error("LLM batch error: %s", e)
        return ""


async def _run_all_batches(unique_qs: list, cfg: FunnelConfig) -> list[str]:
    client     = AsyncOpenAI(api_key=cfg.LLM_API_KEY, base_url=cfg.LLM_BASE_URL)
    batch_size = 20
    batches    = [
        [(i, q) for i, q in enumerate(unique_qs)][j:j + batch_size]
        for j in range(0, len(unique_qs), batch_size)
    ]
    sem = asyncio.Semaphore(15)

    async def sem_task(b):
        async with sem:
            return await _process_batch(client, b, cfg)

    logger.info("  Dispatching %d async LLM requests ...", len(batches))
    return await asyncio.gather(*[sem_task(b) for b in batches])


def _semantic_filter(df_markets: pd.DataFrame,
                     active_cids: set,
                     cfg: FunnelConfig) -> set:
    """
    LLM-based semantic filter: classify each unique question as KEEP or REJECT.
    Results are cached in data/processed/semantic_clusters.parquet.
    Also builds a 2D TF-IDF + SVD embedding on LLM-generated keywords for visualization.
    """
    cache_path = PROCESSED_DIR / "semantic_clusters.parquet"

    if cache_path.exists():
        logger.info("  [Cache] Loading existing semantic_clusters.parquet ...")
        cached = pd.read_parquet(cache_path)
        return set(cached.loc[cached["is_tradfi"], "condition_id"]).intersection(active_cids)

    logger.info("  Building semantic filter from scratch ...")
    mkt_sub = df_markets[df_markets["condition_id"].isin(active_cids)].copy()
    mkt_sub["clean_q"] = mkt_sub["question"].fillna("").apply(_clean_question)

    unique_qs = mkt_sub["clean_q"].unique().tolist()
    logger.info("  %d unique questions to classify ...", len(unique_qs))

    results_raw = asyncio.run(_run_all_batches(unique_qs, cfg))

    # Parse LLM responses
    results_dict: dict[int, dict] = {}
    for res in results_raw:
        if not res:
            continue
        for line in res.strip().split("\n"):
            if ":" in line and "|" in line:
                try:
                    idx_str, rest = line.split(":", 1)
                    decision, kw_str = rest.split("|", 1)
                    results_dict[int(idx_str.strip())] = {
                        "keep":     "KEEP" in decision.upper(),
                        "keywords": [k.strip() for k in kw_str.split(",")],
                    }
                except Exception:
                    pass

    logger.info("  %d questions successfully parsed.", len(results_dict))

    q_to_res = {
        q: results_dict.get(i, {"keep": True, "keywords": ["finance", "crypto", "macro", "trading", "market"]})
        for i, q in enumerate(unique_qs)
    }

    mkt_sub["llm_res"]  = mkt_sub["clean_q"].map(q_to_res)
    mkt_sub["is_tradfi"] = mkt_sub["llm_res"].apply(lambda x: x["keep"])
    mkt_sub["keywords"]  = mkt_sub["llm_res"].apply(lambda x: " ".join(x["keywords"][:5]))

    # 2D semantic embedding via TF-IDF + SVD (useful for visualization)
    vectorizer = TfidfVectorizer(max_features=1000)
    X = vectorizer.fit_transform(mkt_sub["keywords"])
    svd = TruncatedSVD(n_components=2, random_state=42)
    X_2d = svd.fit_transform(X)
    mkt_sub["svd_x"] = X_2d[:, 0]
    mkt_sub["svd_y"] = X_2d[:, 1]

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    mkt_sub[["condition_id", "svd_x", "svd_y", "is_tradfi", "keywords"]].to_parquet(cache_path)

    return set(mkt_sub.loc[mkt_sub["is_tradfi"], "condition_id"])


# ── Main funnel ───────────────────────────────────────────────────────────────

def apply_usability_funnel(
    df_markets:    pd.DataFrame,
    df_bars:       pd.DataFrame,
    df_token:      pd.DataFrame,
    df_filter_tag: pd.DataFrame,
    df_market_tag: pd.DataFrame,
    per_market:    pd.DataFrame,
    cfg:           FunnelConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    """
    Step 2 — 7-filter usability funnel.

    Returns
    -------
    (df_usable_markets, df_usable_bars, attrition_log)
    """
    if cfg is None:
        cfg = FunnelConfig()

    logger.info("=== STEP 2: USABILITY FUNNEL ===")
    attrition: list[dict] = []

    active_cids = set(df_markets["condition_id"].unique())
    n0 = len(active_cids)
    logger.info("Start: %d unique markets", n0)

    # Filter 1 — Standard binary type
    logger.info("Filter 1: Standard binary type (Yes/No or Up/Down) ...")
    standard_cids = _standard_binary_cids(df_token, cfg)
    active_cids  &= standard_cids
    attrition.append(_log_step(1, "Standard binary (Yes/No or Up/Down)", n0, len(active_cids)))

    # Filter 2 — Inferred resolution
    logger.info("Filter 2: Inferred resolution ...")
    n_before   = len(active_cids)
    resolution = _infer_resolution(df_bars, df_token, cfg)
    active_cids &= set(resolution.loc[resolution["is_resolved"], "condition_id"])
    attrition.append(_log_step(2, "Inferred resolution (last price → 0 or 1)", n_before, len(active_cids)))

    # Filter 3 — Temporal validity
    logger.info("Filter 3: Temporal validity ...")
    n_before = len(active_cids)
    mkt_sub  = df_markets[df_markets["condition_id"].isin(active_cids)].copy()
    duration_h = (mkt_sub["end_date"] - mkt_sub["start_date"]).dt.total_seconds() / 3600.0
    valid_t = (
        (duration_h >= cfg.MIN_DURATION_HOURS)
        & (duration_h <= cfg.MAX_DURATION_DAYS * 24)
        & duration_h.notna()
    )
    active_cids &= set(mkt_sub.loc[valid_t, "condition_id"])
    attrition.append(_log_step(3, "Temporal validity [1h, 2yr]", n_before, len(active_cids)))

    # Filter 4 — Volume / liquidity
    logger.info("Filter 4: Volume >= $%.0f USDC ...", cfg.MIN_VOLUME_USDC)
    n_before = len(active_cids)
    pm_sub   = per_market[per_market["condition_id"].isin(active_cids)]
    active_cids &= set(pm_sub.loc[pm_sub["total_volume_usdc"] >= cfg.MIN_VOLUME_USDC, "condition_id"])
    attrition.append(_log_step(4, f"Volume >= ${cfg.MIN_VOLUME_USDC:,.0f} USDC", n_before, len(active_cids)))

    # Filter 5 — Minimum activity
    logger.info("Filter 5: Activity >= %d trades ...", cfg.MIN_TRADES)
    n_before = len(active_cids)
    pm_sub   = per_market[per_market["condition_id"].isin(active_cids)]
    active_cids &= set(pm_sub.loc[pm_sub["total_trades"] >= cfg.MIN_TRADES, "condition_id"])
    attrition.append(_log_step(5, f"Activity >= {cfg.MIN_TRADES:,} trades", n_before, len(active_cids)))

    # Filter 6 — Price continuity
    logger.info("Filter 6: Price continuity (max gap <= %.0fh) ...", cfg.MAX_GAP_HOURS)
    n_before = len(active_cids)
    bars_sub = df_bars[df_bars["condition_id"].isin(active_cids)].copy()
    bars_sub = bars_sub[bars_sub["trades_count_1m"] > 0]

    if len(bars_sub) > 0:
        bars_sub = bars_sub.sort_values(["condition_id", "minute_ts"])
        bars_sub["prev_ts"] = bars_sub.groupby("condition_id", observed=True)["minute_ts"].shift(1)
        bars_sub["gap_h"]   = (bars_sub["minute_ts"] - bars_sub["prev_ts"]).dt.total_seconds() / 3600.0

        max_gap = (
            bars_sub.groupby("condition_id", observed=True)["gap_h"]
            .max().reset_index().rename(columns={"gap_h": "max_gap_h"})
        )
        n_active = (
            bars_sub.groupby("condition_id", observed=True).size()
            .reset_index(name="n_active_bars")
        )
        cont_df = max_gap.merge(n_active, on="condition_id")
        continuous = cont_df[
            (cont_df["max_gap_h"]    <= cfg.MAX_GAP_HOURS) &
            (cont_df["n_active_bars"] >= cfg.MIN_ACTIVE_BARS)
        ]
        active_cids &= set(continuous["condition_id"])
    else:
        active_cids = set()

    attrition.append(_log_step(6, f"Continuity <= {cfg.MAX_GAP_HOURS:.0f}h gap", n_before, len(active_cids)))

    # Filter 7 — Semantic TradFi relevance
    logger.info("Filter 7: LLM semantic filter (TradFi relevance) ...")
    n_before    = len(active_cids)
    tradfi_cids = _semantic_filter(df_markets, active_cids, cfg)
    active_cids &= tradfi_cids
    attrition.append(_log_step(7, "Semantic TradFi filter (LLM + TF-IDF)", n_before, len(active_cids)))

    n_final = len(active_cids)
    logger.info("=== FUNNEL RESULT: %d usable markets (%.1f%% of total) ===",
                n_final, n_final / n0 * 100)

    df_usable_markets = df_markets[df_markets["condition_id"].isin(active_cids)].copy()
    df_usable_bars    = df_bars[df_bars["condition_id"].isin(active_cids)].copy()

    df_usable_markets = df_usable_markets.merge(
        resolution[["condition_id", "last_close_yes", "resolved_yes", "resolved_no"]],
        on="condition_id", how="left",
    )
    return df_usable_markets, df_usable_bars, attrition
