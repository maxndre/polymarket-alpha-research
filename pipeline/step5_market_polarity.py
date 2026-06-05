"""
step5_market_polarity.py — Step 5: Sentiment polarity labeling.

Assigns a market-level polarity to each YES token relative to global equity sentiment:
  +1 = Bullish (YES outcome is positive for equity markets)
  -1 = Bearish
   0 = Neutral / Unrelated

Heuristics applied first (Crypto, Equities & Earnings → +1; Sports → 0);
remaining categories are classified via LLM in batches of 100.
"""

from __future__ import annotations
import json
import logging
import time
from pathlib import Path

import pandas as pd
from openai import OpenAI

from .config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL, PROCESSED_DIR

logger = logging.getLogger(__name__)

BATCH_SIZE  = 100
MAX_RETRIES = 3
RETRY_DELAY = 5

CHECKPOINT_FILE = PROCESSED_DIR / "market_polarities_checkpoint.jsonl"

_SYSTEM_PROMPT = """You are a financial analyst classifying the sentiment polarity of prediction markets.
For each question, determine whether a YES outcome is Bullish (+1), Bearish (-1), or Neutral/Uncertain (0)
for global equity markets (S&P 500 / major indices).

Guidelines:
- Rate cuts, economic growth, trade agreements, peace, deregulation → +1 (Bullish)
- Rate hikes, inflation, trade wars, geopolitical escalation, sanctions → -1 (Bearish)
- Earnings beat, company-specific positive news → +1
- Earnings miss, corporate failures, lawsuits → -1
- Entertainment, sports, pop culture, unrelated topics → 0

Return ONLY a valid JSON object mapping each market_id (string) to its polarity integer.
No markdown, no explanation. Example: {"500502": 1, "502120": -1, "503901": 0}"""

# Categories classified via heuristic (no LLM call needed)
_HEURISTIC_POSITIVE = {"Crypto Prices", "Equities & Earnings"}
_HEURISTIC_NEUTRAL  = {"Sports", "Entertainment & Showbiz"}

# Categories sent to LLM
_LLM_CATEGORIES = {
    "Fed Policy", "Trade & Tariffs", "Inflation", "Macro / Economy",
    "War & Conflict", "Geopolitics", "Crypto Regulation", "Law & Justice",
    "Energy & Commodities", "AI & Technology", "Science & Tech",
    "Health & Pharma", "Other", "Immigration", "Acquisitions",
}


def _build_prompt(batch: list[dict]) -> str:
    lines = [f'[{m["market_id"]}] "{m["question"][:120].replace(chr(10), " ")}" (category: {m["llm_category"]})']
    return "Classify each market below (polarity: 1, -1, or 0).\n\n" + "\n".join(lines)


def _call_llm(client: OpenAI, batch: list[dict]) -> dict[str, int]:
    prompt = _build_prompt(batch)
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt},
                ],
                temperature=0.0,
                max_tokens=1024,
            )
            raw = resp.choices[0].message.content.strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            return {str(k): int(v) for k, v in json.loads(raw).items()}
        except Exception as e:
            logger.warning("LLM attempt %d/%d failed: %s", attempt, MAX_RETRIES, e)
            time.sleep(RETRY_DELAY * attempt)
    logger.error("All attempts failed — defaulting to 0 for this batch.")
    return {str(m["market_id"]): 0 for m in batch}


def _load_checkpoint() -> dict[str, int]:
    if not CHECKPOINT_FILE.exists():
        return {}
    results: dict[str, int] = {}
    with open(CHECKPOINT_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    results.update(json.loads(line))
                except Exception:
                    pass
    logger.info("Checkpoint loaded: %d polarities already labeled.", len(results))
    return results


def _save_checkpoint(batch_result: dict[str, int]) -> None:
    with open(CHECKPOINT_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(batch_result, ensure_ascii=False) + "\n")


def run_polarity_labeling() -> pd.DataFrame:
    """
    Step 5 — Sentiment polarity labeling.
    Reads market.parquet + market_categories.parquet, writes market_polarities.parquet.
    """
    logger.info("=== STEP 5: MARKET POLARITY LABELING ===")

    mkt  = pd.read_parquet(PROCESSED_DIR / "market.parquet",    columns=["market_id", "question"])
    cats = pd.read_parquet(PROCESSED_DIR / "market_categories.parquet", columns=["market_id", "llm_category"])
    df   = mkt.merge(cats, on="market_id")
    logger.info("Total markets to process: %d", len(df))

    # Heuristic labels
    labeled: dict[str, int] = {}
    for _, row in df[df["llm_category"].isin(_HEURISTIC_POSITIVE)].iterrows():
        labeled[str(row["market_id"])] = 1
    for _, row in df[df["llm_category"].isin(_HEURISTIC_NEUTRAL)].iterrows():
        labeled[str(row["market_id"])] = 0
    logger.info("Heuristic labels: %d markets", len(labeled))

    # LLM labels (with checkpoint resume)
    checkpoint = _load_checkpoint()
    already_labeled = {**labeled, **checkpoint}
    labeled_ids = set(already_labeled.keys())

    to_llm = df[
        df["llm_category"].isin(_LLM_CATEGORIES) &
        (~df["market_id"].astype(str).isin(labeled_ids))
    ].copy()
    logger.info("Remaining markets to label via LLM: %d", len(to_llm))

    if not to_llm.empty:
        client  = OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)
        records = to_llm[["market_id", "question", "llm_category"]].to_dict(orient="records")
        n_batches = (len(records) + BATCH_SIZE - 1) // BATCH_SIZE

        for idx in range(n_batches):
            batch  = records[idx * BATCH_SIZE : (idx + 1) * BATCH_SIZE]
            logger.info("Batch %d/%d (%d markets)", idx + 1, n_batches, len(batch))
            result = _call_llm(client, batch)
            _save_checkpoint(result)
            already_labeled.update(result)
            time.sleep(0.3)

    # Default 0 for anything not labeled
    for _, row in df.iterrows():
        mid = str(row["market_id"])
        if mid not in already_labeled:
            already_labeled[mid] = 0

    # Build and save final DataFrame
    out_path = PROCESSED_DIR / "market_polarities.parquet"
    pol_df = pd.DataFrame(list(already_labeled.items()), columns=["market_id", "polarity"])
    df["market_id"]    = df["market_id"].astype(int)
    pol_df["market_id"] = pol_df["market_id"].astype(int)
    pol_df["polarity"]  = pol_df["polarity"].astype(int)

    final_df = df[["market_id", "question", "llm_category"]].merge(pol_df, on="market_id", how="left")
    final_df["polarity"] = final_df["polarity"].fillna(0).astype(int)
    final_df.to_parquet(out_path, index=False)

    logger.info("Saved %d rows → %s", len(final_df), out_path)
    logger.info("Polarity distribution:\n%s", final_df["polarity"].value_counts().to_dict())

    if CHECKPOINT_FILE.exists():
        CHECKPOINT_FILE.unlink()

    return final_df
