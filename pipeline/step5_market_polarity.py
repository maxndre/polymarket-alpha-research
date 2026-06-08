"""
step5_b_market_polarity.py – Assigns a market-level sentiment polarity to each YES token
relative to general equity market sentiment (Bullish (+1), Bearish (-1), or Neutral (0)).

Heuristics used:
- Crypto Prices: +1 (higher crypto price = positive risk-on / crypto equity sentiment).
- Equities & Earnings: +1 (beating earnings or rising stock price = positive).
- Others: Classified via Mistral LLM in fast batches.
"""

from __future__ import annotations
import json
import logging
import time
from pathlib import Path

import pandas as pd
from openai import OpenAI

# ── Paths & Config ────────────────────────────────────────────────────────────
from pipeline.config import PARQUET_MARKET, PARQUET_DIR, MISTRAL_API_KEY, MISTRAL_BASE_URL, MISTRAL_MODEL

PARQUET_CATS    = PARQUET_DIR / "market_categories.parquet"
OUT_POLARITIES  = PARQUET_DIR / "market_polarities.parquet"
CHECKPOINT_FILE = PARQUET_DIR / "market_polarities_checkpoint.jsonl"

BATCH_SIZE   = 100
MAX_RETRIES  = 3
RETRY_DELAY  = 5

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a financial analyst classifying the sentiment polarity of prediction markets.
For each prediction market question, determine whether a YES outcome is generally Bullish (+1), Bearish (-1), or Neutral/Uncertain (0) for the global equity market (S&P 500 / major stock indices).

Guidelines:
- Rate cuts, economic growth, trade agreements, peace, ceasefire, pro-business deregulation -> +1 (Bullish / Risk-on)
- Rate hikes, higher inflation, trade tariffs, trade wars, geopolitical escalation, sanctions, strict regulations -> -1 (Bearish / Risk-off)
- Earnings beat, company-specific positive achievements -> +1 (Bullish)
- Earnings miss, lawsuits against companies, corporate failures -> -1 (Bearish)
- Entertainment, pop culture, sports, or completely unrelated to macroeconomics/corporate health -> 0 (Neutral)

Return ONLY a valid JSON object mapping each market_id (string) to its polarity integer (+1, -1, or 0). No markdown, no explanation.

Example output:
{"500502": 1, "502120": -1, "503901": 0}"""


def build_user_prompt(batch: list[dict]) -> str:
    lines = []
    for m in batch:
        q = m["question"][:120].replace("\n", " ")
        lines.append(f'[{m["market_id"]}] "{q}" (category: {m["llm_category"]})')
    return "Classify each market below into polarity: 1, -1, or 0.\n\n" + "\n".join(lines)


def call_llm(client: OpenAI, batch: list[dict]) -> dict[str, int]:
    prompt = build_user_prompt(batch)
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=MISTRAL_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt},
                ],
                temperature=0.0,
                max_tokens=1024,
            )
            raw = response.choices[0].message.content.strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            result = json.loads(raw)
            return {str(k): int(v) for k, v in result.items()}
        except Exception as e:
            logger.warning("Attempt %d failed: %s", attempt, e)
            time.sleep(RETRY_DELAY * attempt)
    # No fallback: raise RuntimeError if Mistral does not respond/fails
    logger.error("All attempts failed for batch of %d markets", len(batch))
    raise RuntimeError(f"Mistral API failed for polarity labeling after {MAX_RETRIES} attempts.")


def load_checkpoint() -> dict[str, int]:
    if not CHECKPOINT_FILE.exists():
        return {}
    results = {}
    with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    results.update(json.loads(line))
                except Exception:
                    pass
    logger.info("Checkpoint loaded: %d polarities already labeled.", len(results))
    return results


def save_checkpoint(batch_result: dict[str, int]) -> None:
    with open(CHECKPOINT_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(batch_result, ensure_ascii=False) + "\n")


def run_polarity_labeling():
    logger.info("=== STEP 5b: Market Sentiment Polarity Labeling ===")
    
    # 1. Load data
    mkt = pd.read_parquet(PARQUET_MARKET, columns=["market_id", "question"])
    cats = pd.read_parquet(PARQUET_CATS, columns=["market_id", "llm_category"])
    df = mkt.merge(cats, on="market_id")
    
    logger.info("Total markets to process: %d", len(df))
    
    # 2. Check heuristics
    # Crypto Prices and Equities & Earnings and Sports/Entertainment heuristics
    crypto_mask = df["llm_category"].isin(["Crypto Prices"])
    earnings_mask = df["llm_category"].isin(["Equities & Earnings"])
    neutral_mask = df["llm_category"].isin(["Sports", "Entertainment & Showbiz"])
    
    heuristics = {}
    for _, row in df[crypto_mask].iterrows():
        heuristics[str(row["market_id"])] = 1
    for _, row in df[earnings_mask].iterrows():
        heuristics[str(row["market_id"])] = 1
    for _, row in df[neutral_mask].iterrows():
        heuristics[str(row["market_id"])] = 0
        
    logger.info("Apply heuristics: %d markets automatically labeled (+1 for Crypto/Earnings, 0 for Sports/Showbiz)", len(heuristics))
    
    # 3. Process remaining with LLM
    checkpoint = load_checkpoint()
    already_labeled = {**heuristics, **checkpoint}
    labeled_ids = set(already_labeled.keys())
    
    # Remaining categories to process via LLM
    categories_for_llm = [
        "Fed Policy",
        "Trade & Tariffs",
        "Inflation",
        "Macro / Economy",
        "War & Conflict",
        "Geopolitics",
        "Crypto Regulation",
        "Law & Justice",
        "Energy & Commodities",
        "AI & Technology",
        "Science & Tech",
        "Health & Pharma",
        "Other",
        "Immigration",
        "Immigration/Border",
        "Acquisitions",
    ]
    
    to_llm = df[df["llm_category"].isin(categories_for_llm) & (~df["market_id"].astype(str).isin(labeled_ids))].copy()
    logger.info("Remaining markets to label via LLM: %d", len(to_llm))
    
    if not to_llm.empty:
        client = OpenAI(api_key=MISTRAL_API_KEY, base_url=MISTRAL_BASE_URL)
        records = to_llm[["market_id", "question", "llm_category"]].to_dict(orient="records")
        n_batches = (len(records) + BATCH_SIZE - 1) // BATCH_SIZE
        
        for idx in range(n_batches):
            batch = records[idx * BATCH_SIZE : (idx + 1) * BATCH_SIZE]
            logger.info("Batch %d/%d  (%d markets)", idx + 1, n_batches, len(batch))
            result = call_llm(client, batch)
            save_checkpoint(result)
            already_labeled.update(result)
            time.sleep(0.3)
            
    # Set default 0 for anything else remaining
    for _, row in df.iterrows():
        mid = str(row["market_id"])
        if mid not in already_labeled:
            already_labeled[mid] = 0
            
    # 4. Save final parquet
    logger.info("Saving polarities to %s ...", OUT_POLARITIES)
    pol_df = pd.DataFrame(list(already_labeled.items()), columns=["market_id", "polarity"])
    # Force market_id to int for robust merging
    df["market_id"] = df["market_id"].astype(int)
    pol_df["market_id"] = pol_df["market_id"].astype(int)
    pol_df["polarity"] = pol_df["polarity"].astype(int)

    final_df = df[["market_id", "question", "llm_category"]].merge(pol_df, on="market_id", how="left")
    final_df["polarity"] = final_df["polarity"].fillna(0).astype(int)
    
    final_df.to_parquet(OUT_POLARITIES, index=False)
    logger.info("Successfully saved %d rows", len(final_df))
    
    logger.info("Polarity distribution:\n%s", final_df["polarity"].value_counts())
    
    # Checkpoint cleanup if done
    if len(final_df) == len(df) and CHECKPOINT_FILE.exists():
        CHECKPOINT_FILE.unlink()
        logger.info("Checkpoint file removed.")
        
    return final_df


if __name__ == "__main__":
    run_polarity_labeling()
