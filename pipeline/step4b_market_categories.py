"""
step4b_market_categories.py — Step 4b: LLM-based market category classification.

Classifies each usable Polymarket question into one of ~19 fine-grained
thematic categories. Results are consumed by:
  - step5_market_polarity.py  (heuristic polarity assignment per category)
  - step6_features.py         (1-min feature aggregation per category)

Categories
----------
Crypto Prices, Equities & Earnings, Sports, Entertainment & Showbiz,
Fed Policy, Trade & Tariffs, Inflation, Macro / Economy, War & Conflict,
Geopolitics, Crypto Regulation, Law & Justice, Energy & Commodities,
AI & Technology, Science & Tech, Health & Pharma, Immigration,
Acquisitions, Other
"""

from __future__ import annotations
import json
import logging
import time

import pandas as pd
from openai import OpenAI

from .config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL, PROCESSED_DIR

logger = logging.getLogger(__name__)

BATCH_SIZE  = 50
MAX_RETRIES = 3
RETRY_DELAY = 5

CHECKPOINT_FILE = PROCESSED_DIR / "market_categories_checkpoint.jsonl"

ALL_CATEGORIES: list[str] = [
    "Crypto Prices",
    "Equities & Earnings",
    "Sports",
    "Entertainment & Showbiz",
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
    "Immigration",
    "Acquisitions",
    "Other",
]

_CATEGORY_LIST_STR = "\n".join(f"- {c}" for c in ALL_CATEGORIES)

_SYSTEM_PROMPT = f"""You are a financial data analyst classifying prediction market questions into thematic categories.

Available categories:
{_CATEGORY_LIST_STR}

Rules:
- Assign exactly ONE category per question (the most specific fit).
- If nothing fits clearly, use "Other".
- Return ONLY a valid JSON object mapping each market_id (string) to its category (string).
- No markdown, no explanation.

Example: {{"123": "Fed Policy", "456": "Equities & Earnings", "789": "Other"}}"""


def _build_prompt(batch: list[dict]) -> str:
    lines = [
        f'[{m["market_id"]}] "{m["question"][:120].replace(chr(10), " ")}"'
        for m in batch
    ]
    return "Classify each market into exactly one category.\n\n" + "\n".join(lines)


def _call_llm(client: OpenAI, batch: list[dict]) -> dict[str, str]:
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
            parsed = json.loads(raw)
            # Reject unknown category labels to avoid silent data corruption
            return {
                str(k): v if v in ALL_CATEGORIES else "Other"
                for k, v in parsed.items()
            }
        except Exception as e:
            logger.warning("LLM attempt %d/%d failed: %s", attempt, MAX_RETRIES, e)
            time.sleep(RETRY_DELAY * attempt)
    logger.error("All attempts failed — defaulting to 'Other' for this batch.")
    return {str(m["market_id"]): "Other" for m in batch}


def _load_checkpoint() -> dict[str, str]:
    if not CHECKPOINT_FILE.exists():
        return {}
    results: dict[str, str] = {}
    with open(CHECKPOINT_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    results.update(json.loads(line))
                except Exception:
                    pass
    logger.info("Checkpoint loaded: %d markets already categorized.", len(results))
    return results


def _save_checkpoint(batch_result: dict[str, str]) -> None:
    with open(CHECKPOINT_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(batch_result, ensure_ascii=False) + "\n")


def run_market_categorization(df_usable_markets: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Step 4b — LLM-based thematic classification of usable markets.

    Parameters
    ----------
    df_usable_markets : DataFrame, optional
        Must contain columns [market_id, question]. If None, reads from
        data/processed/per_market_usable.parquet + market.parquet.

    Returns
    -------
    DataFrame[market_id: int, llm_category: str]
    Saves data/processed/market_categories.parquet as a side-effect.
    """
    logger.info("=== STEP 4b: MARKET CATEGORY CLASSIFICATION ===")

    if df_usable_markets is None:
        usable_ids = pd.read_parquet(
            PROCESSED_DIR / "per_market_usable.parquet", columns=["market_id"]
        )["market_id"].unique()
        df = pd.read_parquet(
            PROCESSED_DIR / "market.parquet", columns=["market_id", "question"]
        )
        df = df[df["market_id"].isin(usable_ids)].copy()
    else:
        df = (
            df_usable_markets[["market_id", "question"]]
            .drop_duplicates("market_id")
            .copy()
        )

    logger.info("Total markets to categorize: %d", len(df))

    checkpoint   = _load_checkpoint()
    already_done = set(checkpoint.keys())
    to_classify  = df[~df["market_id"].astype(str).isin(already_done)].copy()
    logger.info("Remaining (not in checkpoint): %d", len(to_classify))

    if not to_classify.empty:
        client    = OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)
        records   = to_classify[["market_id", "question"]].to_dict(orient="records")
        n_batches = (len(records) + BATCH_SIZE - 1) // BATCH_SIZE

        for idx in range(n_batches):
            batch  = records[idx * BATCH_SIZE : (idx + 1) * BATCH_SIZE]
            logger.info("Batch %d/%d (%d markets)", idx + 1, n_batches, len(batch))
            result = _call_llm(client, batch)
            _save_checkpoint(result)
            checkpoint.update(result)
            time.sleep(0.3)

    # Any market not returned by LLM defaults to "Other"
    for _, row in df.iterrows():
        mid = str(row["market_id"])
        if mid not in checkpoint:
            checkpoint[mid] = "Other"

    out_df = pd.DataFrame(
        [(int(k), v) for k, v in checkpoint.items()],
        columns=["market_id", "llm_category"],
    )
    usable_mids = df["market_id"].astype(int).unique()
    out_df = out_df[out_df["market_id"].isin(usable_mids)].reset_index(drop=True)

    out_path = PROCESSED_DIR / "market_categories.parquet"
    out_df.to_parquet(out_path, index=False)

    logger.info("Saved %d rows → %s", len(out_df), out_path)
    logger.info("Category distribution:\n%s", out_df["llm_category"].value_counts().to_string())

    if CHECKPOINT_FILE.exists():
        CHECKPOINT_FILE.unlink()

    return out_df
