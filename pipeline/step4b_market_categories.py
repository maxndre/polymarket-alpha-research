"""
step5_llm_categories.py – LLM-based categorization of ALL Polymarket markets.

Strategy
--------
• Send markets in batches of 80 to Mistral.
• Each batch is a single prompt: the LLM returns JSON {market_id: category}.
• Progress is saved incrementally to a JSONL checkpoint file so the job
  is fully resumable.
• Final output: Donnees_Netoyees/market_categories.parquet

Canonical category list (extendable by LLM when nothing fits):
  Elections, US Politics, Geopolitics, War & Conflict, Macro / Economy,
  Fed Policy, Inflation, Trade & Tariffs, Energy & Commodities,
  Crypto Prices, Crypto Regulation, Equities & Earnings, Sports,
  Entertainment & Showbiz, Science & Tech, Climate & Weather,
  Law & Justice, Health & Pharma, AI & Technology, Other
"""

from __future__ import annotations
import json
import logging
import time
from pathlib import Path

import pandas as pd
from openai import OpenAI

# ── Paths & config ────────────────────────────────────────────────────────────
from pipeline.config import PARQUET_MARKET, PARQUET_DIR, MISTRAL_API_KEY, MISTRAL_BASE_URL, MISTRAL_MODEL, ROOT

OUT_PARQUET     = PARQUET_DIR / "market_categories.parquet"
CHECKPOINT_FILE = PARQUET_DIR / "market_categories_checkpoint.jsonl"

BATCH_SIZE   = 80    # markets per API call
MAX_RETRIES  = 3
RETRY_DELAY  = 5     # seconds between retries

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Canonical categories ──────────────────────────────────────────────────────
CANONICAL_CATEGORIES = [
    "Elections",
    "US Politics",
    "Geopolitics",
    "War & Conflict",
    "Macro / Economy",
    "Fed Policy",
    "Inflation",
    "Trade & Tariffs",
    "Energy & Commodities",
    "Crypto Prices",
    "Crypto Regulation",
    "Equities & Earnings",
    "Sports",
    "Entertainment & Showbiz",
    "Science & Tech",
    "Climate & Weather",
    "Law & Justice",
    "Health & Pharma",
    "AI & Technology",
    "Other",
]

SYSTEM_PROMPT = f"""You are a financial-market analyst classifying prediction markets into broad thematic categories.

Canonical categories (prefer these):
{chr(10).join(f'  - {c}' for c in CANONICAL_CATEGORIES)}

Rules:
1. Assign exactly ONE category per market.
2. If a market clearly belongs to a canonical category, use that exact string.
3. If no canonical category fits, invent a short, clear category name (e.g. "Immigration", "Space Exploration").
4. Base your classification on the question text and tags provided.
5. Return ONLY a valid JSON object mapping each market_id (string) to its category string.
   No markdown, no explanation, no extra keys.

Example output:
{{"12345": "Elections", "67890": "Crypto Prices", "11111": "Sports"}}"""


def build_user_prompt(batch: list[dict]) -> str:
    """Format a batch of markets into a compact prompt."""
    lines = []
    for m in batch:
        q   = m["question"][:120].replace("\n", " ")
        tags = ", ".join(m["tags"][:8]) if m["tags"] else "—"
        lines.append(f'[{m["market_id"]}] "{q}" | tags: {tags}')
    markets_text = "\n".join(lines)
    return (
        "Classify each market below into one category.\n"
        "Return JSON {market_id: category}.\n\n"
        + markets_text
    )


def call_llm(client: OpenAI, batch: list[dict]) -> dict[str, str]:
    """Call Mistral and parse the JSON response. Returns {market_id: category}."""
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
            # Strip potential markdown code fences
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            result: dict = json.loads(raw)
            # Ensure all market_ids are strings
            return {str(k): str(v) for k, v in result.items()}
        except json.JSONDecodeError as e:
            log.warning("JSON parse error on attempt %d: %s | raw=%s", attempt, e, raw[:200])
        except Exception as e:
            log.warning("API error on attempt %d: %s", attempt, e)
        if attempt < MAX_RETRIES:
            time.sleep(RETRY_DELAY * attempt)
    # No fallback: raise RuntimeError if Mistral does not respond/fails
    log.error("All retries failed for batch of %d markets", len(batch))
    raise RuntimeError(f"Mistral API failed for category labeling after {MAX_RETRIES} attempts.")


def load_checkpoint() -> dict[str, str]:
    """Load previously saved results from the checkpoint file."""
    if not CHECKPOINT_FILE.exists():
        return {}
    results = {}
    with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    results.update(json.loads(line))
                except json.JSONDecodeError:
                    pass
    log.info("Checkpoint loaded: %d markets already categorized.", len(results))
    return results


def save_checkpoint(batch_result: dict[str, str]) -> None:
    """Append a batch result to the checkpoint file (one JSON object per line)."""
    with open(CHECKPOINT_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(batch_result, ensure_ascii=False) + "\n")


def parse_tags(tags_json) -> list[str]:
    """Extract tag labels from tags_json (string or list)."""
    if tags_json is None:
        return []
    if isinstance(tags_json, str):
        try:
            tags_json = json.loads(tags_json)
        except Exception:
            return []
    if isinstance(tags_json, list):
        return [t.get("label", "") for t in tags_json if isinstance(t, dict)]
    return []


def run():
    log.info("=== LLM Market Categorization ===")
    log.info("Loading market data from %s …", PARQUET_MARKET)
    mkt = pd.read_parquet(PARQUET_MARKET, columns=["market_id", "question", "tags_json"])
    log.info("Total markets: %d", len(mkt))

    # Parse tags
    mkt["tags"] = mkt["tags_json"].apply(parse_tags)

    # Load checkpoint
    already_done: dict[str, str] = load_checkpoint()
    done_ids = set(already_done.keys())

    # Filter remaining markets
    remaining = mkt[~mkt["market_id"].astype(str).isin(done_ids)].copy()
    log.info("Remaining to classify: %d  (already done: %d)", len(remaining), len(done_ids))

    if remaining.empty:
        log.info("All markets already categorized. Proceeding to final export.")
    else:
        client = OpenAI(api_key=MISTRAL_API_KEY, base_url=MISTRAL_BASE_URL)

        # Build list of dicts for easy batching
        records = remaining[["market_id", "question", "tags"]].to_dict(orient="records")
        n_batches = (len(records) + BATCH_SIZE - 1) // BATCH_SIZE

        for batch_idx in range(n_batches):
            batch = records[batch_idx * BATCH_SIZE : (batch_idx + 1) * BATCH_SIZE]
            log.info(
                "Batch %d/%d  (%d markets, ids %s … %s)",
                batch_idx + 1, n_batches, len(batch),
                batch[0]["market_id"], batch[-1]["market_id"],
            )
            result = call_llm(client, batch)
            save_checkpoint(result)
            already_done.update(result)

            # Small courtesy delay to avoid rate-limiting
            if batch_idx < n_batches - 1:
                time.sleep(0.3)

    # ── Build final DataFrame ─────────────────────────────────────────────────
    log.info("Building final parquet …")
    cat_df = pd.DataFrame(
        list(already_done.items()),
        columns=["market_id", "llm_category"],
    )
    cat_df["market_id"] = cat_df["market_id"].astype(int)

    # Left-join onto all markets to keep order
    final = mkt[["market_id", "question", "tags_json"]].copy()
    final = final.merge(cat_df[["market_id", "llm_category"]], on="market_id", how="left")
    final["llm_category"] = final["llm_category"].fillna("Other")

    final.to_parquet(OUT_PARQUET, index=False)
    log.info("Saved → %s  (%d rows)", OUT_PARQUET, len(final))

    # ── Summary stats ─────────────────────────────────────────────────────────
    counts = final["llm_category"].value_counts()
    log.info("\n=== Category distribution ===\n%s", counts.to_string())

    # Save CSV summary too
    counts_df = counts.reset_index()
    counts_df.columns = ["category", "count"]
    csv_path = PARQUET_DIR / "market_categories_summary.csv"
    counts_df.to_csv(csv_path, index=False)
    log.info("Summary CSV saved → %s", csv_path)

    log.info("=== Done ===")
    return final


if __name__ == "__main__":
    run()
