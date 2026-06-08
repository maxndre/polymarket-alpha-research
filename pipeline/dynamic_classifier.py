"""
dynamic_classifier.py - Dynamically classifies and polarizes new markets using Mistral API at runtime,
keeping the cached classification tables updated without creating new categories.
"""

from __future__ import annotations
import json
import logging
import time
import asyncio
from pathlib import Path
import pandas as pd
import numpy as np
from openai import OpenAI, AsyncOpenAI
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

from pipeline.config import FunnelConfig, RESULTS_DIR, MISTRAL_API_KEY, MISTRAL_BASE_URL, MISTRAL_MODEL

logger = logging.getLogger(__name__)

# ── Clean Helper ─────────────────────────────────────────────────────────────
def clean_for_clustering(text: str) -> str:
    import re
    text = re.sub(r'(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2}(?:st|nd|rd|th)?', '', text, flags=re.IGNORECASE)
    text = re.sub(r'(?:January|February|March|April|May|June|July|August|September|October|November|December)', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\d{1,2}(?::\d{2})?\s*(?:AM|PM)\s*(?:ET|EST|EDT|UTC)?', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\$[\d,]+(?:\.\d+)?', '', text)
    text = re.sub(r'\d+(?:\.\d+)?%', '', text)
    text = re.sub(r'\b202[4-9]\b', '', text)
    return re.sub(r'\s+', ' ', text).strip()

# ── Part 1: Usability Funnel Semantic Filter ──────────────────────────────────
async def _process_funnel_batch(client: AsyncOpenAI, batch: list[tuple[int, str]], model: str) -> str:
    prompt = "Voici plusieurs modèles de questions de marchés de prédiction.\n"
    prompt += "Pour chacune, dis si c'est pertinent pour du trading (finance, macro, crypto) avec 'KEEP', sinon 'REJECT' (bruit, pop-culture, influenceurs).\n"
    prompt += "Donne aussi exactement 5 mots-clés qui la décrivent séparés par des virgules.\n"
    prompt += "Format strict attendu par ligne: ID: KEEP | mot1, mot2, mot3, mot4, mot5\n\n"
    for i, q in batch:
        prompt += f"{i}: {q}\n"

    try:
        resp = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )
        return resp.choices[0].message.content
    except Exception as e:
        logger.error(f"  Mistral API batch call error: {e}")
        raise RuntimeError(f"Mistral API call failed: {e}") from e

async def _run_funnel_batches(unique_qs: list[str]) -> list[str]:
    client = AsyncOpenAI(api_key=MISTRAL_API_KEY, base_url=MISTRAL_BASE_URL)
    batch_size = 20
    batches = []
    current_batch = []
    for i, q in enumerate(unique_qs):
        current_batch.append((i, q))
        if len(current_batch) == batch_size:
            batches.append(current_batch)
            current_batch = []
    if current_batch:
        batches.append(current_batch)

    sem = asyncio.Semaphore(15)
    async def sem_task(b):
        async with sem:
            return await _process_funnel_batch(client, b, MISTRAL_MODEL)

    logger.info("  Lancement de %d requêtes asynchrones à Mistral pour le filtre sémantique...", len(batches))
    tasks = [sem_task(b) for b in batches]
    return await asyncio.gather(*tasks)

def classify_new_markets_in_funnel(df_markets: pd.DataFrame, active_cids: set, cfg: FunnelConfig) -> set:
    out_path = RESULTS_DIR / "semantic_clusters.parquet"
    cached_df = None
    if out_path.exists():
        logger.info("  [Cache] Chargement de semantic_clusters.parquet existant...")
        try:
            cached_df = pd.read_parquet(out_path)
            # Ensure necessary columns exist
            for col in ["condition_id", "is_tradfi", "keywords"]:
                if col not in cached_df.columns:
                    cached_df = None
                    break
        except Exception as e:
            logger.warning(f"Failed to read semantic cache: {e}. Rebuilding...")
            cached_df = None

    mkt_sub = df_markets[df_markets["condition_id"].isin(active_cids)].copy()
    mkt_sub["clean_q"] = mkt_sub["question"].fillna("").apply(clean_for_clustering)

    if cached_df is not None:
        cached_cids = set(cached_df["condition_id"])
        missing_cids = active_cids - cached_cids

        if not missing_cids:
            logger.info("  Tous les marchés actifs sont déjà dans le cache sémantique.")
            tradfi_cids = set(cached_df.loc[cached_df["is_tradfi"], "condition_id"])
            return tradfi_cids.intersection(active_cids)

        logger.info("  %d nouveaux marchés détectés absents du cache sémantique. Classification en cours...", len(missing_cids))
        mkt_missing = mkt_sub[mkt_sub["condition_id"].isin(missing_cids)].copy()
        unique_qs = mkt_missing["clean_q"].unique()

        results = _run_funnel_batches_sync(unique_qs)

        results_dict = {}
        for res in results:
            if not res: continue
            for line in res.strip().split('\n'):
                if ':' in line and '|' in line:
                    try:
                        idx_str, rest = line.split(':', 1)
                        idx = int(idx_str.strip())
                        decision, kw_str = rest.split('|', 1)
                        is_keep = "KEEP" in decision.upper()
                        keywords = [k.strip() for k in kw_str.split(',')]
                        results_dict[idx] = {"keep": is_keep, "keywords": keywords}
                    except Exception:
                        pass

        q_to_res = {}
        for i, q in enumerate(unique_qs):
            if i in results_dict:
                q_to_res[q] = results_dict[i]
            else:
                raise RuntimeError(f"Mistral semantic filter classification failed: question index {i} was not parsed successfully.")

        mkt_missing["llm_res"] = mkt_missing["clean_q"].map(q_to_res)
        mkt_missing["is_tradfi"] = mkt_missing["llm_res"].apply(lambda x: x["keep"])
        mkt_missing["keywords"] = mkt_missing["llm_res"].apply(lambda x: " ".join(x["keywords"][:5]))

        # Combine old cache + new classifications
        new_rows = mkt_missing[["condition_id", "is_tradfi", "keywords"]].drop_duplicates(subset=["condition_id"])
        combined_df = pd.concat([
            cached_df[["condition_id", "is_tradfi", "keywords"]],
            new_rows
        ], ignore_index=True).drop_duplicates(subset=["condition_id"])

        logger.info("  Génération de l'espace sémantique 2D (TF-IDF + SVD) sur l'ensemble des mots-clés...")
        vectorizer = TfidfVectorizer(max_features=1000)
        X = vectorizer.fit_transform(combined_df["keywords"])
        svd = TruncatedSVD(n_components=2, random_state=42)
        X_reduced = svd.fit_transform(X)

        combined_df["svd_x"] = X_reduced[:, 0]
        combined_df["svd_y"] = X_reduced[:, 1]

        out_path.parent.mkdir(exist_ok=True, parents=True)
        combined_df.to_parquet(out_path, index=False)

        tradfi_cids = set(combined_df.loc[combined_df["is_tradfi"], "condition_id"])
        return tradfi_cids.intersection(active_cids)

    else:
        logger.info("  Extraction du texte pour le NLP (pas de cache existant)...")
        unique_qs = mkt_sub["clean_q"].unique()
        logger.info("  %d questions uniques identifiées à envoyer par batch...", len(unique_qs))

        results = _run_funnel_batches_sync(unique_qs)

        results_dict = {}
        for res in results:
            if not res: continue
            for line in res.strip().split('\n'):
                if ':' in line and '|' in line:
                    try:
                        idx_str, rest = line.split(':', 1)
                        idx = int(idx_str.strip())
                        decision, kw_str = rest.split('|', 1)
                        is_keep = "KEEP" in decision.upper()
                        keywords = [k.strip() for k in kw_str.split(',')]
                        results_dict[idx] = {"keep": is_keep, "keywords": keywords}
                    except Exception:
                        pass

        q_to_res = {}
        for i, q in enumerate(unique_qs):
            if i in results_dict:
                q_to_res[q] = results_dict[i]
            else:
                raise RuntimeError(f"Mistral semantic filter classification failed: question index {i} was not parsed successfully.")

        mkt_sub["llm_res"] = mkt_sub["clean_q"].map(q_to_res)
        mkt_sub["is_tradfi"] = mkt_sub["llm_res"].apply(lambda x: x["keep"])
        mkt_sub["keywords"] = mkt_sub["llm_res"].apply(lambda x: " ".join(x["keywords"][:5]))

        logger.info("  Génération de l'espace sémantique 2D (TF-IDF + SVD) sur les mots-clés générés...")
        vectorizer = TfidfVectorizer(max_features=1000)
        X = vectorizer.fit_transform(mkt_sub["keywords"])
        svd = TruncatedSVD(n_components=2, random_state=42)
        X_reduced = svd.fit_transform(X)

        mkt_sub["svd_x"] = X_reduced[:, 0]
        mkt_sub["svd_y"] = X_reduced[:, 1]

        out_path.parent.mkdir(exist_ok=True, parents=True)
        mkt_sub[["condition_id", "svd_x", "svd_y", "is_tradfi", "keywords"]].to_parquet(out_path, index=False)

        tradfi_cids = set(mkt_sub.loc[mkt_sub["is_tradfi"], "condition_id"])
        return tradfi_cids

def _run_funnel_batches_sync(unique_qs: list[str]) -> list[str]:
    # Run the async loop inside a sync context safely
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    if loop.is_running():
        # If in a jupyter notebook, nest_asyncio might be needed, or we run in executor.
        # But we can just use a separate thread or call future directly.
        # Let's import nest_asyncio just in case to be fully safe in notebooks!
        try:
            import nest_asyncio
            nest_asyncio.apply()
        except ImportError:
            pass
        return loop.run_until_complete(_run_funnel_batches(unique_qs))
    else:
        return asyncio.run(_run_funnel_batches(unique_qs))


# ── Part 2: Categorization & Polarity Mapping ─────────────────────────────────
CANONICAL_CATEGORIES = [
    'AI & Technology', 'Acquisitions', 'Crypto Prices', 'Crypto Regulation',
    'Elections', 'Energy & Commodities', 'Equities & Earnings', 'Fed Policy',
    'Geopolitics', 'Inflation', 'Macro / Economy', 'Science & Tech',
    'Trade & Tariffs', 'US Politics', 'War & Conflict', 'Other'
]

SYSTEM_PROMPT_CAT = f"""You are a financial-market analyst classifying prediction markets into broad thematic categories.

Allowed categories (prefer these, select exactly one):
{chr(10).join(f'  - {c}' for c in CANONICAL_CATEGORIES)}

Rules:
1. Assign exactly ONE category per market from the allowed categories list above. Do NOT invent new categories.
2. Return ONLY a valid JSON object mapping each market_id (string) to its category string.
   No markdown, no explanation, no extra keys.

Example output:
{{"12345": "Elections", "67890": "Crypto Prices", "11111": "Other"}}"""

def _call_mistral_category_batch(client: OpenAI, batch: list[dict]) -> dict[str, str]:
    # Build user prompt
    lines = []
    for m in batch:
        q = m["question"][:120].replace("\n", " ")
        lines.append(f'[{m["market_id"]}] "{q}"')
    user_prompt = "Classify each market below into one category.\nReturn JSON {market_id: category}.\n\n" + "\n".join(lines)

    for attempt in range(1, 4):
        try:
            response = client.chat.completions.create(
                model=MISTRAL_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT_CAT},
                    {"role": "user",   "content": user_prompt},
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
            return {str(k): str(v) for k, v in result.items()}
        except Exception as e:
            logger.warning(f"Mistral category classification attempt {attempt} failed: {e}")
            if attempt < 3:
                time.sleep(1)
    raise RuntimeError(f"Mistral category classification failed after 3 attempts.")

SYSTEM_PROMPT_POLARITY = """You are a financial analyst classifying the sentiment polarity of prediction markets.
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

def _call_mistral_polarity_batch(client: OpenAI, batch: list[dict]) -> dict[str, int]:
    # Build user prompt
    lines = []
    for m in batch:
        q = m["question"][:120].replace("\n", " ")
        lines.append(f'[{m["market_id"]}] "{q}" (category: {m["llm_category"]})')
    user_prompt = "Classify each market below into polarity: 1, -1, or 0.\n\n" + "\n".join(lines)

    for attempt in range(1, 4):
        try:
            response = client.chat.completions.create(
                model=MISTRAL_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT_POLARITY},
                    {"role": "user",   "content": user_prompt},
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
            logger.warning(f"Mistral polarity classification attempt {attempt} failed: {e}")
            if attempt < 3:
                time.sleep(1)
    raise RuntimeError(f"Mistral polarity classification failed after 3 attempts.")

def classify_and_polarize_new_markets(
    df_usable_markets: pd.DataFrame,
    df_categories: pd.DataFrame,
    df_polarities: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Identifies any market_id in df_usable_markets that is missing from df_categories or df_polarities.
    Uses Mistral to classify them into the canonical categories and polarities, updates the parquet files on disk,
    and returns the fully updated df_categories and df_polarities.
    """
    categories_path = RESULTS_DIR / "market_categories.parquet"
    polarities_path = RESULTS_DIR / "market_polarities.parquet"

    # Harmonize IDs to strings/ints
    df_usable_markets = df_usable_markets.copy()
    df_usable_markets["market_id"] = df_usable_markets["market_id"].astype(int)
    
    df_categories = df_categories.copy()
    df_categories["market_id"] = df_categories["market_id"].astype(int)
    
    df_polarities = df_polarities.copy()
    df_polarities["market_id"] = df_polarities["market_id"].astype(int)

    active_mids = set(df_usable_markets["market_id"])
    cached_cat_mids = set(df_categories["market_id"])
    cached_pol_mids = set(df_polarities["market_id"])

    missing_cat_mids = active_mids - cached_cat_mids
    missing_pol_mids = active_mids - cached_pol_mids

    client = None

    # Step 1: Classify Categories for missing markets
    if missing_cat_mids:
        logger.info("  %d nouveaux marchés sans catégorie détectés. Classification via Mistral...", len(missing_cat_mids))
        df_missing = df_usable_markets[df_usable_markets["market_id"].isin(missing_cat_mids)].copy()
        records = df_missing[["market_id", "question"]].to_dict(orient="records")
        
        client = OpenAI(api_key=MISTRAL_API_KEY, base_url=MISTRAL_BASE_URL)
        
        batch_size = 80
        new_cats = {}
        for idx in range(0, len(records), batch_size):
            batch = records[idx : idx + batch_size]
            result = _call_mistral_category_batch(client, batch)
            new_cats.update(result)
            
        # Add to df_categories
        new_cat_rows = []
        for mid in missing_cat_mids:
            cat = new_cats.get(str(mid), "Other")
            if cat not in CANONICAL_CATEGORIES:
                cat = "Other"
            new_cat_rows.append({"market_id": int(mid), "llm_category": cat})
        
        df_new_cats = pd.DataFrame(new_cat_rows)
        df_categories = pd.concat([df_categories, df_new_cats], ignore_index=True).drop_duplicates(subset=["market_id"])
        df_categories.to_parquet(categories_path, index=False)
        logger.info("  Cache des catégories mis à jour avec %d nouveaux marchés.", len(missing_cat_mids))

    # Step 2: Classify Polarities for missing markets
    if missing_pol_mids:
        logger.info("  %d nouveaux marchés sans polarité détectés. Labellisation via Mistral...", len(missing_pol_mids))
        # Ensure we have categories for them in df_categories
        df_missing = df_usable_markets[df_usable_markets["market_id"].isin(missing_pol_mids)].copy()
        df_missing = df_missing.merge(df_categories[["market_id", "llm_category"]], on="market_id", how="left")
        df_missing["llm_category"] = df_missing["llm_category"].fillna("Other")
        
        # Apply heuristics first
        # Crypto Prices -> 1
        # Equities & Earnings -> 1
        new_pols = {}
        remaining_records = []
        
        for rec in df_missing[["market_id", "question", "llm_category"]].to_dict(orient="records"):
            mid = rec["market_id"]
            cat = rec["llm_category"]
            if cat == "Crypto Prices" or cat == "Equities & Earnings":
                new_pols[str(mid)] = 1
            else:
                remaining_records.append(rec)
                
        if remaining_records:
            if client is None:
                client = OpenAI(api_key=MISTRAL_API_KEY, base_url=MISTRAL_BASE_URL)
            batch_size = 100
            for idx in range(0, len(remaining_records), batch_size):
                batch = remaining_records[idx : idx + batch_size]
                result = _call_mistral_polarity_batch(client, batch)
                new_pols.update(result)
                
        # Add to df_polarities
        new_pol_rows = []
        for mid in missing_pol_mids:
            pol = new_pols.get(str(mid), 0)
            new_pol_rows.append({"market_id": int(mid), "polarity": int(pol)})
            
        df_new_pols = pd.DataFrame(new_pol_rows)
        df_polarities = pd.concat([df_polarities, df_new_pols], ignore_index=True).drop_duplicates(subset=["market_id"])
        df_polarities.to_parquet(polarities_path, index=False)
        logger.info("  Cache des polarités mis à jour avec %d nouveaux marchés.", len(missing_pol_mids))

    return df_categories, df_polarities
