"""
step2_funnel.py – ÉTAPE 2 : L'Entonnoir de Sélection (Usability Funnel).

7 filtres séquentiels avec log d'attrition à chaque étape.
Tous les filtrages sont vectorisés (pas d'iterrows/itertuples).

Ordre des filtres
-----------------
1. Résolution inférée
2. Validité temporelle
3. Type de pari standardisé (Yes/No ou Up/Down uniquement)
4. Filtre sémantique TradFi (buckets macro/finance)
5. Activité minimale (≥ 1 000 trades)
6. Volume / Liquidité (≥ 10 000 USDC)
7. Continuité des prix (gap ≤ 24h)

Note résolution :
  Pas de colonne resolved dans la base (masquée train/test).
  On infère depuis le dernier close_price du token YES/Up.
"""

from __future__ import annotations
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd

import re

import asyncio
from openai import OpenAI, AsyncOpenAI
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

def clean_for_clustering(text: str) -> str:
    text = re.sub(r'(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2}(?:st|nd|rd|th)?', '', text, flags=re.IGNORECASE)
    text = re.sub(r'(?:January|February|March|April|May|June|July|August|September|October|November|December)', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\d{1,2}(?::\d{2})?\s*(?:AM|PM)\s*(?:ET|EST|EDT|UTC)?', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\$[\d,]+(?:\.\d+)?', '', text)
    text = re.sub(r'\d+(?:\.\d+)?%', '', text)
    text = re.sub(r'\b202[4-9]\b', '', text)
    return re.sub(r'\s+', ' ', text).strip()

from .config import FunnelConfig, RESULTS_DIR

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Attrition logger
# ─────────────────────────────────────────────────────────────────────────────

def _log_step(step: int, label: str, n_before: int, n_after: int) -> dict:
    removed = n_before - n_after
    pct_removed  = removed / n_before * 100 if n_before > 0 else 0.0
    pct_remaining= n_after  / n_before * 100 if n_before > 0 else 0.0
    logger.info(
        "  [FILTRE %d – %s]  %d → %d  (−%d, −%.1f%%  |  %.1f%% restants)",
        step, label, n_before, n_after, removed, pct_removed, pct_remaining,
    )
    return {
        "step": step, "label": label,
        "n_before": n_before, "n_after": n_after,
        "n_removed": removed,
        "pct_removed":   round(pct_removed,  2),
        "pct_remaining": round(pct_remaining, 2),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Resolution inference
# ─────────────────────────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────────────────────────
# Bet-type filter helper
# ─────────────────────────────────────────────────────────────────────────────

def _standard_binary_cids(df_token: pd.DataFrame, cfg: FunnelConfig) -> set:
    """
    Keep only condition_ids whose outcome set (lowercased) exactly matches
    one of the accepted pairs: {yes, no} or {up, down}.
    """
    # Build per-market set of outcome labels (lowercase, stripped)
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


# ─────────────────────────────────────────────────────────────────────────────
# Semantic NLP filter helper
# ─────────────────────────────────────────────────────────────────────────────

async def _process_batch(client, batch, cfg):
    prompt = "Voici plusieurs modèles de questions de marchés de prédiction.\n"
    prompt += "Pour chacune, dis si c'est pertinent pour du trading (finance, macro, crypto) avec 'KEEP', sinon 'REJECT' (bruit, pop-culture, influenceurs).\n"
    prompt += "Donne aussi exactement 5 mots-clés qui la décrivent séparés par des virgules.\n"
    prompt += "Format strict attendu par ligne: ID: KEEP | mot1, mot2, mot3, mot4, mot5\n\n"
    for i, q in batch:
        prompt += f"{i}: {q}\n"

    try:
        resp = await client.chat.completions.create(
            model=cfg.MISTRAL_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )
        return resp.choices[0].message.content
    except Exception as e:
        logger.error(f"  Erreur API Mistral pour un batch : {e}")
        raise RuntimeError(f"Mistral API call failed: {e}") from e

async def _run_all_batches(unique_qs, cfg):
    client = AsyncOpenAI(api_key=cfg.MISTRAL_API_KEY, base_url=cfg.MISTRAL_BASE_URL)
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
            return await _process_batch(client, b, cfg)
            
    logger.info("  Lancement de %d requêtes asynchrones à Mistral...", len(batches))
    tasks = [sem_task(b) for b in batches]
    return await asyncio.gather(*tasks)

def _semantic_clustering_filter(df_markets: pd.DataFrame, active_cids: set, cfg: FunnelConfig) -> set:
    """
    Filtre sémantique 100% IA :
    Traite toutes les questions de marché uniques via Mistral.
    """
    from pipeline.dynamic_classifier import classify_new_markets_in_funnel
    return classify_new_markets_in_funnel(df_markets, active_cids, cfg)


# ─────────────────────────────────────────────────────────────────────────────
# Main funnel
# ─────────────────────────────────────────────────────────────────────────────

def apply_usability_funnel(
    df_markets:   pd.DataFrame,
    df_bars:      pd.DataFrame,
    df_token:     pd.DataFrame,
    df_filter_tag:pd.DataFrame,   # selected_filter_tag
    df_market_tag:pd.DataFrame,   # market_tag join table
    per_market:   pd.DataFrame,   # pre-computed in step1
    cfg:          FunnelConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    """
    ÉTAPE 2 – Entonnoir de sélection (7 filtres).

    Returns
    -------
    df_usable_markets, df_usable_bars, attrition_log
    """
    if cfg is None:
        cfg = FunnelConfig()

    logger.info("=== ÉTAPE 2 : ENTONNOIR DE SÉLECTION ===")
    attrition: list[dict] = []

    active_cids = set(df_markets["condition_id"].unique())
    n0 = len(active_cids)
    logger.info("Début : %d marchés uniques", n0)

    # ── FILTRE 1 : Type de pari standardisé ─────────────────────────────────
    logger.info("Filtre 1 : Type de pari standardisé (Yes/No ou Up/Down) …")
    standard_cids = _standard_binary_cids(df_token, cfg)
    active_cids  &= standard_cids
    attrition.append(_log_step(1, "Paris binaires standard (Yes/No ou Up/Down)", n0, len(active_cids)))

    # ── FILTRE 2 : Résolution ────────────────────────────────────────────────
    logger.info("Filtre 2 : Résolution inférée …")
    n_before = len(active_cids)
    resolution    = _infer_resolution(df_bars, df_token, cfg)
    resolved_cids = set(resolution.loc[resolution["is_resolved"], "condition_id"])
    active_cids  &= resolved_cids
    attrition.append(_log_step(2, "Résolution inférée (last price → 0 ou 1)", n_before, len(active_cids)))

    # ── FILTRE 3 : Validité temporelle ───────────────────────────────────────
    logger.info("Filtre 3 : Validité temporelle …")
    n_before = len(active_cids)
    mkt_sub  = df_markets[df_markets["condition_id"].isin(active_cids)].copy()
    duration_h = (mkt_sub["end_date"] - mkt_sub["start_date"]).dt.total_seconds() / 3600.0
    valid_t  = (duration_h >= cfg.MIN_DURATION_HOURS) & \
               (duration_h <= cfg.MAX_DURATION_DAYS * 24) & \
               duration_h.notna()
    active_cids &= set(mkt_sub.loc[valid_t, "condition_id"])
    attrition.append(_log_step(3, "Validité temporelle (durée ∈ [1h, 2ans])", n_before, len(active_cids)))

    # ── FILTRE 4 : Volume / Liquidité ────────────────────────────────────────
    logger.info("Filtre 4 : Volume USDC >= %.0f …", cfg.MIN_VOLUME_USDC)
    n_before = len(active_cids)
    pm_sub   = per_market[per_market["condition_id"].isin(active_cids)]
    active_cids &= set(pm_sub.loc[pm_sub["total_volume_usdc"] >= cfg.MIN_VOLUME_USDC, "condition_id"])
    attrition.append(_log_step(4, f"Volume >= {cfg.MIN_VOLUME_USDC:,.0f} USDC", n_before, len(active_cids)))

    # ── FILTRE 5 : Activité minimale ─────────────────────────────────────────
    logger.info("Filtre 5 : Activité minimale (min_trades=%d) …", cfg.MIN_TRADES)
    n_before = len(active_cids)
    pm_sub   = per_market[per_market["condition_id"].isin(active_cids)]
    active_cids &= set(pm_sub.loc[pm_sub["total_trades"] >= cfg.MIN_TRADES, "condition_id"])
    attrition.append(_log_step(5, f"Activité >= {cfg.MIN_TRADES:,} trades", n_before, len(active_cids)))

    # ── FILTRE 6 : Continuité des prix ───────────────────────────────────────
    logger.info("Filtre 6 : Continuité des prix (max gap = %.0fh) …", cfg.MAX_GAP_HOURS)
    n_before  = len(active_cids)
    bars_sub  = df_bars[df_bars["condition_id"].isin(active_cids)].copy()
    bars_sub  = bars_sub[bars_sub["trades_count_1m"] > 0]

    if len(bars_sub) > 0:
        bars_sub  = bars_sub.sort_values(["condition_id", "minute_ts"])
        bars_sub["prev_ts"] = bars_sub.groupby("condition_id", observed=True)["minute_ts"].shift(1)
        bars_sub["gap_h"]   = (bars_sub["minute_ts"] - bars_sub["prev_ts"]).dt.total_seconds() / 3600.0

        max_gap = (
            bars_sub.groupby("condition_id", observed=True)["gap_h"]
            .max().reset_index().rename(columns={"gap_h": "max_gap_h"})
        )
        n_active_bars = (
            bars_sub.groupby("condition_id", observed=True).size()
            .reset_index(name="n_active_bars_post")
        )
        cont_df = max_gap.merge(n_active_bars, on="condition_id")
        continuous = cont_df[
            (cont_df["max_gap_h"]         <= cfg.MAX_GAP_HOURS) &
            (cont_df["n_active_bars_post"] >= cfg.MIN_ACTIVE_BARS)
        ]
        active_cids &= set(continuous["condition_id"])
    else:
        active_cids = set()

    attrition.append(_log_step(6, f"Continuité <= {cfg.MAX_GAP_HOURS:.0f}h gap", n_before, len(active_cids)))

    # ── FILTRE 7 : Filtre sémantique TradFi ─────────────────────────────────
    logger.info("Filtre 7 : Filtre sémantique NLP Avancé (K-Means/HDBSCAN + Mistral) …")
    n_before = len(active_cids)
    tradfi_cids = _semantic_clustering_filter(df_markets, active_cids, cfg)
    active_cids &= tradfi_cids
    attrition.append(_log_step(7, "Filtre sémantique NLP (Clustering + Mistral)", n_before, len(active_cids)))

    # ── Final ────────────────────────────────────────────────────────────────
    n_final = len(active_cids)
    logger.info(
        "═══ RÉSULTAT FINAL : %d marchés exploitables (%.1f%% du total) ═══",
        n_final, n_final / n0 * 100
    )

    df_usable_markets = df_markets[df_markets["condition_id"].isin(active_cids)].copy()
    df_usable_bars    = df_bars   [df_bars   ["condition_id"].isin(active_cids)].copy()

    df_usable_markets = df_usable_markets.merge(
        resolution[["condition_id", "last_close_yes", "resolved_yes", "resolved_no"]],
        on="condition_id", how="left"
    )
    return df_usable_markets, df_usable_bars, attrition
