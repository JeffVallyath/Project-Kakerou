"""Novelty / Repetition Discount — reduces marginal impact of repeated content.

Repeated content still signals persistence, but should not keep counting
as fresh independent evidence. This module computes a novelty factor
that discounts reliability on near-duplicate turns.

Design:
  - Word-set Jaccard similarity against recent same-speaker turns
  - Three tiers: exact duplicate, paraphrase overlap, substantially different
  - Escalation bonus: longer/stronger reformulations get less discount
  - Applies to signal reliability, not raw extraction values
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List

from btom_engine.config import (
    NOVELTY_EXACT_THRESHOLD,
    NOVELTY_PARAPHRASE_THRESHOLD,
    NOVELTY_EXACT_FLOOR,
    NOVELTY_PARAPHRASE_FLOOR,
    NOVELTY_ESCALATION_BONUS,
)


@dataclass
class NoveltyResult:
    """Diagnostic output for novelty computation."""

    novelty_factor: float = 1.0
    max_similarity: float = 0.0
    matched_turn: str = ""
    tier: str = "novel"          # "exact_duplicate", "paraphrase", "novel"
    escalation_bonus: float = 0.0


def _normalize_text(text: str) -> set[str]:
    """Lowercase, strip punctuation, split into word set."""
    cleaned = re.sub(r"[^\w\s]", "", text.lower())
    return set(cleaned.split())


def _jaccard(a: set[str], b: set[str]) -> float:
    """Jaccard similarity between two word sets."""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def compute_novelty(
    current_text: str,
    recent_texts: list[str],
) -> NoveltyResult:
    """Compute novelty factor for current text vs recent same-speaker turns.

    Returns a NoveltyResult with novelty_factor in [0.0, 1.0].
    1.0 = fully novel, no discount.
    Lower = more repetitive, stronger discount.
    """
    if not recent_texts:
        return NoveltyResult()

    current_words = _normalize_text(current_text)
    if not current_words:
        return NoveltyResult()

    max_sim = 0.0
    matched = ""

    for prev in recent_texts:
        prev_words = _normalize_text(prev)
        sim = _jaccard(current_words, prev_words)
        if sim > max_sim:
            max_sim = sim
            matched = prev

    # Determine tier and base novelty
    if max_sim >= NOVELTY_EXACT_THRESHOLD:
        tier = "exact_duplicate"
        base_novelty = NOVELTY_EXACT_FLOOR
    elif max_sim >= NOVELTY_PARAPHRASE_THRESHOLD:
        tier = "paraphrase"
        # Linear interpolation between paraphrase floor and 1.0
        t = (max_sim - NOVELTY_PARAPHRASE_THRESHOLD) / (NOVELTY_EXACT_THRESHOLD - NOVELTY_PARAPHRASE_THRESHOLD)
        base_novelty = NOVELTY_PARAPHRASE_FLOOR + (1.0 - NOVELTY_PARAPHRASE_FLOOR) * (1.0 - t)
    else:
        tier = "novel"
        base_novelty = 1.0

    # Escalation detection: if current turn is meaningfully longer and contains
    # new words not seen before, it may be an escalatory restatement
    escalation_bonus = 0.0
    if tier != "novel" and matched:
        matched_words = _normalize_text(matched)
        new_words = current_words - matched_words
        if len(current_words) > len(matched_words) and len(new_words) >= 2:
            # Scale bonus by how many genuinely new words were added
            bonus_scale = min(1.0, len(new_words) / 5.0)
            escalation_bonus = NOVELTY_ESCALATION_BONUS * bonus_scale

    novelty_factor = min(1.0, base_novelty + escalation_bonus)

    return NoveltyResult(
        novelty_factor=novelty_factor,
        max_similarity=max_sim,
        matched_turn=matched[:60] + ("..." if len(matched) > 60 else ""),
        tier=tier,
        escalation_bonus=escalation_bonus,
    )
