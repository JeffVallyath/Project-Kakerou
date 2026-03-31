"""LIWC Signal Extraction — psycholinguistic deception signals from text.

Pure Python, no LLM. Computes word-category rates from published
psycholinguistic research and returns a bluff delta based on tunable weights.

These signals catch STRATEGIC deception that speech acts miss:
- A fluent liar who gives a direct answer but uses abstract language
- Someone who over-explains with cognitive mechanism words
- A target who drops first-person pronouns to distance from their lie
- Overcompensation with certainty words ("I DEFINITELY was there")

Sources: Pennebaker et al. (2015), Newman et al. (2003), Tausczik & Pennebaker (2010)
"""

from __future__ import annotations

import re
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Import word lists from behavioral baseline
from btom_engine.osint.behavioral_baseline import (
    _COGNITIVE_WORDS, _EXCLUSIVE_WORDS, _CERTAINTY_WORDS,
    _TENTATIVE_WORDS, _CONCRETE_WORDS, _FILLER_WORDS,
    _SELF_REFERENCE,
)


@dataclass
class LiwcSignals:
    """Raw LIWC signal values from a text sample."""
    word_count: int = 0
    cognitive_rate: float = 0.0      # cognitive words / total words
    exclusive_rate: float = 0.0      # exclusive words / total words
    certainty_rate: float = 0.0      # certainty words / total words
    tentative_rate: float = 0.0      # tentative words / total words
    concrete_rate: float = 0.0       # concrete words / total words
    filler_rate: float = 0.0         # filler words / total words
    self_ref_rate: float = 0.0       # first-person pronouns / total words
    unique_word_ratio: float = 0.0   # vocabulary diversity

    # Computed composite signals
    bluff_delta: float = 0.0         # overall deception signal
    rationale: str = ""              # human-readable explanation


def extract_liwc_signals(text: str) -> LiwcSignals:
    """Extract LIWC-derived signals from a text sample.

    Returns rates (per-word frequencies) for each psycholinguistic category.
    Pure Python — no LLM, no external dependencies.
    """
    if not text or len(text.strip()) < 3:
        return LiwcSignals()

    words = re.findall(r'\b\w+\b', text.lower())
    n = len(words)
    if n == 0:
        return LiwcSignals()

    word_set = set(words)

    signals = LiwcSignals(word_count=n)
    signals.cognitive_rate = len(word_set & _COGNITIVE_WORDS) / n
    signals.exclusive_rate = len(word_set & _EXCLUSIVE_WORDS) / n
    signals.certainty_rate = len(word_set & _CERTAINTY_WORDS) / n
    signals.tentative_rate = len(word_set & _TENTATIVE_WORDS) / n
    signals.concrete_rate = len(word_set & _CONCRETE_WORDS) / n
    signals.self_ref_rate = sum(1 for w in words if w in _SELF_REFERENCE) / n
    signals.unique_word_ratio = len(word_set) / n

    # Filler uses phrase matching (multi-word patterns)
    text_lower = text.lower()
    filler_count = sum(1 for f in _FILLER_WORDS if f in text_lower)
    signals.filler_rate = filler_count / n

    return signals


def compute_liwc_bluff_delta(
    signals: LiwcSignals,
    weights: "EngineWeights | None" = None,
) -> float:
    """Compute bluff probability delta from LIWC signals.

    Each signal contributes proportionally to its weight.
    Positive weights increase suspicion, negative decrease it.

    Returns a float that should be added to the bluff probability.
    """
    try:
        from btom_engine.weights import WEIGHTS
        w = weights or WEIGHTS
    except Exception:
        return 0.0

    if signals.word_count < 3:
        return 0.0

    delta = 0.0
    rationale_parts = []

    # Cognitive load — high = rationalizing
    cog_contrib = signals.cognitive_rate * w.liwc_cognitive_weight
    if abs(cog_contrib) > 0.005:
        delta += cog_contrib
        rationale_parts.append(f"cognitive={signals.cognitive_rate:.2f}")

    # Exclusive words — low rate = cognitive overload (can't handle complex logic while lying)
    # Weight is negative: fewer exclusive words = more suspicious
    excl_contrib = signals.exclusive_rate * w.liwc_exclusive_weight
    if abs(excl_contrib) > 0.005:
        delta += excl_contrib
        rationale_parts.append(f"exclusive={signals.exclusive_rate:.2f}")

    # Certainty overcompensation — high = trying too hard to convince
    cert_contrib = signals.certainty_rate * w.liwc_certainty_weight
    if abs(cert_contrib) > 0.005:
        delta += cert_contrib
        rationale_parts.append(f"certainty={signals.certainty_rate:.2f}")

    # Tentative language — mild suspicion
    tent_contrib = signals.tentative_rate * w.liwc_tentative_weight
    if abs(tent_contrib) > 0.005:
        delta += tent_contrib
        rationale_parts.append(f"tentative={signals.tentative_rate:.2f}")

    # Concreteness — high = truthful (real episodic memory)
    # Weight is negative: more concrete = less suspicious
    conc_contrib = signals.concrete_rate * w.liwc_concrete_weight
    if abs(conc_contrib) > 0.005:
        delta += conc_contrib
        rationale_parts.append(f"concrete={signals.concrete_rate:.2f}")

    # Filler padding — high = buying time
    fill_contrib = signals.filler_rate * w.liwc_filler_weight
    if abs(fill_contrib) > 0.005:
        delta += fill_contrib
        rationale_parts.append(f"filler={signals.filler_rate:.2f}")

    # Self-reference drop — low = distancing from statement
    # Weight is negative: fewer self-refs = more suspicious
    self_contrib = signals.self_ref_rate * w.liwc_self_ref_weight
    if abs(self_contrib) > 0.005:
        delta += self_contrib
        rationale_parts.append(f"self_ref={signals.self_ref_rate:.2f}")

    # Scale by integration weight
    delta *= w.liwc_integration_weight

    signals.bluff_delta = delta
    signals.rationale = ", ".join(rationale_parts) if rationale_parts else "no significant signals"

    return delta
