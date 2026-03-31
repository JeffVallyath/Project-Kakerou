"""Prior Integration — maps comparison outcomes to bounded hypothesis adjustments.

This does NOT modify the math engine. It applies small direct adjustments
to hypothesis probabilities after the math update, as a parallel evidence channel.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from btom_engine.osint.evidence_schema import ComparisonResult


@dataclass
class PriorContextEffect:
    """The bounded effect of prior-context evidence on hypotheses."""

    bluffing_delta: float = 0.0
    withholding_delta: float = 0.0
    rationale: str = ""
    comparisons_used: int = 0


# Outcome -> (bluffing_delta, withholding_delta) per comparison
_OUTCOME_EFFECTS = {
    "direct_contradiction":   (+0.08, +0.03),   # strong support for bluffing
    "supported_by_prior":     (-0.04, -0.02),   # weak de-escalation
    "weak_tension":           (+0.02, +0.01),   # mild suspicion
    "insufficient_evidence":  (0.0, 0.0),        # no effect
}

# Maximum total adjustment from all comparisons combined
_MAX_PRIOR_DELTA = 0.12


def compute_prior_effect(comparisons: list[ComparisonResult]) -> PriorContextEffect:
    """Map comparison outcomes to bounded hypothesis adjustments.

    Aggregates across all comparisons, capped at ±MAX_PRIOR_DELTA.
    Scales by comparison confidence.
    """
    if not comparisons:
        return PriorContextEffect()

    total_bluff = 0.0
    total_withhold = 0.0
    rationale_parts = []
    used = 0

    for comp in comparisons:
        if comp.outcome == "insufficient_evidence":
            continue

        base_bluff, base_withhold = _OUTCOME_EFFECTS.get(comp.outcome, (0.0, 0.0))

        # Scale by comparison confidence
        conf = max(0.0, min(1.0, comp.comparison_confidence))
        total_bluff += base_bluff * conf
        total_withhold += base_withhold * conf
        used += 1

        rationale_parts.append(f"{comp.outcome}(conf={conf:.2f})")

    # Clamp
    total_bluff = max(-_MAX_PRIOR_DELTA, min(_MAX_PRIOR_DELTA, total_bluff))
    total_withhold = max(-_MAX_PRIOR_DELTA, min(_MAX_PRIOR_DELTA, total_withhold))

    rationale = "; ".join(rationale_parts) if rationale_parts else "no actionable comparisons"

    return PriorContextEffect(
        bluffing_delta=total_bluff,
        withholding_delta=total_withhold,
        rationale=rationale,
        comparisons_used=used,
    )


def apply_prior_effect(
    hypotheses: dict,
    effect: PriorContextEffect,
) -> None:
    """Apply the bounded prior-context effect to hypothesis probabilities.

    Mutates hypotheses in place. Clamps to [0, 1].
    """
    if abs(effect.bluffing_delta) < 0.001 and abs(effect.withholding_delta) < 0.001:
        return

    bluff = hypotheses.get("target_is_bluffing")
    if bluff:
        bluff.probability = max(0.0, min(1.0, bluff.probability + effect.bluffing_delta))

    withhold = hypotheses.get("target_is_withholding_info")
    if withhold:
        withhold.probability = max(0.0, min(1.0, withhold.probability + effect.withholding_delta))
