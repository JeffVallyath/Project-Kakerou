"""Interaction Context — user-pressure estimation and contextual discounting.

Computes recent user-side pressure from the last few user turns using
grouped phrase-family heuristics. This is NOT an LLM pass — user turns are
context, not evidence, so a bounded heuristic is appropriate.

The pressure estimate is used to contextually discount ambiguous
target-side signals (e.g., defensiveness that may be provoked rather
than deceptive).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Tuple

from btom_engine.schema import ExtractedSignals


# ---------------------------------------------------------------------------
# Pressure representation
# ---------------------------------------------------------------------------

@dataclass
class PressureDebug:
    """Detailed trace of how pressure was computed. For UI diagnostics."""

    per_turn_hits: list[dict] = field(default_factory=list)
    # Each entry: {"text": str, "accusation_hits": [...], "repetition_hits": [...], "hostility_hits": [...]}
    raw_scores: dict[str, float] = field(default_factory=dict)
    # {"accusation": float, "repetition": float, "hostility": float}
    discount_factors: dict[str, float] = field(default_factory=dict)
    # {"emotional_intensity": float, "defensive_justification": float, ...}


@dataclass
class UserPressure:
    """Compact representation of recent user-side interaction pressure."""

    accusation: float = 0.0
    repetition: float = 0.0
    hostility: float = 0.0
    debug: PressureDebug = field(default_factory=PressureDebug)

    @property
    def aggregate(self) -> float:
        """Weighted aggregate. Accusation and hostility matter most."""
        return min(1.0, 0.40 * self.accusation + 0.25 * self.repetition + 0.35 * self.hostility)


# ---------------------------------------------------------------------------
# Phrase families — grouped by semantic meaning, not random keywords.
# Each family: (label, weight, list_of_patterns)
#   - label: human-readable name for debug
#   - weight: how much a single match from this family contributes (0-1)
#   - patterns: regex patterns (case-insensitive)
# ---------------------------------------------------------------------------

_ACCUSATION_FAMILIES: list[tuple[str, float, list[str]]] = [
    ("lying_accusation", 0.6, [
        r"\byou('re| are) lying\b",
        r"\byou lied\b",
        r"\byou('re| are) (a )?liar\b",
        r"\bstop lying\b",
        r"\bdon'?t lie\b",
    ]),
    ("dodging_accusation", 0.55, [
        r"\bstop (dodging|avoiding|evading|deflecting)\b",
        r"\byou('re| are) (dodging|avoiding|evading|sidestepping)\b",
        r"\bstop (pretending|playing dumb)\b",
        r"\bdo(n'?t| not) play dumb\b",
        r"\bwriggle out\b",
        r"\byou('re| are) avoiding the question\b",
        r"\bwhy are you still (dodging|avoiding)\b",
    ]),
    ("admission_demand", 0.5, [
        r"\badmit (it|that|what)\b",
        r"\byou (could have|should have) (just )?admitted\b",
        r"\bjust (admit|confess|own up)\b",
        r"\bconfess\b",
        r"\bown up to\b",
    ]),
    ("blame_framing", 0.45, [
        r"\byou (always|never)\b",
        r"\bhow could you\b",
        r"\bwhy did(n'?t)? you\b",
        r"\bexplain yourself\b",
        r"\bthe truth\b",
        r"\bbe honest\b",
        r"\btell me the truth\b",
        r"\b(your fault|on you|because of you)\b",
        r"\byou('re| are) the (issue|problem|reason)\b",
        r"\byou did this\b",
        r"\byou (ruin|ruined|destroy|destroyed)\b",
    ]),
    ("compliance_demand", 0.4, [
        r"\banswer (clearly|directly|me|the question|for once)\b",
        r"\bgive me a (straight|direct|clear) answer\b",
        r"\bjust answer\b",
        r"\bstop (stalling|wasting)\b",
        r"\bwast(e|ing) my time\b",
    ]),
]

_HOSTILITY_FAMILIES: list[tuple[str, float, list[str]]] = [
    ("slur_identity_attack", 0.70, [
        # Explicit slurs and identity-targeted demeaning language.
        # Distinct from generic profanity — these are unambiguously hostile.
        r"\b(nigger|nigga|nig|kike|spic|wetback|chink|gook|raghead|towelhead)\b",
        r"\b(faggot|fag|dyke|tranny|homo)\b",
        r"\b(retard|retarded|tard)\b",
        r"\b(cripple|freak|mongol|subhuman|vermin|animal)\b",
        r"\bkill yourself\b",
        r"\bkys\b",
        r"\bgo die\b",
    ]),
    ("profanity", 0.55, [
        r"\b(fuck|shit|damn|bullshit|ass|asshole)\b",
        r"\bscrew you\b",
        r"\bgo to hell\b",
    ]),
    ("contempt_insult", 0.5, [
        r"\b(pathetic|disgusting|worthless|useless|coward|spineless)\b",
        r"\byou('re| are) (pathetic|disgusting|worthless|useless)\b",
        r"\bgrow up\b",
        r"\bwhat('s| is) wrong with you\b",
    ]),
    ("aggressive_command", 0.5, [
        # Cross-category: these are accusatory AND hostile in tone
        r"\bdo(n'?t| not) play dumb\b",
        r"\bstop (pretending|dodging|lying)\b",
        r"\bwast(e|ing) my time\b",
        r"\banswer .* for once\b",
        r"\bhow dare you\b",
    ]),
    ("aggressive_tone", 0.45, [
        r"\b(shut up|get out|go away|leave me alone)\b",
        r"\b(sick of|tired of|done with|fed up)\b",
        r"\bi('m| am) done (with|talking)\b",
        r"\bdon'?t (test|push) me\b",
        r"[!]{2,}",
        r"[?]{3,}",
    ]),
    ("contemptuous_blame", 0.5, [
        # Identity-level attacks: "you ARE the problem" not "you DID something"
        r"\byou('re| are) the (issue|problem|reason|cause)\b",
        r"\byou('re| are) (the )?villain\b",
        r"\bthis is (your fault|on you|because of you)\b",
        r"\byou (make|made) me (the |a )?(villain|bad guy|problem)\b",
        r"\byou (ruin|ruined|destroy|destroyed) (everything|this|it)\b",
        r"\beverything is (always )?your fault\b",
        r"\byou did this\b",
    ]),
    ("adversarial_universals", 0.4, [
        # Universal blame patterns — hostile when combined with "you always/never"
        r"\byou always (do|make|ruin|start|cause|mess)\b",
        r"\byou never (listen|care|change|admit|take)\b",
        r"\bevery (single )?time.+you\b",
    ]),
    ("belittling", 0.4, [
        r"\byou('re| are) (ridiculous|unbelievable|impossible)\b",
        r"\bunbelievable\b",
        r"\bseriously\?\b",
        r"\bshame on you\b",
        r"\bwriggle out\b",
    ]),
]

_REPETITION_FAMILIES: list[tuple[str, float, list[str]]] = [
    ("explicit_repeat", 0.5, [
        r"\b(i('ll| will))? ask (you )?(again|one more time)\b",
        r"\bfor the last time\b",
        r"\balready asked\b",
        r"\bi just asked\b",
        r"\bhow many times\b",
    ]),
    ("narrowing_demand", 0.45, [
        r"\bthat'?s not what i asked\b",
        r"\byou (still )?haven'?t answered\b",
        r"\byou didn'?t answer\b",
        r"\bstop changing the (subject|topic)\b",
        r"\bback to (the|my) question\b",
        r"\bstay on topic\b",
    ]),
    ("persistence_marker", 0.35, [
        r"\b(again|once more|one more time)\b",
        r"\bi('m| am) (still )?waiting\b",
        r"\bwell\?\b",
        r"\bso\?\b",
    ]),
]


def _score_families(
    text: str,
    families: list[tuple[str, float, list[str]]],
) -> tuple[float, list[str]]:
    """Score text against phrase families. Returns (score, list_of_hit_labels).

    Uses max-per-family scoring: each family contributes its weight at most once,
    even if multiple patterns within it match. This prevents keyword-pile inflation.
    Final score saturates toward 1.0 via diminishing returns.
    """
    text_lower = text.lower()
    total = 0.0
    hits: list[str] = []

    for label, weight, patterns in families:
        for p in patterns:
            if re.search(p, text_lower):
                total += weight
                hits.append(label)
                break  # max one hit per family

    # Diminishing returns with steeper initial ramp:
    # 0.4 -> 0.43, 0.55 -> 0.52, 0.8 -> 0.62, 1.0 -> 0.67, 1.5 -> 0.78, 2.0 -> 0.84
    if total <= 0:
        return 0.0, hits
    score = min(1.0, 1.0 - 1.0 / (1.0 + 1.5 * total))
    return score, hits


def compute_pressure(recent_user_texts: list[str]) -> UserPressure:
    """Estimate user pressure from the last few user turns.

    Recency-weighted: most recent turn counts most.
    Expects texts in chronological order (oldest first).
    """
    if not recent_user_texts:
        return UserPressure()

    # Recency weights: most recent = 1.0, previous = 0.6, before that = 0.3
    weights = [0.3, 0.6, 1.0]
    texts = recent_user_texts[-3:]
    aligned_weights = weights[-len(texts):]

    acc_total = 0.0
    rep_total = 0.0
    hos_total = 0.0
    weight_sum = sum(aligned_weights)

    per_turn_hits: list[dict] = []

    for text, w in zip(texts, aligned_weights):
        acc_score, acc_hits = _score_families(text, _ACCUSATION_FAMILIES)
        rep_score, rep_hits = _score_families(text, _REPETITION_FAMILIES)
        hos_score, hos_hits = _score_families(text, _HOSTILITY_FAMILIES)

        acc_total += acc_score * w
        rep_total += rep_score * w
        hos_total += hos_score * w

        per_turn_hits.append({
            "text": text[:80] + ("..." if len(text) > 80 else ""),
            "weight": w,
            "accusation_hits": acc_hits,
            "repetition_hits": rep_hits,
            "hostility_hits": hos_hits,
            "accusation_score": round(acc_score, 3),
            "repetition_score": round(rep_score, 3),
            "hostility_score": round(hos_score, 3),
        })

    acc_final = min(1.0, acc_total / weight_sum)
    rep_final = min(1.0, rep_total / weight_sum)
    hos_final = min(1.0, hos_total / weight_sum)

    debug = PressureDebug(
        per_turn_hits=per_turn_hits,
        raw_scores={
            "accusation": round(acc_final, 4),
            "repetition": round(rep_final, 4),
            "hostility": round(hos_final, 4),
        },
    )

    return UserPressure(
        accusation=acc_final,
        repetition=rep_final,
        hostility=hos_final,
        debug=debug,
    )


# ---------------------------------------------------------------------------
# Contextual discounting
# ---------------------------------------------------------------------------

# Which target signals are ambiguous under user pressure, and how much to discount.
# Format: {signal_name: [(pressure_component, max_discount), ...]}
# Multiple pressure sources can compound (additively, capped).
_DISCOUNT_MAP: dict[str, list[tuple[str, float]]] = {
    "emotional_intensity": [
        ("hostility", 0.50),       # user hostility strongly explains target emotion
        ("accusation", 0.25),      # accusation also provokes emotion
    ],
    "defensive_justification": [
        ("accusation", 0.50),      # user accusation strongly provokes defensiveness
        ("repetition", 0.20),      # repeated questioning provokes justification
    ],
    "evasive_deflection": [
        ("accusation", 0.30),      # accused targets may deflect legitimately
        ("hostility", 0.20),       # hostile environment provokes shutdown
    ],
}


def apply_contextual_discounting(
    signals: ExtractedSignals,
    pressure: UserPressure,
) -> ExtractedSignals:
    """Reduce signal reliability when user pressure explains the target's behavior.

    This does NOT zero out signals. It reduces reliability proportionally,
    so strong signals still contribute while weak/ambiguous ones are dampened.
    Returns the mutated signals and populates pressure.debug.discount_factors.
    """
    discount_factors: dict[str, float] = {}

    if pressure.aggregate < 0.05:
        pressure.debug.discount_factors = {}
        return signals

    for signal_name, sources in _DISCOUNT_MAP.items():
        reading = getattr(signals, signal_name, None)
        if reading is None:
            continue

        total_discount = 0.0
        for pressure_field, max_discount in sources:
            p_val = getattr(pressure, pressure_field, 0.0)
            total_discount += p_val * max_discount

        total_discount = min(0.70, total_discount)  # never discount more than 70%

        if total_discount > 0.01:
            original = reading.signal_reliability
            reading.signal_reliability = original * (1.0 - total_discount)
            discount_factors[signal_name] = round(total_discount, 4)

    pressure.debug.discount_factors = discount_factors
    return signals
