"""Bounded Semantic Review Layer — motif-based classification for user pressure.

Architecture:
  rules (prior) -> trigger check -> [motif classification if triggered] -> deterministic mapping -> bounded merge -> final pressure

Two paths:
  1. Generic path: Qwen classifies user utterance into discourse pressure motifs.
     Python deterministically maps motifs to accusation/repetition/hostility.
  2. Slur path: Qwen classifies slur usage context into 5 classes.
     Python maps class to fixed hostility value.

Design principles:
  - LLM = bounded classifier (motif selection, not raw pressure scores)
  - Python = owner of all consequences (mapping, merge, final state)
  - Structured JSON output only, no freeform reasoning
  - Falls back cleanly to rule prior on any failure
  - Modular: swap model/prompt without touching engine
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any

from btom_engine.interaction_context import UserPressure

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Review result
# ---------------------------------------------------------------------------

@dataclass
class SemanticReviewResult:
    """Output of the Qwen semantic review. For diagnostics and merge."""

    ran: bool = False
    trigger_reason: str = ""
    rule_prior: dict[str, float] = field(default_factory=dict)
    qwen_adjusted: dict[str, float] = field(default_factory=dict)
    final_merged: dict[str, float] = field(default_factory=dict)
    confidence: float = 0.0
    rationale_short: str = ""
    fallback_used: bool = False
    fallback_reason: str = ""
    raw_llm_output: str = ""
    # Slur-specific adjudication fields
    slur_path: bool = False
    slur_context_class: str = ""
    # Motif classification fields (generic path)
    primary_class: str = ""
    motif_classes: list[dict] = field(default_factory=list)
    strategy_class: str = ""


# ---------------------------------------------------------------------------
# Trigger logic — when does Qwen review run?
# ---------------------------------------------------------------------------

_PROFANITY_RE = re.compile(
    r"\b(fuck|fucking|fucked|shit|shitty|damn|bullshit|ass|asshole|bitch|cunt|dick|bastard|"
    r"screw you|go to hell|kill yourself|kys)\b",
    re.IGNORECASE,
)

_POSITIVE_INTENSIFIER_RE = re.compile(
    r"\b(awesome|amazing|great|love|beautiful|incredible|fantastic|brilliant)\b",
    re.IGNORECASE,
)

_EMOTIONAL_PUNCTUATION_RE = re.compile(r"[!]{2,}|[?]{3,}")

_IMPERATIVE_COMMAND_RE = re.compile(
    r"\b(stop|quit|don'?t|do not|cut out|knock off|enough with)\b.{0,20}"
    r"\b(acting|being|playing|pretending|making|getting|giving|trying)\b",
    re.IGNORECASE,
)

_HIGH_SALIENCE_RE = re.compile(
    r"\b(kill yourself|kys|die|threat|destroy you|end you)\b",
    re.IGNORECASE,
)

_SLUR_RE = re.compile(
    r"\b(nigger|nigga|nig|kike|spic|wetback|chink|gook|raghead|towelhead|"
    r"faggot|fag|dyke|tranny|homo|retard|retarded|tard|"
    r"cripple|freak|mongol|subhuman|vermin|"
    r"kill yourself|kys|go die)\b",
    re.IGNORECASE,
)


def _should_trigger(text: str, pressure: UserPressure) -> tuple[bool, str]:
    """Decide whether Qwen review should run. Returns (should_run, reason)."""
    text_lower = text.lower()
    words = text.split()

    if _SLUR_RE.search(text_lower):
        return True, "slur_identity_attack"
    if _HIGH_SALIENCE_RE.search(text_lower):
        return True, "high_salience_content"

    has_profanity = bool(_PROFANITY_RE.search(text_lower))
    has_positive = bool(_POSITIVE_INTENSIFIER_RE.search(text_lower))
    has_emotional_punct = bool(_EMOTIONAL_PUNCTUATION_RE.search(text))

    if has_profanity:
        if has_positive:
            return True, "profanity_positive_ambiguity"
        if pressure.aggregate < 0.20:
            return True, "profanity_low_rules_disambiguation"
        return True, "profanity_disambiguation"

    if pressure.aggregate < 0.10 and _IMPERATIVE_COMMAND_RE.search(text_lower):
        return True, "imperative_command_rule_miss"

    if pressure.aggregate < 0.15 and len(words) > 5:
        if has_emotional_punct:
            return True, "low_rules_but_emotional_punctuation"

    cats = [pressure.accusation, pressure.repetition, pressure.hostility]
    above = [c for c in cats if c > 0.2]
    if len(above) >= 2:
        spread = max(above) - min(above)
        if spread < 0.15:
            return True, "category_collision"

    return False, ""


# ---------------------------------------------------------------------------
# Motif classification prompt (generic path)
# ---------------------------------------------------------------------------

_VALID_MOTIF_CLASSES = {
    "neutral_or_information_seeking",
    "accountability_probe",
    "accusation_of_dodging",
    "admission_demand",
    "narrowing_repeat",
    "hostile_imperative",
    "contemptuous_blame",
    "direct_insult_or_belittling",
    "hostile_profanity",
    "affiliative_or_nonhostile_profanity",
    "quoted_or_referential_offensive_term",
    "adversarial_universal",
}

_VALID_STRATEGY_CLASSES = {
    "none",
    "repeated_narrowing",
    "escalating_blame",
    "evasive_accusation_pattern",
}

# Form flags that the model might mistakenly place in the class list
_VALID_FORM_FLAGS = {
    "profanity_present",
    "emotional_punctuation",
    "imperative_form",
    "quoted_reference",
    "fragmented_or_noisy",
}

_MOTIF_PROMPT = """You are a discourse pressure classifier for online text. Output raw JSON only. No prose.

Classify this user utterance into operational discourse pressure motifs.

User utterance: "{user_text}"
{context_line}

MOTIF CLASSES (select 1 primary, up to 2 secondary if genuinely mixed):
- neutral_or_information_seeking: genuine question, no pressure
- accountability_probe: questioning actions without accusation
- accusation_of_dodging: accusing of evasion, deflection, avoidance
- admission_demand: demanding confession or acknowledgment
- narrowing_repeat: repeating/narrowing the same demand
- hostile_imperative: aggressive command ("stop", "shut up", "answer now")
- contemptuous_blame: identity-level blame ("you're the problem", "your fault")
- direct_insult_or_belittling: personal insult, degrading remark
- hostile_profanity: profanity used aggressively toward someone
- affiliative_or_nonhostile_profanity: profanity as emphasis, humor, or bonding
- quoted_or_referential_offensive_term: quoting/referencing offensive language
- adversarial_universal: "you always/never" universal blame pattern

STRATEGY CLASS (if clearly evident from immediate context):
- none: no discernible multi-turn strategy
- repeated_narrowing: systematically narrowing toward a single point
- escalating_blame: progressively intensifying blame
- evasive_accusation_pattern: accusing to deflect own responsibility

FORM FLAGS (select any that apply):
- profanity_present
- emotional_punctuation
- imperative_form
- quoted_reference
- fragmented_or_noisy

RULES:
- Classify ONLY operational discourse pressure patterns
- Do NOT infer hidden motives or personality
- If weak evidence, prefer neutral_or_information_seeking or accountability_probe
- Allow up to 3 classes total (1 primary + up to 2 secondary)
- Prefer fewer classes unless genuinely mixed
- Membership values 0.0-1.0 reflect how strongly the text fits each class

Return ONLY:
{{"primary_class": "<class>", "primary_membership": <float>, "secondary_classes": [{{"class": "<class>", "membership": <float>}}], "form_flags": ["<flag>"], "strategy_class": "<strategy>", "ambiguity": <float>, "confidence": <float>, "rationale_short": "<max 12 words>"}}"""


# ---------------------------------------------------------------------------
# Deterministic motif -> pressure mapping (Python-owned)
# ---------------------------------------------------------------------------

# [accusation, repetition, hostility]
_MOTIF_PRESSURE_MAP: dict[str, list[float]] = {
    "neutral_or_information_seeking":       [0.00, 0.00, 0.00],
    "accountability_probe":                 [0.20, 0.05, 0.00],
    "accusation_of_dodging":                [0.75, 0.05, 0.20],
    "admission_demand":                     [0.70, 0.10, 0.10],
    "narrowing_repeat":                     [0.10, 0.80, 0.05],
    "hostile_imperative":                   [0.10, 0.10, 0.75],
    "contemptuous_blame":                   [0.65, 0.05, 0.60],
    "direct_insult_or_belittling":          [0.10, 0.00, 0.85],
    "hostile_profanity":                    [0.05, 0.00, 0.80],
    "affiliative_or_nonhostile_profanity":  [0.00, 0.00, 0.05],
    "quoted_or_referential_offensive_term": [0.00, 0.00, 0.10],
    "adversarial_universal":                [0.50, 0.10, 0.35],
}

_STRATEGY_ADJUSTMENT: dict[str, list[float]] = {
    "none":                        [0.00, 0.00, 0.00],
    "repeated_narrowing":          [0.00, 0.15, 0.00],
    "escalating_blame":            [0.10, 0.00, 0.10],
    "evasive_accusation_pattern":  [0.10, 0.05, 0.00],
}

_LAMBDA = 0.60   # rank discount for secondary classes
_BETA = 0.67     # merge strength
_DELTA = 0.40    # max adjustment per category


def _compute_motif_pressure(
    classes: list[dict],
    strategy: str,
) -> dict[str, float]:
    """Deterministic weighted mapping from motif classes to pressure vector.

    1. Rank classes by membership, take top 3.
    2. Weight: membership_k * lambda^(k-1), normalize.
    3. Weighted sum of class vectors.
    4. Add strategy adjustment (bounded).
    """
    if not classes:
        return {"accusation": 0.0, "repetition": 0.0, "hostility": 0.0}

    # Sort by membership descending, take top 3
    ranked = sorted(classes, key=lambda c: c.get("membership", 0.0), reverse=True)[:3]

    # Compute raw weights with rank discount
    raw_weights = []
    for i, entry in enumerate(ranked):
        m = max(0.0, min(1.0, float(entry.get("membership", 0.0))))
        raw_weights.append(m * (_LAMBDA ** i))

    total_weight = sum(raw_weights)
    if total_weight < 0.001:
        return {"accusation": 0.0, "repetition": 0.0, "hostility": 0.0}

    # Normalize
    norm_weights = [w / total_weight for w in raw_weights]

    # Weighted sum of class vectors
    acc, rep, hos = 0.0, 0.0, 0.0
    for entry, w in zip(ranked, norm_weights):
        cls_name = entry.get("class", "neutral_or_information_seeking")
        vec = _MOTIF_PRESSURE_MAP.get(cls_name, [0.0, 0.0, 0.0])
        acc += vec[0] * w
        rep += vec[1] * w
        hos += vec[2] * w

    # Strategy adjustment
    strat_vec = _STRATEGY_ADJUSTMENT.get(strategy, [0.0, 0.0, 0.0])
    acc = min(1.0, acc + strat_vec[0])
    rep = min(1.0, rep + strat_vec[1])
    hos = min(1.0, hos + strat_vec[2])

    return {"accusation": acc, "repetition": rep, "hostility": hos}


def _motif_merge(
    rule_prior: dict[str, float],
    motif_pressure: dict[str, float],
    effective_confidence: float,
) -> dict[str, float]:
    """Bounded merge: rule_prior + clamp(beta * conf * (motif - prior), -delta, +delta)."""
    merged = {}
    for key in ["accusation", "repetition", "hostility"]:
        rp = rule_prior.get(key, 0.0)
        mp = motif_pressure.get(key, 0.0)
        adjustment = _BETA * effective_confidence * (mp - rp)
        clamped = max(-_DELTA, min(_DELTA, adjustment))
        merged[key] = max(0.0, min(1.0, rp + clamped))
    return merged


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------

def _sanitize_json(raw: str) -> str:
    """Extract JSON from potentially wrapped LLM output."""
    raw = re.sub(r"```(?:json)?\s*", "", raw)
    raw = raw.replace("```", "")
    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end != -1 and end > start:
        return raw[start : end + 1]
    return raw.strip()


def _call_qwen(prompt: str, max_tokens: int = 400) -> tuple[dict[str, Any], str]:
    """LLM call for semantic review. Routes to remote if available."""
    from btom_engine.remote_llm import remote_chat

    result = remote_chat(
        user=prompt,
        max_tokens=max_tokens,
        temperature=0.1,
    )
    raw = result["content"]
    sanitized = _sanitize_json(raw)
    parsed = json.loads(sanitized)
    return parsed, raw


# ---------------------------------------------------------------------------
# Generic motif classification path
# ---------------------------------------------------------------------------

def _build_motif_prompt(
    user_text: str,
    prev_target_text: str = "",
) -> str:
    """Build the motif classification prompt."""
    context_line = ""
    if prev_target_text:
        short = prev_target_text[:100] + ("..." if len(prev_target_text) > 100 else "")
        context_line = f'Previous target utterance: "{short}"'

    return _MOTIF_PROMPT.format(
        user_text=user_text,
        context_line=context_line,
    )


def _parse_motif_response(
    parsed: dict[str, Any],
) -> tuple[list[dict], str, float, float, str, list[str], list[str]]:
    """Extract and validate motif classification from LLM response.

    Returns (classes, strategy, ambiguity, confidence, rationale, form_flags, dropped).
    Dropped items are logged for diagnostics — form flags misplaced as classes, etc.
    """
    dropped: list[str] = []

    # Primary class — if model emits a form flag as primary, remap to neutral
    primary = str(parsed.get("primary_class", "neutral_or_information_seeking")).strip().lower()
    if primary in _VALID_FORM_FLAGS:
        dropped.append(f"primary '{primary}' is a form flag, remapped to neutral")
        primary = "neutral_or_information_seeking"
    elif primary not in _VALID_MOTIF_CLASSES:
        dropped.append(f"primary '{primary}' invalid, remapped to neutral")
        primary = "neutral_or_information_seeking"
    primary_m = max(0.0, min(1.0, float(parsed.get("primary_membership", 0.5))))

    classes = [{"class": primary, "membership": primary_m}]

    # Secondary classes — reclassify form flags, drop truly invalid entries
    form_flags_from_classes: list[str] = []
    secondaries = parsed.get("secondary_classes", [])
    if isinstance(secondaries, list):
        for s in secondaries[:2]:
            if isinstance(s, dict):
                cls = str(s.get("class", "")).strip().lower()
                mem = max(0.0, min(1.0, float(s.get("membership", 0.0))))

                if cls in _VALID_FORM_FLAGS:
                    # Misplaced form flag — reclassify into form_flags list
                    form_flags_from_classes.append(cls)
                    dropped.append(f"secondary '{cls}' is a form flag, moved to form_flags")
                elif cls in _VALID_MOTIF_CLASSES and cls != primary and mem > 0.05:
                    classes.append({"class": cls, "membership": mem})
                elif cls and cls not in _VALID_MOTIF_CLASSES:
                    dropped.append(f"secondary '{cls}' invalid, dropped")

    # Strategy
    strategy = str(parsed.get("strategy_class", "none")).strip().lower()
    if strategy not in _VALID_STRATEGY_CLASSES:
        dropped.append(f"strategy '{strategy}' invalid, defaulted to none")
        strategy = "none"

    # Scalars
    ambiguity = max(0.0, min(1.0, float(parsed.get("ambiguity", 0.3))))
    confidence = max(0.0, min(1.0, float(parsed.get("confidence", 0.5))))
    rationale = str(parsed.get("rationale_short", ""))[:80]

    # Form flags — merge explicit flags with reclassified ones
    raw_flags = parsed.get("form_flags", [])
    form_flags = [str(f) for f in raw_flags] if isinstance(raw_flags, list) else []
    for reclassified in form_flags_from_classes:
        if reclassified not in form_flags:
            form_flags.append(reclassified)

    return classes, strategy, ambiguity, confidence, rationale, form_flags, dropped


def _run_generic_review(
    user_text: str,
    pressure: UserPressure,
    rule_prior: dict[str, float],
    trigger_reason: str,
    prev_target_text: str = "",
) -> tuple[UserPressure, SemanticReviewResult]:
    """Generic motif classification path."""
    result = SemanticReviewResult(
        ran=True,
        trigger_reason=trigger_reason,
        rule_prior=dict(rule_prior),
    )

    prompt = _build_motif_prompt(user_text, prev_target_text)

    try:
        parsed, raw = _call_qwen(prompt, max_tokens=250)
        result.raw_llm_output = raw
        logger.debug("MOTIF REVIEW RAW: %s", raw)

        classes, strategy, ambiguity, confidence, rationale, form_flags, dropped = _parse_motif_response(parsed)

        if dropped:
            for d in dropped:
                logger.info("Motif parse: %s", d)

        result.primary_class = classes[0]["class"] if classes else ""
        result.motif_classes = classes
        result.strategy_class = strategy
        result.confidence = confidence
        result.rationale_short = rationale
        result.qwen_adjusted = {
            "classes": classes,
            "strategy": strategy,
            "ambiguity": ambiguity,
            "form_flags": form_flags,
            "dropped": dropped,
        }

        # Deterministic mapping: motifs -> pressure vector
        motif_pressure = _compute_motif_pressure(classes, strategy)

        # Effective confidence: penalize by ambiguity
        effective_confidence = confidence * (1.0 - 0.5 * ambiguity)

        # Bounded merge with rule prior
        merged = _motif_merge(rule_prior, motif_pressure, effective_confidence)
        result.final_merged = merged

        final_pressure = UserPressure(
            accusation=merged["accusation"],
            repetition=merged["repetition"],
            hostility=merged["hostility"],
            debug=pressure.debug,
        )
        return final_pressure, result

    except httpx.ConnectError as e:
        result.fallback_used = True
        result.fallback_reason = f"Connection Error: {e}"
        result.final_merged = dict(rule_prior)
        logger.warning("Semantic review: LLM unreachable, using rule prior")
        return pressure, result

    except (json.JSONDecodeError, httpx.HTTPError, KeyError, ValueError, TypeError) as e:
        result.fallback_used = True
        result.fallback_reason = f"{type(e).__name__}: {e}"
        result.final_merged = dict(rule_prior)
        logger.warning("Semantic review failed (%s): %s", type(e).__name__, e)
        return pressure, result


# ---------------------------------------------------------------------------
# Slur-specific contextual adjudication (unchanged)
# ---------------------------------------------------------------------------

_SLUR_CLASSIFICATION_PROMPT = """You are a contextual language classifier. Output raw JSON only. No prose.

A rule-based detector found a slur or identity-targeted term in this user utterance.
Your job: classify HOW the term is being used in context.

User utterance: "{user_text}"
{context_line}

Classify the usage into exactly ONE of these classes:
- explicit_hostile_attack: directed at someone with intent to harm, demean, or threaten
- demeaning_contemptuous: used to belittle or degrade, possibly indirectly
- quoted_referential: reporting, quoting, discussing the word itself, or asking about it
- affiliative_banter: in-group use, joking, friendly intensifier, not directed as attack
- ambiguous: insufficient context to classify confidently

Also provide:
- confidence (0.0-1.0)
- rationale (max 15 words)

Return ONLY:
{{"context_class": "<one of the five classes>", "confidence": <float>, "rationale": "<string>"}}"""

_SLUR_CLASS_HOSTILITY_MAP = {
    "explicit_hostile_attack": 0.85,
    "demeaning_contemptuous": 0.65,
    "ambiguous": 0.45,
    "quoted_referential": 0.10,
    "affiliative_banter": 0.05,
}

_VALID_SLUR_CLASSES = set(_SLUR_CLASS_HOSTILITY_MAP.keys())


def _run_slur_adjudication(
    user_text: str,
    pressure: UserPressure,
    rule_prior: dict[str, float],
    prev_target_text: str = "",
) -> tuple[UserPressure, SemanticReviewResult]:
    """Slur-specific path: Qwen classifies usage context, Python maps to hostility."""
    result = SemanticReviewResult(
        ran=True,
        trigger_reason="slur_identity_attack",
        rule_prior=dict(rule_prior),
        slur_path=True,
    )

    context_line = ""
    if prev_target_text:
        short = prev_target_text[:100] + ("..." if len(prev_target_text) > 100 else "")
        context_line = f'Previous target utterance: "{short}"'

    prompt = _SLUR_CLASSIFICATION_PROMPT.format(
        user_text=user_text,
        context_line=context_line,
    )

    try:
        parsed, raw = _call_qwen(prompt)
        result.raw_llm_output = raw
        logger.debug("SLUR ADJUDICATION RAW: %s", raw)

        context_class = str(parsed.get("context_class", "ambiguous")).strip().lower()
        if context_class not in _VALID_SLUR_CLASSES:
            context_class = "ambiguous"

        confidence = max(0.0, min(1.0, float(parsed.get("confidence", 0.5))))
        rationale = str(parsed.get("rationale", ""))[:80]

        result.slur_context_class = context_class
        result.confidence = confidence
        result.rationale_short = rationale
        result.qwen_adjusted = {"context_class": context_class}

        mapped_hostility = _SLUR_CLASS_HOSTILITY_MAP[context_class]

        final = {
            "accusation": rule_prior["accusation"],
            "repetition": rule_prior["repetition"],
            "hostility": mapped_hostility,
        }
        result.final_merged = final

        final_pressure = UserPressure(
            accusation=final["accusation"],
            repetition=final["repetition"],
            hostility=final["hostility"],
            debug=pressure.debug,
        )
        return final_pressure, result

    except httpx.ConnectError as e:
        result.fallback_used = True
        result.fallback_reason = f"Connection Error: {e}"
        result.final_merged = dict(rule_prior)
        return pressure, result

    except (json.JSONDecodeError, httpx.HTTPError, KeyError, ValueError, TypeError) as e:
        result.fallback_used = True
        result.fallback_reason = f"{type(e).__name__}: {e}"
        result.final_merged = dict(rule_prior)
        return pressure, result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_semantic_review(
    user_text: str,
    pressure: UserPressure,
    prev_target_text: str = "",
) -> tuple[UserPressure, SemanticReviewResult]:
    """Run the bounded Qwen semantic review on user pressure.

    Returns (final_pressure, review_result).
    If review doesn't trigger or fails, returns the original pressure unchanged.
    """
    rule_prior = {
        "accusation": pressure.accusation,
        "repetition": pressure.repetition,
        "hostility": pressure.hostility,
    }

    result = SemanticReviewResult(rule_prior=dict(rule_prior))

    # Check trigger
    should_run, trigger_reason = _should_trigger(user_text, pressure)
    if not should_run:
        result.final_merged = dict(rule_prior)
        return pressure, result

    # Route slur matches through dedicated adjudication path
    if trigger_reason == "slur_identity_attack":
        return _run_slur_adjudication(user_text, pressure, rule_prior, prev_target_text)

    # Generic path: motif classification
    return _run_generic_review(user_text, pressure, rule_prior, trigger_reason, prev_target_text)
