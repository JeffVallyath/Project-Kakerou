"""Speech Act Classification — structural conversation analysis.

Instead of scoring surface-level linguistic features (fragmentation, emotional
intensity), this module classifies WHAT each utterance is DOING in the
conversation. This is vocabulary-agnostic and slang-proof.

Based on:
- Speech Act Theory (Austin 1962, Searle 1969)
- Computational Pragmatics (Jurafsky & Martin)
- Conversation Analysis adjacency pairs (Schegloff & Sacks 1973)

Speech Acts (what the utterance DOES):
- INFORM: provides concrete, verifiable information
- EVADE: redirects, refuses, or avoids the question
- DISMISS: shuts down the topic without engaging
- DEFLECT: answers a different question than what was asked
- DEFEND: justifies, rationalizes, or pushes back
- CHALLENGE: questions the questioner's authority or motives
- ACKNOWLEDGE: confirms, agrees, or accepts
- QUALIFY: partially answers but hedges or limits scope

Adjacency Pair Analysis (structural expectation violations):
- Question → Inform = normal (compliance)
- Question → Evade = suspicious (structural violation)
- Accusation → Defend = expected (but tracked for escalation)
- Accusation → Dismiss = highly suspicious
- Request → Challenge = suspicious (deflecting accountability)

The combination of speech act classification + adjacency pair tracking
gives the Bayesian engine structurally grounded signals that don't depend
on vocabulary, slang, or cultural context.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Speech Act taxonomy
# ---------------------------------------------------------------------------

SPEECH_ACTS = {
    "INFORM": "Provides concrete, specific, verifiable information",
    "EVADE": "Redirects, refuses, or avoids addressing the topic",
    "DISMISS": "Shuts down the topic without substantive engagement",
    "DEFLECT": "Answers a different question than what was asked",
    "DEFEND": "Justifies, rationalizes, or pushes back against scrutiny",
    "CHALLENGE": "Questions the questioner's authority, motives, or right to ask",
    "ACKNOWLEDGE": "Confirms, agrees, accepts, or validates",
    "QUALIFY": "Partially answers but hedges, limits scope, or adds caveats",
}

# Speech act weights loaded from optimized config
# These are the defaults — overridden by weights.json if it exists
def _get_act_maps() -> tuple[dict[str, float], dict[str, float]]:
    """Load speech act weight maps from tunable weights config."""
    try:
        from btom_engine.weights import WEIGHTS
        return WEIGHTS.get_act_bluff_map(), WEIGHTS.get_act_withhold_map()
    except Exception:
        # Fallback: construct from canonical EngineWeights defaults (no drift)
        from btom_engine.weights import EngineWeights
        default_w = EngineWeights()
        return default_w.get_act_bluff_map(), default_w.get_act_withhold_map()


# ---------------------------------------------------------------------------
# Adjacency Pairs — structural expectations
# ---------------------------------------------------------------------------

@dataclass
class AdjacencyPair:
    """A linked pair of conversational moves."""
    first_act: str = ""       # what the user did (question, accusation, request)
    second_act: str = ""      # what the target responded with
    expected: list[str] = field(default_factory=list)  # what was structurally expected
    violation: bool = False   # did the target violate the expectation?
    severity: float = 0.0     # how suspicious is the violation (0-1)


# What responses are structurally expected for each user move
_EXPECTED_RESPONSES = {
    "QUESTION": {
        "expected": ["INFORM", "ACKNOWLEDGE", "QUALIFY"],
        "suspicious": {"EVADE": 0.8, "DISMISS": 0.7, "DEFLECT": 0.85, "CHALLENGE": 0.6, "DEFEND": 0.4},
    },
    "ACCUSATION": {
        "expected": ["DEFEND", "ACKNOWLEDGE", "INFORM"],
        "suspicious": {"EVADE": 0.9, "DISMISS": 0.85, "DEFLECT": 0.8, "CHALLENGE": 0.5},
    },
    "REQUEST": {
        "expected": ["ACKNOWLEDGE", "INFORM", "QUALIFY"],
        "suspicious": {"EVADE": 0.7, "DISMISS": 0.75, "CHALLENGE": 0.65, "DEFLECT": 0.7},
    },
    "STATEMENT": {
        "expected": ["ACKNOWLEDGE", "INFORM", "QUALIFY", "CHALLENGE"],
        "suspicious": {"EVADE": 0.5, "DISMISS": 0.4},
    },
}


# ---------------------------------------------------------------------------
# User utterance classification (what did the USER do?)
# ---------------------------------------------------------------------------

def classify_user_act(text: str) -> str:
    """Classify the user's utterance into a conversational move type.

    Pure Python — pattern matching on structure, not vocabulary.
    """
    text_lower = text.lower().strip()

    # Question detection
    if text.rstrip().endswith("?"):
        return "QUESTION"
    if any(text_lower.startswith(w) for w in [
        "what", "where", "when", "who", "why", "how", "did", "do", "does",
        "is", "are", "was", "were", "can", "could", "would", "will",
        "have you", "had you", "tell me",
    ]):
        return "QUESTION"

    # Accusation detection
    accusation_markers = [
        "you lied", "you're lying", "that's not true", "that's a lie",
        "you didn't", "you never", "you always", "you said",
        "how come you", "why didn't you", "explain why",
        "that contradicts", "that doesn't match", "that's inconsistent",
    ]
    if any(marker in text_lower for marker in accusation_markers):
        return "ACCUSATION"

    # Request detection
    request_markers = [
        "tell me", "show me", "explain", "describe", "give me",
        "i need", "i want", "please", "can you", "could you",
    ]
    if any(marker in text_lower for marker in request_markers):
        return "REQUEST"

    return "STATEMENT"


# ---------------------------------------------------------------------------
# Target speech act classification
# ---------------------------------------------------------------------------

def classify_target_act(text: str) -> str:
    """Classify the target's utterance into a speech act.

    Uses structural patterns, not vocabulary meaning.
    This is the core classification that feeds the Bayesian engine.
    """
    text_lower = text.lower().strip()
    words = text_lower.split()
    word_count = len(words)

    # Very short responses — classify by structure
    if word_count <= 3:
        # Single word/emoji acknowledgments
        if text_lower in {"ok", "okay", "yes", "yeah", "yep", "sure", "fine",
                          "right", "true", "agreed", "correct", "absolutely"}:
            return "ACKNOWLEDGE"
        # Single word/short dismissals
        if text_lower in {"no", "nah", "nope", "whatever", "idk", "idc",
                          "bruh", "lol", "lmao", "k", "bye", "bet", "cap",
                          "ngl", "ion know", "meh", "pass", "next",
                          "bro what", "bruh what", "lmfao"}:
            return "DISMISS"
        # Short combos that are dismissive
        if any(p in text_lower for p in ["idk", "idc", "ion know", "who cares",
                                          "dont care", "dont know", "no idea",
                                          "lol idk", "lmao idk"]):
            return "DISMISS"

    # Evasion patterns (structural, not vocabulary-dependent)
    evasion_patterns = [
        r"(?:i |i'?d )(?:don'?t|do not) (?:want|wish|care|need) to (?:talk|discuss|get into)",
        r"(?:can|let'?s) (?:we |us )?(?:talk|discuss|focus|move) (?:about |on )?(?:something|something else)",
        r"(?:i |i'?d )(?:rather|prefer) not",
        r"that'?s not (?:really |)(?:the |my |)(?:point|issue|question|problem|concern)",
        r"(?:why (?:do |would |are )?you|what makes you) (?:want|need|think|ask|care)",
        r"i (?:don'?t |do not )(?:remember|recall|know)",
        r"(?:let me |i'?ll )(?:get back|think about|check)",
        # Legal/formal evasion
        r"(?:i )?(?:plead|invoke|take) the fifth",
        r"(?:i )?(?:have )?no (?:recollection|memory|knowledge)",
        r"(?:i )?(?:can'?t|cannot) (?:recall|remember|recollect)",
        r"not to my knowledge",
        r"(?:my )?(?:lawyer|attorney|counsel) (?:advised|told|said)",
        r"(?:i'?m |i am )?not (?:at liberty|authorized|able) to",
        # Slang evasion
        r"(?:ion|i dont|idk) (?:even |)(?:know|remember|care)",
        r"(?:thats|that is|its) not (?:that |even )?(?:deep|serious|important)",
        r"(?:chill|relax|calm down)",
        # Deflection (answering question with question)
        r"^(?:why|how|what|who) (?:do|did|would|are|is) (?:you|that)",
    ]
    for pattern in evasion_patterns:
        if re.search(pattern, text_lower):
            return "EVADE"

    # Challenge patterns
    challenge_patterns = [
        r"(?:who |what |why )(?:told|asked|made|gives) you",
        r"(?:that'?s |it'?s )(?:not (?:your|any of your)|none of your)",
        r"you (?:don'?t|do not) (?:know|understand|get)",
        r"(?:why|how) (?:is|would) that (?:any of |)your (?:business|concern|problem)",
        r"you (?:have|got) no (?:right|authority|business)",
    ]
    for pattern in challenge_patterns:
        if re.search(pattern, text_lower):
            return "CHALLENGE"

    # Defense patterns
    defense_patterns = [
        r"i (?:had|have) to (?:because|since)",
        r"(?:the |)(?:reason|thing) (?:is|was) (?:that |because )",
        r"(?:i |)already (?:told|explained|said|mentioned)",
        r"(?:it'?s |that'?s )not (?:my |)(?:fault|responsibility|problem)",
        r"(?:anyone|everybody|you) (?:would have|would'?ve|in my (?:position|place|shoes))",
        r"(?:you (?:have|need) to |)understand (?:that |the )",
        r"(?:i |)(?:was |am )(?:just |only )(?:trying|doing|following)",
        # Denials
        r"i (?:never|didn'?t|did not|don'?t|do not|wasn'?t|was not|haven'?t|have not) ",
        r"(?:i'?m |i am )not (?:lying|the one|responsible|guilty|involved)",
        r"(?:why would i|how could i) (?:lie|do that|steal|take)",
        # Slang defense
        r"(?:thats|that is) cap",
        r"(?:i'?m |)not (?:cap|capping|lying|playing)",
        r"on (?:my |)(?:life|god|everything|mama)",
    ]
    for pattern in defense_patterns:
        if re.search(pattern, text_lower):
            return "DEFEND"

    # Qualify patterns (partial answer with hedging)
    qualify_patterns = [
        r"(?:i think|i believe|i guess|i suppose|maybe|perhaps|probably)",
        r"(?:as far as|to the best of|from what) (?:i |my )",
        r"(?:it |that )(?:depends|varies|could be)",
        r"(?:sort of|kind of|in a way|to some extent)",
        r"(?:not exactly|not really|not quite|not entirely)",
    ]
    qualify_count = sum(1 for p in qualify_patterns if re.search(p, text_lower))

    # Inform patterns (concrete, specific information)
    has_numbers = bool(re.search(r'\d+', text))
    has_names = bool(re.search(r'[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?', text))
    has_temporal = bool(re.search(
        r'(?:yesterday|today|tomorrow|monday|tuesday|wednesday|thursday|friday|'
        r'saturday|sunday|morning|afternoon|evening|january|february|march|april|'
        r'may|june|july|august|september|october|november|december|\d+\s*(?:am|pm|o\'?clock))',
        text_lower,
    ))
    has_location = bool(re.search(
        r'(?:at |in |to |from |near )(?:the |my |our |a )',
        text_lower,
    ))

    concreteness_score = sum([has_numbers, has_names, has_temporal, has_location])

    # Decision logic
    if qualify_count >= 2:
        return "QUALIFY"
    if concreteness_score >= 2 and qualify_count == 0:
        return "INFORM"
    if concreteness_score >= 1 and word_count > 5:
        if qualify_count > 0:
            return "QUALIFY"
        return "INFORM"
    if word_count > 10 and qualify_count == 0 and concreteness_score == 0:
        # Long response with no concrete details — likely defending or evading
        return "DEFEND"

    # Default: acknowledge for short, qualify for medium, inform for long with specifics
    if word_count <= 5:
        return "ACKNOWLEDGE"
    return "QUALIFY"


# ---------------------------------------------------------------------------
# Adjacency pair analysis
# ---------------------------------------------------------------------------

def analyze_adjacency_pair(
    user_text: str,
    target_text: str,
) -> AdjacencyPair:
    """Analyze the structural relationship between a user utterance and target response.

    Returns an AdjacencyPair indicating whether the response violated
    conversational expectations — the core structural deception signal.
    """
    user_act = classify_user_act(user_text)
    target_act = classify_target_act(target_text)

    pair = AdjacencyPair(
        first_act=user_act,
        second_act=target_act,
    )

    expectations = _EXPECTED_RESPONSES.get(user_act, _EXPECTED_RESPONSES["STATEMENT"])
    pair.expected = expectations["expected"]

    if target_act in expectations["expected"]:
        pair.violation = False
        pair.severity = 0.0
    elif target_act in expectations.get("suspicious", {}):
        pair.violation = True
        pair.severity = expectations["suspicious"][target_act]
    else:
        pair.violation = False
        pair.severity = 0.0

    return pair


# ---------------------------------------------------------------------------
# Hypothesis adjustment from speech acts
# ---------------------------------------------------------------------------

@dataclass
class SpeechActResult:
    """Result of speech act analysis for one turn."""
    target_act: str = ""
    user_act: str = ""
    adjacency_pair: AdjacencyPair = field(default_factory=AdjacencyPair)
    bluffing_delta: float = 0.0
    withholding_delta: float = 0.0
    structural_violation: bool = False
    violation_severity: float = 0.0
    rationale: str = ""


def analyze_turn(
    target_text: str,
    user_text: str = "",
    turn_number: int = 0,
) -> SpeechActResult:
    """Full speech act analysis for one conversational turn.

    Returns hypothesis adjustments based on:
    1. What speech act the target performed
    2. Whether it violated adjacency pair expectations
    3. The severity of any structural violation
    """
    target_act = classify_target_act(target_text)
    result = SpeechActResult(target_act=target_act)

    # Load optimized weights
    bluff_map, withhold_map = _get_act_maps()
    try:
        from btom_engine.weights import WEIGHTS
        viol_boost_factor = WEIGHTS.violation_boost_factor
    except Exception:
        viol_boost_factor = 0.25

    # Base adjustment from speech act type (Optuna-optimized)
    result.bluffing_delta = bluff_map.get(target_act, 0.0)
    result.withholding_delta = withhold_map.get(target_act, 0.0)

    # Adjacency pair analysis (if we have the user's utterance)
    if user_text:
        result.user_act = classify_user_act(user_text)
        pair = analyze_adjacency_pair(user_text, target_text)
        result.adjacency_pair = pair
        result.structural_violation = pair.violation
        result.violation_severity = pair.severity

        # Amplify suspicion on structural violations (weight from Optuna)
        if pair.violation:
            violation_boost = pair.severity * viol_boost_factor
            result.bluffing_delta += violation_boost
            result.withholding_delta += violation_boost * 0.8

    # Build rationale
    parts = [f"Speech act: {target_act}"]
    if user_text:
        parts.append(f"User move: {result.user_act}")
    if result.structural_violation:
        parts.append(
            f"STRUCTURAL VIOLATION: {result.user_act}→{target_act} "
            f"(expected: {', '.join(result.adjacency_pair.expected)}, "
            f"severity: {result.violation_severity:.2f})"
        )
    result.rationale = " | ".join(parts)

    return result
