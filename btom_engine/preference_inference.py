"""Preference Inference — ToM Level 2.

Catches strategic deception by computing Action-Claim Divergence.
Strategic liars have zero cognitive load (no linguistic stress signals),
but their ACTIONS betray their true preferences:
- "I don't need water" + refuses every deal giving away water = lying
- "I'm flexible on food" + demands food in every offer = lying

Architecture:
- Claims: verbal statements about preferences ("I don't need X")
- Actions: behavioral signals (offers, refusals, requests, trades)
- Each action decomposes into per-item impacts (multi-item trades handled)
- Divergence: |stated_value - revealed_value| per item
- Output: standard SignalReading for Bayesian integration

Update rule: latest extraction overwrites previous (no averaging).
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from btom_engine.schema import SignalReading

logger = logging.getLogger(__name__)

_CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "pref_cache"


# ---------------------------------------------------------------------------
# LLM extraction cache
# ---------------------------------------------------------------------------

def _cache_key(text: str, valid_items: list[str] | None, context: str = "") -> str:
    """Deterministic hash of text + valid_items + context for cache lookup."""
    items_str = ",".join(sorted(valid_items)) if valid_items else ""
    raw = f"{text[:500]}|{items_str}|{context[:300]}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


class ExtractionCache:
    """Disk-backed cache for LLM extraction results.

    Stores all entries in a single JSON file per dataset context.
    Loads lazily, writes on save().
    """

    def __init__(self, cache_file: str = "default"):
        self._path = _CACHE_DIR / f"{cache_file}.json"
        self._data: dict[str, Any] | None = None  # lazy load

    def _load(self) -> dict[str, Any]:
        if self._data is None:
            if self._path.exists():
                try:
                    self._data = json.loads(self._path.read_text(encoding="utf-8"))
                except (json.JSONDecodeError, OSError):
                    self._data = {}
            else:
                self._data = {}
        return self._data

    def get(self, text: str, valid_items: list[str] | None, context: str = "") -> dict[str, Any] | None:
        key = _cache_key(text, valid_items, context)
        result = self._load().get(key)
        if result is None and context:
            # Fallback: check no-context key for backward compatibility
            result = self._load().get(_cache_key(text, valid_items, ""))
        return result

    def put(self, text: str, valid_items: list[str] | None, result: dict[str, Any], context: str = "") -> None:
        key = _cache_key(text, valid_items, context)
        self._load()[key] = result

    def save(self) -> None:
        if self._data is None:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(self._data, ensure_ascii=False), encoding="utf-8")

    def __len__(self) -> int:
        return len(self._load())


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class PreferenceEntry:
    """One item/entity being tracked."""
    item: str = ""
    stated_value: float = 0.0       # -1.0 (claims don't want) to +1.0 (claims want)
    revealed_value: float = 0.0     # -1.0 (acts like don't want) to +1.0 (acts like want)
    stated_turn: int = 0
    revealed_turn: int = 0
    stated_raw: str = ""
    revealed_raw: str = ""
    has_stated: bool = False
    has_revealed: bool = False


@dataclass
class TargetMentalState:
    """Accumulated mental model of one target across a conversation."""
    preferences: dict[str, PreferenceEntry] = field(default_factory=dict)
    max_divergence: float = 0.0
    divergence_signal: SignalReading = field(default_factory=SignalReading)
    rationale: str = ""


# ---------------------------------------------------------------------------
# Divergence math
# ---------------------------------------------------------------------------

def _compute_divergence(stated: float, revealed: float) -> float:
    """Divergence = 0 when aligned, max 2.0 when perfectly opposite."""
    return abs(stated - revealed)


def _compute_signal(preferences: dict[str, PreferenceEntry]) -> tuple[float, float, str]:
    """Compute max divergence across all items with both stated + revealed.

    Returns (max_divergence, signal_value, rationale).
    """
    divergences: list[tuple[str, float]] = []
    for item, pref in preferences.items():
        if pref.has_stated and pref.has_revealed:
            div = _compute_divergence(pref.stated_value, pref.revealed_value)
            if div > 0.5:  # noise filter
                divergences.append((item, div))

    if not divergences:
        n = len(preferences)
        return 0.0, 0.0, f"Tracking {n} item(s), no divergence detected"

    max_item, max_div = max(divergences, key=lambda x: x[1])
    signal_value = min(1.0, max_div / 2.0)

    explanations = []
    for item, div in sorted(divergences, key=lambda x: -x[1])[:3]:
        pref = preferences[item]
        explanations.append(
            f"'{item}': stated={pref.stated_value:+.1f} (T{pref.stated_turn}) "
            f"vs revealed={pref.revealed_value:+.1f} (T{pref.revealed_turn}), "
            f"div={div:.2f}"
        )

    rationale = f"PREFERENCE DIVERGENCE: {'; '.join(explanations)}"
    return max_div, signal_value, rationale


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def _normalize_item(raw: str, valid_items: list[str] | None) -> str | None:
    """Normalize an item name to the controlled vocabulary.

    Returns None if valid_items is set and no match is found.
    """
    cleaned = raw.lower().strip().rstrip(".")
    if not cleaned or len(cleaned) < 2:
        return None

    if valid_items is None:
        return cleaned

    # Exact match
    for vi in valid_items:
        if cleaned == vi.lower():
            return vi

    # Substring match (e.g., "firewood" in "some firewood")
    for vi in valid_items:
        if vi.lower() in cleaned or cleaned in vi.lower():
            return vi

    return None  # no match in controlled vocab


# ---------------------------------------------------------------------------
# Regex extraction (fallback, no LLM)
# ---------------------------------------------------------------------------

def _extract_claims_regex(text: str) -> list[dict[str, Any]]:
    """Extract verbal preference claims via regex. Returns list of {item, valence}."""
    claims = []
    text_lower = text.lower().strip()

    # Positive: "I want/need/like X"
    want_patterns = [
        r"(?:i |we )(?:really |definitely )?(?:need|want|like|prefer|must have) (?:the |some |more )?(\w[\w\s]{1,20}?)(?:\.|,|$|!| and | but | for | so | if )",
        r"(\w[\w\s]{1,15}?) (?:is |are )(?:really |very )?(?:important|essential|critical|necessary|crucial)",
        r"(?:i |we )(?:can'?t |cannot )(?:do|go|live|survive) without (\w[\w\s]{1,15}?)(?:\.|,|$)",
    ]
    for pattern in want_patterns:
        for match in re.finditer(pattern, text_lower):
            item = match.group(1).strip().rstrip(".")
            if item and 1 < len(item) < 25:
                claims.append({"item": item, "valence": 1.0, "quote": text[:100]})

    # Negative: "I don't want/need X"
    dont_patterns = [
        r"(?:i |we )(?:don'?t |do not )(?:really )?(?:need|want|care about|care for) (?:the |any )?(\w[\w\s]{1,20}?)(?:\.|,|$| anymore| at all| and | but | for | so | if )",
        r"(?:you can have|take|keep) (?:all )?(?:the |my )?(\w[\w\s]{1,15}?)(?:\.|,|$| and | but )",
        r"(?:i'?m |we'?re )(?:not )?(?:interested in|flexible on|willing to give up) (\w[\w\s]{1,15}?)(?:\.|,|$| and | but )",
        r"(\w[\w\s]{1,15}?) (?:is |are )(?:not )?(?:important|a priority|necessary)",
    ]
    for pattern in dont_patterns:
        for match in re.finditer(pattern, text_lower):
            item = match.group(1).strip().rstrip(".")
            if item and 1 < len(item) < 25:
                claims.append({"item": item, "valence": -1.0, "quote": text[:100]})

    return claims


def _extract_actions_regex(text: str) -> list[dict[str, Any]]:
    """Extract actions via regex. Returns list of {action_type, item_impacts}."""
    actions = []
    text_lower = text.lower().strip()

    # Offering/giving away → negative implied_value
    offer_patterns = [
        r"(?:i'?ll |i will |i can )(?:give|offer|trade|let you have) (?:you )?(?:the |my |some |all )?(\w[\w\s]{1,15}?)(?:\.|,|$| for| if)",
        r"(?:take |here'?s? )(?:the |my )?(\w[\w\s]{1,15}?)(?:\.|,|$)",
    ]
    for pattern in offer_patterns:
        for match in re.finditer(pattern, text_lower):
            item = match.group(1).strip().rstrip(".")
            if item and 1 < len(item) < 25:
                actions.append({
                    "action_type": "offer",
                    "item_impacts": [{"item": item, "implied_value": -0.8}],
                    "quote": text[:100],
                })

    # Requesting/demanding → positive implied_value
    request_patterns = [
        r"(?:i'?ll need|give me|i want) (?:the |some |at least )?(\w[\w\s]{1,15}?)(?:\.|,|$| in | and | or | if | please)",
        r"(?:can i (?:have|get|take)) (?:the |some )?(\w[\w\s]{1,15}?)(?:\.|,|$| and | or )",
    ]
    for pattern in request_patterns:
        for match in re.finditer(pattern, text_lower):
            item = match.group(1).strip().rstrip(".")
            if item and 1 < len(item) < 25:
                actions.append({
                    "action_type": "request",
                    "item_impacts": [{"item": item, "implied_value": 0.8}],
                    "quote": text[:100],
                })

    # Refusing → positive implied_value (protecting what they have)
    refuse_patterns = [
        r"(?:no|nah|i can'?t|that doesn'?t work|i won'?t).*?(?:give up |trade |lose )(?:the |my )?(\w[\w\s]{1,15}?)(?:\.|,|$)",
    ]
    for pattern in refuse_patterns:
        for match in re.finditer(pattern, text_lower):
            item = match.group(1).strip().rstrip(".")
            if item and 1 < len(item) < 25:
                actions.append({
                    "action_type": "refuse",
                    "item_impacts": [{"item": item, "implied_value": 0.7}],
                    "quote": text[:100],
                })

    # Accepting a deal: items given away are negative, items received are positive
    accept_patterns = [
        r"(?:deal|ok|okay|sure|agreed|accept).*?(?:i'?ll (?:give|trade) )(?:the |my )?(\w[\w\s]{1,15}?) (?:for |and get |in exchange for )(?:the |your )?(\w[\w\s]{1,15}?)(?:\.|,|$)",
    ]
    for pattern in accept_patterns:
        for match in re.finditer(pattern, text_lower):
            given = match.group(1).strip().rstrip(".")
            received = match.group(2).strip().rstrip(".")
            if given and received:
                actions.append({
                    "action_type": "accept",
                    "item_impacts": [
                        {"item": given, "implied_value": -0.7},
                        {"item": received, "implied_value": 0.7},
                    ],
                    "quote": text[:100],
                })

    return actions


# ---------------------------------------------------------------------------
# LLM extraction (Gemini)
# ---------------------------------------------------------------------------

def _extract_llm(
    text: str,
    valid_items: list[str] | None,
    cache: ExtractionCache | None = None,
    cache_only: bool = False,
    context: str = "",
) -> dict[str, Any]:
    """Extract claims + actions via Gemini LLM. Returns parsed JSON dict.

    If a cache is provided, checks it first and stores new results.
    If cache_only=True, returns empty on cache miss (no live LLM call).
    Context is recent conversation history for multi-turn understanding.
    """
    if len(text.strip()) < 10:
        return {"claims": [], "actions": []}

    # Check cache (context-aware key)
    if cache is not None:
        cached = cache.get(text, valid_items, context)
        if cached is not None:
            return cached
        if cache_only:
            return {"claims": [], "actions": []}

    try:
        from btom_engine.remote_llm import remote_chat_json

        items_str = ", ".join(valid_items) if valid_items else "any"

        prompt = (
            "Analyze this game/negotiation message. Extract what the speaker DOES (actions) "
            "and what they CLAIM (verbal statements about preferences).\n\n"
            f"Valid items: [{items_str}]\n"
        )
        if valid_items:
            prompt += "You MUST map all extracted items to one of the valid item names exactly.\n"

        # Add conversation context if available
        if context:
            prompt += (
                "\nRecent conversation history (for context only — analyze the CURRENT message):\n"
                f"{context}\n"
            )

        prompt += (
            "\nReturn JSON:\n"
            "{\n"
            '  "claims": [\n'
            '    {"item": "water", "valence": -1.0, "quote": "I don\'t need water"}\n'
            "  ],\n"
            '  "actions": [\n'
            '    {"action_type": "refuse|offer|request|protect|accuse|accept",\n'
            '     "item_impacts": [\n'
            '       {"item": "water", "implied_value": -1.0},\n'
            '       {"item": "food", "implied_value": 1.0}\n'
            "     ],\n"
            '     "quote": "I will give you all my water for your food"}\n'
            "  ]\n"
            "}\n\n"
            "Rules:\n"
            "- valence: -1.0 = doesn't want, +1.0 = wants\n"
            "- implied_value: -1.0 = giving away/doesn't want, +1.0 = demanding/wants\n"
            "  Each item in a trade gets its OWN implied_value.\n"
            "- action_type must be one of: refuse, offer, request, protect, accuse, accept\n"
            "- Only extract concrete, specific claims and actions. Skip filler/greetings.\n"
            "- If no claims or actions, return empty arrays.\n\n"
            f'CURRENT message to analyze: "{text[:500]}"'
        )

        parsed = remote_chat_json(user=prompt, max_tokens=400, temperature=0.1)

        # Store in cache (context-aware key)
        if cache is not None and (parsed.get("claims") or parsed.get("actions")):
            cache.put(text, valid_items, parsed, context)

        return parsed

    except Exception as e:
        logger.debug("LLM preference extraction failed: %s", e)
        return {"claims": [], "actions": []}


# ---------------------------------------------------------------------------
# Main tracker
# ---------------------------------------------------------------------------

class PreferenceInferenceTracker:
    """Tracks stated vs revealed preferences to detect strategic deception.

    Computes Action-Claim Divergence: when what they SAY doesn't match
    what they DO, the divergence signal spikes.
    """

    def __init__(
        self,
        use_llm: bool = True,
        valid_items: list[str] | None = None,
        cache: ExtractionCache | None = None,
        cache_only: bool = False,
    ):
        self._mental_state = TargetMentalState()
        self._use_llm = use_llm
        self._valid_items = valid_items
        self._cache = cache
        self._cache_only = cache_only  # no live LLM, only cached results
        self._turn_count = 0
        self._history: list[tuple[str, str]] = []  # (speaker, text) rolling buffer

    def process_turn(
        self, text: str, turn_number: int = 0, opponent_text: str = "",
    ) -> TargetMentalState:
        """Extract claims + actions, update mental state, compute divergence.

        Args:
            text: the target's message this turn
            turn_number: current turn number
            opponent_text: the opponent's most recent message (for context)
        """
        self._turn_count = turn_number or self._turn_count + 1
        turn = self._turn_count

        # Build context string from rolling history (only when opponent_text is provided)
        context = ""
        if opponent_text:
            self._history.append(("Opponent", opponent_text))
            self._history.append(("Speaker", text))
            # Keep last 6 entries (3 exchanges)
            self._history = self._history[-6:]

            if len(self._history) > 1:
                lines = []
                for speaker, msg in self._history[:-1]:
                    lines.append(f'- {speaker}: "{msg[:150]}"')
                context = "\n".join(lines[-5:])

        # --- Step 1: Extract claims and actions ---
        if self._use_llm:
            extracted = _extract_llm(
                text, self._valid_items,
                cache=self._cache, cache_only=self._cache_only,
                context=context,
            )
            claims = extracted.get("claims", [])
            actions = extracted.get("actions", [])
            # Regex fallback if LLM returned nothing
            if not claims and not actions:
                claims = _extract_claims_regex(text)
                actions = _extract_actions_regex(text)
        else:
            claims = _extract_claims_regex(text)
            actions = _extract_actions_regex(text)

        # --- Step 2: Update preferences from claims (latest overwrites) ---
        for claim in claims:
            if not isinstance(claim, dict):
                continue
            raw_item = str(claim.get("item", ""))
            valence = claim.get("valence", 0.0)
            quote = str(claim.get("quote", ""))

            item = _normalize_item(raw_item, self._valid_items)
            if item is None:
                continue

            try:
                valence = max(-1.0, min(1.0, float(valence)))
            except (TypeError, ValueError):
                continue

            if item in self._mental_state.preferences:
                entry = self._mental_state.preferences[item]
                entry.stated_value = valence
                entry.stated_turn = turn
                entry.stated_raw = quote
                entry.has_stated = True
            else:
                self._mental_state.preferences[item] = PreferenceEntry(
                    item=item,
                    stated_value=valence,
                    stated_turn=turn,
                    stated_raw=quote,
                    has_stated=True,
                )

        # --- Step 3: Update preferences from actions (latest overwrites) ---
        for action in actions:
            if not isinstance(action, dict):
                continue
            impacts = action.get("item_impacts", [])
            quote = str(action.get("quote", ""))

            for impact in impacts:
                if not isinstance(impact, dict):
                    continue
                raw_item = str(impact.get("item", ""))
                implied = impact.get("implied_value", 0.0)

                item = _normalize_item(raw_item, self._valid_items)
                if item is None:
                    continue

                try:
                    implied = max(-1.0, min(1.0, float(implied)))
                except (TypeError, ValueError):
                    continue

                if item in self._mental_state.preferences:
                    entry = self._mental_state.preferences[item]
                    entry.revealed_value = implied
                    entry.revealed_turn = turn
                    entry.revealed_raw = quote
                    entry.has_revealed = True
                else:
                    self._mental_state.preferences[item] = PreferenceEntry(
                        item=item,
                        revealed_value=implied,
                        revealed_turn=turn,
                        revealed_raw=quote,
                        has_revealed=True,
                    )

        # --- Step 4: Compute divergence signal (THIS TURN only) ---
        # Only compute divergence for items that were updated this turn,
        # not accumulated max across all turns. This prevents a single
        # past divergence from flagging every subsequent turn.
        this_turn_prefs = {
            k: v for k, v in self._mental_state.preferences.items()
            if (v.has_stated and v.has_revealed)
            and (v.stated_turn == turn or v.revealed_turn == turn)
        }
        max_div, signal_val, rationale = _compute_signal(this_turn_prefs)
        self._mental_state.max_divergence = max_div
        self._mental_state.divergence_signal = SignalReading(
            value=signal_val, signal_reliability=0.7,
        )
        self._mental_state.rationale = rationale

        return self._mental_state

    def reset(self) -> None:
        """Clear for new conversation."""
        self._mental_state = TargetMentalState()
        self._turn_count = 0
        self._history.clear()


# ---------------------------------------------------------------------------
# Cache warming utility
# ---------------------------------------------------------------------------

def warm_cache(
    texts: list[str],
    valid_items: list[str] | None = None,
    cache_name: str = "default",
    rate_limit: float = 0.3,
    contexts: list[str] | None = None,
) -> ExtractionCache:
    """Pre-populate the LLM extraction cache for a list of texts.

    Calls Gemini for any uncached texts, then saves to disk.
    If contexts is provided, each text[i] is cached with contexts[i].
    Returns the cache object for immediate use.
    """
    import time

    cache = ExtractionCache(cache_name)
    total = len(texts)
    hits = 0
    calls = 0

    for i, text in enumerate(texts):
        ctx = contexts[i] if contexts and i < len(contexts) else ""
        if cache.get(text, valid_items, ctx) is not None:
            hits += 1
            continue

        result = _extract_llm(text, valid_items, cache=None, context=ctx)
        if result.get("claims") or result.get("actions"):
            cache.put(text, valid_items, result, ctx)
        calls += 1

        if calls % 50 == 0:
            print(f"  [{calls}/{total - hits}] LLM calls made, {hits} cache hits...")
            cache.save()  # periodic save

        if rate_limit > 0:
            time.sleep(rate_limit)

    cache.save()
    print(f"Cache warmed: {hits} hits, {calls} new LLM calls, {len(cache)} total entries")
    return cache
