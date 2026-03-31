"""LLM Sensor Layer (Layer 2).

Extracts 5 orthogonal behavioral signals from a single line of text.
Uses the remote LLM (Gemini) via a single consolidated call for speed.
Falls back to local LM Studio if no remote is configured.

The LLM never sees history — it only scores the current utterance.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from btom_engine.schema import ExtractedSignals, SignalReading

logger = logging.getLogger(__name__)

# Store last debug info for the cockpit to display
last_debug_info: dict[str, dict[str, str]] = {}

# ---------------------------------------------------------------------------
# Consolidated prompt — all 5 signals in one call
# ---------------------------------------------------------------------------

_CONSOLIDATED_PROMPT = """You are a clinical behavioral analysis API. Output raw JSON only. No prose, no markdown.

Analyze the following text and score these 5 independent behavioral signals.
Each signal is scored 0.0-1.0 with a reliability estimate 0.0-1.0.

SIGNALS:

1. syntactic_fragmentation — Structural breakdown ONLY:
   - Broken grammar, abandoned clauses, self-interruption, incomplete thoughts
   - NOT anger, hostility, or emotional intensity
   - Short neutral confirmations ('ok', 'yes') = 0.0-0.1

2. defensive_justification — Excuse-making and pushback:
   - Explanation ('I did that because...'), rationalization, deflecting blame
   - Pushback against scrutiny ('I already explained...')
   - Pure insults/anger WITHOUT explanation = LOW

3. emotional_intensity — Affective force:
   - Anger, hostility, loaded punctuation (!!!, ???, CAPS)
   - Insults, profanity, confrontational language
   - 0.0-0.1 calm, 0.2-0.4 mild, 0.5-0.7 heated, 0.8-1.0 explosive

4. evasive_deflection — Avoiding information exchange:
   - Explicit refusal, redirecting, shutting down inquiry
   - Answering a question with a question
   - Explanations/excuses are NOT evasion (score LOW)
   - Pure anger is NOT evasion unless combined with refusal

5. direct_answer_compliance — Providing concrete information:
   - Specific facts, names, times, locations
   - Directly addressing what was asked
   - Vague reassurance or generic agreement = LOW
   - 'I sent the report at 3pm to Sarah' = HIGH

IMPORTANT: You are scoring LINGUISTIC and BEHAVIORAL signals only.
You are NOT judging whether the content is true or false.
A bold, confident lie scores the same as a bold, confident truth on these axes.
Truth-checking is handled by a separate verification system.

Text: \"{text}\"

Return ONLY this JSON (no other text). Include a brief rationale for each score:
{{"syntactic_fragmentation": {{"value": <float>, "reliability": <float>, "rationale": "<why>"}}, "defensive_justification": {{"value": <float>, "reliability": <float>, "rationale": "<why>"}}, "emotional_intensity": {{"value": <float>, "reliability": <float>, "rationale": "<why>"}}, "evasive_deflection": {{"value": <float>, "reliability": <float>, "rationale": "<why>"}}, "direct_answer_compliance": {{"value": <float>, "reliability": <float>, "rationale": "<why>"}}}}"""

SIGNAL_NAMES = [
    "syntactic_fragmentation",
    "defensive_justification",
    "emotional_intensity",
    "evasive_deflection",
    "direct_answer_compliance",
]


def _sanitize_llm_json(raw: str) -> str:
    """Strip markdown code blocks and conversational filler from LLM output."""
    raw = re.sub(r"```(?:json)?\s*", "", raw)
    raw = raw.replace("```", "")
    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end != -1 and end > start:
        return raw[start : end + 1]
    return raw.strip()


def _parse_signal(data: dict, name: str) -> dict[str, Any]:
    """Extract value/reliability/rationale from a parsed signal dict."""
    if isinstance(data, dict):
        return {
            "value": max(0.0, min(1.0, float(data.get("value", 0.0)))),
            "reliability": max(0.0, min(1.0, float(data.get("reliability", 0.8)))),
            "rationale": str(data.get("rationale", "")),
        }
    return {"value": 0.0, "reliability": 0.0, "rationale": ""}


def extract_signals_sync(text: str) -> ExtractedSignals:
    """Extract all 5 signals in a single LLM call.

    Routes to Gemini if available, falls back to local.
    This is a pure perception layer — no policy, normalization, or overrides.
    """
    global last_debug_info
    last_debug_info = {}

    from btom_engine.remote_llm import remote_chat

    prompt = _CONSOLIDATED_PROMPT.format(text=text)

    try:
        result = remote_chat(
            user=prompt,
            max_tokens=1500,
            temperature=0.1,
        )
        raw = result["content"]
        provider = result.get("provider", "unknown")

        logger.info("SENSOR [%s]: %s", provider, raw[:300])

        sanitized = _sanitize_llm_json(raw)
        parsed = json.loads(sanitized)

        # Extract each signal
        signals = {}
        for name in SIGNAL_NAMES:
            sig_data = parsed.get(name, {})
            signals[name] = _parse_signal(sig_data, name)
            last_debug_info[name] = {
                "raw_response": json.dumps(sig_data),
                "parsed": str(signals[name]),
                "rationale": signals[name].get("rationale", ""),
                "error": "",
            }

        return ExtractedSignals(
            syntactic_fragmentation=SignalReading(
                value=signals["syntactic_fragmentation"]["value"],
                signal_reliability=signals["syntactic_fragmentation"]["reliability"],
            ),
            defensive_justification=SignalReading(
                value=signals["defensive_justification"]["value"],
                signal_reliability=signals["defensive_justification"]["reliability"],
            ),
            emotional_intensity=SignalReading(
                value=signals["emotional_intensity"]["value"],
                signal_reliability=signals["emotional_intensity"]["reliability"],
            ),
            evasive_deflection=SignalReading(
                value=signals["evasive_deflection"]["value"],
                signal_reliability=signals["evasive_deflection"]["reliability"],
            ),
            direct_answer_compliance=SignalReading(
                value=signals["direct_answer_compliance"]["value"],
                signal_reliability=signals["direct_answer_compliance"]["reliability"],
            ),
        )

    except Exception as exc:
        logger.warning("Sensor failed: %s", exc)
        for name in SIGNAL_NAMES:
            last_debug_info[name] = {"error": str(exc), "raw_response": "", "parsed": ""}

        # Return zero signals on failure
        zero = SignalReading(value=0.0, signal_reliability=0.0)
        return ExtractedSignals(
            syntactic_fragmentation=zero,
            defensive_justification=zero,
            emotional_intensity=zero,
            evasive_deflection=zero,
            direct_answer_compliance=zero,
        )
