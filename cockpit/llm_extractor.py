"""LLM Transcript Extractor — turns messy pasted text into structured turns.

The LLM is used ONLY as a data extractor. It must NOT:
- assign User/Target/Other roles
- infer motives or hypotheses
- paraphrase or summarize
- rewrite message meaning
- invent content
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

@dataclass
class ExtractedTurn:
    """One turn extracted by the LLM."""
    speaker_raw: str = ""
    speaker_canonical: str = ""  # cleaned/canonicalized speaker name
    text_full: str = ""
    text_preview: str = ""      # truncated for display only
    timestamp: Optional[str] = None
    confidence: float = 0.5


@dataclass
class ExtractionResult:
    """Result of LLM transcript extraction."""
    turns: list[ExtractedTurn] = field(default_factory=list)
    unparsed_blocks: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    raw_speakers: list[str] = field(default_factory=list)
    extraction_source: str = "llm"
    success: bool = False


# ---------------------------------------------------------------------------
# LLM prompt
# ---------------------------------------------------------------------------

_EXTRACTION_PROMPT = """You are a transcript extraction tool. Output raw JSON only. No prose before or after the JSON.

Convert the following messy raw copied chat/message text into an ordered JSON transcript.

STRICT RULES:
1. Preserve the EXACT wording of actual messages. Do NOT paraphrase, summarize, clean up slang, or rewrite anything.
2. Do NOT assign User, Target, or Other roles. Only extract raw speaker names as they appear.
3. Ignore obvious UI chrome: profile buttons, reaction controls, "View profile", "sent the following message", connection badges, helper text, platform labels. These are NOT part of the conversation.
4. URLs that are part of a message (shared by the speaker) SHOULD be included in text_full.
5. Extract timestamps as separate metadata when possible. Do NOT include timestamps in text_full.
6. Each turn should contain ONE speaker's complete message. Merge continuation lines from the same speaker.
7. If material is ambiguous (cannot tell if it's message text or UI chrome), place it in unparsed_blocks.
8. Lower confidence when extraction is uncertain.
9. Do NOT silently drop text. If you exclude something, note it in warnings.
10. Preserve original message order.

Raw text to extract:
\"\"\"
{raw_text}
\"\"\"

Return ONLY valid JSON in this exact format:
{{"turns": [{{"speaker_raw": "<exact name>", "text_full": "<exact message text>", "timestamp": "<if found, else null>", "confidence": <0.0-1.0>}}], "unparsed_blocks": ["<any ambiguous material>"], "warnings": ["<any notes>"]}}"""


# ---------------------------------------------------------------------------
# Junk speaker patterns (for validation)
# ---------------------------------------------------------------------------

_JUNK_SPEAKER_PATTERNS = [
    re.compile(r"^https?://", re.IGNORECASE),
    re.compile(r"^\d{1,2}[:/]\d{2}", re.IGNORECASE),  # timestamp-only
    re.compile(r"^(React with|Reply|Share|Report|Save|Permalink|View|Subscribe|Read more)$", re.IGNORECASE),
    re.compile(r"^\s*$"),  # empty/whitespace
    re.compile(r"^\d+\s*(likes?|upvotes?|views?|retweets?)$", re.IGNORECASE),
    re.compile(r"^(Call with|View|Image|Photo|Video|Attachment|File|Link)\s", re.IGNORECASE),
    re.compile(r"^(sent the following|View .* profile)", re.IGNORECASE),
    re.compile(r"^\d+$"),  # bare numbers
]


def _is_junk_speaker(name: str) -> bool:
    """Check if a speaker name is obviously junk."""
    name = name.strip()
    if not name or len(name) > 80:
        return True
    return any(p.match(name) for p in _JUNK_SPEAKER_PATTERNS)


# ---------------------------------------------------------------------------
# Speaker canonicalization
# ---------------------------------------------------------------------------

def canonicalize_speaker(raw: str) -> str:
    """Strip timestamps, pronouns, profile chrome from speaker name.

    "Amir Fischer (He/Him) 9:27 AM" -> "Amir Fischer"
    "View Amir's profileAmir Fischer" -> "Amir Fischer"
    "Jeff Vallyath (He/Him) 9" -> "Jeff Vallyath"
    """
    s = raw.strip()

    # Strip "View X's profile" prefix
    s = re.sub(r"^View\s+.*?(?:profile|Profile)\s*", "", s)

    # Strip "sent the following message at ..." suffix
    s = re.sub(r"\s*sent the following message.*$", "", s, flags=re.IGNORECASE)

    # Strip pronoun tags: (He/Him), (She/Her), (They/Them), etc.
    s = re.sub(r"\s*\([Hh]e/?[Hh]i[ms]?\)", "", s)
    s = re.sub(r"\s*\([Ss]he/?[Hh]er?\)", "", s)
    s = re.sub(r"\s*\([Tt]hey/?[Tt]hem?\)", "", s)
    s = re.sub(r"\s*\([A-Za-z]+/[A-Za-z]+\)", "", s)  # catch-all pronoun pattern

    # Strip trailing timestamps: "9:27 AM", "11", "9", "12:00 PM"
    s = re.sub(r"\s+\d{1,2}[:/]\d{2}\s*[APap]?[Mm]?\s*$", "", s)
    s = re.sub(r"\s+\d{1,2}\s*$", "", s)  # trailing bare number

    # Strip trailing commas, periods, colons
    s = re.sub(r"[,.:;]+$", "", s)

    # Collapse whitespace
    s = re.sub(r"\s+", " ", s).strip()

    return s if s else raw.strip()


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------

def _call_llm_extract(raw_text: str) -> tuple[dict, str]:
    """Call the LLM for transcript extraction. Routes to remote if available."""
    from btom_engine.remote_llm import remote_chat

    prompt = _EXTRACTION_PROMPT.format(raw_text=raw_text[:4000])

    result = remote_chat(user=prompt, max_tokens=3000, temperature=0.1)
    raw_output = result["content"]

    # Extract JSON
    cleaned = re.sub(r"```(?:json)?\s*", "", raw_output)
    cleaned = cleaned.replace("```", "")
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end > start:
        cleaned = cleaned[start:end + 1]

    parsed = json.loads(cleaned)
    return parsed, raw_output


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_extraction(parsed: dict, raw_text_len: int = 0) -> tuple[list[ExtractedTurn], list[str], list[str]]:
    """Validate and convert LLM output to ExtractedTurns.

    Returns (valid_turns, unparsed_blocks, warnings).
    Rejects junk speakers, empty texts, malformed entries.
    """
    warnings = list(parsed.get("warnings", []))
    unparsed = [str(b)[:200] for b in parsed.get("unparsed_blocks", []) if b]

    raw_turns = parsed.get("turns", [])
    if not isinstance(raw_turns, list):
        return [], unparsed, ["LLM returned non-list turns field"]

    valid: list[ExtractedTurn] = []
    for i, t in enumerate(raw_turns):
        if not isinstance(t, dict):
            warnings.append(f"turn {i}: not a dict, skipped")
            continue

        speaker = str(t.get("speaker_raw", "")).strip()
        text = str(t.get("text_full", "")).strip()
        ts = t.get("timestamp")
        conf = float(t.get("confidence", 0.5))

        if not speaker:
            warnings.append(f"turn {i}: empty speaker, skipped")
            continue
        if not text:
            warnings.append(f"turn {i}: empty text for speaker '{speaker}', skipped")
            continue

        # Canonicalize speaker
        canonical = canonicalize_speaker(speaker)

        # Reject junk after canonicalization
        if _is_junk_speaker(canonical):
            warnings.append(f"turn {i}: junk speaker '{speaker}' -> '{canonical}', skipped")
            continue

        valid.append(ExtractedTurn(
            speaker_raw=speaker,
            speaker_canonical=canonical,
            text_full=text,
            text_preview=text[:80],
            timestamp=str(ts) if ts else None,
            confidence=max(0.0, min(1.0, conf)),
        ))

    # Sanity checks
    if len(valid) == 0 and raw_text_len > 100:
        warnings.append("no valid turns extracted from non-trivial input")
    elif len(valid) == 1 and raw_text_len > 500:
        warnings.append("only 1 turn extracted from long input — possible merge error")

    return valid, unparsed, warnings


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_transcript(raw_text: str) -> ExtractionResult:
    """Extract a transcript from raw pasted text using the LLM.

    Returns ExtractionResult. Falls back gracefully on any failure.
    """
    result = ExtractionResult()

    if not raw_text or not raw_text.strip():
        result.warnings.append("empty input")
        return result

    try:
        parsed, raw_output = _call_llm_extract(raw_text)
        print(f"LLM EXTRACTOR RAW: {raw_output[:200]}...")

        turns, unparsed, warnings = validate_extraction(parsed, len(raw_text))

        result.turns = turns
        result.unparsed_blocks = unparsed
        result.warnings = warnings
        result.raw_speakers = sorted(set(t.speaker_canonical for t in turns))
        result.success = len(turns) > 0

    except Exception as e:
        result.warnings.append(f"LLM extraction failed: {type(e).__name__}: {e}")
        result.success = False
        logger.warning("LLM transcript extraction failed: %s", e)

    return result


# ---------------------------------------------------------------------------
# Replay queue conversion
# ---------------------------------------------------------------------------

def extraction_to_replay_queue(
    turns: list[ExtractedTurn],
    role_map: dict[str, str],
) -> list[dict]:
    """Convert extracted turns to replay queue format using manual role map.

    role_map: {speaker_canonical -> "User"/"Target"/"Other"}
    Always uses text_full for replay, never truncated preview.
    """
    queue = []
    for turn in turns:
        # Try canonical first, then raw, then Other
        role = role_map.get(turn.speaker_canonical, role_map.get(turn.speaker_raw, "Other"))
        queue.append({
            "speaker": role,                    # normalized role for engine routing
            "text": turn.text_full,             # FULL text — invariant
            "speaker_raw": turn.speaker_raw,
            "speaker_canonical": turn.speaker_canonical,
        })
    return queue
