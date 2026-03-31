"""Transcript Parser — hybrid deterministic + LLM-repair parser for pasted transcripts.

Deterministic-first:
  1. Try common inline formats (Name: message, [ts] Name: message, Name - message)
  2. Try header/body format (Name — timestamp \n message)
  3. Score parse quality
  4. LLM repair only when quality is poor and user opts in

Role mapping is always deterministic — LLM never assigns User/Target.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class ParsedTurn:
    """One parsed turn from raw transcript text."""
    speaker_raw: str = ""
    speaker_role: str = "Other"
    text: str = ""
    timestamp: Optional[str] = None
    line_number: int = 0
    parse_source: str = "deterministic"
    confidence: float = 1.0


@dataclass
class ParseQuality:
    """Quality assessment of a parse result."""
    score: float = 1.0          # 0.0 = terrible, 1.0 = perfect
    warnings: list[str] = field(default_factory=list)
    needs_repair: bool = False


@dataclass
class ParseResult:
    """Result of parsing a raw transcript."""
    turns: list[ParsedTurn] = field(default_factory=list)
    dropped_lines: list[str] = field(default_factory=list)
    raw_speakers: list[str] = field(default_factory=list)
    user_count: int = 0
    target_count: int = 0
    other_count: int = 0
    quality: ParseQuality = field(default_factory=ParseQuality)
    parse_source: str = "deterministic"


# ---------------------------------------------------------------------------
# Alias normalization
# ---------------------------------------------------------------------------

def _normalize_alias(name: str) -> str:
    """Normalize for comparison: lowercase, strip, collapse whitespace, strip brackets/punctuation."""
    name = name.strip().lower()
    name = re.sub(r"\s+", " ", name)
    # Strip trailing punctuation
    name = re.sub(r"[,.:;!?]+$", "", name)
    return name


def _normalize_for_matching(name: str) -> str:
    """Extra normalization for fuzzy alias matching: strip bracket tags, collapse."""
    norm = _normalize_alias(name)
    # Strip bracket tags like [BS], (admin), etc.
    norm = re.sub(r"\s*[\[\(][^\]\)]*[\]\)]", "", norm).strip()
    return norm


def _match_alias(speaker_raw: str, alias_set: set[str], alias_base_set: set[str]) -> bool:
    """Check if speaker matches any alias (exact normalized or bracket-stripped)."""
    norm = _normalize_alias(speaker_raw)
    if norm in alias_set:
        return True
    base = _normalize_for_matching(speaker_raw)
    return base in alias_base_set and base != ""


# ---------------------------------------------------------------------------
# Deterministic patterns
# ---------------------------------------------------------------------------

# Inline patterns (speaker and message on same line)
_INLINE_PATTERNS = [
    # [timestamp] Name: message
    re.compile(r"^\[([^\]]+)\]\s*([^:\-—]+?)\s*[:]\s*(.+)$"),
    # [timestamp] Name - message
    re.compile(r"^\[([^\]]+)\]\s*([^:\-—]+?)\s*[-—]\s*(.+)$"),
    # Name: message (allow brackets in names for tags like [BS])
    re.compile(r"^([^:\-—]{1,50}?)\s*[:]\s*(.+)$"),
    # Name - message
    re.compile(r"^([^:\-—]{1,50}?)\s*[-—]\s*(.+)$"),
]

# Header/body pattern: "Name — timestamp" or "Name, timestamp" on its own line
_HEADER_PATTERN = re.compile(
    r"^([A-Za-z0-9_\[\]\(\)\s]{2,50}?)\s*[—\-–,]\s*"
    r"(\d{1,2}[:/]\d{2}(?:\s*[APap][Mm])?(?:\s.*)?)\s*$"
)

# Alternate header: "Name" alone on a line (only if known speaker)
_SPEAKER_ONLY = re.compile(r"^([A-Za-z0-9_\[\]\(\)\s]{2,50})\s*$")


def _try_parse_inline(line: str) -> dict | None:
    """Try inline patterns."""
    for pattern in _INLINE_PATTERNS:
        m = pattern.match(line)
        if m:
            groups = m.groups()
            if len(groups) == 3:
                return {"timestamp": groups[0].strip(), "speaker": groups[1].strip(), "text": groups[2].strip()}
            elif len(groups) == 2:
                return {"speaker": groups[0].strip(), "text": groups[1].strip()}
    return None


def _try_parse_header(line: str) -> dict | None:
    """Try header/body pattern (speaker + timestamp, no message)."""
    m = _HEADER_PATTERN.match(line)
    if m:
        return {"speaker": m.group(1).strip(), "timestamp": m.group(2).strip(), "type": "header"}
    return None


# ---------------------------------------------------------------------------
# Main parser
# ---------------------------------------------------------------------------

def parse_transcript(
    raw_text: str,
    user_aliases: list[str] | None = None,
    target_aliases: list[str] | None = None,
    skip_normalization: bool = False,
) -> ParseResult:
    """Parse raw transcript with hybrid deterministic logic.

    Flow: source normalization -> inline/header parsing -> role mapping.
    Role mapping is always deterministic via aliases.
    """
    # Source normalization: strip platform chrome before parsing
    if not skip_normalization:
        from source_normalizer import normalize_source
        norm = normalize_source(raw_text)
        raw_text = norm.normalized_text

    user_norms = {_normalize_alias(a) for a in (user_aliases or []) if a.strip()}
    target_norms = {_normalize_alias(a) for a in (target_aliases or []) if a.strip()}
    user_bases = {_normalize_for_matching(a) for a in (user_aliases or []) if a.strip()}
    target_bases = {_normalize_for_matching(a) for a in (target_aliases or []) if a.strip()}

    result = ParseResult()
    seen_speakers: set[str] = set()
    lines = raw_text.split("\n")

    current_turn: ParsedTurn | None = None
    pending_header: dict | None = None  # header waiting for body text

    for line_num, raw_line in enumerate(lines, 1):
        line = raw_line.strip()
        if not line:
            continue

        # Try header parse FIRST (speaker + timestamp on own line, no message body)
        header = _try_parse_header(line)
        if header:
            _flush_turn(current_turn, result)
            current_turn = None
            _flush_header(pending_header, result)
            pending_header = {"speaker": header["speaker"], "timestamp": header.get("timestamp"), "line_num": line_num}
            seen_speakers.add(header["speaker"])
            continue

        # Try inline parse (speaker and message on same line)
        inline = _try_parse_inline(line)
        if inline:
            _flush_turn(current_turn, result)
            _flush_header(pending_header, result)
            pending_header = None

            current_turn = _make_turn(
                inline["speaker"], inline.get("text", ""), inline.get("timestamp"),
                line_num, user_norms, target_norms, user_bases, target_bases,
            )
            seen_speakers.add(inline["speaker"])
            continue

        # Body text after a header
        if pending_header and not current_turn:
            current_turn = _make_turn(
                pending_header["speaker"], line, pending_header.get("timestamp"),
                pending_header["line_num"], user_norms, target_norms, user_bases, target_bases,
            )
            pending_header = None
            continue

        # Continuation of current turn
        if current_turn:
            current_turn.text += " " + line
            continue

        # Unparsable
        result.dropped_lines.append(f"L{line_num}: {line[:60]}")

    # Flush remaining
    _flush_turn(current_turn, result)
    _flush_header(pending_header, result)

    # Stats
    result.raw_speakers = sorted(seen_speakers)
    result.user_count = sum(1 for t in result.turns if t.speaker_role == "User")
    result.target_count = sum(1 for t in result.turns if t.speaker_role == "Target")
    result.other_count = sum(1 for t in result.turns if t.speaker_role == "Other")

    # Quality scoring
    result.quality = _assess_quality(result, len(lines))
    result.parse_source = "deterministic"

    return result


def _make_turn(
    speaker: str, text: str, timestamp: str | None, line_num: int,
    user_norms: set, target_norms: set, user_bases: set, target_bases: set,
) -> ParsedTurn:
    """Create a ParsedTurn with role mapping."""
    if _match_alias(speaker, user_norms, user_bases):
        role = "User"
    elif _match_alias(speaker, target_norms, target_bases):
        role = "Target"
    else:
        role = "Other"
    return ParsedTurn(speaker_raw=speaker, speaker_role=role, text=text.strip(),
                      timestamp=timestamp, line_number=line_num)


def _flush_turn(turn: ParsedTurn | None, result: ParseResult) -> None:
    if turn and turn.text.strip():
        result.turns.append(turn)


def _flush_header(header: dict | None, result: ParseResult) -> None:
    """Flush an orphaned header (no body text followed)."""
    if header:
        result.dropped_lines.append(f"L{header['line_num']}: orphan header '{header['speaker']}'")


# ---------------------------------------------------------------------------
# Parse quality scoring
# ---------------------------------------------------------------------------

def _assess_quality(result: ParseResult, total_lines: int) -> ParseQuality:
    """Score parse quality and flag potential issues."""
    q = ParseQuality(score=1.0)

    if total_lines == 0:
        q.score = 0.0
        return q

    non_blank = max(1, total_lines - sum(1 for _ in [] if True))  # approx

    # High drop rate
    drop_rate = len(result.dropped_lines) / max(len(result.turns) + len(result.dropped_lines), 1)
    if drop_rate > 0.3:
        q.score -= 0.4
        q.warnings.append(f"high_drop_rate: {drop_rate:.0%} of lines dropped")

    # Only one speaker when transcript seems longer
    if len(result.raw_speakers) <= 1 and len(result.turns) > 3:
        q.score -= 0.3
        q.warnings.append("single_speaker: only one speaker detected in multi-turn transcript")

    # Suspiciously long messages (accidental concatenation)
    for t in result.turns:
        if len(t.text) > 500:
            q.score -= 0.1
            q.warnings.append(f"long_message: turn L{t.line_number} has {len(t.text)} chars")
            break

    # No turns parsed at all
    if len(result.turns) == 0 and total_lines > 2:
        q.score -= 0.6
        q.warnings.append("no_turns: no turns parsed from non-empty input")

    q.score = max(0.0, min(1.0, q.score))
    q.needs_repair = q.score < 0.6

    return q


# ---------------------------------------------------------------------------
# LLM repair path
# ---------------------------------------------------------------------------

_LLM_REPAIR_PROMPT = """You are a transcript parser. Output raw JSON only. No prose.

Convert this raw transcript text into ordered speaker turns.

RULES:
- Preserve EXACT wording from the raw text. Do not paraphrase, summarize, or clean up.
- Extract: speaker name, message text, timestamp if present.
- If a message spans multiple lines, merge them into one text field.
- If a line cannot be attributed to a speaker, put it in unparsed_lines.
- Do NOT infer who is "User" or "Target". Just extract raw speaker names.
- If boundaries are ambiguous, return lower confidence.

Raw transcript:
\"\"\"
{raw_text}
\"\"\"

Return ONLY:
{{"turns": [{{"speaker_raw": "<name>", "text": "<exact text>", "timestamp": "<if present>", "confidence": <0.0-1.0>}}], "unparsed_lines": ["<line>"], "notes": ["<any notes>"]}}"""


def repair_with_llm(
    raw_text: str,
    user_aliases: list[str] | None = None,
    target_aliases: list[str] | None = None,
) -> ParseResult:
    """Use LLM to parse messy transcript text. Fallback path only.

    Role mapping is still deterministic after LLM extraction.
    """
    from btom_engine.remote_llm import remote_chat

    user_norms = {_normalize_alias(a) for a in (user_aliases or []) if a.strip()}
    target_norms = {_normalize_alias(a) for a in (target_aliases or []) if a.strip()}
    user_bases = {_normalize_for_matching(a) for a in (user_aliases or []) if a.strip()}
    target_bases = {_normalize_for_matching(a) for a in (target_aliases or []) if a.strip()}

    prompt = _LLM_REPAIR_PROMPT.format(raw_text=raw_text[:3000])

    try:
        result = remote_chat(user=prompt, max_tokens=2000, temperature=0.1)
        raw_output = result["content"]

        # Extract JSON
        raw_output = re.sub(r"```(?:json)?\s*", "", raw_output).replace("```", "")
        start = raw_output.find("{")
        end = raw_output.rfind("}")
        if start != -1 and end > start:
            raw_output = raw_output[start:end + 1]
        parsed = json.loads(raw_output)

        return _llm_output_to_result(parsed, user_norms, target_norms, user_bases, target_bases)

    except Exception as e:
        logger.warning("LLM transcript repair failed: %s", e)
        result = ParseResult(parse_source="llm_failed")
        result.quality = ParseQuality(score=0.0, warnings=[f"LLM repair failed: {e}"], needs_repair=True)
        return result


def _llm_output_to_result(
    parsed: dict,
    user_norms: set, target_norms: set, user_bases: set, target_bases: set,
) -> ParseResult:
    """Validate and convert LLM output to ParseResult. Role mapping is deterministic."""
    result = ParseResult(parse_source="llm_repair")
    seen: set[str] = set()

    turns = parsed.get("turns", [])
    if not isinstance(turns, list):
        result.quality = ParseQuality(score=0.0, warnings=["LLM returned invalid turns format"])
        return result

    for i, t in enumerate(turns):
        if not isinstance(t, dict):
            continue
        speaker = str(t.get("speaker_raw", "")).strip()
        text = str(t.get("text", "")).strip()
        ts = t.get("timestamp")
        conf = float(t.get("confidence", 0.5))

        if not speaker or not text:
            continue

        if _match_alias(speaker, user_norms, user_bases):
            role = "User"
        elif _match_alias(speaker, target_norms, target_bases):
            role = "Target"
        else:
            role = "Other"

        result.turns.append(ParsedTurn(
            speaker_raw=speaker, speaker_role=role, text=text,
            timestamp=str(ts) if ts else None, line_number=i + 1,
            parse_source="llm_repair", confidence=conf,
        ))
        seen.add(speaker)

    for line in parsed.get("unparsed_lines", []):
        result.dropped_lines.append(str(line)[:80])

    result.raw_speakers = sorted(seen)
    result.user_count = sum(1 for t in result.turns if t.speaker_role == "User")
    result.target_count = sum(1 for t in result.turns if t.speaker_role == "Target")
    result.other_count = sum(1 for t in result.turns if t.speaker_role == "Other")
    result.quality = ParseQuality(score=0.7, warnings=["LLM-repaired parse"])

    return result


# ---------------------------------------------------------------------------
# Replay queue
# ---------------------------------------------------------------------------

def to_replay_queue(parsed: ParseResult) -> list[dict]:
    """Convert ParseResult to the replay queue format the cockpit expects."""
    queue = []
    for turn in parsed.turns:
        queue.append({
            "speaker": turn.speaker_role,
            "text": turn.text,
            "speaker_raw": turn.speaker_raw,
        })
    return queue


# ---------------------------------------------------------------------------
# Block-based parse (new primary path)
# ---------------------------------------------------------------------------

def parse_transcript_blocks(
    raw_text: str,
    user_aliases: list[str] | None = None,
    target_aliases: list[str] | None = None,
) -> ParseResult:
    """Parse using the block normalizer pipeline.

    Falls back to line-based parse if block normalization produces nothing.
    """
    from block_normalizer import normalize_blocks

    norm = normalize_blocks(raw_text)

    if not norm.turns:
        # Fallback to line-based parser
        return parse_transcript(raw_text, user_aliases, target_aliases, skip_normalization=True)

    # Build alias sets for role mapping
    user_norms = {_normalize_alias(a) for a in (user_aliases or []) if a.strip()}
    target_norms = {_normalize_alias(a) for a in (target_aliases or []) if a.strip()}
    user_bases = {_normalize_for_matching(a) for a in (user_aliases or []) if a.strip()}
    target_bases = {_normalize_for_matching(a) for a in (target_aliases or []) if a.strip()}

    result = ParseResult()
    seen: set[str] = set()

    for turn in norm.turns:
        # Role mapping
        if _match_alias(turn.speaker_raw, user_norms, user_bases):
            role = "User"
        elif _match_alias(turn.speaker_raw, target_norms, target_bases):
            role = "Target"
        else:
            role = "Other"

        result.turns.append(ParsedTurn(
            speaker_raw=turn.speaker_raw,
            speaker_role=role,
            text=turn.text_full,  # FULL text, not preview
            timestamp=turn.timestamp,
            parse_source="block",
            confidence=turn.confidence,
        ))
        seen.add(turn.speaker_raw)

    for db in norm.dropped_blocks:
        result.dropped_lines.append(f"[{db.block_type}] {db.raw_text[:50]}")

    result.raw_speakers = sorted(seen)
    result.user_count = sum(1 for t in result.turns if t.speaker_role == "User")
    result.target_count = sum(1 for t in result.turns if t.speaker_role == "Target")
    result.other_count = sum(1 for t in result.turns if t.speaker_role == "Other")
    result.parse_source = "block"

    # Quality
    result.quality = _assess_quality(result, len(raw_text.split("\n")))
    result.quality.warnings.extend(norm.warnings)

    return result


def block_replay_queue(parsed: ParseResult) -> list[dict]:
    """Convert block-parsed result to replay queue. Uses text_full."""
    queue = []
    for turn in parsed.turns:
        queue.append({
            "speaker": turn.speaker_role,
            "text": turn.text,  # This is text_full from block normalization
            "speaker_raw": turn.speaker_raw,
        })
    return queue
