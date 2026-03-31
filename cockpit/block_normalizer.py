"""Block-based transcript normalizer.

Pipeline: raw text -> block segmentation -> classification -> turn assembly

Treats pasted text as a semi-structured document, not a flat line stream.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Block schemas
# ---------------------------------------------------------------------------

@dataclass
class BlockCandidate:
    """A candidate block from segmentation."""
    block_id: int = 0
    raw_text: str = ""
    line_start: int = 0
    line_end: int = 0


@dataclass
class ClassifiedBlock:
    """A classified block with type and extracted fields."""
    block_id: int = 0
    raw_text: str = ""
    block_type: str = "unknown"  # speaker_header, message_body, timestamp, date_separator,
                                  # profile_chrome, attachment_or_link_card, reaction_or_ui_junk, unknown
    confidence: float = 0.5
    warnings: list[str] = field(default_factory=list)
    # Extracted fields
    speaker_raw: str = ""
    timestamp: str = ""
    url: str = ""


@dataclass
class NormalizedTurn:
    """A fully assembled turn from classified blocks."""
    speaker_raw: str = ""
    speaker_role: str = "Other"
    timestamp: Optional[str] = None
    text_full: str = ""           # full text for engine/replay
    text_preview: str = ""        # truncated for UI display
    attachments: list[str] = field(default_factory=list)
    confidence: float = 0.5
    warnings: list[str] = field(default_factory=list)
    raw_blocks: list[int] = field(default_factory=list)  # block_ids


@dataclass
class BlockNormalizationResult:
    """Complete output of block normalization."""
    turns: list[NormalizedTurn] = field(default_factory=list)
    blocks: list[ClassifiedBlock] = field(default_factory=list)
    dropped_blocks: list[ClassifiedBlock] = field(default_factory=list)
    source_detected: str = "generic"
    warnings: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Classification patterns
# ---------------------------------------------------------------------------

# Speaker header: name-like text followed by optional timestamp separator
_SPEAKER_HEADER_PATTERNS = [
    # Name — timestamp  or  Name - timestamp  or  Name, timestamp (numeric)
    re.compile(
        r"^([A-Za-z][A-Za-z0-9_\s\[\]\(\)/]{1,50}?)\s*[—\-–,]\s*"
        r"(\d{1,2}[:/]\d{2}(?:\s*[APap][Mm])?(?:\s.*)?)\s*$"
    ),
    # Name - Today/Yesterday at time (Discord-style)
    re.compile(
        r"^([A-Za-z][A-Za-z0-9_\s\[\]\(\)/]{1,50}?)\s*[—\-–]\s*"
        r"((Today|Yesterday)\s+at\s+\d{1,2}:\d{2}\s*[APap]?[Mm]?.*)\s*$",
        re.IGNORECASE,
    ),
    # Name (with optional tags): just a short name line
    re.compile(r"^([A-Za-z][A-Za-z0-9_\s\[\]\(\)/]{1,40})\s*$"),
]

# Timestamp-only line
_TIMESTAMP_ONLY = re.compile(
    r"^\s*(\d{1,2}[:/]\d{2}(?:\s*[APap][Mm])?)\s*$"
    r"|^\s*(Today at|Yesterday at)\s+\d{1,2}:\d{2}\s*[APap]?[Mm]?\s*$"
    r"|^\s*\d{1,2}/\d{1,2}/\d{2,4}\s*,?\s*\d{1,2}:\d{2}\s*[APap]?[Mm]?\s*$",
    re.IGNORECASE,
)

# Date separator
_DATE_SEPARATOR = re.compile(
    r"^\s*[-—–]+\s*$"
    r"|^\s*(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)"
    r"|^\s*\d{1,2}\s+(January|February|March|April|May|June|July|August|September|October|November|December)"
    r"|^\s*(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}",
    re.IGNORECASE,
)

# URL / link card
_URL_LINE = re.compile(r"^\s*https?://\S+\s*$")
_LINK_CARD = re.compile(
    r"^\s*(Image|Photo|Video|Attachment|File|Document|Link|Preview)[\s:]*",
    re.IGNORECASE,
)

# Profile chrome
_PROFILE_CHROME = re.compile(
    r"(sent the following message|View\s+.*profile|^\s*\d+\s*(connections?|followers?|following)"
    r"|^\s*(1st|2nd|3rd)\s*\|"
    r"|^\s*\d+\s*(mo|yr|wk|hr|min|day)s?\s+ago\s*$)",
    re.IGNORECASE | re.MULTILINE,
)

# Reaction / UI junk
_UI_JUNK = re.compile(
    r"^\s*(\d+\s*(upvotes?|downvotes?|likes?|dislikes?|points?|views?|retweets?|replies)"
    r"|Reply|Permalink|Share|Report|Save|Give Award|Subscribe|SUBSCRIBE"
    r"|Read more|Show this thread|Replying to\s+@\w+"
    r"|Liked a message|Seen|Pinned a message"
    r"|\(edited\))\s*$",
    re.IGNORECASE,
)

# Inline message: Name: text  or  Name - text
_INLINE_MSG = re.compile(r"^([A-Za-z][A-Za-z0-9_\s\[\]\(\)/]{0,49}?)\s*[:]\s*(.+)$")


# ---------------------------------------------------------------------------
# Block segmentation
# ---------------------------------------------------------------------------

def segment_blocks(raw_text: str) -> list[BlockCandidate]:
    """Segment raw text into candidate blocks.

    Uses blank lines as primary separators. Each non-blank run of lines
    becomes one block candidate.
    """
    blocks = []
    lines = raw_text.split("\n")
    current_lines: list[str] = []
    start_line = 0

    for i, line in enumerate(lines):
        if not line.strip():
            if current_lines:
                blocks.append(BlockCandidate(
                    block_id=len(blocks),
                    raw_text="\n".join(current_lines),
                    line_start=start_line,
                    line_end=i - 1,
                ))
                current_lines = []
            start_line = i + 1
        else:
            if not current_lines:
                start_line = i
            current_lines.append(line)

    if current_lines:
        blocks.append(BlockCandidate(
            block_id=len(blocks),
            raw_text="\n".join(current_lines),
            line_start=start_line,
            line_end=len(lines) - 1,
        ))

    # If few blocks and many lines, split into per-line blocks
    # This handles no-blank-line chat formats like "Alice: Hello\nBob: Hi"
    if len(blocks) <= 1 and len(lines) >= 2:
        blocks = []
        for i, line in enumerate(lines):
            if line.strip():
                blocks.append(BlockCandidate(
                    block_id=len(blocks),
                    raw_text=line.strip(),
                    line_start=i,
                    line_end=i,
                ))

    # Also split multi-line blocks that contain multiple inline speaker patterns
    refined = []
    for block in blocks:
        sub_lines = block.raw_text.split("\n")
        if len(sub_lines) > 1:
            # Check if multiple lines match inline speaker pattern
            inline_count = sum(1 for l in sub_lines if _INLINE_MSG.match(l.strip()))
            if inline_count >= 2:
                # Split into per-line blocks
                for j, l in enumerate(sub_lines):
                    if l.strip():
                        refined.append(BlockCandidate(
                            block_id=len(refined),
                            raw_text=l.strip(),
                            line_start=block.line_start + j,
                            line_end=block.line_start + j,
                        ))
                continue
        refined.append(BlockCandidate(block_id=len(refined), raw_text=block.raw_text,
                                       line_start=block.line_start, line_end=block.line_end))

    return refined


# ---------------------------------------------------------------------------
# Block classification
# ---------------------------------------------------------------------------

def classify_block(block: BlockCandidate) -> ClassifiedBlock:
    """Classify a single block by type."""
    text = block.raw_text.strip()
    cb = ClassifiedBlock(block_id=block.block_id, raw_text=text)

    if not text:
        cb.block_type = "unknown"
        return cb

    # Check single-line blocks
    lines = text.split("\n")
    first_line = lines[0].strip()

    # URL / link card
    if _URL_LINE.match(text):
        cb.block_type = "attachment_or_link_card"
        cb.url = text.strip()
        cb.confidence = 0.9
        return cb

    if _LINK_CARD.match(first_line):
        cb.block_type = "attachment_or_link_card"
        cb.confidence = 0.8
        return cb

    # UI junk / reactions
    if all(_UI_JUNK.match(l.strip()) for l in lines if l.strip()):
        cb.block_type = "reaction_or_ui_junk"
        cb.confidence = 0.85
        return cb

    # Profile chrome
    if _PROFILE_CHROME.search(text):
        cb.block_type = "profile_chrome"
        cb.confidence = 0.75
        return cb

    # Timestamp only
    if _TIMESTAMP_ONLY.match(text):
        cb.block_type = "timestamp"
        cb.timestamp = text.strip()
        cb.confidence = 0.9
        return cb

    # Date separator
    if _DATE_SEPARATOR.match(text):
        cb.block_type = "date_separator"
        cb.confidence = 0.8
        return cb

    # Inline message: Name: text (on a single line)
    if len(lines) == 1:
        m = _INLINE_MSG.match(text)
        if m:
            # Check it's not a URL or junk masquerading as name:text
            name_part = m.group(1).strip()
            if len(name_part) < 50 and not name_part.startswith("http"):
                cb.block_type = "speaker_header"
                cb.speaker_raw = name_part
                cb.confidence = 0.8
                # The rest is the message — store in a special way
                # We'll handle this in assembly
                return cb

    # Speaker header patterns (multi-line: header + body)
    for pat in _SPEAKER_HEADER_PATTERNS:
        m = pat.match(first_line)
        if m:
            name = m.group(1).strip()
            ts = m.group(2).strip() if m.lastindex and m.lastindex >= 2 else ""
            if len(name) < 50 and not name.startswith("http"):
                cb.block_type = "speaker_header"
                cb.speaker_raw = name
                cb.timestamp = ts
                cb.confidence = 0.7
                return cb

    # Multi-line block: check if first line is a speaker header (header/body pattern)
    if len(lines) > 1:
        for pat in _SPEAKER_HEADER_PATTERNS:
            m = pat.match(first_line)
            if m:
                name = m.group(1).strip()
                ts = m.group(2).strip() if m.lastindex and m.lastindex >= 2 else ""
                if len(name) < 50 and not name.startswith("http"):
                    cb.block_type = "speaker_header"
                    cb.speaker_raw = name
                    cb.timestamp = ts
                    cb.confidence = 0.75
                    return cb

        cb.block_type = "message_body"
        cb.confidence = 0.6
        return cb

    # Single line, no pattern matched — could be message body or unknown
    if len(text) > 10:
        cb.block_type = "message_body"
        cb.confidence = 0.5
    else:
        cb.block_type = "unknown"
        cb.confidence = 0.3

    return cb


# ---------------------------------------------------------------------------
# Turn assembly
# ---------------------------------------------------------------------------

def assemble_turns(classified: list[ClassifiedBlock]) -> tuple[list[NormalizedTurn], list[ClassifiedBlock]]:
    """Assemble classified blocks into normalized turns.

    Returns (turns, dropped_blocks).
    """
    turns: list[NormalizedTurn] = []
    dropped: list[ClassifiedBlock] = []
    current_turn: NormalizedTurn | None = None
    last_timestamp: str = ""

    for cb in classified:
        if cb.block_type == "speaker_header":
            # Flush current turn
            if current_turn:
                _finalize_turn(current_turn)
                turns.append(current_turn)

            # Check if inline message (Name: text format)
            inline_text = ""
            m = _INLINE_MSG.match(cb.raw_text)
            if m and len(cb.raw_text.split("\n")) == 1:
                inline_text = m.group(2).strip()
            elif len(cb.raw_text.split("\n")) > 1:
                # Multi-line block: first line is header, rest is body
                body_lines = cb.raw_text.split("\n")[1:]
                inline_text = "\n".join(l.strip() for l in body_lines if l.strip())

            current_turn = NormalizedTurn(
                speaker_raw=cb.speaker_raw,
                timestamp=cb.timestamp or last_timestamp or None,
                text_full=inline_text,
                raw_blocks=[cb.block_id],
            )

        elif cb.block_type == "message_body":
            if current_turn:
                # Append to current turn
                if current_turn.text_full:
                    current_turn.text_full += "\n" + cb.raw_text
                else:
                    current_turn.text_full = cb.raw_text
                current_turn.raw_blocks.append(cb.block_id)
            else:
                # Orphan message body — create a turn with unknown speaker
                current_turn = NormalizedTurn(
                    speaker_raw="[unknown]",
                    text_full=cb.raw_text,
                    timestamp=last_timestamp or None,
                    raw_blocks=[cb.block_id],
                    warnings=["orphan message body — no preceding speaker header"],
                )

        elif cb.block_type == "timestamp":
            last_timestamp = cb.timestamp
            if current_turn:
                current_turn.timestamp = current_turn.timestamp or cb.timestamp

        elif cb.block_type == "attachment_or_link_card":
            if current_turn:
                current_turn.attachments.append(cb.url or cb.raw_text[:100])
                current_turn.raw_blocks.append(cb.block_id)
            # else: dropped (no turn to attach to)

        elif cb.block_type in ("profile_chrome", "reaction_or_ui_junk", "date_separator"):
            dropped.append(cb)

        elif cb.block_type == "unknown":
            if current_turn:
                # Conservatively append to current turn with warning
                current_turn.text_full += "\n" + cb.raw_text if current_turn.text_full else cb.raw_text
                current_turn.warnings.append(f"unknown block appended: {cb.raw_text[:30]}")
                current_turn.raw_blocks.append(cb.block_id)
            else:
                dropped.append(cb)

    # Flush last turn
    if current_turn:
        _finalize_turn(current_turn)
        turns.append(current_turn)

    return turns, dropped


def _finalize_turn(turn: NormalizedTurn) -> None:
    """Finalize a turn: clean text, set preview, compute confidence."""
    turn.text_full = turn.text_full.strip()
    turn.text_preview = turn.text_full[:80]
    turn.confidence = 0.8 if turn.speaker_raw != "[unknown]" else 0.4


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def normalize_blocks(raw_text: str) -> BlockNormalizationResult:
    """Run the full block normalization pipeline.

    raw text -> segment -> classify -> assemble -> result
    """
    result = BlockNormalizationResult()

    if not raw_text or not raw_text.strip():
        return result

    # Source normalization (chrome stripping) first
    from source_normalizer import normalize_source
    norm = normalize_source(raw_text)
    result.source_detected = norm.source_detected
    cleaned = norm.normalized_text

    # Segment
    blocks = segment_blocks(cleaned)

    # Classify
    classified = [classify_block(b) for b in blocks]
    result.blocks = classified

    # Assemble
    turns, dropped = assemble_turns(classified)
    result.turns = turns
    result.dropped_blocks = dropped

    # Warnings
    if not turns:
        result.warnings.append("no turns assembled from input")
    orphan_count = sum(1 for t in turns if t.speaker_raw == "[unknown]")
    if orphan_count > 0:
        result.warnings.append(f"{orphan_count} turn(s) with unknown speaker")

    return result
