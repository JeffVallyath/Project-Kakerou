"""Source Normalizer — detects platform and strips chrome from pasted text.

Flow: raw pasted text -> sniff source -> normalize -> clean text for parser

Supports (best-effort):
  Chat: Discord, LinkedIn DMs, Instagram DMs, Reddit chat, X/Twitter DMs
  Thread: Reddit comments, X threads, YouTube comments
  Profile: LinkedIn/X/Instagram/Reddit/YouTube profile snippets
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class NormalizationResult:
    """Output of source normalization."""
    source_detected: str = "generic"       # discord, linkedin, instagram, reddit, x_twitter, youtube, generic
    source_family: str = "chat"            # chat, thread, profile, generic
    normalized_text: str = ""
    confidence: float = 0.5
    warnings: list[str] = field(default_factory=list)
    chrome_stripped: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Source signatures for sniffing
# ---------------------------------------------------------------------------

_SIGNATURES: list[tuple[str, str, list[re.Pattern]]] = [
    ("discord", "chat", [
        re.compile(r"(Today at|Yesterday at)\s+\d{1,2}:\d{2}\s*(AM|PM)", re.IGNORECASE),
        re.compile(r"#\w+.*\d{2}/\d{2}/\d{4}"),
        re.compile(r"BOT\s", re.IGNORECASE),
    ]),
    ("linkedin", "chat", [
        re.compile(r"sent the following message", re.IGNORECASE),
        re.compile(r"View\s+.*profile", re.IGNORECASE),
        re.compile(r"LinkedIn", re.IGNORECASE),
        re.compile(r"\d+\s*(mo|yr|wk|hr|min)\s*ago", re.IGNORECASE),
    ]),
    ("instagram", "chat", [
        re.compile(r"Liked a message", re.IGNORECASE),
        re.compile(r"Seen\s+\d", re.IGNORECASE),
        re.compile(r"Instagram", re.IGNORECASE),
    ]),
    ("reddit", "thread", [
        re.compile(r"(u/\w+|r/\w+)"),
        re.compile(r"\d+\s*(upvotes?|downvotes?|points?)", re.IGNORECASE),
        re.compile(r"(Reply|Permalink|Share|Report|Save)", re.IGNORECASE),
        re.compile(r"level\s+\d+", re.IGNORECASE),
    ]),
    ("x_twitter", "thread", [
        re.compile(r"@\w+\s"),
        re.compile(r"(Retweet|Quote Tweet|Replying to)", re.IGNORECASE),
        re.compile(r"\d+\s*(Retweets?|Likes?|Views?|Replies)", re.IGNORECASE),
        re.compile(r"(twitter\.com|x\.com)", re.IGNORECASE),
    ]),
    ("youtube", "thread", [
        re.compile(r"(\d+\s*(likes?|dislikes?|replies))", re.IGNORECASE),
        re.compile(r"(Subscribe|subscribers)", re.IGNORECASE),
        re.compile(r"(youtube\.com|youtu\.be)", re.IGNORECASE),
        re.compile(r"\d+\s*(views?|watching)", re.IGNORECASE),
    ]),
]


def sniff_source(raw_text: str) -> tuple[str, str, float]:
    """Detect likely source platform from text signatures.

    Returns (source_name, source_family, confidence).
    Falls back to ("generic", "generic", 0.3).
    """
    text_sample = raw_text[:2000]
    best_source = "generic"
    best_family = "generic"
    best_score = 0

    for source, family, patterns in _SIGNATURES:
        score = sum(1 for p in patterns if p.search(text_sample))
        if score > best_score:
            best_score = score
            best_source = source
            best_family = family

    if best_score >= 2:
        confidence = min(0.9, 0.4 + best_score * 0.15)
        return best_source, best_family, confidence
    elif best_score == 1:
        return best_source, best_family, 0.35
    else:
        return "generic", "generic", 0.3


# ---------------------------------------------------------------------------
# Chrome stripping patterns (per source)
# ---------------------------------------------------------------------------

# Common platform UI chrome to strip (preserving actual message text)
_CHROME_PATTERNS: dict[str, list[tuple[str, re.Pattern]]] = {
    "linkedin": [
        ("linkedin_sent_msg", re.compile(r".*sent the following message.*\n?", re.IGNORECASE)),
        ("linkedin_view_profile", re.compile(r"View\s+.*?'s?\s+profile\n?", re.IGNORECASE)),
        ("linkedin_connection", re.compile(r"(1st|2nd|3rd)\s*\|\s*", re.IGNORECASE)),
        ("linkedin_time_ago", re.compile(r"^\s*\d+\s*(mo|yr|wk|hr|min|day)s?\s*ago\s*$", re.MULTILINE | re.IGNORECASE)),
    ],
    "discord": [
        ("discord_bot_tag", re.compile(r"\s*BOT\s*", re.IGNORECASE)),
        ("discord_edited", re.compile(r"\s*\(edited\)\s*", re.IGNORECASE)),
        ("discord_pinned", re.compile(r"^\s*Pinned\s+a\s+message.*$", re.MULTILINE | re.IGNORECASE)),
    ],
    "instagram": [
        ("ig_liked", re.compile(r"^\s*Liked a message\s*$", re.MULTILINE | re.IGNORECASE)),
        ("ig_seen", re.compile(r"^\s*Seen\s*$", re.MULTILINE | re.IGNORECASE)),
        ("ig_sent_photo", re.compile(r"^\s*Sent an? (photo|attachment|reel|story)\s*$", re.MULTILINE | re.IGNORECASE)),
    ],
    "reddit": [
        ("reddit_upvotes", re.compile(r"^\s*\d+\s*(upvotes?|downvotes?|points?)\s*$", re.MULTILINE | re.IGNORECASE)),
        ("reddit_actions", re.compile(r"^\s*(Reply|Permalink|Share|Report|Save|Give Award)\s*$", re.MULTILINE | re.IGNORECASE)),
        ("reddit_level", re.compile(r"^\s*level\s+\d+\s*$", re.MULTILINE | re.IGNORECASE)),
        ("reddit_ago", re.compile(r"^\s*\d+\s*(hours?|minutes?|days?|months?|years?)\s+ago\s*$", re.MULTILINE | re.IGNORECASE)),
    ],
    "x_twitter": [
        ("x_metrics", re.compile(r"^\s*\d+\s*(Retweets?|Likes?|Views?|Replies|Quote Tweets?)\s*$", re.MULTILINE | re.IGNORECASE)),
        ("x_replying", re.compile(r"^\s*Replying to\s+@\w+\s*$", re.MULTILINE | re.IGNORECASE)),
        ("x_show_thread", re.compile(r"^\s*Show this thread\s*$", re.MULTILINE | re.IGNORECASE)),
    ],
    "youtube": [
        ("yt_metrics", re.compile(r"^\s*\d+\s*(likes?|dislikes?|replies)\s*$", re.MULTILINE | re.IGNORECASE)),
        ("yt_subscribe", re.compile(r"^\s*(Subscribe|SUBSCRIBE|subscribed)\s*$", re.MULTILINE | re.IGNORECASE)),
        ("yt_read_more", re.compile(r"^\s*Read more\s*$", re.MULTILINE | re.IGNORECASE)),
    ],
}

# Universal chrome (applies to all sources)
_UNIVERSAL_CHROME = [
    ("empty_lines_excess", re.compile(r"\n{4,}")),  # collapse 4+ newlines to 2
]


def _strip_chrome(text: str, source: str) -> tuple[str, list[str]]:
    """Strip platform-specific chrome from text. Returns (cleaned, stripped_items)."""
    stripped = []

    # Source-specific patterns
    patterns = _CHROME_PATTERNS.get(source, [])
    for label, pattern in patterns:
        if pattern.search(text):
            text = pattern.sub("", text)
            stripped.append(label)

    # Universal cleanup
    for label, pattern in _UNIVERSAL_CHROME:
        if pattern.search(text):
            text = pattern.sub("\n\n", text)
            stripped.append(label)

    # Clean up leftover blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip(), stripped


# ---------------------------------------------------------------------------
# Timestamp extraction helpers
# ---------------------------------------------------------------------------

# Common timestamp patterns to extract from speaker lines
_TIMESTAMP_INLINE = re.compile(
    r"(\d{1,2}[:/]\d{2}(?:\s*[APap][Mm])?)"
    r"|(\d{1,2}/\d{1,2}/\d{2,4}\s+\d{1,2}:\d{2})"
    r"|(Today at \d{1,2}:\d{2}\s*[APap][Mm])"
    r"|(Yesterday at \d{1,2}:\d{2}\s*[APap][Mm])"
    r"|(\d{1,2}\s*(mo|yr|wk|hr|min|day)s?\s+ago)",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Profile snippet detection
# ---------------------------------------------------------------------------

_PROFILE_MARKERS = [
    re.compile(r"(About|Bio|Description|Summary)\s*\n", re.IGNORECASE),
    re.compile(r"(Followers?|Following|Posts?|Connections?)\s*[:\s]*\d", re.IGNORECASE),
    re.compile(r"(Joined|Member since)\s+", re.IGNORECASE),
]


def _is_profile_snippet(text: str) -> bool:
    """Check if text looks like a profile snippet rather than a conversation."""
    return sum(1 for p in _PROFILE_MARKERS if p.search(text[:500])) >= 2


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def normalize_source(raw_text: str) -> NormalizationResult:
    """Detect source platform, strip chrome, return clean text for parser.

    This runs BEFORE the existing transcript parser.
    """
    if not raw_text or not raw_text.strip():
        return NormalizationResult(normalized_text="")

    # Sniff source
    source, family, confidence = sniff_source(raw_text)

    # Check if it's a profile snippet
    if _is_profile_snippet(raw_text):
        family = "profile"

    # Strip chrome
    cleaned, chrome_stripped = _strip_chrome(raw_text, source)

    warnings = []
    if len(chrome_stripped) > 5:
        warnings.append(f"heavy chrome stripping: {len(chrome_stripped)} patterns removed")

    return NormalizationResult(
        source_detected=source,
        source_family=family,
        normalized_text=cleaned,
        confidence=confidence,
        warnings=warnings,
        chrome_stripped=chrome_stripped,
    )
