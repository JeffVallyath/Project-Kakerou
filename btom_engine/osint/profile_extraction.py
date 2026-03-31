"""Profile Extraction — lightweight structured field extraction from page text.

Extracts name, role, org, bio from public profile/about pages using
simple heuristic patterns. Conservative — prefers no result over false extraction.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ProfileFields:
    """Structured fields extracted from a public profile page."""

    display_name: Optional[str] = None
    role_or_title: Optional[str] = None
    organization: Optional[str] = None
    bio_excerpt: Optional[str] = None
    links: list[str] = field(default_factory=list)
    raw_text_used: str = ""
    extraction_confidence: float = 0.3


# Common role/title patterns
_ROLE_PATTERNS = re.compile(
    r"\b(CEO|CTO|CFO|COO|founder|co-founder|director|manager|engineer|developer|"
    r"researcher|professor|analyst|consultant|designer|writer|author|editor|"
    r"scientist|architect|lead|head of|VP|vice president|senior|junior|"
    r"intern|specialist|coordinator|associate)\b",
    re.IGNORECASE,
)

# Organization indicators
_ORG_PATTERNS = re.compile(
    r"\b(at|@)\s+([A-Z][A-Za-z0-9\s&.,']+?)(?:\.|,|\s{2}|\n|$)",
)

# Bio section markers
_BIO_MARKERS = re.compile(
    r"\b(about|bio|biography|description|summary|who i am|who we are)\b",
    re.IGNORECASE,
)


def extract_profile(page_text: str, page_title: str = "") -> ProfileFields:
    """Extract structured profile fields from page text.

    Conservative: returns None for fields where extraction is uncertain.
    """
    result = ProfileFields()

    if not page_text or len(page_text.strip()) < 20:
        return result

    # Use first 2000 chars for extraction (profile info is usually near the top)
    text = page_text[:2000]
    result.raw_text_used = text[:200]

    # Display name: try page title first, then look for name-like patterns
    if page_title and len(page_title) < 60:
        # Clean common suffixes
        name = re.sub(r"\s*[-|–—]\s*(LinkedIn|Twitter|X|GitHub|About|Profile|Bio).*", "", page_title, flags=re.IGNORECASE)
        name = name.strip()
        if 2 < len(name) < 50:
            result.display_name = name

    # Role/title: find role keywords
    role_match = _ROLE_PATTERNS.search(text)
    if role_match:
        # Extract surrounding context for the role
        start = max(0, role_match.start() - 30)
        end = min(len(text), role_match.end() + 30)
        role_context = text[start:end].strip()
        # Clean up to one line
        role_context = role_context.split("\n")[0].strip()
        if len(role_context) < 80:
            result.role_or_title = role_context

    # Organization: look for "at [OrgName]" patterns
    org_match = _ORG_PATTERNS.search(text)
    if org_match:
        org = org_match.group(2).strip()
        if 2 < len(org) < 60:
            result.organization = org

    # Bio excerpt: look for bio section or use first paragraph
    bio_match = _BIO_MARKERS.search(text)
    if bio_match:
        bio_start = bio_match.end()
        bio_text = text[bio_start:bio_start + 300].strip()
        if len(bio_text) > 20:
            result.bio_excerpt = bio_text
    elif len(text) > 50:
        # Use first substantial paragraph as bio fallback
        paragraphs = [p.strip() for p in text.split("\n") if len(p.strip()) > 30]
        if paragraphs:
            result.bio_excerpt = paragraphs[0][:300]

    # Links: extract URLs from text
    urls = re.findall(r"https?://[^\s<>\"']+", text)
    result.links = urls[:5]

    # Confidence: based on how many fields were extracted
    filled = sum(1 for f in [result.display_name, result.role_or_title, result.organization, result.bio_excerpt] if f)
    result.extraction_confidence = min(0.8, 0.2 + filled * 0.15)

    return result
