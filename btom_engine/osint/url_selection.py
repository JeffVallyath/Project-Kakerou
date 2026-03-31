"""URL Selection — chooses which search results to page-read.

Bounded, deterministic, inspectable.
Default: 2 pages. Hard cap: 3.
Scores by: snippet quality, query overlap, domain authority, HTTPS.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from btom_engine.osint.evidence_schema import EvidenceRecord


@dataclass
class SelectedURL:
    """A URL chosen for page reading, with rationale."""
    url: str
    title: str
    score: float
    reason: str


# High-authority domains for verification (bonus scoring)
_AUTHORITY_DOMAINS = {
    "linkedin.com": 1.5,
    "crunchbase.com": 1.5,
    "bloomberg.com": 1.3,
    "techcrunch.com": 1.2,
    "sec.gov": 1.5,
    "wikipedia.org": 1.0,
    "reuters.com": 1.2,
    "pitchbook.com": 1.3,
    "github.com": 0.8,
    "medium.com": 0.5,
    "twitter.com": 0.5,
    "x.com": 0.5,
}

# Domains to deprioritize (content farms, aggregators)
_LOW_QUALITY_DOMAINS = {
    "pinterest.com", "quora.com", "reddit.com",
    "facebook.com", "instagram.com", "tiktok.com",
}


def _extract_domain(url: str) -> str:
    """Extract the base domain from a URL."""
    try:
        # Remove protocol
        domain = url.split("://", 1)[-1].split("/", 1)[0]
        # Remove www.
        if domain.startswith("www."):
            domain = domain[4:]
        return domain.lower()
    except Exception:
        return ""


def select_urls(
    search_records: list[EvidenceRecord],
    query_text: str = "",
    max_pages: int = 2,
) -> list[SelectedURL]:
    """Select top URL(s) from search results for page reading.

    Scoring:
    - non-empty snippet: +1.0
    - title/snippet word overlap with query: +overlap_count * 0.5
    - HTTPS: +0.5
    - authority domain bonus: +0.5 to +1.5
    - low-quality domain penalty: -1.0
    - longer snippets (more info): +0.3

    Returns up to max_pages (hard cap 3) selected URLs.
    """
    max_pages = min(max_pages, 3)
    query_words = set(re.sub(r"[^\w\s]", "", query_text.lower()).split()) if query_text else set()

    candidates = []
    for rec in search_records:
        url = rec.url_or_citation or ""
        if not url or not url.startswith("http"):
            continue

        score = 0.0
        reasons = []
        domain = _extract_domain(url)

        # Non-empty snippet
        if rec.snippet and len(rec.snippet) > 10:
            score += 1.0
            reasons.append("has_snippet")
            # Longer snippets suggest more useful content
            if len(rec.snippet) > 100:
                score += 0.3
                reasons.append("rich_snippet")

        # Word overlap with query
        if query_words:
            combined = f"{rec.title} {rec.snippet}".lower()
            combined_words = set(re.sub(r"[^\w\s]", "", combined).split())
            overlap = len(query_words & combined_words)
            if overlap > 0:
                score += overlap * 0.5
                reasons.append(f"overlap={overlap}")

        # HTTPS preferred
        if url.startswith("https://"):
            score += 0.5
            reasons.append("https")

        # Authority domain bonus
        for auth_domain, bonus in _AUTHORITY_DOMAINS.items():
            if auth_domain in domain:
                score += bonus
                reasons.append(f"authority:{auth_domain}")
                break

        # Low-quality domain penalty
        if domain in _LOW_QUALITY_DOMAINS:
            score -= 1.0
            reasons.append(f"low_quality:{domain}")

        candidates.append(SelectedURL(
            url=url,
            title=rec.title or url,
            score=score,
            reason=", ".join(reasons),
        ))

    # Sort by score descending, take top max_pages
    candidates.sort(key=lambda c: c.score, reverse=True)
    return candidates[:max_pages]
