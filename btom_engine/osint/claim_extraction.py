"""Claim Extraction — hybrid regex + LLM general-purpose claim flagger.

Fast path: regex catches obvious prior-statement references (already proven).
General path: LLM judges whether the text contains ANY verifiable claims.
This eliminates the brittleness of hardcoded claim-type patterns.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import List

logger = logging.getLogger(__name__)


@dataclass
class ExtractedClaim:
    """A typed claim extracted from a target turn."""

    claim_type: str
    claim_text: str
    retrieval_query: str
    search_queries: list[str] = field(default_factory=list)  # multiple refined queries
    needs_external: bool = False
    extraction_method: str = "regex"  # "regex" or "llm"


# ---------------------------------------------------------------------------
# Fast path: regex for prior-statement references (proven, keep these)
# ---------------------------------------------------------------------------

_PRIOR_PATTERNS: list[tuple[str, re.Pattern, str]] = [
    (
        "prior_explanation_claim",
        re.compile(r"\b(i (already|just) (explained|told|said|mentioned|stated|covered))\b", re.IGNORECASE),
        "prior explanation or statement",
    ),
    (
        "denial_of_prior_statement",
        re.compile(r"\b(i never said|i didn'?t say|that'?s not what i said|i never (claimed|stated|mentioned))\b", re.IGNORECASE),
        "prior statement being denied",
    ),
    (
        "self_consistency_claim",
        re.compile(r"\b(i('ve| have) (always|been consistent|been clear)|as i (said|mentioned) before)\b", re.IGNORECASE),
        "consistency with prior statements",
    ),
    (
        "prior_statement_reference",
        re.compile(r"\b(like i said|as i told you|i said (earlier|before|already)|remember i said)\b", re.IGNORECASE),
        "referenced prior statement",
    ),
]


def _extract_prior_claims(text: str) -> list[ExtractedClaim]:
    """Fast regex path for prior-statement references."""
    claims = []
    seen = set()
    for claim_type, pattern, query_hint in _PRIOR_PATTERNS:
        if claim_type in seen:
            continue
        m = pattern.search(text)
        if m:
            start = max(0, m.start() - 20)
            end = min(len(text), m.end() + 40)
            snippet = text[start:end].strip()
            claims.append(ExtractedClaim(
                claim_type=claim_type,
                claim_text=snippet,
                retrieval_query=query_hint,
                needs_external=False,
                extraction_method="regex",
            ))
            seen.add(claim_type)
        if len(claims) >= 2:
            break
    return claims


# ---------------------------------------------------------------------------
# General path: LLM claim flagger (catches everything regex misses)
# ---------------------------------------------------------------------------

_CLAIM_FLAGGER_PROMPT = """You are a verifiable-claim detector. Output raw JSON only. No prose.

Does this message contain specific, verifiable claims about any of the following?
- Identity, employment, role, or title
- Company, organization, or institutional affiliation
- Funding, investment, financial backing, or fundraising
- Credentials, qualifications, or expertise
- Events attended, presentations given, publications authored
- Partnerships, endorsements, or third-party backing
- Factual assertions that could be checked against public records

Message:
\"\"\"{text}\"\"\"

If the message contains verifiable claims, extract up to 3 of the most important ones.
For each claim, provide:
- claim_type: a short label (e.g., "funding_claim", "employment_claim", "credential_claim", "backing_claim", "affiliation_claim")
- claim_text: the exact relevant excerpt from the message
- search_query: a concise search query to verify this claim (e.g., "Amir Fischer BCV Ribbit Capital fund")

If the message contains NO verifiable claims (just casual chat, greetings, vague statements), return an empty list.

Return ONLY:
{{"claims": [{{"claim_type": "<type>", "claim_text": "<exact excerpt>", "search_query": "<query>"}}]}}"""


def _extract_llm_claims(text: str) -> list[ExtractedClaim]:
    """LLM general-purpose claim flagger. Catches what regex can't."""
    from btom_engine.config import LLM_BASE_URL, LLM_MODEL, LLM_TIMEOUT_SECONDS
    import httpx

    if len(text.strip()) < 15:
        return []  # too short to contain verifiable claims

    prompt = _CLAIM_FLAGGER_PROMPT.format(text=text[:1500])

    try:
        resp = httpx.post(
            f"{LLM_BASE_URL}/v1/chat/completions",
            json={
                "model": LLM_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.1,
                "max_tokens": 400,
                "stream": False,
            },
            timeout=LLM_TIMEOUT_SECONDS,
        )
        resp.raise_for_status()
        raw = resp.json()["choices"][0]["message"]["content"]

        # Extract JSON
        raw = re.sub(r"```(?:json)?\s*", "", raw).replace("```", "")
        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end > start:
            raw = raw[start:end + 1]

        parsed = json.loads(raw)
        raw_claims = parsed.get("claims", [])

        if not isinstance(raw_claims, list):
            return []

        claims = []
        for c in raw_claims[:3]:
            if not isinstance(c, dict):
                continue
            ct = str(c.get("claim_type", "verifiable_claim")).strip()
            text_excerpt = str(c.get("claim_text", "")).strip()
            query = str(c.get("search_query", text_excerpt)).strip()

            if not text_excerpt or len(text_excerpt) < 5:
                continue

            claims.append(ExtractedClaim(
                claim_type=ct,
                claim_text=text_excerpt,
                retrieval_query=query,
                needs_external=True,
                extraction_method="llm",
            ))

        return claims

    except Exception as e:
        logger.warning("LLM claim flagger failed: %s", e)
        return []


# ---------------------------------------------------------------------------
# Query expansion — generate multiple targeted search queries per claim
# ---------------------------------------------------------------------------

# Site-specific targets by claim type
_SITE_TARGETS: dict[str, list[str]] = {
    "employment_claim": ["site:linkedin.com", "site:crunchbase.com"],
    "affiliation_claim": ["site:linkedin.com", ""],
    "role_claim": ["site:linkedin.com", ""],
    "funding_claim": ["site:crunchbase.com", "site:techcrunch.com", ""],
    "backing_claim": ["site:crunchbase.com", "portfolio companies", ""],
    "credential_claim": ["site:linkedin.com", ""],
    "investment_claim": ["site:crunchbase.com", "site:sec.gov", ""],
    "company_claim": ["site:crunchbase.com", "site:bloomberg.com", ""],
}


def _expand_queries(claim: ExtractedClaim) -> list[str]:
    """Generate 2-3 targeted search query variants for a claim.

    Base query from LLM + site-specific variants based on claim type.
    """
    base = claim.retrieval_query
    if not base:
        return []

    queries = [base]  # always include the base query

    # Extract key entities for focused queries
    # Simple heuristic: words that start with uppercase are likely entities
    words = base.split()
    entities = [w for w in words if w[0:1].isupper() and len(w) > 2]

    # Add site-targeted variants
    sites = _SITE_TARGETS.get(claim.claim_type, [])
    for site in sites[:2]:
        if site:
            # Combine entities with site target
            entity_str = " ".join(entities[:3]) if entities else base
            variant = f"{entity_str} {site}"
            if variant not in queries:
                queries.append(variant)

    # Add a "person + role/org" focused query if we can detect them
    if len(entities) >= 2:
        person_org = f'"{entities[0]}" "{entities[1]}"'
        if person_org not in queries:
            queries.append(person_org)

    return queries[:3]  # cap at 3 queries per claim


# ---------------------------------------------------------------------------
# Public API: hybrid extraction
# ---------------------------------------------------------------------------

def extract_claims(text: str) -> list[ExtractedClaim]:
    """Extract retrieval-worthy claims using hybrid regex + LLM approach.

    1. Regex catches prior-statement references (fast, proven).
    2. LLM catches everything else (general, no brittleness).
    3. Query expansion generates multiple targeted queries per claim.
    4. Combined, deduplicated, capped at 3.
    """
    # Fast path: prior-statement regex
    prior_claims = _extract_prior_claims(text)

    # General path: LLM claim flagger
    llm_claims = _extract_llm_claims(text)

    # Combine: prior claims first, then LLM claims
    all_claims = prior_claims + llm_claims

    # Deduplicate by claim_text similarity
    seen_texts = set()
    unique = []
    for c in all_claims:
        key = c.claim_text[:50].lower()
        if key not in seen_texts:
            seen_texts.add(key)
            # Expand queries for external claims
            if c.needs_external:
                c.search_queries = _expand_queries(c)
            unique.append(c)

    return unique[:3]
