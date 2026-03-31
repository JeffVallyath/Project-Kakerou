"""Claim Comparison — compares extracted claims against retrieved evidence.

Conservative comparator. Does not overreason.
Uses word overlap and simple contradiction detection.
"""

from __future__ import annotations

import re
from typing import List

from btom_engine.osint.evidence_schema import ComparisonResult, EvidenceRecord
from btom_engine.osint.claim_extraction import ExtractedClaim


_NEGATION_PATTERNS = re.compile(
    r"\b(never|didn'?t|not|no|wasn'?t|haven'?t|couldn'?t|wouldn'?t|don'?t)\b",
    re.IGNORECASE,
)


def _word_set(text: str) -> set[str]:
    """Normalize text to a word set for comparison."""
    return set(re.sub(r"[^\w\s]", "", text.lower()).split())


def _has_negation(text: str) -> bool:
    return bool(_NEGATION_PATTERNS.search(text))


def compare_claim(
    claim: ExtractedClaim,
    evidence: list[EvidenceRecord],
) -> ComparisonResult:
    """Compare one claim against retrieved evidence records.

    Returns a typed comparison outcome.
    """
    if not evidence:
        return ComparisonResult(
            claim_text=claim.claim_text,
            outcome="insufficient_evidence",
            comparison_confidence=0.0,
            rationale="No prior statements found for comparison.",
        )

    claim_words = _word_set(claim.claim_text)
    claim_negated = _has_negation(claim.claim_text)

    best_match = None
    best_overlap = 0.0

    for rec in evidence:
        rec_text = rec.snippet or ""
        rec_words = _word_set(rec_text)

        if not rec_words or not claim_words:
            continue

        overlap = len(claim_words & rec_words) / max(len(claim_words | rec_words), 1)
        if overlap > best_overlap:
            best_overlap = overlap
            best_match = rec

    if best_match is None or best_overlap < 0.1:
        return ComparisonResult(
            claim_text=claim.claim_text,
            outcome="insufficient_evidence",
            comparison_confidence=0.2,
            rationale="No sufficiently relevant prior statement found.",
        )

    # Check for contradiction: claim denies something that evidence supports
    evidence_negated = _has_negation(best_match.snippet or "")

    if claim.claim_type == "denial_of_prior_statement":
        # "I never said X" — if evidence shows they DID say X, that's a contradiction
        if best_overlap > 0.25 and not evidence_negated:
            return ComparisonResult(
                claim_text=claim.claim_text,
                matched_evidence_ids=[best_match.evidence_id],
                outcome="direct_contradiction",
                comparison_confidence=min(0.85, best_overlap + 0.3),
                rationale=f"Target denies prior statement, but evidence contains: '{best_match.snippet[:50]}'",
            )

    if claim.claim_type in ("prior_explanation_claim", "prior_statement_reference", "self_consistency_claim"):
        # "I already explained" — if evidence confirms they did, that's supportive
        if best_overlap > 0.2:
            return ComparisonResult(
                claim_text=claim.claim_text,
                matched_evidence_ids=[best_match.evidence_id],
                outcome="supported_by_prior",
                comparison_confidence=min(0.80, best_overlap + 0.2),
                rationale=f"Prior statement found: '{best_match.snippet[:50]}'",
            )

    # External claims — check if public evidence supports or contradicts
    if claim.claim_type in ("factual_assertion", "public_record_claim", "event_claim", "article_reference"):
        if best_overlap > 0.25:
            # Check for negation mismatch
            if claim_negated != evidence_negated and best_overlap > 0.3:
                return ComparisonResult(
                    claim_text=claim.claim_text,
                    matched_evidence_ids=[best_match.evidence_id],
                    outcome="direct_contradiction",
                    comparison_confidence=min(0.75, best_overlap + 0.2),
                    rationale=f"External evidence contradicts claim: '{best_match.snippet[:50]}'",
                )
            return ComparisonResult(
                claim_text=claim.claim_text,
                matched_evidence_ids=[best_match.evidence_id],
                outcome="supported_by_prior",
                comparison_confidence=min(0.65, best_overlap + 0.1),
                rationale=f"External evidence supports claim: '{best_match.snippet[:50]}'",
            )

    # Profile/affiliation claims — check if profile evidence supports
    if claim.claim_type in ("affiliation_claim", "role_claim", "authorship_claim"):
        if best_overlap > 0.2:
            if claim_negated != evidence_negated and best_overlap > 0.3:
                return ComparisonResult(
                    claim_text=claim.claim_text,
                    matched_evidence_ids=[best_match.evidence_id],
                    outcome="direct_contradiction",
                    comparison_confidence=min(0.70, best_overlap + 0.15),
                    rationale=f"Profile evidence contradicts: '{best_match.snippet[:50]}'",
                )
            return ComparisonResult(
                claim_text=claim.claim_text,
                matched_evidence_ids=[best_match.evidence_id],
                outcome="supported_by_prior",
                comparison_confidence=min(0.60, best_overlap + 0.1),
                rationale=f"Profile evidence supports: '{best_match.snippet[:50]}'",
            )

    # Low overlap or ambiguous
    if best_overlap > 0.15:
        return ComparisonResult(
            claim_text=claim.claim_text,
            matched_evidence_ids=[best_match.evidence_id],
            outcome="weak_tension",
            comparison_confidence=best_overlap,
            rationale=f"Partial overlap with prior: '{best_match.snippet[:50]}'",
        )

    return ComparisonResult(
        claim_text=claim.claim_text,
        outcome="insufficient_evidence",
        comparison_confidence=0.15,
        rationale="Overlap too low for reliable comparison.",
    )
