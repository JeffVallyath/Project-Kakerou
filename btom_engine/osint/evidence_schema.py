"""Normalized evidence schema for all OSINT adapters.

Every adapter normalizes its output into EvidenceRecord objects.
The engine never sees raw source payloads — only these typed records.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class EvidenceRecord:
    """One piece of normalized evidence from any source."""

    evidence_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    source_id: str = ""
    source_family: str = ""           # prior_statements, web_search, page_read, etc.
    target_type: str = ""             # username, url, claim, session, etc.
    target_value: str = ""            # the actual lookup key
    content_type: str = "text"        # text, profile, record, snippet
    title: str = ""
    snippet: str = ""                 # short excerpt (always populated)
    full_text: Optional[str] = None   # full content if available
    url_or_citation: str = ""
    timestamp: Optional[str] = None
    author: Optional[str] = None
    confidence: float = 0.5           # adapter's confidence in this record
    reliability_tier: str = "medium"  # low, medium, high, verified
    relevance_score: float = 0.5      # how relevant to the query
    extraction_notes: Optional[str] = None


@dataclass
class RetrievalRequest:
    """A typed, bounded request for evidence retrieval."""

    request_id: str = field(default_factory=lambda: uuid.uuid4().hex[:10])
    target_type: str = ""             # username, url, claim, session, entity_id
    target_value: str = ""
    query_type: str = ""              # prior_statements, web_search, page_read, etc.
    query_text: str = ""              # natural language query or claim to check
    session_id: Optional[str] = None
    target_entity_id: Optional[str] = None
    top_k: int = 5
    allowed_source_families: list[str] = field(default_factory=list)
    bounded_mode: bool = True         # if True, restrict to registered bounded sources
    time_window: Optional[str] = None # e.g., "24h", "7d", "30d"


@dataclass
class SourceRun:
    """Diagnostics for one adapter execution within a retrieval."""

    source_id: str = ""
    adapter_class: str = ""
    success: bool = False
    records_returned: int = 0
    error: Optional[str] = None
    latency_ms: float = 0.0


@dataclass
class RetrievalPlan:
    """Plan produced by the router before execution."""

    selected_sources: list[str] = field(default_factory=list)
    rationale: str = ""
    max_results: int = 5
    bounded: bool = True


@dataclass
class RetrievalResult:
    """Unified result from the retrieval manager."""

    records: list[EvidenceRecord] = field(default_factory=list)
    source_runs: list[SourceRun] = field(default_factory=list)
    overall_confidence: float = 0.0
    unresolved_gaps: list[str] = field(default_factory=list)
    plan: Optional[RetrievalPlan] = None


@dataclass
class ComparisonResult:
    """Result of comparing a claim against retrieved evidence."""

    claim_text: str = ""
    matched_evidence_ids: list[str] = field(default_factory=list)
    outcome: str = "unknown"    # consistent, inconsistent, no_evidence, ambiguous
    comparison_confidence: float = 0.0
    rationale: str = ""
