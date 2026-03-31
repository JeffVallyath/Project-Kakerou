"""Retrieval Router — selects adapters for a given retrieval request.

Uses deterministic rules to match requests against the SourceRegistry.
No LLM planner — just typed routing.
"""

from __future__ import annotations

from btom_engine.osint.evidence_schema import RetrievalPlan, RetrievalRequest
from btom_engine.osint.source_registry import SourceRegistry


def route(request: RetrievalRequest, registry: SourceRegistry) -> RetrievalPlan:
    """Select the smallest set of valid adapters for this request.

    Returns a RetrievalPlan. If no sources match, returns an empty plan.
    """
    families = request.allowed_source_families or None

    suitable = registry.find_suitable(
        target_type=request.target_type,
        query_type=request.query_type,
        families=families,
        bounded_only=request.bounded_mode,
    )

    if not suitable:
        return RetrievalPlan(
            selected_sources=[],
            rationale="No suitable sources found for this request.",
            max_results=request.top_k,
            bounded=request.bounded_mode,
        )

    selected_ids = [s.source_id for s in suitable]
    families_used = list(set(s.source_family for s in suitable))

    return RetrievalPlan(
        selected_sources=selected_ids,
        rationale=f"Matched {len(selected_ids)} source(s) from families: {', '.join(families_used)}",
        max_results=request.top_k,
        bounded=request.bounded_mode,
    )
