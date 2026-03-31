"""Retrieval Manager — orchestrates plan execution across adapters.

Calls the router, executes selected adapters, aggregates results
into a unified RetrievalResult with full diagnostics.
"""

from __future__ import annotations

import time
from typing import Any

from btom_engine.osint.adapters.base import BaseAdapter
from btom_engine.osint.evidence_schema import (
    RetrievalRequest,
    RetrievalResult,
    SourceRun,
)
from btom_engine.osint.retrieval_router import route
from btom_engine.osint.source_registry import SourceRegistry


class RetrievalManager:
    """Orchestrates OSINT retrieval across registered adapters."""

    def __init__(self, registry: SourceRegistry) -> None:
        self._registry = registry
        self._adapters: dict[str, BaseAdapter] = {}

    def register_adapter(self, source_id: str, adapter: BaseAdapter) -> None:
        """Register a live adapter instance for a source_id."""
        self._adapters[source_id] = adapter

    def retrieve(self, request: RetrievalRequest) -> RetrievalResult:
        """Execute a bounded retrieval.

        1. Route the request to select sources.
        2. Execute each selected adapter.
        3. Aggregate results with diagnostics.
        """
        plan = route(request, self._registry)
        result = RetrievalResult(plan=plan)

        if not plan.selected_sources:
            result.unresolved_gaps.append("No sources available for this request.")
            return result

        all_records = []

        for source_id in plan.selected_sources:
            adapter = self._adapters.get(source_id)
            if adapter is None:
                result.source_runs.append(SourceRun(
                    source_id=source_id,
                    success=False,
                    error="No adapter instance registered.",
                ))
                result.unresolved_gaps.append(f"Source '{source_id}' has no adapter.")
                continue

            if not adapter.can_handle(request):
                result.source_runs.append(SourceRun(
                    source_id=source_id,
                    adapter_class=type(adapter).__name__,
                    success=False,
                    error="Adapter cannot handle this request.",
                ))
                continue

            t0 = time.time()
            try:
                records, run = adapter.execute(request)
                run.latency_ms = (time.time() - t0) * 1000
                result.source_runs.append(run)
                all_records.extend(records)
            except Exception as e:
                result.source_runs.append(SourceRun(
                    source_id=source_id,
                    adapter_class=type(adapter).__name__,
                    success=False,
                    error=f"{type(e).__name__}: {e}",
                    latency_ms=(time.time() - t0) * 1000,
                ))

        # Trim to top_k by relevance
        all_records.sort(key=lambda r: r.relevance_score, reverse=True)
        result.records = all_records[:plan.max_results]

        # Overall confidence: average of record confidences, or 0
        if result.records:
            result.overall_confidence = sum(r.confidence for r in result.records) / len(result.records)

        return result
