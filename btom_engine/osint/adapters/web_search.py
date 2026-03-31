"""Web Search Adapter — bounded search via pluggable SearchProvider.

Uses a provider interface so tests work offline with mocks
while live search is opt-in.
"""

from __future__ import annotations

from btom_engine.osint.adapters.base import AdapterHealth, BaseAdapter
from btom_engine.osint.evidence_schema import EvidenceRecord, RetrievalRequest, SourceRun
from btom_engine.osint.providers import SearchProvider


class WebSearchAdapter(BaseAdapter):
    """Bounded web search adapter. Requires a SearchProvider."""

    source_id = "web_search_generic"

    def __init__(self, provider: SearchProvider | None = None) -> None:
        self._provider = provider

    def can_handle(self, request: RetrievalRequest) -> bool:
        return (
            request.query_type in ("web_search", "fact_check", "external_check")
            and request.target_type in ("claim", "entity_name", "url", "username")
            and self._provider is not None
        )

    def execute(self, request: RetrievalRequest) -> tuple[list[EvidenceRecord], SourceRun]:
        run = SourceRun(
            source_id=self.source_id,
            adapter_class="WebSearchAdapter",
        )

        if self._provider is None:
            run.success = False
            run.error = "No search provider configured."
            return [], run

        try:
            results = self._provider.search(request.query_text, top_k=min(request.top_k, 3))
            records = []
            for r in results:
                records.append(EvidenceRecord(
                    source_id=self.source_id,
                    source_family="web_search",
                    target_type=request.target_type,
                    target_value=request.target_value,
                    content_type="snippet",
                    title=r.title,
                    snippet=r.snippet[:300],
                    url_or_citation=r.url,
                    confidence=0.4,  # web search results are low-trust
                    reliability_tier="low",
                    relevance_score=0.5,  # default; could be improved
                ))

            run.success = True
            run.records_returned = len(records)
            return records, run

        except Exception as e:
            run.success = False
            run.error = f"{type(e).__name__}: {e}"
            return [], run

    def healthcheck(self) -> AdapterHealth:
        if self._provider is None:
            return AdapterHealth(available=False, message="No search provider configured.")
        return AdapterHealth(available=True, message="Search provider available.")
