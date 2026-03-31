"""Page Read Adapter — fetches and extracts content from web pages.

Bounded: default 1 page, max 2. Conservative text extraction.
Uses a PageProvider interface so tests work offline.
"""

from __future__ import annotations

from btom_engine.osint.adapters.base import AdapterHealth, BaseAdapter
from btom_engine.osint.evidence_schema import EvidenceRecord, RetrievalRequest, SourceRun
from btom_engine.osint.providers import PageProvider


class PageReadAdapter(BaseAdapter):
    """Bounded page content extraction. Requires a PageProvider."""

    source_id = "page_read_generic"

    def __init__(self, provider: PageProvider | None = None, max_pages: int = 2) -> None:
        self._provider = provider
        self._max_pages = max_pages

    def can_handle(self, request: RetrievalRequest) -> bool:
        return (
            request.query_type in ("page_read", "content_extract")
            and request.target_type == "url"
            and self._provider is not None
        )

    def execute(self, request: RetrievalRequest) -> tuple[list[EvidenceRecord], SourceRun]:
        run = SourceRun(
            source_id=self.source_id,
            adapter_class="PageReadAdapter",
        )

        if self._provider is None:
            run.success = False
            run.error = "No page provider configured."
            return [], run

        try:
            # target_value should be a URL or comma-separated URLs
            urls = [u.strip() for u in request.target_value.split(",") if u.strip()]
            urls = urls[:self._max_pages]

            records = []
            for url in urls:
                page = self._provider.fetch(url, max_chars=3000)
                if page.fetch_success:
                    # Extract a useful excerpt (first ~500 chars of body)
                    excerpt = page.text[:500].strip()
                    records.append(EvidenceRecord(
                        source_id=self.source_id,
                        source_family="page_read",
                        target_type="url",
                        target_value=url,
                        content_type="text",
                        title=page.title or url,
                        snippet=excerpt,
                        full_text=page.text if len(page.text) > 500 else None,
                        url_or_citation=url,
                        confidence=0.5,
                        reliability_tier="medium",
                        relevance_score=0.6,
                    ))

            run.success = len(records) > 0
            run.records_returned = len(records)
            if not records:
                run.error = "No pages fetched successfully."
            return records, run

        except Exception as e:
            run.success = False
            run.error = f"{type(e).__name__}: {e}"
            return [], run

    def healthcheck(self) -> AdapterHealth:
        if self._provider is None:
            return AdapterHealth(available=False, message="No page provider configured.")
        return AdapterHealth(available=True, message="Page provider available.")
