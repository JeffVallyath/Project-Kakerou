"""Public Profile Adapter — reads a public profile/about page and extracts structured context.

Bounded: reads one page, extracts compact profile fields, normalizes to EvidenceRecord.
Reuses the existing PageProvider interface — no new provider needed.
"""

from __future__ import annotations

from btom_engine.osint.adapters.base import AdapterHealth, BaseAdapter
from btom_engine.osint.evidence_schema import EvidenceRecord, RetrievalRequest, SourceRun
from btom_engine.osint.profile_extraction import extract_profile
from btom_engine.osint.providers import PageProvider


class PublicProfileAdapter(BaseAdapter):
    """Reads a public profile page and extracts structured target context."""

    source_id = "public_profile_page"

    def __init__(self, provider: PageProvider | None = None) -> None:
        self._provider = provider

    def can_handle(self, request: RetrievalRequest) -> bool:
        return (
            request.query_type in ("public_profile", "profile_check", "affiliation_check")
            and request.target_type in ("url", "profile_url")
            and self._provider is not None
        )

    def execute(self, request: RetrievalRequest) -> tuple[list[EvidenceRecord], SourceRun]:
        run = SourceRun(
            source_id=self.source_id,
            adapter_class="PublicProfileAdapter",
        )

        if self._provider is None:
            run.success = False
            run.error = "No page provider configured."
            return [], run

        try:
            url = request.target_value.strip()
            if not url:
                run.success = False
                run.error = "No URL provided."
                return [], run

            page = self._provider.fetch(url, max_chars=3000)
            if not page.fetch_success:
                run.success = False
                run.error = page.error or "Page fetch failed."
                return [], run

            # Extract structured profile fields
            profile = extract_profile(page.text, page.title)

            # Build rich snippet from extracted fields
            parts = []
            if profile.display_name:
                parts.append(f"Name: {profile.display_name}")
            if profile.role_or_title:
                parts.append(f"Role: {profile.role_or_title}")
            if profile.organization:
                parts.append(f"Org: {profile.organization}")
            if profile.bio_excerpt:
                parts.append(f"Bio: {profile.bio_excerpt[:150]}")

            snippet = " | ".join(parts) if parts else page.text[:200]

            record = EvidenceRecord(
                source_id=self.source_id,
                source_family="public_profile",
                target_type=request.target_type,
                target_value=url,
                content_type="profile",
                title=profile.display_name or page.title or url,
                snippet=snippet,
                full_text=page.text[:1000] if len(page.text) > len(snippet) else None,
                url_or_citation=url,
                confidence=profile.extraction_confidence,
                reliability_tier="medium",
                relevance_score=0.7 if profile.extraction_confidence > 0.4 else 0.4,
                extraction_notes=f"fields: name={'Y' if profile.display_name else 'N'} "
                                 f"role={'Y' if profile.role_or_title else 'N'} "
                                 f"org={'Y' if profile.organization else 'N'} "
                                 f"bio={'Y' if profile.bio_excerpt else 'N'}",
            )

            run.success = True
            run.records_returned = 1
            return [record], run

        except Exception as e:
            run.success = False
            run.error = f"{type(e).__name__}: {e}"
            return [], run

    def healthcheck(self) -> AdapterHealth:
        if self._provider is None:
            return AdapterHealth(available=False, message="No page provider configured.")
        return AdapterHealth(available=True, message="Profile adapter available.")
