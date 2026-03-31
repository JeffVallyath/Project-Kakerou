"""Source Registry — metadata catalog for all OSINT adapters.

Each source is registered with typed metadata so the router can
discover suitable adapters by capability, target type, and constraints.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Type


@dataclass
class SourceEntry:
    """Metadata for one registered source."""

    source_id: str
    display_name: str
    source_family: str          # prior_statements, web_search, page_read, public_profile, etc.
    supported_target_types: list[str] = field(default_factory=list)
    supported_query_types: list[str] = field(default_factory=list)
    requires_auth: bool = False
    requires_manual_steps: bool = False
    local_install_only: bool = False
    automatable: bool = True
    reliability_tier: str = "medium"   # low, medium, high, verified
    freshness_tier: str = "live"       # live, daily, weekly, archive
    cost_tier: str = "free"            # free, cheap, moderate, expensive
    risk_notes: str = ""
    adapter_class_name: str = ""       # fully qualified or short name
    enabled: bool = True


class SourceRegistry:
    """Registry of all known OSINT source adapters."""

    def __init__(self) -> None:
        self._sources: dict[str, SourceEntry] = {}

    def register(self, entry: SourceEntry) -> None:
        """Register a source adapter."""
        self._sources[entry.source_id] = entry

    def get(self, source_id: str) -> Optional[SourceEntry]:
        return self._sources.get(source_id)

    def list_all(self) -> list[SourceEntry]:
        return list(self._sources.values())

    def list_enabled(self) -> list[SourceEntry]:
        return [s for s in self._sources.values() if s.enabled]

    def find_by_family(self, family: str) -> list[SourceEntry]:
        return [s for s in self._sources.values() if s.source_family == family and s.enabled]

    def find_by_target_type(self, target_type: str) -> list[SourceEntry]:
        return [s for s in self._sources.values()
                if target_type in s.supported_target_types and s.enabled]

    def find_by_query_type(self, query_type: str) -> list[SourceEntry]:
        return [s for s in self._sources.values()
                if query_type in s.supported_query_types and s.enabled]

    def find_suitable(
        self,
        target_type: str = "",
        query_type: str = "",
        families: list[str] | None = None,
        bounded_only: bool = True,
    ) -> list[SourceEntry]:
        """Find sources matching constraints."""
        results = []
        for s in self._sources.values():
            if not s.enabled:
                continue
            if bounded_only and not s.automatable:
                continue
            if target_type and target_type not in s.supported_target_types:
                continue
            if query_type and query_type not in s.supported_query_types:
                continue
            if families and s.source_family not in families:
                continue
            results.append(s)
        return results


def build_default_registry() -> SourceRegistry:
    """Build the default registry with currently implemented sources."""
    reg = SourceRegistry()

    reg.register(SourceEntry(
        source_id="prior_statements_session",
        display_name="Session Prior Statements",
        source_family="prior_statements",
        supported_target_types=["session", "target_entity"],
        supported_query_types=["prior_statements", "claim_check"],
        requires_auth=False,
        automatable=True,
        reliability_tier="high",
        freshness_tier="live",
        cost_tier="free",
        adapter_class_name="PriorStatementsAdapter",
        risk_notes="Limited to current session transcript.",
    ))

    reg.register(SourceEntry(
        source_id="web_search_generic",
        display_name="Web Search (Generic)",
        source_family="web_search",
        supported_target_types=["username", "claim", "entity_name", "url"],
        supported_query_types=["web_search", "fact_check", "external_check"],
        requires_auth=False,
        automatable=True,
        reliability_tier="low",
        freshness_tier="live",
        cost_tier="free",
        adapter_class_name="WebSearchAdapter",
        risk_notes="Unverified web results. Reliability depends on source quality.",
        enabled=True,  # real implementation with provider interface
    ))

    reg.register(SourceEntry(
        source_id="page_read_generic",
        display_name="Web Page Reader",
        source_family="page_read",
        supported_target_types=["url"],
        supported_query_types=["page_read", "content_extract"],
        requires_auth=False,
        automatable=True,
        reliability_tier="medium",
        freshness_tier="live",
        cost_tier="free",
        adapter_class_name="PageReadAdapter",
        risk_notes="Reads public web pages. Content reliability varies.",
        enabled=False,  # scaffold — not yet implemented
    ))

    reg.register(SourceEntry(
        source_id="public_profile_page",
        display_name="Public Profile / About Page",
        source_family="public_profile",
        supported_target_types=["url", "profile_url"],
        supported_query_types=["public_profile", "profile_check", "affiliation_check"],
        requires_auth=False,
        automatable=True,
        reliability_tier="medium",
        freshness_tier="live",
        cost_tier="free",
        adapter_class_name="PublicProfileAdapter",
        risk_notes="Reads public profile/about pages. Structured extraction is conservative.",
        enabled=True,
    ))

    return reg
