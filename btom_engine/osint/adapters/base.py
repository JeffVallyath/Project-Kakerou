"""Base adapter interface for all OSINT sources.

Every adapter must implement this interface. The retrieval manager
calls adapters through this contract — it never touches raw source APIs.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from btom_engine.osint.evidence_schema import (
    EvidenceRecord,
    RetrievalRequest,
    SourceRun,
)


@dataclass
class AdapterHealth:
    """Health status of an adapter."""

    available: bool = True
    message: str = ""


class BaseAdapter(ABC):
    """Base interface for OSINT source adapters."""

    source_id: str = ""

    @abstractmethod
    def can_handle(self, request: RetrievalRequest) -> bool:
        """Whether this adapter can handle the given request."""
        ...

    @abstractmethod
    def execute(self, request: RetrievalRequest) -> tuple[list[EvidenceRecord], SourceRun]:
        """Execute the retrieval and return normalized records + diagnostics."""
        ...

    def healthcheck(self) -> AdapterHealth:
        """Check if the adapter is available. Override if needed."""
        return AdapterHealth(available=True, message="ok")

    def estimate_cost(self, request: RetrievalRequest) -> float:
        """Estimated cost (0.0 = free). Override if needed."""
        return 0.0
