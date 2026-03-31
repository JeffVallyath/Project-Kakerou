"""Prior Statements Adapter — retrieves target's own prior statements from session transcript.

This is the first real OSINT adapter. It works with data already available
in the engine's session — no external API calls needed.
"""

from __future__ import annotations

import re
from typing import Any

from btom_engine.osint.adapters.base import BaseAdapter
from btom_engine.osint.evidence_schema import EvidenceRecord, RetrievalRequest, SourceRun


class PriorStatementsAdapter(BaseAdapter):
    """Retrieves prior target statements from a conversation transcript.

    The transcript is passed as context at execution time, not stored
    internally. This adapter has no external dependencies.
    """

    source_id = "prior_statements_session"

    def __init__(self) -> None:
        self._transcript: list[dict] = []

    def set_transcript(self, transcript: list[dict]) -> None:
        """Set the conversation transcript for retrieval.

        Each entry: {"speaker": "target"|"user", "text": "...", "turn": int|None}
        """
        self._transcript = transcript

    def can_handle(self, request: RetrievalRequest) -> bool:
        return (
            request.query_type in ("prior_statements", "claim_check")
            and request.target_type in ("session", "target_entity")
        )

    def execute(self, request: RetrievalRequest) -> tuple[list[EvidenceRecord], SourceRun]:
        run = SourceRun(
            source_id=self.source_id,
            adapter_class="PriorStatementsAdapter",
        )

        try:
            records = self._search_transcript(request)
            run.success = True
            run.records_returned = len(records)
            return records, run

        except Exception as e:
            run.success = False
            run.error = f"{type(e).__name__}: {e}"
            return [], run

    def _search_transcript(self, request: RetrievalRequest) -> list[EvidenceRecord]:
        """Search the transcript for relevant prior statements."""
        target_turns = [
            entry for entry in self._transcript
            if entry.get("speaker", "").lower() == "target"
        ]

        if not target_turns:
            return []

        query_lower = request.query_text.lower() if request.query_text else ""
        query_words = set(re.sub(r"[^\w\s]", "", query_lower).split())

        scored: list[tuple[float, dict]] = []
        for entry in target_turns:
            text = entry.get("text", "")
            text_lower = text.lower()
            text_words = set(re.sub(r"[^\w\s]", "", text_lower).split())

            # Simple word-overlap relevance scoring
            if query_words and text_words:
                overlap = len(query_words & text_words)
                relevance = overlap / max(len(query_words), 1)
            else:
                relevance = 0.1  # low default if no query

            scored.append((relevance, entry))

        # Sort by relevance, take top_k
        scored.sort(key=lambda x: x[0], reverse=True)
        top = scored[:request.top_k]

        records = []
        for relevance, entry in top:
            text = entry.get("text", "")
            turn = entry.get("turn", None)
            records.append(EvidenceRecord(
                source_id=self.source_id,
                source_family="prior_statements",
                target_type=request.target_type,
                target_value=request.target_value,
                content_type="text",
                title=f"Target statement (turn {turn})" if turn else "Target statement",
                snippet=text[:200],
                full_text=text if len(text) > 200 else None,
                url_or_citation=f"session:{request.session_id or 'current'}:turn:{turn or '?'}",
                confidence=0.95,  # high — it's the actual transcript
                reliability_tier="high",
                relevance_score=round(relevance, 3),
            ))

        return records
