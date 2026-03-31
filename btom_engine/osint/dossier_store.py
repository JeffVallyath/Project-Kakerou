"""DossierStore — bounded persistent storage for TargetContext dossiers.

JSON-file-backed, one file per target_id. Simple and auditable.
Identity is explicit: caller provides target_id. No fuzzy matching.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Optional

from btom_engine.osint.target_context import TargetContext

logger = logging.getLogger(__name__)


def _sanitize_filename(target_id: str) -> str:
    """Make target_id safe for use as a filename."""
    return re.sub(r"[^\w\-.]", "_", target_id)[:80]


class DossierStore:
    """Persistent store for TargetContext dossiers. One JSON file per target."""

    def __init__(self, store_dir: Path) -> None:
        self._dir = store_dir
        self._dir.mkdir(parents=True, exist_ok=True)

    def _path_for(self, target_id: str) -> Path:
        safe = _sanitize_filename(target_id)
        return self._dir / f"{safe}.json"

    def exists(self, target_id: str) -> bool:
        return self._path_for(target_id).exists()

    def load(self, target_id: str) -> Optional[TargetContext]:
        """Load a dossier by exact target_id. Returns None if not found."""
        path = self._path_for(target_id)
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            ctx = TargetContext.from_dict(data)
            logger.info("Loaded dossier for '%s' (%d evidence records)", target_id, ctx.evidence_count)
            return ctx
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.warning("Failed to load dossier for '%s': %s", target_id, e)
            return None

    def save(self, ctx: TargetContext) -> bool:
        """Save/overwrite a dossier. Returns True on success."""
        if not ctx.target_id:
            logger.warning("Cannot save dossier with empty target_id")
            return False
        path = self._path_for(ctx.target_id)
        try:
            path.write_text(json.dumps(ctx.to_dict(), indent=2, default=str), encoding="utf-8")
            logger.info("Saved dossier for '%s' (%d evidence records)", ctx.target_id, ctx.evidence_count)
            return True
        except Exception as e:
            logger.warning("Failed to save dossier for '%s': %s", ctx.target_id, e)
            return False

    def delete(self, target_id: str) -> bool:
        """Delete a dossier."""
        path = self._path_for(target_id)
        if path.exists():
            path.unlink()
            return True
        return False

    def list_targets(self) -> list[str]:
        """List all stored target_ids."""
        return [p.stem for p in self._dir.glob("*.json")]

    def resolve_identity(self, target_id: str, aliases: list[str] | None = None) -> Optional[str]:
        """Conservative identity resolution.

        Returns the matching stored target_id, or None if no match.
        Only matches on exact target_id or exact alias match.
        Does NOT do fuzzy matching.
        """
        # Exact match
        if self.exists(target_id):
            return target_id

        # Check if any alias matches a stored target
        if aliases:
            for stored in self.list_targets():
                stored_ctx = self.load(stored)
                if stored_ctx and stored_ctx.aliases:
                    for alias in aliases:
                        if alias in stored_ctx.aliases:
                            logger.info("Identity resolved: '%s' matched alias '%s' in dossier '%s'",
                                        target_id, alias, stored)
                            return stored

        return None
