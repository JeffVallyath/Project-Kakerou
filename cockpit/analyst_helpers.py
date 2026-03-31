"""Analyst helpers — speaker remap, turn ledger, OSINT trace formatting.

All functions are pure data transformers — no Streamlit dependency.
Testable independently.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Speaker remap
# ---------------------------------------------------------------------------

@dataclass
class SpeakerMapping:
    """One speaker's role assignment."""
    speaker_raw: str
    assigned_role: str = "Other"  # User, Target, Other
    auto_matched: bool = False    # True if set by alias matching


def build_speaker_mappings(
    raw_speakers: list[str],
    auto_roles: dict[str, str],
) -> list[SpeakerMapping]:
    """Build the initial speaker mapping table from parse results.

    auto_roles: {raw_speaker -> role} from alias matching.
    """
    mappings = []
    for spk in raw_speakers:
        role = auto_roles.get(spk, "Other")
        mappings.append(SpeakerMapping(
            speaker_raw=spk,
            assigned_role=role,
            auto_matched=spk in auto_roles,
        ))
    return mappings


def apply_remap(
    replay_queue: list[dict],
    remap: dict[str, str],
) -> list[dict]:
    """Apply manual speaker remap to a replay queue.

    remap: {raw_speaker -> role}
    Returns a new queue with updated speaker roles.
    """
    updated = []
    for entry in replay_queue:
        raw = entry.get("speaker_raw", "")
        new_role = remap.get(raw, entry.get("speaker", "Other"))
        updated.append({
            **entry,
            "speaker": new_role,
        })
    return updated


# ---------------------------------------------------------------------------
# Turn ledger
# ---------------------------------------------------------------------------

@dataclass
class LedgerEntry:
    """One row in the replay analysis ledger."""
    turn_index: int = 0
    engine_turn: int = 0  # engine turn number (only increments on Target turns)
    speaker_raw: str = ""
    speaker_role: str = ""
    text_excerpt: str = ""
    text_full_length: int = 0     # length of full text (to prove it's preserved)
    processed: bool = False       # True if it went through engine
    # Target turn fields
    bluffing: Optional[float] = None
    withholding: Optional[float] = None
    bluffing_delta: Optional[float] = None
    withholding_delta: Optional[float] = None
    # Pressure
    pressure_agg: Optional[float] = None
    # Motif / review
    motif_primary: str = ""
    review_ran: bool = False
    # OSINT
    osint_fired: bool = False
    osint_path: str = ""
    claims_count: int = 0
    # Dossier
    dossier_updated: bool = False
    patterns_active: int = 0


def build_ledger_entry(
    turn_index: int,
    entry: dict,
    result: Any = None,
    engine_turn: int = 0,
) -> LedgerEntry:
    """Build a ledger entry from a replay queue entry and optional engine result."""
    full_text = entry.get("text", "")
    le = LedgerEntry(
        turn_index=turn_index,
        engine_turn=engine_turn,
        speaker_raw=entry.get("speaker_raw", ""),
        speaker_role=entry.get("speaker", "Other"),
        text_excerpt=full_text[:80],
        text_full_length=len(full_text),
        processed=entry.get("speaker", "") in ("Target", "User"),
    )

    if result is None:
        return le

    # Extract from TurnResult
    state = getattr(result, "state", None)
    if state:
        hyps = getattr(state, "active_hypotheses", {})
        bluff = hyps.get("target_is_bluffing")
        withhold = hyps.get("target_is_withholding_info")
        if bluff:
            le.bluffing = round(bluff.probability, 4)
            le.bluffing_delta = round(bluff.momentum, 4)
        if withhold:
            le.withholding = round(withhold.probability, 4)
            le.withholding_delta = round(withhold.momentum, 4)

    pressure = getattr(result, "user_pressure", None)
    if pressure:
        le.pressure_agg = round(getattr(pressure, "aggregate", 0.0), 3)

    review = getattr(result, "semantic_review", None)
    if review:
        le.review_ran = getattr(review, "ran", False)
        le.motif_primary = getattr(review, "primary_class", "")

    prior_ctx = getattr(result, "prior_context", {})
    if isinstance(prior_ctx, dict):
        claims = prior_ctx.get("claims", [])
        le.claims_count = len(claims)
        path = prior_ctx.get("retrieval_path", "none")
        le.osint_path = path
        le.osint_fired = path != "none" and len(claims) > 0

        effect = prior_ctx.get("effect")
        if effect and getattr(effect, "comparisons_used", 0) > 0:
            le.dossier_updated = True

    ctx_summary = getattr(result, "target_context_summary", {})
    if isinstance(ctx_summary, dict):
        patterns = ctx_summary.get("behavioral_patterns", [])
        le.patterns_active = len(patterns)

    return le


# ---------------------------------------------------------------------------
# OSINT trace
# ---------------------------------------------------------------------------

@dataclass
class OsintTrace:
    """Compact OSINT trace for one turn."""
    fired: bool = False
    reason: str = ""           # why it fired or didn't
    claims: list[dict] = field(default_factory=list)
    retrieval_path: str = ""
    search_query: str = ""
    selected_urls: list[str] = field(default_factory=list)
    evidence_snippets: list[str] = field(default_factory=list)
    comparison_outcomes: list[str] = field(default_factory=list)
    effect_summary: str = ""


def build_osint_trace(prior_ctx: dict) -> OsintTrace:
    """Build an OSINT trace from prior_context diagnostics."""
    trace = OsintTrace()

    if not prior_ctx or not isinstance(prior_ctx, dict):
        trace.reason = "no prior context available"
        return trace

    claims = prior_ctx.get("claims", [])
    if not claims:
        trace.reason = "no retrieval-worthy claims extracted"
        return trace

    trace.claims = claims
    trace.retrieval_path = prior_ctx.get("retrieval_path", "none")

    if trace.retrieval_path == "none":
        trace.reason = "planner chose no_retrieval"
        return trace

    trace.fired = True
    trace.reason = f"retrieval via {trace.retrieval_path}"

    # Selected URLs
    for url_info in prior_ctx.get("selected_urls", []):
        if isinstance(url_info, dict):
            trace.selected_urls.append(url_info.get("url", ""))
        elif isinstance(url_info, str):
            trace.selected_urls.append(url_info)

    # Evidence
    for rec in prior_ctx.get("retrieval_records", []):
        if isinstance(rec, dict) and rec.get("snippet"):
            trace.evidence_snippets.append(f"[{rec.get('source', '?')}] {rec['snippet'][:80]}")

    # Comparisons
    for comp in prior_ctx.get("comparisons", []):
        if isinstance(comp, dict):
            trace.comparison_outcomes.append(
                f"{comp.get('outcome', '?')} (conf={comp.get('confidence', 0):.2f}): {comp.get('rationale', '')[:60]}"
            )

    # Effect
    effect = prior_ctx.get("effect")
    if effect and getattr(effect, "comparisons_used", 0) > 0:
        trace.effect_summary = (
            f"bluff_delta={getattr(effect, 'bluffing_delta', 0):+.3f} "
            f"withhold_delta={getattr(effect, 'withholding_delta', 0):+.3f}"
        )
    else:
        trace.effect_summary = "no actionable effect"

    return trace
