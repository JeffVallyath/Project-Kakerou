"""TargetContext — bounded synthesized dossier for reusable target baseline.

Fuses retrieved evidence into a compact, citation-backed context object
that persists across turns within a session. The engine uses it as an
optional soft conditioning layer — not a replacement for behavioral signals.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class EvidenceLink:
    """Citation for one piece of supporting evidence."""
    source_family: str = ""       # prior_statements, web_search, public_profile
    snippet: str = ""
    url_or_citation: str = ""
    confidence: float = 0.0
    turn_added: int = 0


@dataclass
class SupportedClaim:
    """A claim that has been supported by retrieved evidence."""
    claim_text: str = ""
    claim_type: str = ""
    support_count: int = 1
    evidence: list[EvidenceLink] = field(default_factory=list)
    confidence: float = 0.0


@dataclass
class ContradictedClaim:
    """A claim that has been contradicted by retrieved evidence."""
    claim_text: str = ""
    claim_type: str = ""
    contradiction_count: int = 1
    evidence: list[EvidenceLink] = field(default_factory=list)
    confidence: float = 0.0


@dataclass
class PatternObservation:
    """One observation supporting a behavioral pattern."""
    turn: int = 0
    session_id: str = ""
    cue: str = ""          # what was observed
    confidence: float = 0.0
    condition: str = ""    # interaction condition (e.g., "high_accusation", "low_pressure")


_RECENCY_HALF_LIFE = 15  # observations older than this many turns count at half weight


@dataclass
class BehavioralPattern:
    """An operational behavioral pattern observed across turns/sessions."""
    pattern_id: str = ""
    description: str = ""
    support_count: int = 0
    contrary_count: int = 0       # observations that cut against this pattern
    confidence: float = 0.0
    last_seen_turn: int = 0
    last_seen_session: str = ""
    observations: list[PatternObservation] = field(default_factory=list)
    weakened_reason: str = ""     # why confidence was lowered, if applicable

    _MAX_OBSERVATIONS = 10

    def add_observation(self, obs: PatternObservation) -> None:
        self.observations.append(obs)
        if len(self.observations) > self._MAX_OBSERVATIONS:
            self.observations = self.observations[-self._MAX_OBSERVATIONS:]
        self.support_count += 1
        self._recompute_confidence(obs.turn)
        self.last_seen_turn = obs.turn
        self.last_seen_session = obs.session_id

    def add_contrary(self, turn: int, reason: str = "") -> None:
        """Record evidence that cuts against this pattern."""
        self.contrary_count += 1
        self.weakened_reason = reason or f"contrary at turn {turn}"
        self._recompute_confidence(turn)

    def _recompute_confidence(self, current_turn: int = 0) -> None:
        """Recompute confidence with recency weighting and contrary evidence."""
        if not self.observations:
            self.confidence = 0.0
            return

        # Recency-weighted effective support
        weighted_support = 0.0
        for obs in self.observations:
            age = max(0, current_turn - obs.turn) if current_turn > 0 else 0
            weight = 0.5 ** (age / _RECENCY_HALF_LIFE) if age > 0 else 1.0
            weighted_support += weight

        # Contrary evidence reduces confidence
        net_support = weighted_support - self.contrary_count * 0.5

        # Confidence formula: bounded, requires net positive support
        if net_support <= 0:
            self.confidence = max(0.0, 0.10)  # near-zero but not fully deleted
        else:
            self.confidence = min(0.90, 0.15 + net_support * 0.10)

    @property
    def is_weakened(self) -> bool:
        return self.contrary_count > 0 and self.contrary_count >= self.support_count // 2


# Valid pattern IDs — the system can only produce these
VALID_PATTERNS = {
    "references_prior_explanations",
    "denial_contradiction_recurs",
    "tends_to_answer_directly",
    "tends_to_evade_direct_questions",
    "role_claims_stable",
    "role_claims_inconsistent",
    # Conditional patterns
    "evasive_under_accusation",
    "tone_shift_under_accusation",
    "narrowing_response_is_evasive",
    "direct_when_low_pressure_only",
    # Strategy patterns (higher-order)
    "deny_then_reframe",
    "partial_answer_then_evasion",
    "blame_displacement_under_pressure",
}

_PATTERN_DESCRIPTIONS = {
    "references_prior_explanations": "Target frequently references prior explanations with supporting evidence",
    "denial_contradiction_recurs": "Target's denials are repeatedly contradicted by evidence",
    "tends_to_answer_directly": "Target tends to provide direct, concrete answers",
    "tends_to_evade_direct_questions": "Target tends to evade or deflect direct questions",
    "role_claims_stable": "Target's role/affiliation claims are consistently supported",
    "role_claims_inconsistent": "Target's role/affiliation claims show inconsistencies",
    "evasive_under_accusation": "Target becomes evasive specifically when accused",
    "tone_shift_under_accusation": "Target shifts to defensive/emotional tone when accused",
    "narrowing_response_is_evasive": "Target evades when questions narrow / repeat",
    "direct_when_low_pressure_only": "Target is direct only when pressure is low",
    "deny_then_reframe": "Target denies, then shifts to tone/process/blame reframing under pressure",
    "partial_answer_then_evasion": "Target partially answers then evades the core pressured question",
    "blame_displacement_under_pressure": "Target displaces blame to others/process when pressured on substance",
}

# Pressure band thresholds
_HIGH_PRESSURE = 0.30
_LOW_PRESSURE = 0.10

_MAX_PATTERNS = 8


@dataclass
class TargetContext:
    """Compact synthesized target dossier. Bounded and auditable."""

    target_id: str = ""
    aliases: list[str] = field(default_factory=list)
    known_affiliations: list[str] = field(default_factory=list)
    known_roles: list[str] = field(default_factory=list)
    supported_claims: list[SupportedClaim] = field(default_factory=list)
    contradicted_claims: list[ContradictedClaim] = field(default_factory=list)
    behavioral_patterns: list[BehavioralPattern] = field(default_factory=list)
    statement_patterns: list[str] = field(default_factory=list)  # legacy simple patterns
    overall_consistency: float = 0.5
    evidence_count: int = 0
    last_updated_turn: int = 0
    created_session: str = ""
    last_updated_session: str = ""

    @property
    def has_content(self) -> bool:
        return self.evidence_count > 0

    @property
    def contradiction_ratio(self) -> float:
        """Ratio of contradicted claims to total claims. 0 if no claims."""
        total = len(self.supported_claims) + len(self.contradicted_claims)
        if total == 0:
            return 0.0
        return len(self.contradicted_claims) / total

    def summary(self) -> dict[str, Any]:
        """Compact summary for diagnostics."""
        return {
            "target_id": self.target_id,
            "evidence_count": self.evidence_count,
            "affiliations": self.known_affiliations[:3],
            "roles": self.known_roles[:3],
            "supported": len(self.supported_claims),
            "contradicted": len(self.contradicted_claims),
            "behavioral_patterns": [
                {"id": p.pattern_id, "conf": round(p.confidence, 2), "count": p.support_count}
                for p in self.behavioral_patterns if p.confidence > 0.2
            ],
            "consistency": round(self.overall_consistency, 2),
            "contradiction_ratio": round(self.contradiction_ratio, 2),
            "loaded_from_store": bool(self.created_session and self.created_session != self.last_updated_session),
        }

    def get_pattern(self, pattern_id: str) -> BehavioralPattern | None:
        """Get a specific pattern by ID."""
        return next((p for p in self.behavioral_patterns if p.pattern_id == pattern_id), None)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict for JSON persistence."""
        def _ev(e: EvidenceLink) -> dict:
            return {"source_family": e.source_family, "snippet": e.snippet,
                    "url_or_citation": e.url_or_citation, "confidence": e.confidence, "turn_added": e.turn_added}
        def _sc(s: SupportedClaim) -> dict:
            return {"claim_text": s.claim_text, "claim_type": s.claim_type, "support_count": s.support_count,
                    "evidence": [_ev(e) for e in s.evidence], "confidence": s.confidence}
        def _cc(c: ContradictedClaim) -> dict:
            return {"claim_text": c.claim_text, "claim_type": c.claim_type, "contradiction_count": c.contradiction_count,
                    "evidence": [_ev(e) for e in c.evidence], "confidence": c.confidence}
        def _po(o: PatternObservation) -> dict:
            return {"turn": o.turn, "session_id": o.session_id, "cue": o.cue, "confidence": o.confidence, "condition": o.condition}
        def _bp(p: BehavioralPattern) -> dict:
            return {"pattern_id": p.pattern_id, "description": p.description, "support_count": p.support_count,
                    "contrary_count": p.contrary_count, "confidence": p.confidence, "last_seen_turn": p.last_seen_turn,
                    "last_seen_session": p.last_seen_session, "weakened_reason": p.weakened_reason,
                    "observations": [_po(o) for o in p.observations]}
        return {
            "target_id": self.target_id, "aliases": self.aliases,
            "known_affiliations": self.known_affiliations, "known_roles": self.known_roles,
            "supported_claims": [_sc(s) for s in self.supported_claims],
            "contradicted_claims": [_cc(c) for c in self.contradicted_claims],
            "behavioral_patterns": [_bp(p) for p in self.behavioral_patterns],
            "statement_patterns": self.statement_patterns,
            "overall_consistency": self.overall_consistency, "evidence_count": self.evidence_count,
            "last_updated_turn": self.last_updated_turn,
            "created_session": self.created_session, "last_updated_session": self.last_updated_session,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> TargetContext:
        """Deserialize from dict."""
        def _ev(e: dict) -> EvidenceLink:
            return EvidenceLink(**{k: e.get(k, v) for k, v in EvidenceLink().__dict__.items()})
        def _sc(s: dict) -> SupportedClaim:
            return SupportedClaim(claim_text=s.get("claim_text", ""), claim_type=s.get("claim_type", ""),
                                  support_count=s.get("support_count", 1),
                                  evidence=[_ev(e) for e in s.get("evidence", [])], confidence=s.get("confidence", 0.0))
        def _cc(c: dict) -> ContradictedClaim:
            return ContradictedClaim(claim_text=c.get("claim_text", ""), claim_type=c.get("claim_type", ""),
                                     contradiction_count=c.get("contradiction_count", 1),
                                     evidence=[_ev(e) for e in c.get("evidence", [])], confidence=c.get("confidence", 0.0))
        def _po(o: dict) -> PatternObservation:
            return PatternObservation(turn=o.get("turn", 0), session_id=o.get("session_id", ""),
                                      cue=o.get("cue", ""), confidence=o.get("confidence", 0.0),
                                      condition=o.get("condition", ""))
        def _bp(p: dict) -> BehavioralPattern:
            return BehavioralPattern(pattern_id=p.get("pattern_id", ""), description=p.get("description", ""),
                                     support_count=p.get("support_count", 0), contrary_count=p.get("contrary_count", 0),
                                     confidence=p.get("confidence", 0.0),
                                     last_seen_turn=p.get("last_seen_turn", 0), last_seen_session=p.get("last_seen_session", ""),
                                     weakened_reason=p.get("weakened_reason", ""),
                                     observations=[_po(o) for o in p.get("observations", [])])
        return cls(
            target_id=d.get("target_id", ""), aliases=d.get("aliases", []),
            known_affiliations=d.get("known_affiliations", []), known_roles=d.get("known_roles", []),
            supported_claims=[_sc(s) for s in d.get("supported_claims", [])],
            contradicted_claims=[_cc(c) for c in d.get("contradicted_claims", [])],
            behavioral_patterns=[_bp(p) for p in d.get("behavioral_patterns", [])],
            statement_patterns=d.get("statement_patterns", []),
            overall_consistency=d.get("overall_consistency", 0.5), evidence_count=d.get("evidence_count", 0),
            last_updated_turn=d.get("last_updated_turn", 0),
            created_session=d.get("created_session", ""), last_updated_session=d.get("last_updated_session", ""),
        )


# ---------------------------------------------------------------------------
# Synthesis — update TargetContext from comparison results
# ---------------------------------------------------------------------------

_MAX_CLAIMS_PER_TYPE = 5  # prevent unbounded growth
_MAX_AFFILIATIONS = 5
_MAX_ROLES = 5
_MAX_PATTERNS = 5


def synthesize_context(
    ctx: TargetContext,
    comparisons: list[dict],
    claims: list[dict],
    retrieval_records: list[dict],
    current_turn: int,
) -> TargetContext:
    """Update TargetContext conservatively from turn results.

    Only adds what evidence actually supports. Prefers omission over fake certainty.
    """
    if not comparisons:
        return ctx

    for comp in comparisons:
        outcome = comp.get("outcome", "insufficient_evidence")
        claim_text = comp.get("claim", "")
        confidence = comp.get("confidence", 0.0)
        rationale = comp.get("rationale", "")

        if outcome == "insufficient_evidence":
            continue

        # Build evidence link from retrieval records
        ev_link = EvidenceLink(
            snippet=rationale[:100],
            confidence=confidence,
            turn_added=current_turn,
        )
        # Try to attach source info from records
        for rec in retrieval_records:
            if rec.get("snippet", "")[:30] in rationale:
                ev_link.source_family = rec.get("source", "")
                ev_link.url_or_citation = rec.get("url", "")
                break

        if outcome == "supported_by_prior":
            # Check if we already have this claim supported
            existing = next((s for s in ctx.supported_claims if s.claim_text == claim_text), None)
            if existing:
                existing.support_count += 1
                existing.evidence.append(ev_link)
                existing.confidence = min(0.9, existing.confidence + 0.05)
            elif len(ctx.supported_claims) < _MAX_CLAIMS_PER_TYPE:
                ctx.supported_claims.append(SupportedClaim(
                    claim_text=claim_text,
                    claim_type=_infer_claim_type(claims, claim_text),
                    evidence=[ev_link],
                    confidence=confidence,
                ))

        elif outcome == "direct_contradiction":
            existing = next((c for c in ctx.contradicted_claims if c.claim_text == claim_text), None)
            if existing:
                existing.contradiction_count += 1
                existing.evidence.append(ev_link)
                existing.confidence = min(0.9, existing.confidence + 0.05)
            elif len(ctx.contradicted_claims) < _MAX_CLAIMS_PER_TYPE:
                ctx.contradicted_claims.append(ContradictedClaim(
                    claim_text=claim_text,
                    claim_type=_infer_claim_type(claims, claim_text),
                    evidence=[ev_link],
                    confidence=confidence,
                ))

        elif outcome == "weak_tension":
            # Don't add to contradicted unless it recurs
            pass

    # Extract affiliations/roles from supported claims
    for claim in claims:
        ctype = claim.get("type", "")
        ctext = claim.get("text", "")
        if ctype == "affiliation_claim" and ctext:
            short = ctext[:60]
            if short not in ctx.known_affiliations and len(ctx.known_affiliations) < _MAX_AFFILIATIONS:
                # Only add if supported
                if any(s.claim_text == ctext for s in ctx.supported_claims):
                    ctx.known_affiliations.append(short)
        elif ctype == "role_claim" and ctext:
            short = ctext[:60]
            if short not in ctx.known_roles and len(ctx.known_roles) < _MAX_ROLES:
                if any(s.claim_text == ctext for s in ctx.supported_claims):
                    ctx.known_roles.append(short)

    # Update patterns
    if len(ctx.contradicted_claims) >= 2 and len(ctx.statement_patterns) < _MAX_PATTERNS:
        pattern = "repeated contradictions detected"
        if pattern not in ctx.statement_patterns:
            ctx.statement_patterns.append(pattern)

    supported_count = sum(s.support_count for s in ctx.supported_claims)
    contradicted_count = sum(c.contradiction_count for c in ctx.contradicted_claims)
    total = supported_count + contradicted_count
    if total > 0:
        ctx.overall_consistency = supported_count / total
    else:
        ctx.overall_consistency = 0.5

    ctx.evidence_count += len([c for c in comparisons if c.get("outcome") != "insufficient_evidence"])
    ctx.last_updated_turn = current_turn

    return ctx


def _infer_claim_type(claims: list[dict], claim_text: str) -> str:
    for c in claims:
        if c.get("text", "") == claim_text:
            return c.get("type", "unknown")
    return "unknown"


# ---------------------------------------------------------------------------
# Behavioral pattern synthesis
# ---------------------------------------------------------------------------

def _get_or_create_pattern(ctx: TargetContext, pattern_id: str) -> BehavioralPattern:
    """Get existing pattern or create a new one."""
    existing = ctx.get_pattern(pattern_id)
    if existing:
        return existing
    if len(ctx.behavioral_patterns) >= _MAX_PATTERNS:
        return None  # type: ignore  # bounded
    bp = BehavioralPattern(
        pattern_id=pattern_id,
        description=_PATTERN_DESCRIPTIONS.get(pattern_id, pattern_id),
    )
    ctx.behavioral_patterns.append(bp)
    return bp


def synthesize_patterns(
    ctx: TargetContext,
    comparisons: list[dict],
    claims: list[dict],
    current_turn: int,
    session_id: str = "",
    signals: Any = None,
    pressure: Any = None,
) -> TargetContext:
    """Synthesize behavioral patterns from current turn evidence + signals + pressure.

    Claim-based patterns: from comparisons.
    Signal-based patterns: from turn-level signal observations.
    Conditional patterns: from signal + pressure combinations.
    Also handles cross-pattern weakening.
    """
    # --- Claim-based patterns (unchanged) ---
    for comp in comparisons:
        outcome = comp.get("outcome", "insufficient_evidence")
        confidence = comp.get("confidence", 0.0)
        if outcome == "insufficient_evidence":
            continue

        claim_text = comp.get("claim", "")
        claim_type = _infer_claim_type(claims, claim_text)

        obs = PatternObservation(
            turn=current_turn, session_id=session_id,
            cue=f"{claim_type}:{outcome}", confidence=confidence,
        )

        if claim_type in ("prior_explanation_claim", "prior_statement_reference") and outcome == "supported_by_prior":
            bp = _get_or_create_pattern(ctx, "references_prior_explanations")
            if bp:
                bp.add_observation(obs)

        if claim_type == "denial_of_prior_statement" and outcome == "direct_contradiction":
            bp = _get_or_create_pattern(ctx, "denial_contradiction_recurs")
            if bp:
                bp.add_observation(obs)

        if claim_type in ("role_claim", "affiliation_claim") and outcome == "supported_by_prior":
            bp = _get_or_create_pattern(ctx, "role_claims_stable")
            if bp:
                bp.add_observation(obs)
            # Weaken the opposite pattern if it exists
            opposite = ctx.get_pattern("role_claims_inconsistent")
            if opposite:
                opposite.add_contrary(current_turn, "role support contradicts inconsistency")

        if claim_type in ("role_claim", "affiliation_claim") and outcome == "direct_contradiction":
            bp = _get_or_create_pattern(ctx, "role_claims_inconsistent")
            if bp:
                bp.add_observation(obs)
            opposite = ctx.get_pattern("role_claims_stable")
            if opposite:
                opposite.add_contrary(current_turn, "role contradiction contradicts stability")

    # --- Signal-based patterns ---
    if signals is not None:
        compliance_val = getattr(getattr(signals, "direct_answer_compliance", None), "value", 0.0)
        compliance_rel = getattr(getattr(signals, "direct_answer_compliance", None), "signal_reliability", 0.0)
        evasion_val = getattr(getattr(signals, "evasive_deflection", None), "value", 0.0)
        evasion_rel = getattr(getattr(signals, "evasive_deflection", None), "signal_reliability", 0.0)

        compliance_eff = compliance_val * compliance_rel
        evasion_eff = evasion_val * evasion_rel

        # tends_to_answer_directly: high compliance, not offset by evasion
        if compliance_eff > 0.20 and evasion_eff < 0.15:
            bp = _get_or_create_pattern(ctx, "tends_to_answer_directly")
            if bp:
                obs = PatternObservation(
                    turn=current_turn, session_id=session_id,
                    cue=f"compliance={compliance_val:.2f}*{compliance_rel:.2f}", confidence=compliance_eff,
                )
                bp.add_observation(obs)
            # Weaken opposite
            opposite = ctx.get_pattern("tends_to_evade_direct_questions")
            if opposite:
                opposite.add_contrary(current_turn, "direct answer contradicts evasion pattern")

        # tends_to_evade_direct_questions: high evasion, low compliance
        if evasion_eff > 0.20 and compliance_eff < 0.15:
            bp = _get_or_create_pattern(ctx, "tends_to_evade_direct_questions")
            if bp:
                obs = PatternObservation(
                    turn=current_turn, session_id=session_id,
                    cue=f"evasion={evasion_val:.2f}*{evasion_rel:.2f}", confidence=evasion_eff,
                )
                bp.add_observation(obs)
            # Weaken opposite
            opposite = ctx.get_pattern("tends_to_answer_directly")
            if opposite:
                opposite.add_contrary(current_turn, "evasion contradicts direct-answer pattern")

    # --- Conditional patterns (pressure + signal combinations) ---
    if signals is not None and pressure is not None:
        acc = getattr(pressure, "accusation", 0.0)
        rep = getattr(pressure, "repetition", 0.0)
        agg = getattr(pressure, "aggregate", 0.0)

        defense_val = getattr(getattr(signals, "defensive_justification", None), "value", 0.0)
        defense_rel = getattr(getattr(signals, "defensive_justification", None), "signal_reliability", 0.0)
        defense_eff = defense_val * defense_rel

        # Determine pressure condition label
        if acc >= _HIGH_PRESSURE:
            condition = "high_accusation"
        elif rep >= _HIGH_PRESSURE:
            condition = "high_repetition"
        elif agg >= _HIGH_PRESSURE:
            condition = "high_pressure"
        elif agg <= _LOW_PRESSURE:
            condition = "low_pressure"
        else:
            condition = "moderate"

        # evasive_under_accusation: high accusation + high evasion
        if acc >= _HIGH_PRESSURE and evasion_eff > 0.15:
            bp = _get_or_create_pattern(ctx, "evasive_under_accusation")
            if bp:
                bp.add_observation(PatternObservation(
                    turn=current_turn, session_id=session_id,
                    cue=f"acc={acc:.2f}+evasion={evasion_eff:.2f}",
                    confidence=min(acc, evasion_eff), condition=condition,
                ))

        # tone_shift_under_accusation: high accusation + high defense + low compliance
        if acc >= _HIGH_PRESSURE and defense_eff > 0.15 and compliance_eff < 0.15:
            bp = _get_or_create_pattern(ctx, "tone_shift_under_accusation")
            if bp:
                bp.add_observation(PatternObservation(
                    turn=current_turn, session_id=session_id,
                    cue=f"acc={acc:.2f}+defense={defense_eff:.2f}",
                    confidence=min(acc, defense_eff), condition=condition,
                ))

        # narrowing_response_is_evasive: high repetition + high evasion
        if rep >= _HIGH_PRESSURE and evasion_eff > 0.15:
            bp = _get_or_create_pattern(ctx, "narrowing_response_is_evasive")
            if bp:
                bp.add_observation(PatternObservation(
                    turn=current_turn, session_id=session_id,
                    cue=f"rep={rep:.2f}+evasion={evasion_eff:.2f}",
                    confidence=min(rep, evasion_eff), condition=condition,
                ))

        # direct_when_low_pressure_only: low pressure + high compliance
        if agg <= _LOW_PRESSURE and compliance_eff > 0.20:
            bp = _get_or_create_pattern(ctx, "direct_when_low_pressure_only")
            if bp:
                bp.add_observation(PatternObservation(
                    turn=current_turn, session_id=session_id,
                    cue=f"low_pressure+compliance={compliance_eff:.2f}",
                    confidence=compliance_eff, condition="low_pressure",
                ))
            # If this target is also evasive under high pressure, weaken tends_to_answer_directly
            if ctx.get_pattern("evasive_under_accusation"):
                opp = ctx.get_pattern("tends_to_answer_directly")
                if opp:
                    opp.add_contrary(current_turn, "direct only under low pressure")

    # --- Idle decay: recompute all pattern confidences with current turn ---
    for p in ctx.behavioral_patterns:
        p._recompute_confidence(current_turn)

    return ctx


# ---------------------------------------------------------------------------
# Strategy synthesis — higher-order patterns from signal/pressure co-occurrence
# ---------------------------------------------------------------------------

_MAX_STRATEGIES = 5


def synthesize_strategies(
    ctx: TargetContext,
    current_turn: int,
    session_id: str = "",
    signals: Any = None,
    pressure: Any = None,
    comparisons: list[dict] | None = None,
) -> TargetContext:
    """Synthesize higher-order strategy patterns from turn-local signal + pressure co-occurrence.

    Strategies are detected from multi-signal combinations within one turn,
    not inferred from stored lower-level patterns. Requires repeated evidence.
    """
    if signals is None or pressure is None:
        return ctx

    acc = getattr(pressure, "accusation", 0.0)
    rep = getattr(pressure, "repetition", 0.0)
    agg = getattr(pressure, "aggregate", 0.0)

    compliance_eff = getattr(getattr(signals, "direct_answer_compliance", None), "value", 0.0) * \
                     getattr(getattr(signals, "direct_answer_compliance", None), "signal_reliability", 0.0)
    evasion_eff = getattr(getattr(signals, "evasive_deflection", None), "value", 0.0) * \
                  getattr(getattr(signals, "evasive_deflection", None), "signal_reliability", 0.0)
    defense_eff = getattr(getattr(signals, "defensive_justification", None), "value", 0.0) * \
                  getattr(getattr(signals, "defensive_justification", None), "signal_reliability", 0.0)
    emotion_eff = getattr(getattr(signals, "emotional_intensity", None), "value", 0.0) * \
                  getattr(getattr(signals, "emotional_intensity", None), "signal_reliability", 0.0)

    # Check for denial in comparisons
    has_denial_contradiction = False
    if comparisons:
        for c in comparisons:
            if c.get("outcome") == "direct_contradiction":
                has_denial_contradiction = True
                break

    # Strategy: deny_then_reframe
    # Denial/contradiction + defensive tone shift + low compliance, under accusation
    if has_denial_contradiction and defense_eff > 0.15 and compliance_eff < 0.15 and acc >= _HIGH_PRESSURE:
        bp = _get_or_create_pattern(ctx, "deny_then_reframe")
        if bp:
            bp.add_observation(PatternObservation(
                turn=current_turn, session_id=session_id,
                cue=f"denial+defense={defense_eff:.2f}+acc={acc:.2f}",
                confidence=min(acc, defense_eff),
                condition="high_accusation",
            ))

    # Strategy: partial_answer_then_evasion
    # Some compliance + meaningful evasion in the same turn
    if compliance_eff > 0.10 and evasion_eff > 0.15 and agg >= _LOW_PRESSURE:
        bp = _get_or_create_pattern(ctx, "partial_answer_then_evasion")
        if bp:
            bp.add_observation(PatternObservation(
                turn=current_turn, session_id=session_id,
                cue=f"compliance={compliance_eff:.2f}+evasion={evasion_eff:.2f}",
                confidence=min(compliance_eff, evasion_eff),
                condition="mixed_response",
            ))

    # Strategy: blame_displacement_under_pressure
    # High accusation + high defense (blame-shifting tone) + high emotion + low compliance
    if acc >= _HIGH_PRESSURE and defense_eff > 0.20 and emotion_eff > 0.15 and compliance_eff < 0.10:
        bp = _get_or_create_pattern(ctx, "blame_displacement_under_pressure")
        if bp:
            bp.add_observation(PatternObservation(
                turn=current_turn, session_id=session_id,
                cue=f"acc={acc:.2f}+defense={defense_eff:.2f}+emotion={emotion_eff:.2f}",
                confidence=min(acc, defense_eff, emotion_eff),
                condition="high_accusation",
            ))

    # Idle decay for all patterns
    for p in ctx.behavioral_patterns:
        p._recompute_confidence(current_turn)

    return ctx


# ---------------------------------------------------------------------------
# Soft conditioning — bounded effect on hypotheses
# ---------------------------------------------------------------------------

_MAX_CONTEXT_DELTA = 0.05  # very small bounded effect


@dataclass
class ContextEffect:
    """Bounded effect of TargetContext on hypotheses."""
    bluffing_delta: float = 0.0
    withholding_delta: float = 0.0
    rationale: str = ""


def compute_context_conditioning(ctx: TargetContext) -> ContextEffect:
    """Compute a bounded soft effect from accumulated TargetContext.

    Rules:
    - High consistency (many supported, few contradicted) → mild de-escalation
    - High contradiction ratio → mild scrutiny increase
    - Weak/empty context → no effect
    """
    if not ctx.has_content or ctx.evidence_count < 2:
        return ContextEffect(rationale="insufficient context")

    ratio = ctx.contradiction_ratio

    if ratio > 0.5 and len(ctx.contradicted_claims) >= 2:
        # Repeated contradictions → raise scrutiny
        delta = min(_MAX_CONTEXT_DELTA, ratio * 0.08)
        return ContextEffect(
            bluffing_delta=delta,
            withholding_delta=delta * 0.5,
            rationale=f"contradiction_ratio={ratio:.2f}, {len(ctx.contradicted_claims)} contradicted claims",
        )

    if ctx.overall_consistency > 0.7 and len(ctx.supported_claims) >= 2:
        # Consistent target → mild de-escalation
        delta = min(_MAX_CONTEXT_DELTA, (ctx.overall_consistency - 0.5) * 0.1)
        return ContextEffect(
            bluffing_delta=-delta,
            withholding_delta=-delta * 0.5,
            rationale=f"consistency={ctx.overall_consistency:.2f}, {len(ctx.supported_claims)} supported claims",
        )

    # Pattern-based conditioning (bounded additions)
    pattern_delta_bluff = 0.0
    pattern_delta_withhold = 0.0
    pattern_reasons = []

    for p in ctx.behavioral_patterns:
        if p.confidence < 0.3 or p.support_count < 2:
            continue  # not stable enough

        if p.pattern_id == "denial_contradiction_recurs":
            pattern_delta_bluff += 0.02
            pattern_reasons.append(f"denial_recurs(conf={p.confidence:.2f})")
        elif p.pattern_id == "references_prior_explanations":
            pattern_delta_bluff -= 0.01
            pattern_reasons.append(f"prior_refs(conf={p.confidence:.2f})")
        elif p.pattern_id == "role_claims_stable":
            pattern_delta_bluff -= 0.01
            pattern_reasons.append(f"role_stable(conf={p.confidence:.2f})")
        elif p.pattern_id == "role_claims_inconsistent":
            pattern_delta_bluff += 0.02
            pattern_reasons.append(f"role_inconsistent(conf={p.confidence:.2f})")
        elif p.pattern_id == "tends_to_answer_directly":
            pattern_delta_bluff -= 0.015
            pattern_reasons.append(f"direct_answers(conf={p.confidence:.2f})")
        elif p.pattern_id == "tends_to_evade_direct_questions":
            pattern_delta_bluff += 0.02
            pattern_delta_withhold += 0.01
            pattern_reasons.append(f"evasive_pattern(conf={p.confidence:.2f})")
        elif p.pattern_id == "evasive_under_accusation":
            pattern_delta_bluff += 0.015
            pattern_delta_withhold += 0.01
            pattern_reasons.append(f"evasive_under_acc(conf={p.confidence:.2f})")
        elif p.pattern_id == "tone_shift_under_accusation":
            pattern_delta_bluff += 0.01
            pattern_reasons.append(f"tone_shift_acc(conf={p.confidence:.2f})")
        elif p.pattern_id == "narrowing_response_is_evasive":
            pattern_delta_withhold += 0.015
            pattern_reasons.append(f"narrow_evasive(conf={p.confidence:.2f})")
        elif p.pattern_id == "direct_when_low_pressure_only":
            pattern_delta_bluff += 0.01
            pattern_reasons.append(f"direct_low_only(conf={p.confidence:.2f})")
        # Strategy patterns
        elif p.pattern_id == "deny_then_reframe":
            pattern_delta_bluff += 0.025
            pattern_reasons.append(f"deny_reframe(conf={p.confidence:.2f})")
        elif p.pattern_id == "partial_answer_then_evasion":
            pattern_delta_bluff += 0.02
            pattern_delta_withhold += 0.015
            pattern_reasons.append(f"partial_evade(conf={p.confidence:.2f})")
        elif p.pattern_id == "blame_displacement_under_pressure":
            pattern_delta_bluff += 0.02
            pattern_reasons.append(f"blame_displace(conf={p.confidence:.2f})")

    if abs(pattern_delta_bluff) > 0.001 or abs(pattern_delta_withhold) > 0.001:
        total_bluff = max(-_MAX_CONTEXT_DELTA, min(_MAX_CONTEXT_DELTA, pattern_delta_bluff))
        total_withhold = max(-_MAX_CONTEXT_DELTA, min(_MAX_CONTEXT_DELTA, pattern_delta_withhold))
        return ContextEffect(
            bluffing_delta=total_bluff,
            withholding_delta=total_withhold,
            rationale=f"patterns: {'; '.join(pattern_reasons)}",
        )

    return ContextEffect(rationale="context neutral or ambiguous")


def apply_context_conditioning(
    hypotheses: dict,
    effect: ContextEffect,
) -> None:
    """Apply bounded TargetContext effect to hypotheses. Mutates in place."""
    if abs(effect.bluffing_delta) < 0.001 and abs(effect.withholding_delta) < 0.001:
        return

    bluff = hypotheses.get("target_is_bluffing")
    if bluff:
        bluff.probability = max(0.0, min(1.0, bluff.probability + effect.bluffing_delta))

    withhold = hypotheses.get("target_is_withholding_info")
    if withhold:
        withhold.probability = max(0.0, min(1.0, withhold.probability + effect.withholding_delta))
